import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from fomo_tune.backbone import fit_to_shape, rescale

Images = dict[str, nib.Nifti1Image]


class Task1SupportVolumeTransform:
    """Normalize Task 1 images using the supplied DWI's nonzero support."""

    def __init__(self, target_volume_ml: float, img_size: tuple[int, int, int]):
        assert target_volume_ml > 0
        self.target_volume_ml = target_volume_ml
        self.img_size = img_size

    def image_and_support(
        self, image: nib.Nifti1Image
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        image = nib.Nifti1Image(image.dataobj, image.affine, image.header)
        image = nib.as_closest_canonical(nib.funcs.squeeze_image(image))
        data = torch.from_numpy(np.ascontiguousarray(image.get_fdata(dtype=np.float32)))
        assert data.ndim == 3, f"expected a 3D volume, got {tuple(data.shape)}"
        support = data > 0
        affine = np.asarray(image.affine)

        source_spacing = image.header.get_zooms()[:3]
        if max(abs(value - 1.0) for value in source_spacing) > 0.05:
            data, affine = rescale(data, affine, source_spacing)
            support = F.interpolate(
                support[None, None].float(), size=data.shape, mode="nearest-exact"
            )[0, 0].bool()

        data, affine = fit_to_shape(data, affine, self.img_size)
        support, _ = fit_to_shape(support, np.eye(4), self.img_size)
        assert support.any()
        return data, support, affine

    def scale_volume(
        self, volume: torch.Tensor, scale: float, center: np.ndarray, order: int
    ) -> torch.Tensor:
        matrix = np.eye(3) / scale
        offset = center - matrix @ center
        scaled = ndimage.affine_transform(
            volume.numpy(),
            matrix=matrix,
            offset=offset,
            output_shape=self.img_size,
            order=order,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        return torch.from_numpy(scaled)

    def __call__(
        self, images: Images, modalities: list[str]
    ) -> dict[str, dict[str, torch.Tensor]]:
        grids = {modality: self.image_and_support(images[modality]) for modality in modalities}
        dwi_grid = grids.get("dwi_b1000")
        if dwi_grid is None:
            dwi_grid = self.image_and_support(images["dwi_b1000"])
        _, dwi_support, _ = dwi_grid

        center = torch.argwhere(dwi_support).float().mean(0).numpy()
        original_volume_ml = float(dwi_support.sum()) / 1000
        scale = float((self.target_volume_ml / original_volume_ml) ** (1 / 3))
        support = self.scale_volume(dwi_support, scale, center, order=0).bool()

        samples = {}
        for modality in modalities:
            data, _, affine = grids[modality]
            data = self.scale_volume(data, scale, center, order=1).float()
            values = data[support]
            mean = values.mean()
            std = values.std(correction=0).clamp_min(1e-6)
            normalized = torch.where(support, (data - mean) / std, 0.0)
            samples[modality] = {
                "image": normalized.unsqueeze(0),
                "mask": support.unsqueeze(0),
                "affine": torch.as_tensor(affine, dtype=torch.float32),
            }
        return samples
