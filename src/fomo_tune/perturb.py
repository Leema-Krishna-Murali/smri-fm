import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output, smooth_image


def canonical(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Repacked, because the hf Nifti feature hands back a wrapper `as_reoriented` cannot rebuild."""
    return nib.as_closest_canonical(nib.Nifti1Image(img.dataobj, img.affine, img.header))


def resample(
    img: nib.Nifti1Image,
    voxel_sizes: float | tuple[float, float, float],
    fwhm: float | tuple[float, float, float] | None,
) -> nib.Nifti1Image:
    """Resample image to target voxel size, with optional smoothing."""
    data = np.asarray(img.dataobj, dtype=np.float32)
    mask = nib.Nifti1Image((data > 0).astype(np.float32), img.affine)

    if fwhm is not None:
        img = smooth_image(img, fwhm)

    resampled = resample_to_output(img, voxel_sizes, order=1)
    resampled_mask = resample_to_output(mask, voxel_sizes, order=1)

    # apply the mask to prevent nonzero data "bleeding" into the background.
    kept = np.where(np.asarray(resampled_mask.dataobj) > 0.5, np.asarray(resampled.dataobj), 0.0)
    return nib.Nifti1Image(kept.astype(np.float32), resampled.affine)


def acquired_at(img: nib.Nifti1Image, mm: float) -> nib.Nifti1Image:
    """An isotropic acquisition at `mm`."""
    return resample(canonical(img), mm, mm)


def thick_slice(img: nib.Nifti1Image, mm: float) -> nib.Nifti1Image:
    """An anisotropic axial acquisition: `mm` slices, in-plane resolution untouched."""
    img = canonical(img)
    in_plane = nib.affines.voxel_sizes(img.affine)[:2]
    return resample(img, (*in_plane, mm), (0, 0, mm))


def random_scale(img: nib.Nifti1Image, low: float, high: float) -> nib.Nifti1Image:
    """Claim a different voxel size, so the head reaches the backbone at a different physical size.

    Header only, so the array is untouched and this isolates apparent scale from any resampling.
    """
    scale = np.random.default_rng().uniform(low, high)
    return nib.Nifti1Image(img.dataobj, img.affine @ np.diag([scale, scale, scale, 1.0]))


PERTURBATIONS = {
    "thick_slice_5mm": lambda img: thick_slice(img, 5.0),
    "acquired_at_2mm": lambda img: acquired_at(img, 2.0),
    "random_scale": lambda img: random_scale(img, 0.9, 1.1),
}
