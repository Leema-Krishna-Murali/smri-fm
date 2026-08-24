import gzip
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

if TYPE_CHECKING:
    from datasets import Dataset

# `map` keys its cache on fn_kwargs and never on this file, so bump this after any changes
SYNTHSEG_HF_CACHE_VERSION = 1

SYNTHSEG_WEIGHTS = os.getenv("SYNTHSEG_WEIGHTS")


@lru_cache(maxsize=1)
def synthseg_predictor(device: str | None = None):
    from SynthSeg_pytorch import SynthSegPredictor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SynthSegPredictor(device=device, path_model=SYNTHSEG_WEIGHTS)


def synthseg(image: nib.Nifti1Image, device: str | None = None) -> nib.Nifti1Image:
    """SynthSeg's label map, resampled back onto the grid `data` came in on."""
    predictor = synthseg_predictor(device=device)
    seg, _, _, seg_affine, _ = predictor.segment(image)
    seg_image = nib.Nifti1Image(seg.astype(np.uint8), seg_affine)
    seg_image = resample_from_to(seg_image, image, order=0)
    seg_image.set_data_dtype(np.uint8)
    return seg_image


def applymask(image: nib.Nifti1Image, mask: nib.Nifti1Image) -> nib.Nifti1Image:
    data = image.get_fdata(dtype=np.float32)
    mask_data = mask.get_fdata(dtype=np.float32)
    data = np.where(mask_data > 0, data, 0.0)
    masked = nib.Nifti1Image(data, image.affine)
    masked.set_data_dtype(np.float32)
    return masked


def synthseg_strip_dataset(
    dataset: "Dataset",
    *,
    source: str,
    exclude_columns: list[str] | None = None,
    **map_kwargs,
) -> "Dataset":
    # imported here, not at the top, so the container needs no dataset stack to run `predict`
    from datasets import Features
    from datasets.features.nifti import Nifti

    features = Features({**dataset.features, "synthseg": Nifti()})
    map_kwargs.setdefault("writer_batch_size", 16)
    return dataset.map(
        _synthseg_strip_sample,
        features=features,
        fn_kwargs={
            "source": source,
            "exclude_columns": exclude_columns,
            "version": SYNTHSEG_HF_CACHE_VERSION,
        },
        desc="synthseg",
        **map_kwargs,
    )


def _synthseg_strip_sample(
    sample: dict[str, Any],
    *,
    source: str,
    exclude_columns: list[str] | None = None,
    version: int = SYNTHSEG_HF_CACHE_VERSION,
) -> dict[str, Any]:
    columns = [
        k
        for k, v in sample.items()
        if isinstance(v, nib.Nifti1Image) and (exclude_columns is None or k not in exclude_columns)
    ]
    source_img = repack(sample[source])
    seg = synthseg(source_img)

    stripped = sample.copy()
    for name in columns:
        img = repack(sample[name])
        assert img.shape == seg.shape, f"{name} is {img.shape}, mask is {seg.shape}"
        out = applymask(img, seg)
        stripped[name] = encode_nifti(out)

    stripped["synthseg"] = encode_nifti(seg)
    return stripped


def encode_nifti(img: nib.Nifti1Image) -> dict:
    # level 1 is nibabel's own default
    return {"path": None, "bytes": gzip.compress(img.to_bytes(), compresslevel=1)}


def repack(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.Nifti1Image(img.dataobj, img.affine, img.header)
