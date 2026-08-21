"""K2 acquisition/domain views for FOMO task 3, at the original 1.0x Paul strength.

Draw 0 is the legacy `sum(ord(subject))` seed. Draw 1 uses SHA-256 subject entropy so the two
draws are nested and collision-free. Clean is included once; each family's weight is split across
the two draws so every subject still sums to one.
"""

import hashlib
from collections.abc import Generator

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

FIT_WEIGHTS = {
    "clean": 0.25,
    "acquisition": 0.15,
    "lowres_extreme": 0.10,
    "geometry": 0.15,
    "intensity_artifact": 0.15,
    "motion_coverage": 0.10,
    "domain": 0.10,
}
FAMILIES = tuple(name for name in FIT_WEIGHTS if name != "clean")
K2_DRAWS = (0, 1)
FAMILY_WEIGHTS = np.array([FIT_WEIGHTS[name] for name in FAMILIES], dtype=float)
K2_WEIGHTS = np.concatenate(
    ([FIT_WEIGHTS["clean"]], np.tile(FAMILY_WEIGHTS / len(K2_DRAWS), len(K2_DRAWS)))
)


def subject_entropy(subject: str, seed_namespace: str = "fomo") -> tuple[int, int, int, int]:
    digest = hashlib.sha256(f"{seed_namespace}\0{subject}".encode()).digest()[:16]
    words = (int.from_bytes(digest[offset : offset + 4], "little") for offset in range(0, 16, 4))
    return tuple(words)


def rng_for_view(
    seed: int, subject: str, variant_index: int, draw: int, seed_namespace: str = "fomo"
) -> np.random.Generator:
    if draw == 0:
        entropy = [seed, sum(map(ord, subject)), variant_index]
    else:
        assert draw == 1, draw
        entropy = [seed, *subject_entropy(subject, seed_namespace), variant_index, draw]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def resample_acquisition(
    data: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    target_spacing: np.ndarray,
    profile: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_shape = np.asarray(data.shape)
    spacing = nib.affines.voxel_sizes(affine)
    if profile == "gaussian":
        added_fwhm = np.sqrt(np.maximum(target_spacing**2 - spacing**2, 0))
        data = ndimage.gaussian_filter(data, added_fwhm / (2.355 * spacing))
    else:
        assert profile == "boxcar"
        for axis, width in enumerate(np.maximum(1, np.rint(target_spacing / spacing)).astype(int)):
            data = ndimage.uniform_filter1d(data, width, axis=axis, mode="nearest")
    new_shape = np.maximum(1, np.rint(old_shape * spacing / target_spacing)).astype(int)
    data = (
        F.interpolate(
            torch.from_numpy(data)[None, None],
            size=tuple(new_shape),
            mode="trilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )
    mask = (
        F.interpolate(
            torch.from_numpy(mask.astype(np.float32))[None, None],
            size=tuple(new_shape),
            mode="nearest-exact",
        )
        .squeeze()
        .bool()
        .numpy()
    )
    scale = old_shape / new_shape
    step = np.diag([*scale, 1.0])
    step[:3, 3] = 0.5 * scale - 0.5
    return data, mask, affine @ step


def change_contrast(
    data: np.ndarray, mask: np.ndarray, rng: np.random.Generator, strength: float
) -> np.ndarray:
    brain = data[mask]
    low, high = np.percentile(brain, (1, 99))
    gamma = rng.uniform(1 - 0.35 * strength, 1 + 0.45 * strength)
    data = low + (high - low) * np.clip((data - low) / (high - low), 0, 1) ** gamma
    coefficients = rng.uniform(-0.25 * strength, 0.25 * strength, 3)
    coordinates = np.meshgrid(
        *[np.linspace(-1, 1, n, dtype=np.float32) for n in data.shape], indexing="ij"
    )
    data *= np.clip(
        1
        + sum(
            coefficient * coordinate for coefficient, coordinate in zip(coefficients, coordinates)
        ),
        0.5,
        1.5,
    )
    sigma = 0.025 * strength * (high - low)
    n1 = rng.normal(0, sigma, data.shape).astype(np.float32)
    n2 = rng.normal(0, sigma, data.shape).astype(np.float32)
    return np.sqrt((np.maximum(data, 0) + n1) ** 2 + n2**2)


def change_pose(
    data: np.ndarray, mask: np.ndarray, rng: np.random.Generator, strength: float
) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = np.deg2rad(rng.uniform(-8 * strength, 8 * strength, 3))
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    scale = rng.uniform(1 - 0.10 * strength, 1 + 0.10 * strength)
    inverse = np.linalg.inv(scale * (rz @ ry @ rx))
    shift = rng.uniform(-8 * strength, 8 * strength, 3)
    centre = (np.asarray(data.shape) - 1) / 2
    offset = centre - inverse @ (centre + shift)
    data = ndimage.affine_transform(data, inverse, offset=offset, order=1)
    mask = ndimage.affine_transform(mask, inverse, offset=offset, order=0) > 0
    return data, mask


def augment_draw(
    row: dict, seed: int, draw: int, seed_namespace: str = "fomo"
) -> Generator[dict, None, None]:
    """One clean and six fixed-strength acquisition/domain views for a single draw."""
    image = row["t1w"]
    source_data = image.get_fdata(dtype=np.float32)
    source_mask = source_data > 0
    for variant_index, (variant, weight) in enumerate(FIT_WEIGHTS.items()):
        if variant == "clean":
            yield {
                **row,
                "subject": f"{row['subject']}__clean",
                "base_subject": row["subject"],
                "variant": variant,
                "fit_weight": weight,
                "draw": draw,
            }
            continue

        rng = rng_for_view(seed, row["subject"], variant_index, draw, seed_namespace)
        data, mask, affine = source_data.copy(), source_mask.copy(), image.affine.copy()

        if variant == "acquisition":
            data = change_contrast(data, mask, rng, 0.5)
            family = ("anisotropic", "isotropic", "reconstruction")[int(rng.integers(0, 3))]
            if family == "anisotropic":
                target = nib.affines.voxel_sizes(affine).copy()
                target[int(rng.integers(0, 3))] = rng.uniform(2, 5)
                data, mask, affine = resample_acquisition(data, mask, affine, target, "gaussian")
            elif family == "isotropic":
                target = np.full(3, rng.uniform(1.5, 2.5))
                data, mask, affine = resample_acquisition(data, mask, affine, target, "gaussian")
            else:
                fwhm = rng.uniform(1.5, 3)
                data = ndimage.gaussian_filter(
                    data, fwhm / (2.355 * nib.affines.voxel_sizes(affine))
                )
        elif variant == "lowres_extreme":
            data = change_contrast(data, mask, rng, 0.4)
            family = ("thick_slice", "dual_axis", "isotropic")[int(rng.integers(0, 3))]
            target = nib.affines.voxel_sizes(affine).copy()
            if family == "thick_slice":
                target[int(rng.integers(0, 3))] = rng.uniform(5, 9)
            elif family == "dual_axis":
                axes = rng.choice(3, 2, replace=False)
                target[axes[0]] = rng.uniform(4, 8)
                target[axes[1]] = rng.uniform(1.5, 2.5)
            else:
                target[:] = rng.uniform(2.5, 3.5)
            data, mask, affine = resample_acquisition(data, mask, affine, target, "boxcar")
        elif variant == "geometry":
            data, mask = change_pose(data, mask, rng, 2.0)
            data = change_contrast(data, mask, rng, 0.35)
        elif variant == "intensity_artifact":
            data = change_contrast(data, mask, rng, 1.5)
            fwhm = rng.uniform(1.5, 3.5)
            data = ndimage.gaussian_filter(data, fwhm / (2.355 * nib.affines.voxel_sizes(affine)))
        elif variant == "motion_coverage":
            data, mask = change_pose(data, mask, rng, 0.8)
            data = change_contrast(data, mask, rng, 0.7)
            axis = int(rng.integers(0, 3))
            shift = np.zeros(3)
            shift[axis] = rng.uniform(6, 16)
            mix = rng.uniform(0.15, 0.30)
            data = (1 - mix) * data + mix * ndimage.shift(data, shift, order=1)
            dropout_axis = int(rng.integers(0, 3))
            for _ in range(int(rng.integers(2, 6))):
                start = int(rng.integers(0, data.shape[dropout_axis] - 3))
                width = int(rng.integers(1, 4))
                index = [slice(None)] * 3
                index[dropout_axis] = slice(start, start + width)
                data[tuple(index)] *= rng.uniform(0.25, 0.75)
            crop_axis = int(rng.integers(0, 3))
            crop_mm = int(rng.integers(4, 13))
            crop_voxels = max(1, round(crop_mm / nib.affines.voxel_sizes(affine)[crop_axis]))
            index = [slice(None)] * 3
            index[crop_axis] = (
                slice(0, crop_voxels) if rng.random() < 0.5 else slice(-crop_voxels, None)
            )
            mask[tuple(index)] = False
            erosion = int(rng.integers(0, 3))
            if erosion:
                mask = ndimage.binary_erosion(mask, iterations=erosion)
        else:
            assert variant == "domain"
            data, mask = change_pose(data, mask, rng, 1.15)
            data = change_contrast(data, mask, rng, 1.0)
            axis = int(rng.integers(0, 3))
            shift = np.zeros(3)
            shift[axis] = rng.uniform(6, 12)
            mix = rng.uniform(0.12, 0.22)
            data = (1 - mix) * data + mix * ndimage.shift(data, shift, order=1)
            erosion = int(rng.integers(0, 3))
            if erosion:
                mask = ndimage.binary_erosion(mask, iterations=erosion)
            family = ("anisotropic", "isotropic", "reconstruction")[int(rng.integers(0, 3))]
            if family == "anisotropic":
                target = nib.affines.voxel_sizes(affine).copy()
                target[int(rng.integers(0, 3))] = rng.uniform(3, 6)
                data, mask, affine = resample_acquisition(data, mask, affine, target, "boxcar")
            elif family == "isotropic":
                target = np.full(3, rng.uniform(2, 3))
                data, mask, affine = resample_acquisition(data, mask, affine, target, "gaussian")
            else:
                fwhm = rng.uniform(3, 5)
                data = ndimage.gaussian_filter(
                    data, fwhm / (2.355 * nib.affines.voxel_sizes(affine))
                )

        data = np.where(mask, np.maximum(data, 0), 0).astype(np.float32)
        yield {
            "subject": f"{row['subject']}__{variant}__draw{draw}",
            "base_subject": row["subject"],
            "age": row["age"],
            "variant": variant,
            "fit_weight": weight / len(K2_DRAWS),
            "draw": draw,
            "t1w": nib.Nifti1Image(data, affine),
        }


def k2_views(row: dict, seed: int, seed_namespace: str = "fomo") -> Generator[dict, None, None]:
    """Clean once, then the six families at draw 0 and draw 1. Weights sum to one."""
    for view in augment_draw(row, seed, 0, seed_namespace):
        yield view
    for view in augment_draw(row, seed, 1, seed_namespace):
        if view["variant"] != "clean":
            yield view
