"""What is task 4 made of, and what does a model have to be pointed at?

Task 4 is 40 identical 0.5mm 360x512x512 T2w volumes with two labelled structures, the trigeminal
nerve (1) and the vessel touching it (2), on both sides. Everything here is measured on the native
grid: the structures are a few voxels thick and the 1mm transform is not the frame to look at them
in.

Three questions. What are the structures -- size, shape, orientation, and how they look against
their surroundings. Where are they -- how far the label moves across subjects in the voxel index
frame a crop would live in. And whether the free anatomical anchor, the centroid of the same
`data > data.mean()` mask the transform uses, takes any of that movement out.

    uv run python explore.py     # -> explore.tsv, figures/*.png, cache in output/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).parents[2]
TASK_DIR = ROOT / "data/fomo_eval/Task_4"
OUT_DIR = Path(__file__).parent
CACHE_DIR = OUT_DIR / "output/cache"

# a label-free box, centred on the mean label centroid over the cohort and sized to hold every
# subject's labels; `cache_subject` asserts the containment it claims
CROP_CENTRE = (175, 237, 246)
CROP_SIZE = (192, 176, 192)

NEIGHBOURHOOD = 10  # voxels of margin defining "local background" around a side's labels
CONNECTIVITY = np.ones((3, 3, 3))
N_SLICES = 8
ZOOM_MARGIN = 20
PLANE_HALF = 40


def subjects() -> list[str]:
    return sorted(p.name for p in (TASK_DIR / "preprocessed").iterdir())


def crop_slices() -> tuple[slice, slice, slice]:
    return tuple(
        slice(centre - size // 2, centre + size - size // 2)
        for centre, size in zip(CROP_CENTRE, CROP_SIZE)
    )


def cache_subject(subject: str) -> None:
    """Crop to the fixed box, and take the anchor off the full volume before throwing it away."""
    image = nib.load(TASK_DIR / f"preprocessed/{subject}/ses-01/t2w.nii.gz")
    seg = nib.load(TASK_DIR / f"labels/{subject}/ses-01/seg.nii.gz")
    assert nib.aff2axcodes(image.affine) == ("R", "A", "S"), f"{subject} is not RAS"
    assert np.allclose(image.affine, seg.affine, atol=1e-3), f"{subject} seg is on another grid"

    # a few degrees oblique, so index axes are not world axes; distances in `zooms` still hold
    # because the rotation is rigid and the three scales agree to 2%
    rotation = image.affine[:3, :3]
    zooms = np.array(image.header.get_zooms()[:3])
    assert np.allclose(rotation.T @ rotation, np.diag(zooms**2), atol=1e-2), f"{subject} is skewed"

    data = np.asarray(image.dataobj, dtype=np.float32)
    labels = np.asarray(seg.dataobj).astype(np.uint8)

    mask = data > data.mean()
    mask_centroid = np.array(ndimage.center_of_mass(mask))
    box = np.array([(idx.min(), idx.max()) for idx in np.nonzero(mask)])
    mask_box_centre = box.mean(axis=1)

    foreground = np.argwhere(labels > 0)
    window = crop_slices()
    inside = np.array([[w.start, w.stop - 1] for w in window])
    assert (foreground >= inside[:, 0]).all() and (foreground <= inside[:, 1]).all(), (
        f"{subject}: labels reach outside the fixed crop box"
    )

    np.savez_compressed(
        CACHE_DIR / f"{subject}.npz",
        image=data[window].astype(np.int16),
        seg=labels[window],
        zooms=zooms,
        crop_origin=np.array([w.start for w in window]),
        mask_centroid=mask_centroid,
        mask_box_centre=mask_box_centre,
        mask_voxels=np.array(mask.sum()),
    )
    print(f"cached {subject}", flush=True)


def build_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for subject in subjects():
        if not (CACHE_DIR / f"{subject}.npz").exists():
            cache_subject(subject)


def load(subject: str) -> dict:
    with np.load(CACHE_DIR / f"{subject}.npz") as cached:
        return {key: cached[key] for key in cached.files}


def shape_stats(coords: np.ndarray, zooms: np.ndarray) -> dict:
    """Principal axis of one component, and its extent along that axis and across it, in mm."""
    points = coords * zooms
    centred = points - points.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov(centred.T))
    projections = centred @ axes[:, ::-1]
    extents = projections.max(axis=0) - projections.min(axis=0)
    principal = axes[:, -1]
    return {
        "length_mm": float(extents[0]),
        "thickness_mm": float(extents[1:].mean()),
        "axis_r": float(abs(principal[0])),
        "axis_a": float(abs(principal[1])),
        "axis_s": float(abs(principal[2])),
    }


def measure_subject(subject: str) -> list[dict]:
    """One row per side: the nerve there, the vessel assigned to it, and where they sit."""
    cached = load(subject)
    image = cached["image"].astype(np.float32)
    seg = cached["seg"]
    zooms = cached["zooms"]
    origin = cached["crop_origin"]

    nerves, n_nerves = ndimage.label(seg == 1, structure=CONNECTIVITY)
    vessels, n_vessels = ndimage.label(seg == 2, structure=CONNECTIVITY)
    nerve_ids = np.arange(1, n_nerves + 1)
    vessel_ids = np.arange(1, n_vessels + 1)

    nerve_coords = {i: np.argwhere(nerves == i) for i in nerve_ids}
    vessel_coords = {i: np.argwhere(vessels == i) for i in vessel_ids}
    # the two largest nerve components are the two sides; a third is a fragment of one of them
    sides = sorted(nerve_ids, key=lambda i: -len(nerve_coords[i]))[:2]
    sides = sorted(sides, key=lambda i: nerve_coords[i][:, 0].mean())

    nerve_centres = {i: nerve_coords[i].mean(axis=0) * zooms for i in sides}
    assigned = {i: [] for i in sides}
    for vessel_id, coords in vessel_coords.items():
        centre = coords.mean(axis=0) * zooms
        nearest = min(sides, key=lambda i: np.linalg.norm(centre - nerve_centres[i]))
        assigned[nearest].append(vessel_id)

    rows = []
    for name, nerve_id in zip(("left", "right"), sides):
        nerve = nerve_coords[nerve_id]
        vessel = (
            np.concatenate([vessel_coords[i] for i in assigned[nerve_id]])
            if assigned[nerve_id]
            else np.zeros((0, 3), dtype=int)
        )

        both = np.concatenate([nerve, vessel])
        lo = np.maximum(both.min(axis=0) - NEIGHBOURHOOD, 0)
        hi = np.minimum(both.max(axis=0) + NEIGHBOURHOOD + 1, image.shape)
        box = image[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]

        centroid = nerve.mean(axis=0) + origin
        rows.append(
            {
                "subject": subject,
                "side": name,
                "n_nerve_components": n_nerves,
                "n_vessel_components": n_vessels,
                "nerve_voxels": len(nerve),
                "vessel_voxels": len(vessel),
                "nerve_mm3": float(len(nerve) * np.prod(zooms)),
                "vessel_mm3": float(len(vessel) * np.prod(zooms)),
                **shape_stats(nerve, zooms),
                "contact_mm": float(
                    cKDTree(vessel * zooms).query(nerve * zooms)[0].min() if len(vessel) else np.nan
                ),
                "nerve_intensity": float(np.median(image[tuple(nerve.T)])),
                "vessel_intensity": float(np.median(image[tuple(vessel.T)]))
                if len(vessel)
                else np.nan,
                "local_p50": float(np.percentile(box, 50)),
                "local_p95": float(np.percentile(box, 95)),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "anchor_x": float(cached["mask_centroid"][0]),
                "anchor_y": float(cached["mask_centroid"][1]),
                "anchor_z": float(cached["mask_centroid"][2]),
                "box_anchor_x": float(cached["mask_box_centre"][0]),
                "box_anchor_y": float(cached["mask_box_centre"][1]),
                "box_anchor_z": float(cached["mask_box_centre"][2]),
                "zoom": float(zooms.mean()),
            }
        )
    return rows


def measure() -> pd.DataFrame:
    table = pd.DataFrame([row for subject in subjects() for row in measure_subject(subject)])
    for axis in "xyz":
        table[f"residual_{axis}"] = table[f"centroid_{axis}"] - table[f"anchor_{axis}"]
        table[f"box_residual_{axis}"] = table[f"centroid_{axis}"] - table[f"box_anchor_{axis}"]
    table["nerve_contrast"] = (table["nerve_intensity"] - table["local_p50"]) / (
        table["local_p95"] - table["local_p50"]
    )
    table["vessel_contrast"] = (table["vessel_intensity"] - table["local_p50"]) / (
        table["local_p95"] - table["local_p50"]
    )
    return table


def report_spread(table: pd.DataFrame) -> None:
    """How much a crop has to allow for, in mm, under each label-free placement rule."""
    zoom = table["zoom"].mean()
    print("\n-- spread of the nerve centroid across subjects, mm, within side")
    rules = {
        "fixed index": ["centroid_x", "centroid_y", "centroid_z"],
        "mask centroid": ["residual_x", "residual_y", "residual_z"],
        "mask bbox centre": ["box_residual_x", "box_residual_y", "box_residual_z"],
    }
    for name, keys in rules.items():
        centred = (table[keys] - table.groupby("side")[keys].transform("mean")).to_numpy() * zoom
        sd = centred.std(axis=0)
        span = centred.max(axis=0) - centred.min(axis=0)
        print(
            f"{name:18s} sd=({sd[0]:5.1f} {sd[1]:5.1f} {sd[2]:5.1f})  "
            f"range=({span[0]:5.1f} {span[1]:5.1f} {span[2]:5.1f})",
            flush=True,
        )

    print("\n-- structure, over 80 sides")
    for key in (
        "nerve_mm3",
        "vessel_mm3",
        "length_mm",
        "thickness_mm",
        "contact_mm",
        "axis_r",
        "axis_a",
        "axis_s",
        "nerve_contrast",
        "vessel_contrast",
    ):
        values = table[key].dropna()
        print(
            f"{key:16s} median={values.median():7.2f}  "
            f"[{values.min():7.2f}, {values.max():7.2f}]  n={len(values)}",
            flush=True,
        )


def report_boxes() -> None:
    """The smallest crop that holds every subject's labels, under each label-free placement."""
    bounds = {"fixed index": [], "mask centroid": [], "mask bbox centre": []}
    for subject in subjects():
        cached = load(subject)
        coords = np.argwhere(cached["seg"] > 0) + cached["crop_origin"]
        zooms = cached["zooms"]
        for name, anchor in (
            ("fixed index", np.zeros(3)),
            ("mask centroid", cached["mask_centroid"]),
            ("mask bbox centre", cached["mask_box_centre"]),
        ):
            relative = coords - anchor
            bounds[name].append(np.stack([relative.min(axis=0), relative.max(axis=0)]))

    print("\n-- smallest crop holding all 40 subjects' labels, mm (and 0.5mm voxels)")
    for name, values in bounds.items():
        stacked = np.stack(values)
        size = (stacked[:, 1].max(axis=0) - stacked[:, 0].min(axis=0)) * zooms
        voxels = (size / zooms).round().astype(int)
        print(
            f"{name:18s} {size[0]:5.1f} x {size[1]:5.1f} x {size[2]:5.1f} mm  "
            f"({voxels[0]} x {voxels[1]} x {voxels[2]})  {np.prod(size) / 1000:7.1f} mL",
            flush=True,
        )


def window(panel: np.ndarray) -> dict:
    lo, hi = np.percentile(panel, [1.0, 99.8])
    return {"vmin": lo, "vmax": max(hi, lo + 1)}


def draw(ax, plane: np.ndarray, seg_plane: np.ndarray | None, limits: dict) -> None:
    ax.imshow(np.rot90(plane), cmap="gray", **limits)
    if seg_plane is not None:
        for value, colour in ((1, "red"), (2, "cyan")):
            if (seg_plane == value).any():
                ax.contour(
                    np.rot90(seg_plane == value), levels=[0.5], colors=colour, linewidths=0.6
                )
    ax.set_xticks([])
    ax.set_yticks([])


def slice_columns(seg: np.ndarray) -> np.ndarray:
    z = np.nonzero((seg > 0).any(axis=(0, 1)))[0]
    return np.linspace(z.min(), z.max(), N_SLICES).round().astype(int)


def figure_grid(name: str, zoom: bool) -> None:
    """Two rows per subject, plain over annotated, axial slices spanning the labelled slab."""
    names = subjects()
    fig, axes = plt.subplots(
        2 * len(names),
        N_SLICES,
        figsize=((2.0 if zoom else 1.5) * N_SLICES, (1.3 if zoom else 1.5) * 2 * len(names)),
        squeeze=False,
    )
    for s, subject in enumerate(names):
        cached = load(subject)
        image, seg = cached["image"].astype(np.float32), cached["seg"]
        columns = slice_columns(seg)

        if zoom:
            foreground = np.argwhere(seg > 0)
            lo = np.maximum(foreground.min(axis=0) - ZOOM_MARGIN, 0)
            hi = np.minimum(foreground.max(axis=0) + ZOOM_MARGIN, image.shape)
            image = image[lo[0] : hi[0], lo[1] : hi[1]]
            seg = seg[lo[0] : hi[0], lo[1] : hi[1]]
        else:
            limits = window(image)

        for c, k in enumerate(columns):
            if zoom:
                limits = window(image[:, :, k])
            draw(axes[2 * s][c], image[:, :, k], None, limits)
            draw(axes[2 * s + 1][c], image[:, :, k], seg[:, :, k], limits)
            axes[2 * s][c].set_title(f"z={k}", fontsize=5)
        axes[2 * s][0].set_ylabel(f"{subject}\nt2w", fontsize=6)
        axes[2 * s + 1][0].set_ylabel("+ seg", fontsize=6)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"figures/{name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def figure_planes(table: pd.DataFrame) -> None:
    """Three planes through each nerve, so the structures can be told apart by their shape."""
    names = subjects()
    fig, axes = plt.subplots(len(names), 6, figsize=(2.0 * 6, 2.2 * len(names)), squeeze=False)
    for s, subject in enumerate(names):
        cached = load(subject)
        image, seg = cached["image"].astype(np.float32), cached["seg"]
        rows = table[table["subject"] == subject]

        for side, (_, row) in enumerate(rows.iterrows()):
            centre = (
                (
                    np.array([row["centroid_x"], row["centroid_y"], row["centroid_z"]])
                    - cached["crop_origin"]
                )
                .round()
                .astype(int)
            )
            lo = np.maximum(centre - PLANE_HALF, 0)
            hi = np.minimum(centre + PLANE_HALF, image.shape)
            views = (
                (
                    image[lo[0] : hi[0], lo[1] : hi[1], centre[2]],
                    seg[lo[0] : hi[0], lo[1] : hi[1], centre[2]],
                ),
                (
                    image[lo[0] : hi[0], centre[1], lo[2] : hi[2]],
                    seg[lo[0] : hi[0], centre[1], lo[2] : hi[2]],
                ),
                (
                    image[centre[0], lo[1] : hi[1], lo[2] : hi[2]],
                    seg[centre[0], lo[1] : hi[1], lo[2] : hi[2]],
                ),
            )
            for v, (plane, seg_plane) in enumerate(views):
                ax = axes[s][3 * side + v]
                draw(ax, plane, seg_plane, window(plane))
                if s == 0:
                    ax.set_title(f"{row['side']} {['axial', 'coronal', 'sagittal'][v]}", fontsize=8)
        axes[s][0].set_ylabel(subject, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures/planes.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def figure_geometry(table: pd.DataFrame) -> None:
    zoom = table["zoom"].mean()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    ax = axes[0, 0]
    ax.scatter(table["nerve_mm3"], table["vessel_mm3"])
    ax.set_xlabel("nerve volume (mm³)")
    ax.set_ylabel("vessel volume (mm³)")

    ax = axes[0, 1]
    ax.scatter(table["length_mm"], table["thickness_mm"])
    ax.set_xlabel("nerve length along its principal axis (mm)")
    ax.set_ylabel("nerve mean cross extent (mm)")

    ax = axes[0, 2]
    for key, label in (("axis_r", "R"), ("axis_a", "A"), ("axis_s", "S")):
        ax.hist(table[key], bins=20, range=(0, 1), histtype="step", label=label)
    ax.set_xlabel("|principal axis| component")
    ax.set_ylabel("sides")
    ax.legend()

    ax = axes[1, 0]
    for keys, label in (
        (["centroid_y", "centroid_z"], "fixed index"),
        (["residual_y", "residual_z"], "mask centroid"),
    ):
        centred = (table[keys] - table.groupby("side")[keys].transform("mean")).to_numpy() * zoom
        ax.scatter(centred[:, 0], centred[:, 1], label=label)
    ax.set_xlabel("A displacement from cohort mean (mm)")
    ax.set_ylabel("S displacement from cohort mean (mm)")
    ax.legend()

    ax = axes[1, 1]
    ax.hist(table["contact_mm"].dropna(), bins=20)
    ax.set_xlabel("nerve-to-vessel minimum distance (mm)")
    ax.set_ylabel("sides")

    ax = axes[1, 2]
    ax.scatter(table["nerve_contrast"], table["vessel_contrast"])
    ax.axline((0, 0), slope=1, ls=":", c="k", lw=0.8)
    ax.set_xlabel("nerve intensity, (median - local p50) / (local p95 - p50)")
    ax.set_ylabel("vessel intensity, same scale")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures/geometry.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    (OUT_DIR / "figures").mkdir(exist_ok=True)
    build_cache()
    table = measure()
    table.to_csv(OUT_DIR / "explore.tsv", sep="\t", index=False, float_format="%.3f")
    report_spread(table)
    report_boxes()
    figure_grid("subjects", zoom=False)
    figure_grid("zoom", zoom=True)
    figure_planes(table)
    figure_geometry(table)


if __name__ == "__main__":
    main()
