"""What a model would actually be handed: one fixed box per subject, placed without labels.

The box is 128 x 96 x 96 voxels at 0.5mm, centred (0, +4, -16) voxels from the subject's own
`data > data.mean()` centroid. That is the smallest round box holding all 40 subjects' labels with
margin: a 99% coverage, 95% confidence normal tolerance box over the label extremes is
117 x 84 x 89, and this clears it on every axis while staying a multiple of 16.

Three rows per subject -- full sagittal with the box outlined, the box, the box with labels -- over
six subjects spanning the hard cases rather than the first six. In-plane the panels are exactly the
model's input; the slices are chosen around each side's nerve, which a model could not do, so the
through-plane view is deliberately more favourable than inference.

    uv run python model_box.py     # -> figures/model_box.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage

from explore import TASK_DIR, draw, side_labels, window

OUT_DIR = Path(__file__).parent
BOX = np.array([128, 96, 96])
CENTRE = np.array([0.0, 4.0, -16.0])

# extremes of the S and A anchor residual, then the smallest and largest labels
SUBJECTS = ["sub-07", "sub-31", "sub-03", "sub-25", "sub-16", "sub-01"]
N_SLICES = 8
SLICE_STRIDE = 2


def load_subject(subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Volume, labels, the mask bounding box, and the anchor -- all off the source nifti."""
    image = nib.load(TASK_DIR / f"preprocessed/{subject}/ses-01/t2w.nii.gz")
    seg = nib.load(TASK_DIR / f"labels/{subject}/ses-01/seg.nii.gz")
    data = np.asarray(image.dataobj, dtype=np.float32)
    labels = np.asarray(seg.dataobj).astype(np.uint8)

    mask = data > data.mean()
    anchor = np.array(ndimage.center_of_mass(mask))
    # percentiles, not the bounding box: stray mask voxels out at the FOV edge undo the crop
    head = (
        np.array([np.percentile(idx, [0.5, 99.5]) for idx in np.nonzero(mask)]).round().astype(int)
    )
    return data, labels, head, anchor


def slice_columns(labels: np.ndarray, lo: np.ndarray, zooms: np.ndarray) -> np.ndarray:
    """A few slices around each side's nerve, in box coordinates."""
    columns = []
    for _, nerve, _ in side_labels(labels, zooms):
        centre = int(round(nerve[:, 0].mean() - lo[0]))
        offsets = SLICE_STRIDE * (np.arange(N_SLICES) - N_SLICES // 2)
        columns.append(np.clip(centre + offsets, 0, BOX[0] - 1))
    return np.concatenate(columns)


def main() -> None:
    (OUT_DIR / "figures").mkdir(exist_ok=True)
    n_columns = 2 * N_SLICES
    fig, axes = plt.subplots(
        3 * len(SUBJECTS), n_columns, figsize=(1.6 * n_columns, 1.7 * 3 * len(SUBJECTS))
    )

    for s, subject in enumerate(SUBJECTS):
        data, labels, head, anchor = load_subject(subject)
        zooms = np.array(
            nib.load(TASK_DIR / f"labels/{subject}/ses-01/seg.nii.gz").header.get_zooms()[:3]
        )

        lo = np.round(anchor + CENTRE - BOX / 2).astype(int)
        hi = lo + BOX
        assert (lo >= 0).all() and (hi <= data.shape).all(), f"{subject}: box leaves the volume"

        for c, k in enumerate(slice_columns(labels, lo, zooms)):
            # the full slice cropped to the head, so the panel is not mostly background
            full = data[lo[0] + k, head[1, 0] : head[1, 1], head[2, 0] : head[2, 1]]
            box = data[lo[0] + k, lo[1] : hi[1], lo[2] : hi[2]]
            box_seg = labels[lo[0] + k, lo[1] : hi[1], lo[2] : hi[2]]

            draw(axes[3 * s][c], full, None, window(full))
            # display column is y, display row is z reversed, both relative to the head crop
            axes[3 * s][c].add_patch(
                Rectangle(
                    (lo[1] - head[1, 0] - 0.5, head[2, 1] - hi[2] - 0.5),
                    BOX[1],
                    BOX[2],
                    fill=False,
                    edgecolor="yellow",
                    linewidth=0.8,
                )
            )
            draw(axes[3 * s + 1][c], box, None, window(box))
            draw(axes[3 * s + 2][c], box, box_seg, window(box))
            axes[3 * s][c].set_title(f"{'lr'[c // N_SLICES]} R+{k}", fontsize=6)

        axes[3 * s][0].set_ylabel(f"{subject}\nhead + box", fontsize=7)
        axes[3 * s + 1][0].set_ylabel("box", fontsize=7)
        axes[3 * s + 2][0].set_ylabel("box + seg", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures/model_box.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/model_box.png")


if __name__ == "__main__":
    main()
