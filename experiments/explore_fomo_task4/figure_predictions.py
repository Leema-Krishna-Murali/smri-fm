"""What a baseline run actually claims, one row per label so the two do not overlap.

Four rows per subject: the head with the display box outlined, the box, then the nerve and the
vessel as hit / false / missed. Six subjects spanning the run's own range, best to worst.

The saved folds are cut at each subject's *oracle* pair of thresholds, chosen on that subject's own
labels, so this is the optimistic picture -- the row labels carry the global-cut Dice alongside.

    uv run python figure_predictions.py --run s4_c4_d04   # -> figures/s4_c4_d04_predictions.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch, Rectangle
from scipy import ndimage

from explore import TASK_DIR, side_labels, window

OUT_DIR = Path(__file__).parent
RUNS_DIR = OUT_DIR.parent / "fomo_tune_baseline_task4/output"

# the display box of `model_box.py`: 128 x 96 x 96 at 0.5mm, holding all 40 subjects' labels
BOX = np.array([128, 96, 96])
CENTRE = np.array([0.0, 4.0, -16.0])

LABEL_NAMES = ("nerve", "vessel")
CATEGORY_COLOURS = {"hit": (0.15, 0.9, 0.25), "false": (1.0, 0.25, 0.2), "missed": (0.3, 0.6, 1.0)}
N_SLICES = 6
SLICE_STRIDE = 2


def load_subject(subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Volume, labels, the mask bounding box, and the anchor -- all off the source nifti."""
    image = nib.load(TASK_DIR / f"preprocessed/{subject}/ses-01/t2w.nii.gz")
    seg = nib.load(TASK_DIR / f"labels/{subject}/ses-01/seg.nii.gz")
    data = np.asarray(image.dataobj, dtype=np.float32)
    labels = np.asarray(seg.dataobj).round().astype(np.uint8)

    mask = data > data.mean()
    anchor = np.array(ndimage.center_of_mass(mask))
    head = (
        np.array([np.percentile(idx, [0.5, 99.5]) for idx in np.nonzero(mask)]).round().astype(int)
    )
    return data, labels, head, anchor


def load_prediction(run: str, subject: str, seg: nib.Nifti1Image) -> np.ndarray:
    """The fold's sparse claim, back on the input grid it was written in."""
    saved = np.load(RUNS_DIR / f"{run}/folds/{subject}/prediction.npz")
    assert (saved["shape"] == seg.shape).all(), f"{subject}: prediction is not on the label grid"
    assert np.allclose(saved["affine"], seg.affine), f"{subject}: prediction affine differs"
    claimed = np.zeros(int(np.prod(saved["shape"])), dtype=np.uint8)
    claimed[saved["voxels"]] = saved["labels"]
    return claimed.reshape(saved["shape"])


def pick_subjects(run: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Best two, middle two and worst two by the oracle mean, which is what the folds are cut at."""
    curves = np.load(RUNS_DIR / f"{run}/curves.npz")
    dice = curves["dice"]
    subjects = curves["subjects"]

    shared_cut = np.unravel_index(dice.mean(axis=(0, 1)).argmax(), dice.shape[2:])
    shared = dice[:, :, shared_cut[0], shared_cut[1]]
    by_subject = dice.mean(axis=1)
    oracle = np.array(
        [
            dice[s, :, *np.unravel_index(subject.argmax(), subject.shape)]
            for s, subject in enumerate(by_subject)
        ]
    )

    order = np.argsort(-oracle.mean(axis=1))
    middle = len(order) // 2
    picks = [order[0], order[1], order[middle - 1], order[middle], order[-2], order[-1]]
    return [str(subjects[i]) for i in picks], oracle[picks], shared[picks]


def draw_panel(ax, plane: np.ndarray, limits: dict) -> None:
    ax.imshow(np.rot90(plane), cmap="gray", **limits)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_categories(ax, plane: np.ndarray, truth: np.ndarray, claimed: np.ndarray, limits: dict):
    """One label's plane painted hit / false / missed, filled rather than outlined: at 0.5mm the
    structures are a couple of voxels across and a contour of that is unreadable."""
    draw_panel(ax, plane, limits)
    overlay = np.zeros((*plane.shape, 4), dtype=np.float32)
    for category, hit in (
        ("hit", truth & claimed),
        ("false", ~truth & claimed),
        ("missed", truth & ~claimed),
    ):
        overlay[hit] = (*CATEGORY_COLOURS[category], 0.95)
    ax.imshow(np.rot90(overlay))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="s4_c4_d04")
    args = parser.parse_args()

    subjects, oracle, shared = pick_subjects(args.run)
    n_columns = 2 * N_SLICES
    n_rows = 4 * len(subjects)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(1.7 * n_columns, 1.8 * n_rows))

    for s, subject in enumerate(subjects):
        data, labels, head, anchor = load_subject(subject)
        seg = nib.load(TASK_DIR / f"labels/{subject}/ses-01/seg.nii.gz")
        zooms = np.array(seg.header.get_zooms()[:3])
        claimed = load_prediction(args.run, subject, seg)

        lo = np.round(anchor + CENTRE - BOX / 2).astype(int)
        hi = lo + BOX
        assert (lo >= 0).all() and (hi <= data.shape).all(), f"{subject}: box leaves the volume"
        box = tuple(slice(a, b) for a, b in zip(lo, hi))

        # a claim outside the display box is invisible here, so say how much of it is inside
        shown = [
            (claimed[box] == value).sum() / max((claimed == value).sum(), 1) for value in (1, 2)
        ]
        counts = " ".join(
            f"{name}={(claimed == value).sum()} ({100 * fraction:.0f}% in box)"
            for name, value, fraction in zip(LABEL_NAMES, (1, 2), shown)
        )
        print(f"{subject}: claimed {counts}")

        columns = []
        for _, nerve, _ in side_labels(labels, zooms):
            centre = int(round(nerve[:, 0].mean() - lo[0]))
            offsets = SLICE_STRIDE * (np.arange(N_SLICES) - N_SLICES // 2)
            columns.append(np.clip(centre + offsets, 0, BOX[0] - 1))
        columns = np.concatenate(columns)

        for c, k in enumerate(columns):
            full = data[lo[0] + k, head[1, 0] : head[1, 1], head[2, 0] : head[2, 1]]
            plane = data[lo[0] + k, box[1], box[2]]
            truth_plane = labels[lo[0] + k, box[1], box[2]]
            claimed_plane = claimed[lo[0] + k, box[1], box[2]]
            limits = window(plane)

            draw_panel(axes[4 * s][c], full, window(full))
            axes[4 * s][c].add_patch(
                Rectangle(
                    (lo[1] - head[1, 0] - 0.5, head[2, 1] - hi[2] - 0.5),
                    BOX[1],
                    BOX[2],
                    fill=False,
                    edgecolor="yellow",
                    linewidth=0.8,
                )
            )
            draw_panel(axes[4 * s + 1][c], plane, limits)
            for label, value in enumerate((1, 2)):
                draw_categories(
                    axes[4 * s + 2 + label][c],
                    plane,
                    truth_plane == value,
                    claimed_plane == value,
                    limits,
                )
            axes[4 * s][c].set_title(f"{'lr'[c // N_SLICES]} R+{k}", fontsize=6)

        axes[4 * s][0].set_ylabel(
            f"{subject}\noracle {oracle[s].mean():.3f} / global {shared[s].mean():.3f}", fontsize=7
        )
        axes[4 * s + 1][0].set_ylabel("box", fontsize=7)
        for label, name in enumerate(LABEL_NAMES):
            axes[4 * s + 2 + label][0].set_ylabel(
                f"{name}\n{oracle[s][label]:.3f} / {shared[s][label]:.3f}\n"
                f"{100 * shown[label]:.0f}% in box",
                fontsize=7,
            )

    # after `tight_layout`, so both sit above the panels rather than over the first row
    fig.tight_layout()
    fig.legend(
        handles=[Patch(color=colour, label=name) for name, colour in CATEGORY_COLOURS.items()],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=3,
        fontsize=9,
    )
    fig.suptitle(f"{args.run}: predictions at each subject's oracle cut", fontsize=11, y=1.005)
    (OUT_DIR / "figures").mkdir(exist_ok=True)
    path = OUT_DIR / f"figures/{args.run}_predictions.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
