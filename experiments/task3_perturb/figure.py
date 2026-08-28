"""What corruption does to the task 3 head, and what training on it buys back.

Reads the run dirs written by launch.sh.

    uv run python experiments/task3_perturb/figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent
LIMITS = [0.0, 95.0]

# categorical slots 1-3, in fixed order by arm rather than by rank
RUNS = (
    ("noaug", "clean only", "#2a78d6"),
    ("aug", "+ resolution views", "#eb6834"),
    ("aug_v2", "+ resolution + scale", "#1baf7a"),
)
CONDITIONS = (
    ("preds.json", "FOMO\nout of fold"),
    ("camcan_preds.json", "CamCAN\nclean"),
    ("camcan-thick_slice_5mm_preds.json", "thick slice\n5mm"),
    ("camcan-acquired_at_2mm_preds.json", "acquired at\n2mm"),
    ("camcan-random_scale_preds.json", "random\nscale"),
)


def load(run: str, filename: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        json.loads(x) for x in (OUT_DIR / "output" / run / filename).read_text().splitlines() if x
    ]
    return (
        np.array([row["age"] for row in rows], dtype=float),
        np.array([row["pred"] for row in rows], dtype=float),
    )


def recede(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.3)
    ax.set_axisbelow(True)


def summary() -> None:
    stats = {}
    for run, _, _ in RUNS:
        for filename, _ in CONDITIONS:
            age, pred = load(run, filename)
            bias = (pred - age).mean()
            stats[run, filename] = (
                np.abs(pred - age).mean(),
                np.abs(pred - age - bias).mean(),
                bias,
                np.polyfit(age, pred, 1)[0],
            )

    panels = (
        (0, "MAE (years)", "What corruption costs"),
        (1, "MAE after removing the offset", "The clean-input cost is offset, not accuracy"),
        (3, "slope of prediction on age", "Resolution loss flattens the response"),
    )
    x = np.arange(len(CONDITIONS))
    width = 0.26

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (index, ylabel, title) in zip(axes, panels):
        for offset, (run, label, color) in enumerate(RUNS):
            values = [stats[run, filename][index] for filename, _ in CONDITIONS]
            ax.bar(x + (offset - 1) * width, values, width * 0.92, color=color, label=label)
        ax.set_xticks(x, [name for _, name in CONDITIONS], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        recede(ax)
        if index == 3:
            ax.axhline(1.0, c="k", ls=":", lw=0.9)
            ax.text(-0.4, 1.02, "no shrinkage", va="bottom", fontsize=7, color="0.35")
            ax.set_ylim(0, 1.15)
    axes[0].legend(fontsize=8, frameon=False)

    fig.suptitle("Task 3: training on corruption buys back what corruption costs")
    fig.text(
        0.5,
        -0.03,
        "random scale is a training view only for '+ resolution + scale'; "
        "the other two arms never saw it",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout()
    (OUT_DIR / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "figures/summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def scatter() -> None:
    shown = (RUNS[0], RUNS[2])
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(19, 4.0), sharex=True, sharey=True)
    for ax, (filename, name) in zip(axes, CONDITIONS):
        ax.plot(LIMITS, LIMITS, "k:", lw=0.9)
        for run, label, color in shown:
            age, pred = load(run, filename)
            slope, intercept = np.polyfit(age, pred, 1)
            ax.scatter(age, pred, s=7, alpha=0.35, c=color, lw=0)
            ax.plot(
                LIMITS,
                intercept + slope * np.array(LIMITS),
                color=color,
                lw=2,
                label=f"{label}\nslope {slope:.2f}, r {np.corrcoef(age, pred)[0, 1]:.3f}",
            )
        ax.set_title(name.replace("\n", " "), fontsize=10)
        ax.set_xlabel("age (years)")
        ax.set_xlim(LIMITS)
        ax.set_ylim(LIMITS)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("predicted age (years)")

    fig.suptitle("Task 3: what each corruption does to the age response")
    fig.tight_layout()
    (OUT_DIR / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "figures/scatter.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    summary()
    scatter()
