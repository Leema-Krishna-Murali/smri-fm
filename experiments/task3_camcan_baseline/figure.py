"""The same task 3 head on the cohort it was fit on and on CamCAN, three panels each.

Reads the run dir written by launch.sh.

    uv run python experiments/task3_camcan_baseline/figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent
RUN_DIR = OUT_DIR / "output/task3_camcan"
LIMITS = [0.0, 95.0]


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in (RUN_DIR / name).read_text().splitlines()]
    age = np.array([row["age"] for row in rows], dtype=float)
    pred = np.array([row["pred"] for row in rows], dtype=float)
    return age, pred


def main() -> None:
    cohorts = {
        "FOMO, out-of-fold": (*load("preds.json"), "C0"),
        "CamCAN, transfer": (*load("camcan_preds.json"), "C3"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(LIMITS, LIMITS, "k:", lw=0.9, label="identity")
    for label, (age, pred, color) in cohorts.items():
        slope, intercept = np.polyfit(age, pred, 1)
        ax.scatter(age, pred, s=16, alpha=0.6, c=color, label=f"{label}, slope {slope:.3f}")
        ax.plot(LIMITS, intercept + slope * np.array(LIMITS), color, lw=1)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("prediction")
    ax.set_xlim(LIMITS)
    ax.set_ylim(LIMITS)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhline(0, c="k", ls=":", lw=0.9)
    for label, (age, pred, color) in cohorts.items():
        ax.scatter(age, pred - age, s=16, alpha=0.6, c=color, label=label)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("error (years)")
    ax.legend(fontsize=8)

    ax = axes[2]
    for label, (age, _, color) in cohorts.items():
        ax.hist(age, bins=np.arange(15, 95, 2), alpha=0.6, color=color, label=label)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("subjects")
    ax.legend(fontsize=8)

    fig.suptitle("Task 3 baseline: the same head on FOMO and on CamCAN")
    fig.tight_layout()
    (OUT_DIR / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "figures/scores.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
