"""Every run in one figure: out-of-fold p against the label and the three covariates that
explained the baseline -- lesion volume, brain-mask volume, lesion DWI conspicuity.

One row per run, in the order `collect.py` prints the table. Covariates come from
`experiments/explore_fomo_task1/explore.tsv`; only the probabilities differ between runs, so a
column reads straight down the sweep.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / "output"
FIG_DIR = EXP_DIR / "figures"
TABLE = EXP_DIR.parent / "explore_fomo_task1/explore.tsv"

RUNS = (
    "mask-mean_norm-none_pool-mean",
    "mask-zero_norm-none_pool-mean",
    "mask-zero_norm-test_pool-mean",
    "mask-zero_norm-train_pool-mean",
    "mask-zero_norm-both_pool-mean",
    "mask-zero_norm-none_pool-local",
    "mask-zero_norm-both_pool-local",
    "mask-zero_norm-none_pool-ensemble",
    "mask-zero_norm-both_pool-ensemble",
)

ORDER = RUNS + tuple(f"ckpt-walnut_{run}" for run in RUNS)

# colour is the label, marker is the 4th modality, a site proxy
GROUPS = (("swi", 0, "s"), ("swi", 1, "s"), ("t2s", 0, "o"), ("t2s", 1, "o"))
PANELS = (
    ("label_x", "label", False),
    ("volume_ml", "lesion volume (mL)", True),
    ("brain_ml", "brain-mask volume (mL)", False),
    ("z_dwi_b1000", "lesion mean dwi (z within brain mask)", False),
)


def load_subjects() -> dict[str, dict]:
    rows = {}
    for row in csv.DictReader(TABLE.open(), delimiter="\t"):
        record = {
            "subject": row["subject"],
            "label": int(row["label"]),
            "fourth": row["fourth"],
            "label_x": int(row["label"]) + (0.05 if row["fourth"] == "swi" else -0.05),
        }
        for key in ("brain_ml", "volume_ml", "z_dwi_b1000"):
            record[key] = float(row[key]) if row[key] else None
        rows[row["subject"]] = record
    return rows


def group_scatter(ax, rows: list[dict], x_key: str) -> None:
    for fourth, label, marker in GROUPS:
        subset = [r for r in rows if r["label"] == label and r["fourth"] == fourth]
        if not subset:
            continue
        ax.scatter(
            [r[x_key] for r in subset],
            [r["p"] for r in subset],
            s=22,
            color=f"C{label}",
            marker=marker,
            label=f"{fourth} y={label}",
        )
    for record in rows:
        ax.annotate(
            record["subject"].removeprefix("sub-"),
            (record[x_key], record["p"]),
            fontsize=6,
            xytext=(5, 0),
            textcoords="offset points",
        )


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    subjects = load_subjects()
    fig, axes = plt.subplots(len(ORDER), len(PANELS), figsize=(19, 2.4 * len(ORDER)), sharex="col")
    for row, name in enumerate(ORDER):
        rows = []
        for line in (OUT_DIR / name / "preds.json").read_text().splitlines():
            pred = json.loads(line)
            assert pred["label"] == subjects[pred["subject"]]["label"]
            rows.append(subjects[pred["subject"]] | {"p": pred["pred"]})
        positives = [r for r in rows if r["volume_ml"] is not None]
        metrics = json.loads((OUT_DIR / name / "metrics.json").read_text())
        y = np.array([r["label"] for r in rows])
        p = np.array([r["p"] for r in rows])

        for column, (x_key, x_label, log_x) in enumerate(PANELS):
            ax = axes[row, column]
            group_scatter(ax, rows if x_key in ("label_x", "brain_ml") else positives, x_key)
            if log_x:
                ax.set_xscale("log")
            # every row, not just the bottom one: the figure is too tall to scroll blind
            ax.tick_params(labelbottom=True)
            ax.set_xlabel(x_label, fontsize=8)

        axes[row, 0].axhline(p[y == 0].max(), ls=":", c="k", lw=0.8, label="max negative")
        axes[row, 0].set_xticks([0, 1])
        rho_brain = spearmanr(p, [r["brain_ml"] for r in rows]).statistic
        rho_lesion = spearmanr(
            [r["p"] for r in positives], [r["volume_ml"] for r in positives]
        ).statistic
        axes[row, 0].set_ylabel(
            f"{'walnut-vitl' if name.startswith('ckpt-walnut_') else 'pt-full'}\n"
            f"{name.removeprefix('ckpt-walnut_').replace('_', chr(10))}\n"
            f"\nout-of-fold p",
            fontsize=8,
        )
        axes[row, 1].set_title(
            f"AUROC {metrics['auroc']:.3f} "
            f"({metrics['auroc_ci_low']:.3f}–{metrics['auroc_ci_high']:.3f})",
            fontsize=9,
        )
        axes[row, 2].set_title(f"spearman(p, brain volume) = {rho_brain:+.2f}", fontsize=9)
        axes[row, 3].set_title(f"spearman(p, lesion volume) = {rho_lesion:+.2f}", fontsize=9)

    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Task 1 v2: out-of-fold scores, every run", y=1.002)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scores.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
