"""Three things the per-run panels do not show, across the whole sweep.

Left: where the two checkpoints disagree, subject by subject, for the matched mean-pooled config.
Middle: AUROC overall and within each 4th-modality block, so a cross-site deficit is visible.
Right: |logit| per fold, which is how hard the head was regularized -- the mean-pooled runs sit
pinned at the smallest C in the grid and predict ~0.5 for everyone, except where the inner CV
picked a different C for that fold.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score

EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / "output"
FIG_DIR = EXP_DIR / "figures"
TABLE = EXP_DIR.parent / "explore_fomo_task1/explore.tsv"

BASELINE = "mask-zero_norm-none_pool-mean"


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    info = {r["subject"]: r for r in csv.DictReader(TABLE.open(), delimiter="\t")}
    runs = {
        d.name: {
            json.loads(line)["subject"]: json.loads(line)["pred"]
            for line in (d / "preds.json").read_text().splitlines()
        }
        for d in sorted(OUT_DIR.iterdir())
    }
    subjects = sorted(info)
    y = np.array([int(info[s]["label"]) for s in subjects])
    swi = np.array([info[s]["fourth"] == "swi" for s in subjects])

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    ax = axes[0]
    a = np.array([runs[BASELINE][s] for s in subjects])
    b = np.array([runs["ckpt-walnut_" + BASELINE][s] for s in subjects])
    for label, marker in ((0, "s"), (1, "o")):
        for site, fill in ((True, "full"), (False, "none")):
            keep = (y == label) & (swi == site)
            ax.plot(
                a[keep],
                b[keep],
                marker,
                color=f"C{label}",
                fillstyle=fill,
                label=f"y={label} {'swi' if site else 't2s'}",
                ls="none",
            )
    for s, x, z in zip(subjects, a, b):
        ax.annotate(
            s.removeprefix("sub-"), (x, z), fontsize=7, xytext=(5, 0), textcoords="offset points"
        )
    lims = [min(a.min(), b.min()) - 0.003, max(a.max(), b.max()) + 0.003]
    ax.plot(lims, lims, ":", c="k", lw=0.8)
    ax.axhline(b[y == 0].max(), ls=":", c="C0", lw=0.8)
    ax.axvline(a[y == 0].max(), ls=":", c="C0", lw=0.8)
    ax.set_xlabel(f"pt-full p ({BASELINE})")
    ax.set_ylabel("walnut-vitl p (same config)")
    ax.set_title("dotted lines: each run's highest negative")
    ax.legend(fontsize=8)

    ax = axes[1]
    names = [n for n in runs if not n.startswith("ckpt-walnut_")]
    x = np.arange(len(names))
    for offset, prefix, hatch in ((-0.2, "", None), (0.2, "ckpt-walnut_", "//")):
        for shift, (mask, color) in enumerate(((slice(None), "C7"), (swi, "C0"), (~swi, "C1"))):
            p = np.array([[runs[prefix + n][s] for s in subjects] for n in names])
            auroc = [roc_auc_score(y[mask], row[mask]) for row in p]
            ax.bar(
                x + offset + (shift - 1) * 0.13,
                auroc,
                0.12,
                color=color,
                hatch=hatch,
                label=f"{'walnut' if prefix else 'pt-full'} {['all', 'swi', 't2s'][shift]}",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.axhline(0.5, ls=":", c="k", lw=0.8)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("AUROC")
    ax.set_title("hatched = walnut-vitl; grey all, blue swi (n=16), orange t2s (n=5)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[2]
    for i, name in enumerate(runs):
        p = np.array([runs[name][s] for s in subjects])
        ax.scatter(
            np.abs(logit(np.clip(p, 1e-6, 1 - 1e-6))),
            np.full(len(p), i),
            s=12,
            color="C1" if name.startswith("ckpt-walnut_") else "C0",
        )
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(runs, fontsize=7)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("|logit p| per subject")
    ax.set_title("head confidence: pinned near 0 means C_ hit the grid floor")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "compare.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
