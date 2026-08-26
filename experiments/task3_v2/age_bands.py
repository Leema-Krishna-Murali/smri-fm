"""Scores over fixed age bands, for two runs side by side.

r within a band is invariant to a constant bias, so it reports only ranking; MAE carries the
bias. `range only` is what the band's age spread alone would predict, from the whole cohort's
scatter about its own regression line -- if observed r sits near it, the band is not harder to
predict, only narrower.

    uv run python experiments/task3_v2/age_bands.py [run_a] [run_b]
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from omegaconf import OmegaConf

OUT_DIR = Path(__file__).parent / "output"

DEFAULT_RUNS = ("ckpt-walnut_depth-final_aug-none", "bal_ckpt-walnut_depth-final_aug-none")

BANDS = (
    ("all", 0, 200),
    ("young adults", 18, 30),
    ("adults", 30, 65),
    ("seniors", 65, 95),
    ("pre-senior", 18, 65),
    ("mid/clinical", 40, 70),
    ("narrow clinical", 50, 70),
)

COHORTS = (("preds.json", "FOMO 494 out of fold"), ("camcan_preds.json", "CamCAN 481"))


def load(run: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(x) for x in (OUT_DIR / run / name).read_text().splitlines() if x]
    return (
        np.array([r["age"] for r in rows], dtype=float),
        np.array([r["pred"] for r in rows], dtype=float),
    )


def band_stats(y, p, slope, scatter, lo, hi):
    m = (y >= lo) & (y < hi)
    if m.sum() < 5:
        return None
    yb, pb = y[m], p[m]
    sd_a = float(yb.std())
    return {
        "n": int(m.sum()),
        "sd": sd_a,
        "r": float(np.corrcoef(yb, pb)[0, 1]),
        "mae": float(np.abs(yb - pb).mean()),
        "bias": float((pb - yb).mean()),
        "range_only": slope * sd_a / np.sqrt((slope * sd_a) ** 2 + scatter**2),
    }


def main() -> None:
    runs = sys.argv[1:3] if len(sys.argv) > 2 else list(DEFAULT_RUNS)

    for run in runs:
        cfg = OmegaConf.load(OUT_DIR / run / "config.yaml")
        head = joblib.load(OUT_DIR / run / "model/head.joblib")
        print(
            f"{run}\n  balance_age={cfg.get('balance_age', False)}  "
            f"alpha={cfg.get('alpha', 'ridgecv')} -> selected {head[-1].alpha:g}"
        )

    for file_name, label in COHORTS:
        fits = {}
        for run in runs:
            y, p = load(run, file_name)
            slope, intercept = np.polyfit(y, p, 1)
            fits[run] = (y, p, slope, float(np.std(p - (slope * y + intercept))), intercept)

        print(f"\n=== {label} ===")
        for run in runs:
            _, _, slope, scatter, intercept = fits[run]
            print(
                f"  {run:<38} pred = {intercept:+.2f} + {slope:.3f} x age   scatter {scatter:.2f}y"
            )

        a, b = runs
        print(
            f"\n{'band':>16} {'n':>4} {'sd':>5} {'range only':>11} "
            f"{'r(a)':>7} {'r(b)':>7} {'dr':>7} {'MAE(a)':>8} {'MAE(b)':>8} {'dMAE':>7}"
        )
        for band, lo, hi in BANDS:
            sa = band_stats(*fits[a][:4], lo, hi)
            sb = band_stats(*fits[b][:4], lo, hi)
            if sa is None or sb is None:
                continue
            print(
                f"{band:>16} {sa['n']:>4} {sa['sd']:>5.1f} {sa['range_only']:>11.3f} "
                f"{sa['r']:>7.3f} {sb['r']:>7.3f} {sb['r'] - sa['r']:>+7.3f} "
                f"{sa['mae']:>8.2f} {sb['mae']:>8.2f} {sb['mae'] - sa['mae']:>+7.2f}"
            )
        print(f"  (a) {a}\n  (b) {b}")


if __name__ == "__main__":
    main()
