"""Scores per run per condition, and the same predictions sliced into age bands.

r within a band is invariant to a constant bias, so it reports only ranking; MAE carries the
bias. Bands are a post-hoc slice of one full-range run -- the head is fixed and predictions are
per subject, so scoring a band is identical to having evaluated only that band.

    uv run python experiments/task3_perturb/collect.py
"""

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent / "output"

RUNS = (("noaug", "clean only"), ("aug", "+ res"), ("aug_v2", "+ res + scale"))
CONDITIONS = (
    ("preds.json", "fomo 494 out of fold"),
    ("camcan_preds.json", "camcan clean"),
    ("camcan-thick_slice_5mm_preds.json", "camcan thick slice 5mm"),
    ("camcan-acquired_at_2mm_preds.json", "camcan acquired at 2mm"),
    ("camcan-random_scale_preds.json", "camcan random scale"),
)
BANDS = (("all", 0, 200), ("young", 18, 40), ("mid", 40, 65), ("senior", 65, 95))


def load(run: str, filename: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = OUT_DIR / run / filename
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return (
        np.array([row["age"] for row in rows], dtype=float),
        np.array([row["pred"] for row in rows], dtype=float),
    )


def main() -> None:
    labels = " | ".join(f"r {label} | MAE {label}" for _, label in RUNS)
    print(f"| condition | n | {labels} | d MAE |")
    print("|---|---|" + "---|" * (2 * len(RUNS) + 1))
    for filename, condition in CONDITIONS:
        loaded = [load(run, filename) for run, _ in RUNS]
        if all(pair is None for pair in loaded):
            continue
        cells, maes = [], []
        for pair in loaded:
            if pair is None:
                cells.append("- | -")
                maes.append(None)
                continue
            age, pred = pair
            maes.append(np.abs(pred - age).mean())
            cells.append(f"{np.corrcoef(age, pred)[0, 1]:.3f} | **{maes[-1]:.2f}**")
        n = next(len(pair[0]) for pair in loaded if pair is not None)
        delta = f"{maes[-1] - maes[0]:+.2f}" if None not in maes else "-"
        print(f"| {condition} | {n} | " + " | ".join(cells) + f" | {delta} |")

    print("\n| run | condition | band | n | sd(age) | r | MAE | bias |")
    print("|---|---|---|---|---|---|---|---|")
    for run, label in RUNS:
        for filename, condition in CONDITIONS:
            loaded = load(run, filename)
            if loaded is None:
                continue
            age, pred = loaded
            for band, low, high in BANDS:
                keep = (age >= low) & (age < high)
                if keep.sum() < 5:
                    continue
                a, p = age[keep], pred[keep]
                print(
                    f"| {label} | {condition} | {band} | {keep.sum()} | {a.std():.1f} "
                    f"| {np.corrcoef(a, p)[0, 1]:.3f} | {np.abs(p - a).mean():.2f} "
                    f"| {(p - a).mean():+.2f} |"
                )


if __name__ == "__main__":
    main()
