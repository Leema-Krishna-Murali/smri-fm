"""Markdown table of every run in `output/`, geometry sweep first, then the block sweep."""

import json
from pathlib import Path

from omegaconf import OmegaConf

# measured exactly over all 40 subjects, for a prediction constant on cells of that size
CEILING_BY_CELL_MM = {8.0: 0.074, 4.0: 0.217, 2.0: 0.459, 1.0: 0.714, 0.5: 1.000}

OUT_DIR = Path(__file__).parent / "output"

HEADER = (
    "| run | scale | subcell | cell mm | block | ceiling | dice | nerve | vessel | oracle "
    "| thr | min |"
)
RULE = "|---" * 12 + "|"


def main() -> None:
    rows = []
    for run_dir in sorted(OUT_DIR.iterdir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        cfg = OmegaConf.load(run_dir / "config.yaml")
        metrics = json.loads(metrics_path.read_text())
        cell_mm = 8.0 / (cfg.scale * cfg.subcell)
        rows.append((cfg, metrics, cell_mm))

    rows.sort(
        key=lambda row: (
            row[0].block if row[0].block is not None else -1,
            -row[0].scale,
            row[0].subcell,
        )
    )

    print(HEADER)
    print(RULE)
    for cfg, metrics, cell_mm in rows:
        ceiling = CEILING_BY_CELL_MM.get(round(cell_mm, 3))
        print(
            f"| {cfg.name} | {cfg.scale} | {cfg.subcell} | {cell_mm:.2f} "
            f"| {'final' if cfg.block is None else cfg.block} "
            f"| {'-' if ceiling is None else f'{ceiling:.3f}'} "
            f"| **{metrics['dice']:.3f}** [{metrics['dice_ci_low']:.3f}, "
            f"{metrics['dice_ci_high']:.3f}] "
            f"| {metrics['dice_nerve']:.3f} | {metrics['dice_vessel']:.3f} "
            f"| {metrics['dice_oracle']:.3f} | {metrics['threshold']:.1e} "
            f"| {metrics['run_time'] / 60:.0f} |"
        )


if __name__ == "__main__":
    main()
