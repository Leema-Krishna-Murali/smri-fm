"""Markdown table of sweep C, shallowest depth first. Sweeps A and B are `collect.py`."""

import json
from pathlib import Path

from omegaconf import OmegaConf

# measured exactly over all 40 subjects, for a prediction constant on cells of that size
CEILING_BY_CELL_MM = {8.0: 0.074, 4.0: 0.217, 2.0: 0.459, 1.0: 0.714, 0.5: 1.000}

OUT_DIR = Path(__file__).parent / "output"

HEADER = (
    "| run | scale | depth | cell mm | ceiling | dice | nerve | vessel | oracle "
    "| nerve cut | vessel cut | min |"
)
RULE = "|---" * 12 + "|"


def main() -> None:
    rows = []
    for run_dir in sorted(OUT_DIR.iterdir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        cfg = OmegaConf.load(run_dir / "config.yaml")
        if "depth" not in cfg:  # a sweep A or B run, which shares this output dir
            continue
        rows.append((cfg, json.loads(metrics_path.read_text()), 8.0 / (cfg.scale * cfg.subcell)))

    # the full model is depth 24, which a 0-23 pre-hook cannot reach, so it sorts as such
    rows.sort(key=lambda row: (24 if row[0].depth is None else row[0].depth, -row[0].scale))

    print(HEADER)
    print(RULE)
    for cfg, metrics, cell_mm in rows:
        ceiling = CEILING_BY_CELL_MM.get(round(cell_mm, 3))
        nerve_cut, vessel_cut = metrics["thresholds"]
        print(
            f"| {cfg.name} | {cfg.scale} | {'final' if cfg.depth is None else cfg.depth} "
            f"| {cell_mm:.2f} | {'-' if ceiling is None else f'{ceiling:.3f}'} "
            f"| **{metrics['dice']:.3f}** [{metrics['dice_ci_low']:.3f}, "
            f"{metrics['dice_ci_high']:.3f}] "
            f"| {metrics['dice_nerve']:.3f} | {metrics['dice_vessel']:.3f} "
            f"| {metrics['dice_oracle']:.3f} | {nerve_cut:.1e} | {vessel_cut:.1e} "
            f"| {metrics['run_time'] / 60:.0f} |"
        )


if __name__ == "__main__":
    main()
