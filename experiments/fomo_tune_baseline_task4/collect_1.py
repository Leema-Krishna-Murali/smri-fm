import argparse
import json
import fnmatch
from pathlib import Path

from omegaconf import OmegaConf

# measured exactly over all 40 subjects, for a prediction constant on cells of that size
CEILING_BY_CELL_MM = {8.0: 0.074, 4.0: 0.217, 2.0: 0.459, 1.0: 0.714, 0.5: 1.000}

OUT_DIR = Path(__file__).parent / "output"

HEADER = (
    "| run | scale | subcell | cell mm | depth | ceiling | dice | nerve | vessel | oracle "
    "| nerve cut | vessel cut | min |"
)
RULE = "|---" * 13 + "|"

SWEEPS = {
    "depth": ("s?_d*",),
    "scale": ("s?_c?_d??",),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=sorted(SWEEPS), default="depth")
    args = parser.parse_args()
    globs = SWEEPS[args.sweep]

    rows = []
    for run_dir in sorted(OUT_DIR.iterdir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        cfg = OmegaConf.load(run_dir / "config.yaml")
        if not any(fnmatch.fnmatch(run_dir.name, glob) for glob in globs):
            continue
        metrics = json.loads(metrics_path.read_text())
        cell_mm = 8.0 / (cfg.scale * cfg.subcell)
        rows.append((cfg, metrics, cell_mm))

    rows.sort(
        key=lambda row: (
            row[0].scale,
            row[0].subcell,
            row[0].depth if row[0].depth is not None else 24,
        )
    )

    print(HEADER)
    print(RULE)
    for cfg, metrics, cell_mm in rows:
        ceiling = CEILING_BY_CELL_MM.get(round(cell_mm, 3))
        nerve_cut, vessel_cut = metrics["thresholds"]
        print(
            f"| {cfg.name} | {cfg.scale} | {cfg.subcell} | {cell_mm:.2f} "
            f"| {24 if cfg.depth is None else cfg.depth} "
            f"| {'-' if ceiling is None else f'{ceiling:.3f}'} "
            f"| **{metrics['dice']:.3f}** [{metrics['dice_ci_low']:.3f}, "
            f"{metrics['dice_ci_high']:.3f}] "
            f"| {metrics['dice_nerve']:.3f} | {metrics['dice_vessel']:.3f} "
            f"| {metrics['dice_oracle']:.3f} | {nerve_cut:.1e} | {vessel_cut:.1e} "
            f"| {metrics['run_time'] / 60:.0f} |"
        )


if __name__ == "__main__":
    main()
