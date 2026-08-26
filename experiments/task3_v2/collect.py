"""The sweep as one table: the augmentation 2x2 at two depths, local CV and CamCAN transfer.

`run_time` is `cross_validate` only. Every view is embedded up front by `precompute`, which is
timed separately in the run log, so these numbers are seconds and do not compare with the
leaderboard's.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from omegaconf import OmegaConf

OUT_DIR = Path(__file__).parent / "output"

RUNS = tuple(
    f"{depth}_{aug}"
    for depth in ("depth-final", "depth-b16")
    for aug in ("aug-none", "aug-train", "aug-test", "aug-both")
)

ORDER = RUNS + tuple(f"ckpt-walnut_{run}" for run in RUNS)
ORDER = ORDER + tuple(f"bal_ckpt-walnut_{run}" for run in RUNS)

HEADER = (
    "| ckpt | depth | train aug | test aug | bal | r | 95% CI | MAE | 95% CI "
    "| camcan r | camcan MAE | alpha | ‖w‖ | time |"
)
RULE = "|---" * 14 + "|"

CKPT_NAMES = {
    "pretrain_full_90_10_h100": "pt-full",
    "sub-52k": "walnut-vitl",
}


def main() -> None:
    print(HEADER)
    print(RULE)
    for name in ORDER:
        metrics_path = OUT_DIR / name / "metrics.json"
        if not metrics_path.exists():
            continue
        cfg = OmegaConf.load(OUT_DIR / name / "config.yaml")
        m = json.loads(metrics_path.read_text())
        head = joblib.load(OUT_DIR / name / "model/head.joblib")

        camcan = m.get("evals", {}).get("camcan")
        camcan_r = f"{camcan['pearson_r']:.3f}" if camcan else "—"
        camcan_mae = f"{camcan['mae']:.2f}" if camcan else "—"

        print(
            f"| {CKPT_NAMES[Path(cfg.ckpt_path).parent.name]} "
            f"| {'final' if cfg.depth is None else cfg.depth} "
            f"| {cfg.train_aug} | {cfg.test_aug} "
            f"| {cfg.get('balance_age')} "
            f"| **{m['pearson_r']:.3f}** | {m['pearson_r_ci_low']:.3f} – {m['pearson_r_ci_high']:.3f} "
            f"| **{m['mae']:.2f}** | {m['mae_ci_low']:.2f} – {m['mae_ci_high']:.2f} "
            f"| {camcan_r} | {camcan_mae} "
            f"| {head[-1].alpha:.4g} | {np.linalg.norm(head[-1].coef_):.3f} "
            f"| {m['run_time']:.0f}s |"
        )


if __name__ == "__main__":
    main()
