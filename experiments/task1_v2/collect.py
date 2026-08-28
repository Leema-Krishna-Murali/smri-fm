import json
from pathlib import Path

import joblib
import numpy as np
from omegaconf import OmegaConf

OUT_DIR = Path(__file__).parent / "output"

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

# each config's two checkpoints adjacent, so the comparison reads down the table
ORDER = RUNS + tuple(f"ckpt-walnut_{run}" for run in RUNS)

HEADER = (
    "| ckpt | masking | pooling | norm vol | norm test vol | AUROC | 95% CI "
    "| selected C | ‖w‖ | time |"
)
RULE = "|---" * 10 + "|"

CKPT_NAMES = {
    "pretrain_full_90_10_h100": "pt-full",
    "sub-52k": "walnut-vitl",
}


def selected(state: dict) -> tuple[str, str]:
    """The C the inner CV picked and the coefficient norm, global head then local where there
    is one. These are the head fit on all n, not the fold heads, which the protocol does not save."""
    heads = [state["head"][-1]] + ([state["local_head"][-1]] if "local_head" in state else [])
    return (
        " / ".join(f"{float(np.ravel(head.C_)[0]):.4g}" for head in heads),
        " / ".join(f"{np.linalg.norm(head.coef_):.3f}" for head in heads),
    )


def main() -> None:
    print(HEADER)
    print(RULE)
    for name in ORDER:
        metrics_path = OUT_DIR / name / "metrics.json"
        if not metrics_path.exists():
            continue
        cfg = OmegaConf.load(OUT_DIR / name / "config.yaml")
        m = json.loads(metrics_path.read_text())
        C, norm = selected(joblib.load(OUT_DIR / name / "model/head.joblib"))
        print(
            f"| {CKPT_NAMES[Path(cfg.ckpt_path).parent.name]} | {cfg.masking} | {cfg.pooling} "
            f"| {cfg.normalize_volume} | {cfg.normalize_test_volume} "
            f"| **{m['auroc']:.3f}** | {m['auroc_ci_low']:.3f} – {m['auroc_ci_high']:.3f} "
            f"| {C} | {norm} | {m['run_time']:.0f}s |"
        )


if __name__ == "__main__":
    main()
