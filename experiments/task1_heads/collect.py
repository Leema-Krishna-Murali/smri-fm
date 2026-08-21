import json
from pathlib import Path

import joblib
import numpy as np
from omegaconf import OmegaConf

OUT_DIR = Path(__file__).parent / "output"

HEADS = (
    "head-cv-auc",
    "head-cv-auc-noscale",
    "head-cv-logloss",
    "head-fixed-C0.01",
    "head-fixed-C1",
    "head-fixed-C1-noscale",
    "head-fixed-C100",
    "head-lda",
    "head-lda-noscale",
)
# each head's variants adjacent, so a row block is one head across the three embedding configs
PREFIXES = ("", "ckpt-walnut_", "walnut-ensemble_")
ORDER = tuple(f"{prefix}{head}" for head in HEADS for prefix in PREFIXES)

CKPT_NAMES = {
    "pretrain_full_90_10_h100": "pt-full",
    "sub-52k": "walnut-vitl",
}

HEADER = (
    "| ckpt | embedding | head | scaler | AUROC | 95% CI | selected C | ‖w‖ | p spread | time |"
)
RULE = "|---" * 10 + "|"


def main() -> None:
    print(HEADER)
    print(RULE)
    for name in ORDER:
        metrics_path = OUT_DIR / name / "metrics.json"
        if not metrics_path.exists():
            continue
        cfg = OmegaConf.load(OUT_DIR / name / "config.yaml")
        m = json.loads(metrics_path.read_text())
        state = joblib.load(OUT_DIR / name / "model/head.joblib")
        # ensemble fits a global and a local probe from the same config; report both
        clfs = [state["head"][-1]] + ([state["local_head"][-1]] if "local_head" in state else [])
        selected = " / ".join(
            f"{float(np.ravel(clf.C_)[0]):.4g}" if hasattr(clf, "C_") else "—" for clf in clfs
        )
        norms = " / ".join(f"{np.linalg.norm(clf.coef_):.3f}" for clf in clfs)
        lines = (OUT_DIR / name / "preds.json").read_text().splitlines()
        preds = [json.loads(line)["pred"] for line in lines]
        norm = "norm both" if cfg.normalize_volume else "no norm"
        head = cfg.head if cfg.head != "logistic" else f"logistic C={cfg.head_C:g}"
        if cfg.head == "logistic_cv":
            head = f"logistic_cv ({cfg.head_scoring})"
        print(
            f"| {CKPT_NAMES[Path(cfg.ckpt_path).parent.name]} | {cfg.pooling} / {norm} "
            f"| {head} | {cfg.scaler} "
            f"| **{m['auroc']:.3f}** | {m['auroc_ci_low']:.3f} – {m['auroc_ci_high']:.3f} "
            f"| {selected} | {norms} "
            f"| {max(preds) - min(preds):.3f} | {m['run_time']:.0f}s |"
        )


if __name__ == "__main__":
    main()
