"""The sweep as one table, against the floor a model has to beat.

The audit's diagnostic -- rank-residualize the out-of-fold probability against the edge fraction
-- stops working once the crop is in. `brain_at_ap_edge` separates the classes at AUROC 0.997, so
anything that ranks the labels at all correlates with it, and residualizing against it deletes the
label signal along with the cue. Two replacements:

`rho(p, edge) in y` is the spearman computed *within* each class, controls then cases, where a
model still ranking by coverage shows up and a model ranking by the label does not.

`AUROC | anatomy` residualizes against a leave-one-out head on scalars that need no backbone --
SynthSeg's tissue volumes plus the brain volume left inside the crop. That head is the honest
floor here: the crop removed the export's cue but left the anatomy, and cases have larger
ventricles and slightly smaller brains.
"""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from omegaconf import OmegaConf
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / "output"
EXPLORE = EXP_DIR.parent / "explore_fomo_task5/explore.tsv"
CROPPED = OUT_DIR / "verify_crop_133.tsv"

# each config's two checkpoints adjacent, so the comparison reads down the table
ORDER = tuple(
    f"ckpt-{ckpt}_crop-{crop}" for crop in ("none", "test", "both") for ckpt in ("ptfull", "walnut")
)

ANATOMY = ("cortex_ml", "cortex_frac", "gm_wm_ratio", "ventricle_ml", "folding")

HEADER = (
    "| ckpt | crop train | crop test | AUROC | 95% CI | rho(p, edge) in y | AUROC \\| anatomy "
    "| selected C | ‖w‖ | time |"
)
RULE = "|---" * 10 + "|"

CKPT_NAMES = {
    "pretrain_full_90_10_h100": "pt-full",
    "sub-52k": "walnut-vitl",
}


def read_tsv(path: Path) -> dict[str, dict]:
    return {row["subject"]: row for row in csv.DictReader(path.open(), delimiter="\t")}


def anatomy_scores(subjects: list[str], y: np.ndarray) -> np.ndarray:
    """Out-of-fold probability from a head that never sees an image: SynthSeg's tissue volumes
    before the crop, and the brain volume left inside it."""
    explore, cropped = read_tsv(EXPLORE), read_tsv(CROPPED)
    X = np.array(
        [
            [float(explore[s][k]) for k in ANATOMY] + [float(cropped[s]["brain_ml"])]
            for s in subjects
        ]
    )
    oof = np.zeros(len(y))
    for train, test in LeaveOneOut().split(X):
        head = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                Cs=10,
                class_weight="balanced",
                max_iter=2000,
                l1_ratios=(0,),
                use_legacy_attributes=False,
            ),
        )
        head.fit(X[train], y[train])
        oof[test] = head.predict_proba(X[test])[:, list(head.classes_).index(1)]
    return oof


def residual_auroc(y: np.ndarray, p: np.ndarray, cue: np.ndarray) -> float:
    """AUROC of the model's ranking once the cue's ranking is regressed out of it."""
    ranks, cue_ranks = rankdata(p), rankdata(cue)
    residual = ranks - np.polyval(np.polyfit(cue_ranks, ranks, 1), cue_ranks)
    return float(roc_auc_score(y, residual))


def main() -> None:
    names = [name for name in ORDER if (OUT_DIR / name / "metrics.json").exists()]
    preds = {
        name: [
            json.loads(line) for line in (OUT_DIR / name / "preds.json").read_text().splitlines()
        ]
        for name in names
    }
    subjects = [row["subject"] for row in next(iter(preds.values()))]
    y = np.array([row["label"] for row in next(iter(preds.values()))])
    edge = np.array([float(read_tsv(EXPLORE)[s]["brain_at_ap_edge"]) for s in subjects])
    anatomy = anatomy_scores(subjects, y)

    print(f"model-free anatomy head: AUROC {roc_auc_score(y, anatomy):.3f}\n")
    print(HEADER)
    print(RULE)
    for name in names:
        cfg = OmegaConf.load(OUT_DIR / name / "config.yaml")
        m = json.loads((OUT_DIR / name / "metrics.json").read_text())
        head = joblib.load(OUT_DIR / name / "model/head.joblib")["head"][-1]
        p = np.array([row["pred"] for row in preds[name]])
        within = " / ".join(f"{spearmanr(p[y == c], edge[y == c]).statistic:+.2f}" for c in (0, 1))

        print(
            f"| {CKPT_NAMES[Path(cfg.ckpt_path).parent.name]} | {cfg.crop_ap} | {cfg.crop_test_ap} "
            f"| **{m['auroc']:.3f}** | {m['auroc_ci_low']:.3f} – {m['auroc_ci_high']:.3f} "
            f"| {within} | {residual_auroc(y, p, anatomy):.3f} "
            f"| {float(np.ravel(head.C_)[0]):.4g} | {np.linalg.norm(head.coef_):.3f} "
            f"| {m['run_time']:.0f}s |"
        )


if __name__ == "__main__":
    main()
