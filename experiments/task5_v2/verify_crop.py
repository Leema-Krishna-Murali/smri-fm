"""Does the crop actually remove the coverage cue?

`main_task5.crop_ap` cuts every subject to a common anterior-posterior slab. This scores the
scalars that carried the confound on the volumes the model itself sees -- the stripped nifti the
crop runs on, and the canvas the transform hands the backbone -- so that a model score afterwards
can be read as a model score. `brain_at_ap_edge` is the one to watch: 0.997 uncropped, and it has
to fall to chance.

`auroc_with_perm` and `loo_auroc` are forked from `experiments/explore_fomo_task5/explore.py`, so
the two experiments score a scalar the same way without importing across folders.

    uv run python experiments/task5_v2/verify_crop.py \
        | tee experiments/task5_v2/output/verify_crop.log
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fomo_tune.backbone import SmriMaeTransform
from fomo_tune.datasets import load_fomo_task5
from fomo_tune.main_task5 import AP_EXTENT_MM, crop_ap
from fomo_tune.synthseg import repack, synthseg_strip_dataset

COVERAGE = ["brain_at_ap_edge", "ap_margin_mm", "fov_ap_mm", "n_slices", "brain_centre_ap"]
# brain volume rides along because the crop changes it, but it is anatomy rather than a cue the
# export left behind, so it is scored on its own and not pooled into the coverage head
ANATOMY = ["brain_ml"]


def auroc_with_perm(y: np.ndarray, value: np.ndarray, rng, n_perm: int = 10000) -> tuple:
    """AUROC of one scalar, and the two-sided permutation p of |AUROC - 0.5|."""
    observed = roc_auc_score(y, value)
    null = [roc_auc_score(rng.permutation(y), value) for _ in range(n_perm)]
    p = (np.abs(np.array(null) - 0.5) >= abs(observed - 0.5)).mean()
    return observed, p


def loo_auroc(X: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out logistic head, so a group of scalars is scored the way the model is."""
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
    return roc_auc_score(y, oof)


def coverage_scalars(img: nib.Nifti1Image, transform: SmriMaeTransform, extent_mm: float) -> dict:
    img = crop_ap(img, extent_mm) if extent_mm else nib.as_closest_canonical(repack(img))
    zoom = img.header.get_zooms()[1]

    profile = (img.get_fdata(dtype=np.float32) > 0).sum(axis=(0, 2))
    live = np.flatnonzero(profile)
    edge = max(profile[0], profile[-1])

    # the canvas the backbone is handed, where `fit_to_shape` has centred the scan's own field
    # of view: a brain that sits at a different place in the canvas is a cue of its own
    mask = transform(img)["mask"][0].numpy()
    canvas = np.flatnonzero(mask.sum(axis=(0, 2)))

    return {
        "n_slices": img.shape[1],
        "fov_ap_mm": img.shape[1] * zoom,
        "brain_at_ap_edge": edge / profile.max(),
        "ap_margin_mm": min(live[0], len(profile) - 1 - live[-1]) * zoom,
        "brain_ml": mask.sum() / 1e3,
        "brain_centre_ap": (canvas[0] + canvas[-1]) / 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extent", type=float, nargs="*", default=[0.0, AP_EXTENT_MM])
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    ds = synthseg_strip_dataset(load_fomo_task5(), source="t1w")
    rows = list(ds)
    transform = SmriMaeTransform(masking="zero")
    y = np.array([row["label"] for row in rows])

    for extent in args.extent:
        table = pd.DataFrame([coverage_scalars(row["t1w"], transform, extent) for row in rows])
        table.insert(0, "subject", [row["subject"] for row in rows])
        table["label"] = y

        print(f"\n--- extent {extent or 'uncropped'} " + "-" * 40)
        print(f"{'scalar':<18} {'AUROC':>7} {'perm p':>8}   control mean / case mean")
        for column in COVERAGE + ANATOMY:
            value = table[column].to_numpy(dtype=float)
            if value.std() == 0:
                print(f"{column:<18} {'-':>7} {'-':>8}   constant at {value[0]:.4g}")
                continue
            auroc, p = auroc_with_perm(y, value, rng)
            print(
                f"{column:<18} {auroc:>7.3f} {p:>8.4f}   "
                f"{value[y == 0].mean():.4g} / {value[y == 1].mean():.4g}"
            )
        varying = [c for c in COVERAGE if table[c].std() > 0]
        print(f"{'coverage (LOO head)':<18} {loo_auroc(table[varying].to_numpy(float), y):>7.3f}")
        print(f"{'anatomy (LOO head)':<18} {loo_auroc(table[ANATOMY].to_numpy(float), y):>7.3f}")

        table.to_csv(
            out_dir / f"verify_crop_{int(extent)}.tsv", sep="\t", index=False, float_format="%.4f"
        )


if __name__ == "__main__":
    main()
