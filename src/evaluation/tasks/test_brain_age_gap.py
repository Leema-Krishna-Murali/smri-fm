import numpy as np
from datasets import Dataset

from evaluation.tasks.brain_age_gap import BrainAgeGapTask


def _data():
    # 20 controls, 10 cases
    dx = ["CN"] * 20 + ["AD"] * 10
    return Dataset.from_dict(
        {
            "nifti": list(range(30)),
            "age": [50 + i for i in range(30)],
            "dx": dx,
            "sub": [f"s{i}" for i in range(30)],
        }
    )


def _task():
    return BrainAgeGapTask(
        name="bag",
        data=_data(),
        age_column="age",
        diagnosis_column="dx",
        control_label="CN",
        case_label="AD",
        id_column="sub",
        test_control_frac=0.25,
        seed=0,
    )


def test_train_is_controls_only_and_leakage_free():
    task = _task()
    ((train_idx, test_idx),) = list(task.split())
    dx = np.asarray(_data()["dx"])
    assert set(dx[train_idx]) == {"CN"}  # trains on controls only
    assert set(dx[test_idx]) == {"CN", "AD"}  # test compares both
    assert set(train_idx).isdisjoint(test_idx)  # no held-out control leaks


def test_metrics_score_gap_against_diagnosis():
    task = _task()
    ((_, test_idx),) = list(task.split())
    age = np.asarray(_data()["age"])[test_idx]
    dx = np.asarray(_data()["dx"])[test_idx]
    # simulate cases predicted older than they are (positive gap), controls near-accurate
    rng = np.random.default_rng(0)
    pred = age + np.where(dx == "AD", 5.0, 0.0) + rng.normal(scale=0.5, size=len(age))
    result = task.metrics(age, pred, test_idx)
    assert result["gap_case"] > result["gap_control"]
    assert result["bag_tstat"] > 0
