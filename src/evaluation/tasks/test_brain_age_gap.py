import numpy as np
from datasets import Dataset

from evaluation.tasks.brain_age_gap import BrainAgeGapTask


def _data():
    dx = ["CN"] * 20 + ["AD"] * 10
    return Dataset.from_dict(
        {
            "image": list(range(30)),
            "age": [50 + i for i in range(30)],
            "dx": dx,
        }
    )


def _task():
    return BrainAgeGapTask(
        name="bag",
        data=_data(),
        age_column="age",
        dx_column="dx",
        control_label="CN",
        case_label="AD",
        test_control_frac=0.25,
        seed=0,
    )


def test_split_trains_on_controls_and_is_leakage_free():
    ((train, test),) = list(_task().split())
    dx = np.asarray(_data()["dx"])
    assert set(dx[train]) == {"CN"}  # train is controls only
    assert set(dx[test]) == {"CN", "AD"}  # test compares both
    assert set(train).isdisjoint(test)  # held-out controls don't leak


def test_metrics_positive_tstat_when_cases_predicted_older():
    task = _task()
    ((_, test),) = list(task.split())
    age = np.asarray(_data()["age"])[test]
    dx = np.asarray(_data()["dx"])[test]
    rng = np.random.default_rng(0)
    pred = age + np.where(dx == "AD", 5.0, 0.0) + rng.normal(scale=0.5, size=len(age))
    assert task.metrics(age, pred, test)["bag_tstat"] > 0
