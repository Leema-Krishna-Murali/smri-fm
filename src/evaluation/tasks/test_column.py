import numpy as np
from datasets import Dataset
from sklearn.model_selection import KFold, StratifiedKFold

from evaluation.tasks.column import ColumnTask


def _data(n=20):
    return Dataset.from_dict(
        {
            "nifti": list(range(n)),
            "age": [40 + i for i in range(n)],
            "sex": ["M", "F"] * (n // 2),
            "sub": [f"s{i}" for i in range(n)],
        }
    )


def test_canonical_sample():
    task = ColumnTask(
        name="t",
        kind="regression",
        data=_data(),
        splitter=KFold(2),
        image_column="nifti",
        target_column="age",
        id_column="sub",
    )
    sample = task.dataset()[3]
    assert sample == {"image": 3, "target": 43, "id": "s3"}


def test_split_partitions_dataset():
    task = ColumnTask(
        name="t",
        kind="regression",
        data=_data(20),
        target_column="age",
        splitter=KFold(n_splits=5, shuffle=True, random_state=0),
    )
    folds = list(task.split())
    assert len(folds) == 5
    test_sizes = [len(test) for _, test in folds]
    assert sum(test_sizes) == 20  # every sample tested exactly once
    for train, test in folds:
        assert set(train).isdisjoint(test)


def test_stratified_split_uses_target():
    task = ColumnTask(
        name="t",
        kind="classification",
        data=_data(20),
        target_column="sex",
        splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
    )
    y = np.asarray(_data(20)["sex"])
    for _, test in task.split():
        # balanced classes => each fold keeps both
        assert set(y[test]) == {"M", "F"}


def test_metrics_dispatch_on_kind():
    reg = ColumnTask(name="t", kind="regression", data=_data(), splitter=KFold(2))
    cls = ColumnTask(name="t", kind="classification", data=_data(), splitter=KFold(2))
    idx = np.arange(4)
    assert "mae" in reg.metrics(np.zeros(4), np.ones(4), idx)
    assert "balanced_accuracy" in cls.metrics(["M", "F", "M", "F"], ["M", "F", "F", "F"], idx)
