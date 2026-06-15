import numpy as np
import torch
import torch.nn as nn

from evaluation.main_probe import run_probe
from evaluation.tasks.metrics import regression_metrics


class DummyModel(nn.Module):
    def forward(self, batch):
        return batch["x"]


def dummy_transform(image):
    return {"x": torch.tensor(image, dtype=torch.float32)}


class FakeTask:
    name = "fake"
    kind = "regression"

    def __init__(self, X, y):
        self.samples = [
            {"image": list(x), "target": float(t), "id": str(i)}
            for i, (x, t) in enumerate(zip(X, y))
        ]

    def dataset(self):
        return self.samples

    def split(self):
        idx = np.arange(len(self.samples))
        yield idx[:16], idx[16:]

    def metrics(self, y_true, y_pred, test_idx):
        return regression_metrics(y_true, y_pred)


def test_run_probe_end_to_end():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    weights = rng.normal(size=5)
    y = X @ weights  # exactly linear => ridge should recover it well

    task = FakeTask(X, y)
    metrics, features, predictions = run_probe(
        task,
        DummyModel(),
        dummy_transform,
        device=torch.device("cpu"),
        batch_size=4,
        num_workers=0,
        seed=0,
    )

    assert features["X"].shape == (20, 5)
    assert len(predictions) == 4  # test split size
    assert {p["id"] for p in predictions} == {"16", "17", "18", "19"}
    assert metrics["summary"]["mae"] < 0.1
