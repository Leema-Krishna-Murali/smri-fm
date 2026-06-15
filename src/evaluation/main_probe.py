import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from evaluation.models.base import Model, Transform
from evaluation.models.registry import create_model
from evaluation.tasks.base import Task
from evaluation.tasks.metrics import aggregate_folds
from evaluation.tasks.registry import create_task


# Estimators, keyed by task kind. Hyperparameters are selected by inner CV.
def fit_ridge(X: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))]
    ).fit(X, y)


def fit_logistic(X: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    clf = LogisticRegressionCV(Cs=10, scoring="balanced_accuracy", max_iter=1000, random_state=seed)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)]).fit(X, y)


ESTIMATORS = {"regression": fit_ridge, "classification": fit_logistic}


class TransformDataset(Dataset):
    """Applies the model transform to each canonical sample for batched loading."""

    def __init__(self, dataset: Dataset, transform: Transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        return self.transform(sample["image"]), sample["target"], sample["id"]


def collate_samples(batch):
    inputs = default_collate([item[0] for item in batch])
    targets = [item[1] for item in batch]
    ids = [item[2] for item in batch]
    return inputs, targets, ids


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


@torch.inference_mode()
def extract_features(
    model: Model,
    dataset: Dataset,
    transform: Transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = DataLoader(
        TransformDataset(dataset, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_samples,
    )
    model.eval().to(device)

    features, targets, ids = [], [], []
    for inputs, batch_targets, batch_ids in loader:
        features.append(model(to_device(inputs, device)).cpu().float())
        targets.extend(batch_targets)
        ids.extend(batch_ids)
    return torch.cat(features).numpy(), np.asarray(targets), ids


def run_probe(
    task: Task,
    model: Model,
    transform: Transform,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    X, y, ids = extract_features(model, task.dataset(), transform, device, batch_size, num_workers)

    fit = ESTIMATORS[task.kind]
    fold_metrics, predictions = [], []
    for fold, (train_idx, test_idx) in enumerate(task.split()):
        estimator = fit(X[train_idx], y[train_idx], seed)
        pred = estimator.predict(X[test_idx])
        fold_metrics.append(task.metrics(y[test_idx], pred, test_idx))
        for index, value in zip(test_idx, pred):
            predictions.append(
                {"fold": fold, "id": ids[index], "target": y[index], "prediction": value}
            )

    metrics = {"summary": aggregate_folds(fold_metrics), "folds": fold_metrics}
    features = {"X": X, "y": y, "ids": np.asarray(ids)}
    return metrics, features, predictions


def write_outputs(run_dir: Path, metrics: dict, features: dict, predictions: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez(run_dir / "features.npz", **features)
    with (run_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "id", "target", "prediction"])
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config_path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    cfg = OmegaConf.to_container(cfg, resolve=True)

    set_seed(int(cfg.get("seed", 0)))
    task_cfg = dict(cfg["task"])
    task = create_task(task_cfg.pop("name"), **task_cfg)
    model_cfg = dict(cfg["model"])
    model, transform = create_model(model_cfg.pop("name"), **model_cfg)

    metrics, features, predictions = run_probe(
        task,
        model,
        transform,
        device=torch.device(cfg.get("device", "cpu")),
        batch_size=int(cfg.get("batch_size", 4)),
        num_workers=int(cfg.get("num_workers", 0)),
        seed=int(cfg.get("seed", 0)),
    )

    write_outputs(Path(cfg["output_dir"]) / cfg["name"], metrics, features, predictions)
    print(json.dumps(metrics["summary"], indent=2))
    return metrics


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    main(args.config, args.overrides)


if __name__ == "__main__":
    cli()
