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

DEFAULT_CONFIG = Path(__file__).parent / "config/default_probe.yaml"


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
        return self.transform(sample["image"]), sample["target"]


def collate_samples(batch):
    inputs = default_collate([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return inputs, targets


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
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TransformDataset(dataset, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_samples,
    )
    model.eval().to(device)

    features, targets = [], []
    for inputs, batch_targets in loader:
        features.append(model(to_device(inputs, device)).cpu().float())
        targets.extend(batch_targets)
    return torch.cat(features).numpy(), np.asarray(targets)


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
    X, y = extract_features(model, task.dataset(), transform, device, batch_size, num_workers)

    fit = ESTIMATORS[task.kind]
    fold_metrics, predictions = [], []
    for fold, (train_idx, test_idx) in enumerate(task.split()):
        estimator = fit(X[train_idx], y[train_idx], seed)
        pred = estimator.predict(X[test_idx])
        fold_metrics.append(task.metrics(y[test_idx], pred, test_idx))
        for index, value in zip(test_idx, pred):
            predictions.append(
                {"fold": fold, "index": int(index), "target": y[index], "prediction": value}
            )

    metrics = {"summary": aggregate_folds(fold_metrics), "folds": fold_metrics}
    features = {"X": X, "y": y}
    return metrics, features, predictions


def write_outputs(run_dir: Path, metrics: dict, features: dict, predictions: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez(run_dir / "features.npz", **features)
    with (run_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "index", "target", "prediction"])
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config_path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if config_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(config_path))
    if overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(overrides))

    set_seed(cfg.seed)
    cfg.name = cfg.name or f"{cfg.model}__{cfg.task}"

    task = create_task(cfg.task, **(cfg.task_kwargs or {}))
    model, transform = create_model(cfg.model, **(cfg.model_kwargs or {}))

    metrics, features, predictions = run_probe(
        task,
        model,
        transform,
        device=torch.device(cfg.device),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    write_outputs(Path(cfg.output_root) / cfg.name, metrics, features, predictions)
    print(json.dumps(metrics["summary"], indent=2))
    return metrics


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    main(args.config, args.overrides)


if __name__ == "__main__":
    cli()
