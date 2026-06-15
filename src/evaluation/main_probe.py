import argparse
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

from evaluation.models.base import Model, Transform
from evaluation.models.registry import create_model
from evaluation.tasks.base import Task
from evaluation.tasks.registry import create_task

DEFAULT_CONFIG = Path(__file__).parent / "config/default_probe.yaml"


# Estimators, keyed by task kind. Hyperparameters are selected by inner CV.
def fit_ridge(X: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model = Pipeline([("scaler", StandardScaler()), ("ridge", ridge)])
    return model.fit(X, y)


def fit_logistic(X: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    clf = LogisticRegressionCV(Cs=10, scoring="balanced_accuracy", max_iter=1000, random_state=seed)
    model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return model.fit(X, y)


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
        img = sample["image"]
        target = sample["target"]
        sample = self.transform(img)
        sample["target"] = target
        return sample


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
    )
    model.eval()

    features = []
    targets = []
    for batch in loader:
        targets.extend(batch.pop("target"))
        batch = to_device(batch, device)
        embeddings = model(batch)
        features.append(embeddings.cpu().float())

    X = torch.cat(features).numpy()
    y = np.asarray(targets)
    return X, y


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
    dataset = task.dataset()
    X, y = extract_features(model, dataset, transform, device, batch_size, num_workers)

    fit = ESTIMATORS[task.kind]

    fold_metrics = []
    for train_idx, test_idx in task.split():
        estimator = fit(X[train_idx], y[train_idx], seed)
        pred = estimator.predict(X[test_idx])
        fold_metrics.append(task.metrics(y[test_idx], pred, test_idx))

    metrics = {"summary": aggregate_folds(fold_metrics), "folds": fold_metrics}
    return metrics


def aggregate_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    summary = {}
    for key in fold_metrics[0]:
        values = np.array([fold[key] for fold in fold_metrics])
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    return summary


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

    device = torch.device(cfg.device)

    task = create_task(cfg.task, **(cfg.task_kwargs or {}))
    model, transform = create_model(cfg.model, **(cfg.model_kwargs or {}))
    model.to(device)

    metrics = run_probe(
        task,
        model,
        transform,
        device=device,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    run_dir = Path(cfg.output_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.json").open("w") as f:
        print(json.dumps(metrics), file=f)
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
