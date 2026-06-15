import numpy as np
from sklearn import metrics as skm


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    residuals = y_pred - y_true
    return {
        "mae": float(np.abs(residuals).mean()),
        "rmse": float(np.sqrt((residuals**2).mean())),
        "bias": float(residuals.mean()),
        "r2": float(skm.r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, y_pred)),
    }


def aggregate_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Mean +/- std across folds, flattened to ``<metric>`` and ``<metric>_std``."""
    summary = {}
    for key in fold_metrics[0]:
        values = np.array([fold[key] for fold in fold_metrics], dtype=np.float64)
        summary[key] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    return summary
