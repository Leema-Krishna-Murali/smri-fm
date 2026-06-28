import numpy as np
from sklearn.pipeline import Pipeline


def classification_score(estimator: Pipeline, X: np.ndarray, positive_label) -> np.ndarray | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    proba = estimator.predict_proba(X)
    if positive_label is None:
        return proba
    classes = np.asarray(estimator.classes_)
    matches = np.flatnonzero(classes == positive_label)
    if len(matches) != 1:
        raise ValueError(
            f"positive label {positive_label!r} not found in classes {classes.tolist()}"
        )
    return proba[:, matches[0]]
