from collections.abc import Iterator
from typing import Literal, Protocol

import numpy as np
from torch.utils.data import Dataset

Kind = Literal["regression", "classification"]


class Task(Protocol):
    """Thin, declarative wrapper around a dataset: owns data, splits, and scoring."""

    name: str
    kind: Kind

    def dataset(self) -> Dataset:
        """Indexable dataset of canonical ``{image, target}`` samples, in stable order."""
        ...

    def split(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` into dataset order; one pair, or many for outer CV."""
        ...

    def metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, test_idx: np.ndarray
    ) -> dict[str, float]:
        """Score one fold; ``test_idx`` lets a task pull auxiliary metadata for scoring."""
        ...
