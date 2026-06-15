from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from datasets import Dataset as HFDataset
from sklearn.model_selection import BaseCrossValidator
from torch.utils.data import Dataset

from evaluation.tasks.base import Kind
from evaluation.tasks.metrics import classification_metrics, regression_metrics


class ColumnDataset(Dataset):
    """Adapts HF dataset rows to canonical ``{image, target, id}`` samples."""

    def __init__(
        self, data: HFDataset, image_column: str, target_column: str, id_column: str | None
    ):
        self.data = data
        self.image_column = image_column
        self.target_column = target_column
        self.id_column = id_column

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]
        id_ = str(row[self.id_column]) if self.id_column else str(index)
        return {"image": row[self.image_column], "target": row[self.target_column], "id": id_}


@dataclass
class ColumnTask:
    """Predict a single column of an HF dataset from frozen image features."""

    name: str
    kind: Kind
    data: HFDataset
    splitter: BaseCrossValidator
    image_column: str = "nifti"
    target_column: str = "target"
    group_column: str | None = None
    id_column: str | None = None

    def dataset(self) -> ColumnDataset:
        return ColumnDataset(self.data, self.image_column, self.target_column, self.id_column)

    def split(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(len(self.data))
        targets = np.asarray(self.data[self.target_column])
        groups = np.asarray(self.data[self.group_column]) if self.group_column else None
        return self.splitter.split(indices, y=targets, groups=groups)

    def metrics(self, y_true: np.ndarray, y_pred: np.ndarray, test_idx: np.ndarray) -> dict:
        if self.kind == "regression":
            return regression_metrics(y_true, y_pred)
        return classification_metrics(y_true, y_pred)
