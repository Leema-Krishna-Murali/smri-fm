from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from datasets import Dataset as HFDataset
from scipy import stats

from evaluation.tasks.base import Kind


@dataclass
class BrainAgeGapTask:
    """Train age regression on controls, then evaluate age prediction on controls and cases."""

    name: str
    data: HFDataset
    age_column: str
    dx_column: str
    control_label: str
    case_label: str
    image_column: str = "image"
    test_control_frac: float = 0.2
    seed: int = 0
    kind: Kind = "regression"

    def dataset(self) -> HFDataset:
        column_mapping = {self.image_column: "image", self.age_column: "target"}
        dataset = self.data.select_columns(list(column_mapping)).rename_columns(column_mapping)
        return dataset

    def split(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        dx = np.asarray(self.data[self.dx_column])
        controls = np.where(dx == self.control_label)[0]
        cases = np.where(dx == self.case_label)[0]

        # hold out some controls so the test set is leakage-free
        rng = np.random.default_rng(self.seed)
        controls = rng.permutation(controls)
        n_test = round(self.test_control_frac * len(controls))
        test_controls, train_controls = controls[:n_test], controls[n_test:]

        yield train_controls, np.concatenate([test_controls, cases])

    def metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        test_idx: np.ndarray,
        y_score: np.ndarray | None = None,
    ) -> dict:
        gap = (y_pred - y_true).reshape(-1)
        dx = np.asarray(self.data[self.dx_column])[test_idx]
        case_gap = gap[dx == self.case_label]
        control_gap = gap[dx == self.control_label]
        test = stats.ttest_ind(case_gap, control_gap)
        return {"bag_tstat": float(test.statistic)}
