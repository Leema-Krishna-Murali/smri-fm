from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from datasets import Dataset as HFDataset
from scipy import stats

from evaluation.tasks.base import Kind
from evaluation.tasks.column import ColumnDataset


@dataclass
class BrainAgeGapTask:
    """Train age regression on healthy controls, then score how the brain age
    gap (``age_pred - age_true``) separates cases from controls at test.

    The estimator only ever sees age (``kind="regression"``). The asymmetric
    test objective lives entirely in ``metrics``, which reaches into the
    diagnosis column via ``test_idx``.
    """

    name: str
    data: HFDataset
    age_column: str
    diagnosis_column: str
    control_label: str
    case_label: str
    image_column: str = "nifti"
    test_control_frac: float = 0.2
    seed: int = 0
    kind: Kind = "regression"

    def dataset(self) -> ColumnDataset:
        return ColumnDataset(self.data, self.image_column, self.age_column)

    def split(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        dx = np.asarray(self.data[self.diagnosis_column])
        controls = np.where(dx == self.control_label)[0]
        cases = np.where(dx == self.case_label)[0]

        # hold out some controls so the test set is leakage-free
        controls = np.random.default_rng(self.seed).permutation(controls)
        n_test = round(self.test_control_frac * len(controls))
        test_controls, train_controls = controls[:n_test], controls[n_test:]

        yield train_controls, np.concatenate([test_controls, cases])

    def metrics(self, y_true: np.ndarray, y_pred: np.ndarray, test_idx: np.ndarray) -> dict:
        gap = np.asarray(y_pred).reshape(-1) - np.asarray(y_true).reshape(-1)
        dx = np.asarray(self.data[self.diagnosis_column])[test_idx]
        case_gap = gap[dx == self.case_label]
        control_gap = gap[dx == self.control_label]
        test = stats.ttest_ind(case_gap, control_gap)
        return {
            "bag_tstat": float(test.statistic),
            "bag_pvalue": float(test.pvalue),
            "gap_case": float(case_gap.mean()),
            "gap_control": float(control_gap.mean()),
        }
