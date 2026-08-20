"""Torch ridge regression with grouped k-fold CV.

Uses per-fold Gram statistics `X^T X` and `X^T y` to reduce cost.
"""

from collections.abc import Sequence

import numpy as np
import torch


def group_folds(groups: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    """Group k-fold split. Returns list of fold indices."""
    unique = np.unique(groups)
    assert n_splits <= len(unique), f"{n_splits} splits over {len(unique)} groups"

    shuffled = np.random.default_rng(seed).permutation(unique)
    fold_of_group = {group: i % n_splits for i, group in enumerate(shuffled)}
    labels = np.array([fold_of_group[group] for group in groups])
    return [np.flatnonzero(labels == fold) for fold in range(n_splits)]


def solve(gram: torch.Tensor, cross: torch.Tensor, alpha: float) -> torch.Tensor:
    """Ridge coefficients, leaving the design's trailing ones column unpenalized."""
    penalty = torch.ones_like(cross[:, 0])
    penalty[-1] = 0.0
    return torch.linalg.solve(gram + alpha * torch.diag(penalty), cross)


def squared_error(
    gram: torch.Tensor, cross: torch.Tensor, targets_ss: torch.Tensor, coefs: torch.Tensor
) -> torch.Tensor:
    """Residual sum of squares of `coefs` over the samples behind these statistics."""
    return (coefs * (gram @ coefs)).sum() - 2 * (coefs * cross).sum() + targets_ss.sum()


def fit_ridge(
    features: torch.Tensor,
    targets: torch.Tensor,
    folds: Sequence[np.ndarray],
    alphas: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Fit ridge regression.

    Returns tuple of coefficients, error per alpha, and best alpha.
    """
    n, d = features.shape
    assert targets.ndim == 2 and len(targets) == n, (
        f"targets {tuple(targets.shape)} do not match {n} samples of {d} features"
    )
    ones = torch.ones(n, 1, device=features.device, dtype=features.dtype)
    design = torch.cat([features, ones], dim=1)

    grams, crosses, target_sums = [], [], []
    for fold in folds:
        rows = torch.as_tensor(fold, device=design.device)
        block, target = design[rows], targets[rows]
        # collinear features leave the system conditioned ~1e6 at the smallest alpha, past float32
        grams.append((block.T @ block).double())
        crosses.append((block.T @ target).double())
        target_sums.append((target * target).sum(dim=0).double())

    total_gram = torch.stack(grams).sum(dim=0)
    total_cross = torch.stack(crosses).sum(dim=0)

    errors = torch.zeros(len(alphas), device=design.device, dtype=torch.float64)
    for i, alpha in enumerate(alphas):
        for gram, cross, targets_ss in zip(grams, crosses, target_sums):
            coefs = solve(total_gram - gram, total_cross - cross, alpha)
            errors[i] += squared_error(gram, cross, targets_ss, coefs)

    best = int(errors.argmin())
    coefs = solve(total_gram, total_cross, alphas[best])
    return coefs, errors / (n * targets.shape[1]), alphas[best]


class Ridge:
    """Torch ridge regression with grouped k-fold CV."""

    def __init__(
        self,
        alphas: Sequence[float] = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6),
        n_splits: int = 5,
        standardize: bool = True,
        seed: int = 0,
    ):
        self.alphas = tuple(alphas)
        self.n_splits = n_splits
        self.standardize = standardize
        self.seed = seed

    def fit(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        groups: np.ndarray | None = None,
    ) -> "Ridge":
        if self.standardize:
            self.mean_ = features.mean(dim=0)
            # correction=0 to match sklearn's StandardScaler
            self.scale_ = features.std(dim=0, correction=0).clamp(min=1e-12)
        else:
            self.mean_ = features.new_zeros(features.shape[1])
            self.scale_ = features.new_ones(features.shape[1])

        groups = np.arange(len(features)) if groups is None else np.asarray(groups)
        folds = group_folds(groups, self.n_splits, self.seed)

        scaled = (features - self.mean_) / self.scale_ if self.standardize else features
        coefs, self.cv_mse_, self.alpha_ = fit_ridge(scaled, targets, folds, self.alphas)
        self.coef_ = coefs[:-1].T.to(features.dtype)
        self.intercept_ = coefs[-1].to(features.dtype)
        return self

    def to(self, device: str | torch.device) -> "Ridge":
        """Move the fitted parameters, so a model fit on a device can be saved or run off it."""
        self.coef_ = self.coef_.to(device)
        self.intercept_ = self.intercept_.to(device)
        self.mean_ = self.mean_.to(device)
        self.scale_ = self.scale_.to(device)
        self.cv_mse_ = self.cv_mse_.to(device)
        return self

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        scaled = (features - self.mean_) / self.scale_ if self.standardize else features
        return scaled @ self.coef_.T + self.intercept_
