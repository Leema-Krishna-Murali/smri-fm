"""`Ridge` against sklearn: the closed form, and the held-out error the Gram identity stands in for."""

import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fomo_tune.ridge import Ridge, group_folds

N, D, K, N_GROUPS, N_SPLITS = 500, 20, 3, 10, 4


def sample(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((N, D)) * rng.uniform(0.1, 10.0, D)
    weights = rng.standard_normal((D, K))
    targets = features @ weights + rng.standard_normal((N, K))
    groups = rng.integers(0, N_GROUPS, N)
    return features, targets, groups


def predicted(model: Ridge, features: np.ndarray) -> np.ndarray:
    dtype = model.coef_.dtype
    return model.predict(torch.from_numpy(features).to(dtype)).numpy()


def fit(features: np.ndarray, targets: np.ndarray, dtype: torch.dtype, **kwargs) -> Ridge:
    return Ridge(**kwargs).fit(
        torch.from_numpy(features).to(dtype), torch.from_numpy(targets).to(dtype)
    )


@pytest.mark.parametrize("alpha", [1e-1, 1e1, 1e3])
@pytest.mark.parametrize("standardize", [True, False])
def test_matches_sklearn_closed_form(alpha: float, standardize: bool) -> None:
    features, targets, _ = sample()
    model = fit(
        features,
        targets,
        torch.float64,
        alphas=(alpha,),
        n_splits=2,
        standardize=standardize,
    )

    steps = [StandardScaler()] if standardize else []
    reference = make_pipeline(*steps, SklearnRidge(alpha=alpha)).fit(features, targets)

    np.testing.assert_allclose(predicted(model, features), reference.predict(features), rtol=1e-8)


def test_float32_accumulation_tracks_double() -> None:
    """The default dtype trades accuracy for memory; this is how much it actually costs."""
    features, targets, groups = sample()
    single, double = (
        Ridge(seed=3).fit(
            torch.from_numpy(features).to(dtype), torch.from_numpy(targets).to(dtype), groups
        )
        for dtype in (torch.float32, torch.float64)
    )

    reference = predicted(double, features)
    assert single.alpha_ == double.alpha_
    assert (
        np.linalg.norm(predicted(single, features) - reference) / np.linalg.norm(reference) < 1e-6
    )
    np.testing.assert_allclose(single.cv_mse_.numpy(), double.cv_mse_.numpy(), rtol=1e-4)


def test_cv_error_matches_refitting_each_fold() -> None:
    """The Gram identity for held-out error, against actually fitting and scoring every fold."""
    features, targets, groups = sample()
    model = Ridge(n_splits=N_SPLITS, seed=3).fit(
        torch.from_numpy(features), torch.from_numpy(targets), groups
    )

    # scaled once over everything, as `fit` does, so this compares the identity and not the scaling
    scaled = StandardScaler().fit_transform(features)
    folds = group_folds(groups, N_SPLITS, seed=3)
    expected = []
    for alpha in model.alphas:
        error = 0.0
        for held_out in folds:
            train = np.setdiff1d(np.arange(N), held_out)
            reference = SklearnRidge(alpha=alpha).fit(scaled[train], targets[train])
            residual = reference.predict(scaled[held_out]) - targets[held_out]
            error += float((residual**2).sum())
        expected.append(error / (N * K))

    np.testing.assert_allclose(model.cv_mse_.numpy(), expected, rtol=1e-7)
    assert model.alpha_ == model.alphas[int(np.argmin(expected))]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a device")
def test_cuda_float32_matches_cpu_double() -> None:
    """The configuration we actually run. Fails loudly if TF32 is ever enabled for matmuls."""
    features, targets, groups = sample()
    on_device = Ridge(seed=3).fit(
        torch.from_numpy(features).float().cuda(), torch.from_numpy(targets).float().cuda(), groups
    )
    on_host = Ridge(seed=3).fit(torch.from_numpy(features), torch.from_numpy(targets), groups)

    reference = predicted(on_host, features)
    moved = on_device.predict(torch.from_numpy(features).float().cuda()).cpu().numpy()

    assert on_device.alpha_ == on_host.alpha_
    assert np.linalg.norm(moved - reference) / np.linalg.norm(reference) < 1e-5


def test_groups_stay_whole() -> None:
    _, _, groups = sample()
    folds = group_folds(groups, N_SPLITS, seed=3)

    assert sorted(np.concatenate(folds)) == list(range(N))
    seen = [set(groups[fold]) for fold in folds]
    assert not set.intersection(*seen)
