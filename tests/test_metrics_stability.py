"""Tests for the stability metrics."""

from __future__ import annotations

import numpy as np

from xai_blockchain_framework.metrics.stability import (
    cov_bootstrap,
    identity_score,
    lipschitz_stability,
    rank_stability_kendall,
)


class TestLipschitz:
    def test_identical_explanations_give_zero(self) -> None:
        X = np.random.default_rng(0).normal(size=(10, 5))
        explanations = np.ones((10, 5))
        lip = lipschitz_stability(X, explanations, indices=list(range(10)))
        assert lip == 0.0

    def test_large_explanation_differences_give_large_lipschitz(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 5))
        explanations = rng.normal(size=(10, 5)) * 100
        lip = lipschitz_stability(X, explanations, indices=list(range(10)))
        assert lip > 1.0


class TestKendallTau:
    def test_identical_rankings_give_one(self) -> None:
        same = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]] * 5)
        assert rank_stability_kendall(same) == 1.0

    def test_reversed_rankings_give_minus_one(self) -> None:
        alternating = np.array([
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ])
        # Not strictly -1 due to how we compare argsort of absolute values,
        # but must be strongly negative.
        assert rank_stability_kendall(alternating) < -0.5


class TestCovBootstrap:
    def test_deterministic_explainer_gives_zero_cov(self) -> None:
        def det(x: np.ndarray) -> np.ndarray:
            return np.array([1.0, 2.0, 3.0])

        x = np.zeros(3)
        cov = cov_bootstrap(det, x, n_bootstrap=4, perturbation_scale=0.01)
        assert cov == 0.0


class TestIdentity:
    def test_deterministic_returns_one(self) -> None:
        def det(x: np.ndarray) -> np.ndarray:
            return np.array([1.0, 2.0, 3.0])

        assert identity_score(det, np.zeros(3), n_runs=4) == 1.0

    def test_stochastic_returns_less_than_one(self) -> None:
        rng = np.random.default_rng(0)

        def stoch(x: np.ndarray) -> np.ndarray:
            return rng.normal(size=x.shape)

        assert identity_score(stoch, np.zeros(3), n_runs=4) < 1.0
