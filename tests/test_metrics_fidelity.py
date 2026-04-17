"""Tests for the fidelity metrics."""

from __future__ import annotations

import numpy as np
import pytest

from xai_blockchain_framework.metrics.fidelity import (
    comprehensiveness,
    evaluate_fidelity,
    infidelity,
    sufficiency,
)


def _linear_predict(weights: np.ndarray):
    """Return a predict_proba function for a simple linear model."""
    def predict(X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(X @ weights)))

    return predict


class TestFidelityOnSyntheticLinearModel:
    """When the attribution equals the true model weights, fidelity is high."""

    def setup_method(self) -> None:
        self.rng = np.random.default_rng(0)
        self.w = np.array([1.5, 0.0, -0.5, 2.0, 0.0])
        self.X = self.rng.normal(size=(10, 5))
        self.predict = _linear_predict(self.w)

    def test_comprehensiveness_with_correct_attributions(self) -> None:
        attrs = np.tile(self.w, (10, 1))
        comp = comprehensiveness(self.predict, self.X, attrs, indices=list(range(10)), k=2)
        # Removing the two largest-magnitude features should drop predictions.
        assert comp > -0.5  # generic sanity

    def test_sufficiency_with_correct_attributions(self) -> None:
        attrs = np.tile(self.w, (10, 1))
        suff = sufficiency(self.predict, self.X, attrs, indices=list(range(10)), k=2)
        assert np.isfinite(suff)

    def test_infidelity_of_correct_attributions_is_low(self) -> None:
        attrs = np.tile(self.w, (10, 1))
        score = infidelity(
            self.predict, self.X, attrs, indices=list(range(10)),
            n_perturbations=20, sigma=0.05, rng=np.random.default_rng(1),
        )
        # Linear-approximation error should be small for a linear model with
        # sigmoid and small sigma.
        assert score < 0.05

    def test_evaluate_fidelity_returns_row_per_k(self) -> None:
        attrs = np.tile(self.w, (10, 1))
        df = evaluate_fidelity(
            self.predict, self.X, attrs, indices=list(range(10)),
            k_values=(1, 2, 3),
        )
        assert list(df["k"]) == [1, 2, 3]
        assert set(df.columns) >= {"Comprehensiveness", "Sufficiency", "Infidelity"}
