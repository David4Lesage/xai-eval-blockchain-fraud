"""Uniform SHAP explainer for tree-based classifiers (RF, LightGBM)."""

from __future__ import annotations

import numpy as np
import shap
from numpy.typing import NDArray


class ShapTreeExplainer:
    """Wrapper around :class:`shap.TreeExplainer` with a uniform interface.

    Parameters
    ----------
    model : object
        A fitted scikit-learn / LightGBM tree-based classifier.
    """

    def __init__(self, model: object) -> None:
        self.model = model
        self._explainer = shap.TreeExplainer(model)

    def explain(
        self,
        X: NDArray[np.float64],
        indices: list[int] | NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Return SHAP attributions for the positive class.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix of shape ``(n_total, n_features)``.
        indices : sequence of int, optional
            Row subset to explain. When None, explains the full matrix.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(indices), n_features)``.
        """
        subset = X if indices is None else X[np.asarray(indices)]
        raw = self._explainer.shap_values(subset)
        # Different shap versions return either a list (one array per class)
        # or a single 3-D array. Extract the positive-class slice.
        if isinstance(raw, list):
            return np.asarray(raw[1])
        if raw.ndim == 3:
            return raw[..., 1]
        return raw

    def make_explain_fn(self) -> callable:
        """Return a single-instance explanation function for stability metrics."""
        def fn(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.explain(x.reshape(1, -1))[0]
        return fn
