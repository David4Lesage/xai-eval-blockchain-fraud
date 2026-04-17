"""Uniform LIME explainer for tabular classifiers."""

from __future__ import annotations

from typing import Callable

import numpy as np
from lime.lime_tabular import LimeTabularExplainer as _LimeExplainer
from numpy.typing import NDArray


class LimeTabularExplainer:
    """Wrapper around :class:`lime.lime_tabular.LimeTabularExplainer`.

    Produces dense attribution vectors aligned with the original feature
    order, which is what the downstream evaluation modules expect.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training features used to estimate per-feature distributions.
    feature_names : list of str, optional
        Display names for features.
    num_samples : int, default 500
        Number of perturbed samples per explanation.
    num_features : int, optional
        Maximum number of features with non-zero attribution. Defaults to
        the number of columns (dense explanation).
    random_state : int, optional
        Passed to the underlying LIME explainer.
    """

    def __init__(
        self,
        X_train: NDArray[np.float64],
        feature_names: list[str] | None = None,
        num_samples: int = 500,
        num_features: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self._n_features = X_train.shape[1]
        self._num_samples = num_samples
        self._num_features = num_features or self._n_features
        self._explainer = _LimeExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=["legitimate", "fraud"],
            discretize_continuous=False,
            random_state=random_state,
            mode="classification",
        )

    def explain(
        self,
        X: NDArray[np.float64],
        predict_proba: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        indices: list[int] | NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Return dense LIME attributions aligned with feature indices.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix.
        predict_proba : callable
            Must accept an array of shape ``(n, d)`` and return an array of
            shape ``(n, 2)`` with class probabilities.
        indices : sequence of int, optional
            Row subset to explain.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(indices), n_features)``.
        """
        rows = np.asarray(indices) if indices is not None else np.arange(len(X))
        out = np.zeros((len(rows), self._n_features), dtype=np.float64)
        for i, r in enumerate(rows):
            expl = self._explainer.explain_instance(
                X[r],
                predict_proba,
                num_samples=self._num_samples,
                num_features=self._num_features,
                labels=(1,),
            )
            for feature_idx, weight in expl.local_exp.get(1, []):
                out[i, feature_idx] = weight
        return out

    def make_explain_fn(
        self,
        predict_proba: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """Return a single-instance explanation function for stability metrics."""
        def fn(x: NDArray[np.float64]) -> NDArray[np.float64]:
            out = np.zeros(self._n_features, dtype=np.float64)
            expl = self._explainer.explain_instance(
                x,
                predict_proba,
                num_samples=self._num_samples,
                num_features=self._num_features,
                labels=(1,),
            )
            for feature_idx, weight in expl.local_exp.get(1, []):
                out[feature_idx] = weight
            return out

        return fn
