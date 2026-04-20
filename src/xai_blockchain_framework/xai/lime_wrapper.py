"""Uniform LIME explainer for tabular classifiers.

This wrapper matches the research-code behavior:

- ``discretize_continuous=True`` (LIME's default). Each feature is bucketed
  into quartiles over the training distribution, and LIME fits its local
  linear surrogate on the quartile indicators.
- Attributions are read back from
  :attr:`lime.explanation.Explanation.local_exp` under the positive class
  key, which returns ``(feature_index, weight)`` pairs aligned with the
  **original** feature order. This is numerically equivalent to parsing
  :meth:`~lime.explanation.Explanation.as_list` and greedy-matching feature
  names, but it is simpler and does not depend on feature-name collisions.

These choices are the ones the evaluation framework (Modules 1-4) was
calibrated against, so they are preserved here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from lime.lime_tabular import LimeTabularExplainer as _LimeExplainer
from numpy.typing import NDArray


class LimeTabularExplainer:
    """LIME explainer for tabular classifiers with dense output.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training features used to estimate per-feature distributions.
    feature_names : list of str, optional
        Display names for features. Optional: attributions are always
        returned by feature index, not name.
    num_samples : int, default 500
        Number of perturbed samples per explanation.
    num_features : int, optional
        Maximum number of features with non-zero attribution. Defaults to
        the number of columns (dense explanation).
    random_state : int, optional
        Passed to the underlying LIME explainer for reproducibility.
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
        self._feature_names = list(feature_names) if feature_names is not None else [
            f"f{i}" for i in range(self._n_features)
        ]
        self._explainer = _LimeExplainer(
            training_data=X_train,
            feature_names=self._feature_names,
            class_names=["Legit", "Fraud"],
            mode="classification",
            random_state=random_state,
        )

    def _explain_single(
        self,
        x: NDArray[np.float64],
        predict_proba: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        out: NDArray[np.float64],
    ) -> None:
        expl = self._explainer.explain_instance(
            x,
            predict_proba,
            num_samples=self._num_samples,
            num_features=self._num_features,
        )
        for feature_idx, weight in expl.local_exp.get(1, []):
            if 0 <= feature_idx < self._n_features:
                out[feature_idx] = float(weight)

    def explain(
        self,
        X: NDArray[np.float64],
        predict_proba: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        indices: list[int] | NDArray[np.int64] | None = None,
        verbose_every: int = 0,
        name: str = "",
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
            Row subset to explain. When None, explains all rows.
        verbose_every : int, default 0
            Print a progress line every ``verbose_every`` explanations.
            Disabled when 0.
        name : str
            Label used in progress prints.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(indices), n_features)``.
        """
        rows = np.asarray(indices) if indices is not None else np.arange(len(X))
        out = np.zeros((len(rows), self._n_features), dtype=np.float64)
        for i, r in enumerate(rows):
            self._explain_single(X[r], predict_proba, out[i])
            if verbose_every and (i + 1) % verbose_every == 0:
                label = f"LIME {name}" if name else "LIME"
                print(f"  {label}: {i + 1}/{len(rows)}")
        return out

    def make_explain_fn(
        self,
        predict_proba: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """Return a single-instance explanation function for stability metrics."""
        def fn(x: NDArray[np.float64]) -> NDArray[np.float64]:
            out = np.zeros(self._n_features, dtype=np.float64)
            self._explain_single(x, predict_proba, out)
            return out

        return fn
