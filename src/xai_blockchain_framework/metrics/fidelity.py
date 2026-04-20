"""Module 1: Fidelity metrics for XAI explanations.

Three complementary measures quantify whether an explanation faithfully
reflects the model's decision process:

- :func:`comprehensiveness` — how much the prediction drops when the
  explanation's top-k features are removed (higher is better).
- :func:`sufficiency` — how much the prediction drops when only the top-k
  features are kept (lower is better).
- :func:`infidelity` — mean squared error between the attribution-predicted
  change and the model's actual change under small random perturbations
  (lower is better).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _top_k_mask(attribution: NDArray[np.float64], k: int) -> NDArray[np.bool_]:
    """Return a boolean mask of shape ``(n_features,)`` selecting the top-k."""
    top = np.argsort(-np.abs(attribution))[:k]
    mask = np.zeros_like(attribution, dtype=bool)
    mask[top] = True
    return mask


def comprehensiveness(
    predict_proba: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    X: NDArray[np.float64],
    attributions: NDArray[np.float64],
    indices: list[int] | NDArray[np.int64],
    k: int = 5,
) -> float:
    """Comprehensiveness@k.

    Measures the drop in predicted fraud probability when the top-k most
    important features (as judged by the explanation) are zeroed out. If the
    explanation is faithful, removing these features should substantially
    reduce the model's confidence.

    Parameters
    ----------
    predict_proba : callable
        Function mapping an array of shape ``(n, d)`` to fraud probabilities
        of shape ``(n,)``.
    X : numpy.ndarray
        Feature matrix of shape ``(n_total, n_features)``.
    attributions : numpy.ndarray
        Attribution matrix of shape ``(len(indices), n_features)``.
    indices : sequence of int
        Row indices of ``X`` to evaluate.
    k : int, default 5
        Number of top features to mask out.

    Returns
    -------
    float
        Mean prediction drop across the evaluated instances. Higher is better.
    """
    indices = np.asarray(indices)
    x = X[indices]
    original = predict_proba(x)

    masked = x.copy()
    for i, attr in enumerate(attributions[: len(indices)]):
        mask = _top_k_mask(attr, k)
        masked[i, mask] = 0.0

    masked_probs = predict_proba(masked)
    return float(np.mean(original - masked_probs))


def sufficiency(
    predict_proba: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    X: NDArray[np.float64],
    attributions: NDArray[np.float64],
    indices: list[int] | NDArray[np.int64],
    k: int = 5,
) -> float:
    """Sufficiency@k.

    Measures the prediction drop when only the top-k features are retained
    (all others zeroed). A faithful top-k explanation should by itself
    reproduce most of the original prediction, so the drop should be small.

    Parameters
    ----------
    predict_proba, X, attributions, indices, k
        See :func:`comprehensiveness`.

    Returns
    -------
    float
        Mean prediction drop. Lower is better.
    """
    indices = np.asarray(indices)
    x = X[indices]
    original = predict_proba(x)

    kept = np.zeros_like(x)
    for i, attr in enumerate(attributions[: len(indices)]):
        mask = _top_k_mask(attr, k)
        kept[i, mask] = x[i, mask]

    kept_probs = predict_proba(kept)
    return float(np.mean(original - kept_probs))


def infidelity(
    predict_proba: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    X: NDArray[np.float64],
    attributions: NDArray[np.float64],
    indices: list[int] | NDArray[np.int64],
    n_perturbations: int = 50,
    sigma: float = 0.1,
    rng: np.random.Generator | None = None,
) -> float:
    """Infidelity score (Yeh et al., 2019).

    For each instance and perturbation ``I``, compare:

    - the model's observed change under the perturbation,
      ``f(x) - f(x - I)``, and
    - the change predicted by the linear approximation, ``I · attr``.

    The score is the mean squared difference
    ``(I · attr - (f(x) - f(x - I)))**2``. A faithful explanation makes
    these two quantities agree for small ``I``.

    Parameters
    ----------
    predict_proba, X, attributions, indices
        See :func:`comprehensiveness`.
    n_perturbations : int, default 50
        Number of random perturbations per instance.
    sigma : float, default 0.1
        Standard deviation of the Gaussian noise.
    rng : numpy.random.Generator, optional
        Random number generator. If None, a fresh default generator is used.

    Returns
    -------
    float
        Mean infidelity across instances. Lower is better.
    """
    rng = rng if rng is not None else np.random.default_rng()
    indices = np.asarray(indices)

    scores = []
    for i, idx in enumerate(indices):
        x = X[idx]
        attr = attributions[i]
        perturbations = rng.normal(0.0, sigma, size=(n_perturbations, x.shape[0]))

        f_x = predict_proba(x[None, :])[0]
        f_xp = predict_proba(x[None, :] - perturbations)
        model_delta = f_x - f_xp
        expl_delta = perturbations @ attr
        scores.append(float(np.mean((expl_delta - model_delta) ** 2)))

    return float(np.mean(scores))


def evaluate_fidelity(
    predict_proba: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    X: NDArray[np.float64],
    attributions: NDArray[np.float64],
    indices: list[int] | NDArray[np.int64],
    k_values: list[int] = (1, 3, 5, 10),
    n_perturbations: int = 50,
    sigma: float = 0.1,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Compute all three fidelity metrics for a range of k values.

    Returns a long-format DataFrame with columns
    ``k, Comprehensiveness, Sufficiency, Infidelity``. Infidelity does not
    depend on k, so it is duplicated across rows.
    """
    infid = infidelity(
        predict_proba, X, attributions, indices,
        n_perturbations=n_perturbations, sigma=sigma, rng=rng,
    )
    rows = []
    for k in k_values:
        rows.append({
            "k": k,
            "Comprehensiveness": comprehensiveness(predict_proba, X, attributions, indices, k=k),
            "Sufficiency": sufficiency(predict_proba, X, attributions, indices, k=k),
            "Infidelity": infid,
        })
    return pd.DataFrame(rows)
