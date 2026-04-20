"""Sampling helpers shared across notebooks.

These helpers are small and deterministic; they are factored out of the
research notebooks so that Modules 1-4 all draw exactly the same subset of
instances for their explanations and evaluations. Reproducibility of the
published numbers depends on this.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sample_balanced(
    y: NDArray[np.int_],
    n: int,
    seed: int = 42,
) -> NDArray[np.int64]:
    """Return a balanced subset of indices (``n/2`` fraud + ``n/2`` legitimate).

    The subset is drawn without replacement; if either class has fewer than
    ``n/2`` instances, the subset is clamped to the class size.

    Parameters
    ----------
    y : numpy.ndarray
        Binary label vector (0 = legitimate, 1 = fraud).
    n : int
        Target size of the balanced subset.
    seed : int, default 42
        Seed for the local RNG.

    Returns
    -------
    numpy.ndarray
        Concatenated array of selected indices, fraud first then legitimate.
    """
    rng = np.random.RandomState(seed)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    half = n // 2
    sel_fraud = rng.choice(fraud_idx, min(half, len(fraud_idx)), replace=False)
    sel_legit = rng.choice(legit_idx, min(half, len(legit_idx)), replace=False)
    return np.concatenate([sel_fraud, sel_legit])


def top_features(
    attributions: NDArray[np.float64],
    feature_names: list[str],
    n: int = 10,
) -> list[tuple[str, float]]:
    """Rank features by mean absolute attribution across a batch.

    Parameters
    ----------
    attributions : numpy.ndarray
        Attribution matrix of shape ``(n_instances, n_features)``.
    feature_names : list of str
    n : int, default 10
        Number of top features to return.

    Returns
    -------
    list of (str, float)
        ``(feature_name, mean_abs_importance)`` tuples, sorted descending.
    """
    mean_abs = np.abs(attributions).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:n]
    return [(feature_names[int(i)], float(mean_abs[int(i)])) for i in order]


def jaccard_topk(
    ranking_a: list[tuple[str, float]],
    ranking_b: list[tuple[str, float]],
    k: int = 10,
) -> float:
    """Jaccard overlap between the top-``k`` feature names of two rankings."""
    set_a = {name for name, _ in ranking_a[:k]}
    set_b = {name for name, _ in ranking_b[:k]}
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0
