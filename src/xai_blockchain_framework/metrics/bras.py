"""Module 3: Blockchain Rule Alignment Score (BRAS).

A novel domain-specific metric that quantifies whether an explanation aligns
with known blockchain fraud patterns. Two complementary quantities are
combined:

- :func:`rule_alignment_score` — the fraction of the top-k features that
  belong to the set of domain-relevant features for the instance
  (higher is better).
- :func:`domain_violation_rate` — the fraction of instances whose top-k
  features include at least one feature that is known to contradict the
  domain (lower is better).

The composite :func:`bras_score` is
``alpha * RAS + (1 - alpha) * (1 - DVR)`` with ``alpha = 0.5`` by default.
Rule definitions live in :mod:`xai_blockchain_framework.rules`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

RuleFn = Callable[[NDArray[np.float64], int], tuple[set[int], set[int]]]
"""Type alias: a rule function takes ``(feature_vector, n_features)`` and
returns ``(relevant_features, contradictory_features)`` as sets of indices."""


def top_k_indices(attribution: NDArray[np.float64], k: int = 5) -> set[int]:
    """Return the set of top-k feature indices by absolute attribution."""
    return set(np.argsort(-np.abs(attribution))[:k].tolist())


def rule_alignment_score(
    attribution: NDArray[np.float64],
    relevant_features: set[int],
    k: int = 5,
) -> float:
    """Proportion of top-k features that are in the relevant set.

    Returns 0.0 when ``k == 0`` or when no attribution is available.
    """
    top = top_k_indices(attribution, k)
    if not top:
        return 0.0
    return len(top & relevant_features) / len(top)


def domain_violation_rate(
    attribution: NDArray[np.float64],
    contradictory_features: set[int],
    k: int = 5,
) -> int:
    """Return 1 if any top-k feature is in ``contradictory_features``, else 0.

    The per-instance value is binary; the population-level rate is the mean
    of these indicators across instances.
    """
    if not contradictory_features:
        return 0
    top = top_k_indices(attribution, k)
    return int(bool(top & contradictory_features))


def bras_score(ras: float, dvr: float, alpha: float = 0.5) -> float:
    """Combine RAS and DVR into a single scalar score in ``[0, 1]``.

    Parameters
    ----------
    ras : float
        Mean rule alignment score across instances.
    dvr : float
        Domain violation rate (fraction of violating instances).
    alpha : float, default 0.5
        Weight placed on RAS vs. (1 - DVR).

    Returns
    -------
    float
        Composite BRAS. Higher is better.
    """
    return float(alpha * ras + (1.0 - alpha) * (1.0 - dvr))


def evaluate_bras(
    X: NDArray[np.float64],
    attributions: NDArray[np.float64],
    indices: list[int] | NDArray[np.int64],
    rule_fn: RuleFn,
    k: int = 5,
    alpha: float = 0.5,
) -> dict[str, float | int]:
    """Compute RAS, DVR, and BRAS over a set of instances.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape ``(n_total, n_features)``.
    attributions : numpy.ndarray
        Attribution matrix of shape ``(len(indices), n_features)``.
    indices : sequence of int
        Row indices of ``X`` to evaluate (typically fraud-only).
    rule_fn : callable
        Rule function returning ``(relevant, contradictory)`` feature index
        sets for a given feature vector.
    k : int, default 5
        Number of top features evaluated.
    alpha : float, default 0.5
        BRAS combination weight.

    Returns
    -------
    dict
        Keys: ``RAS``, ``DVR``, ``BRAS``, ``N_eval``.
    """
    indices = np.asarray(indices)
    n_features = X.shape[1]
    ras_values: list[float] = []
    dvr_flags: list[int] = []

    for i, idx in enumerate(indices):
        relevant, contra = rule_fn(X[idx], n_features)
        ras_values.append(rule_alignment_score(attributions[i], relevant, k=k))
        dvr_flags.append(domain_violation_rate(attributions[i], contra, k=k))

    ras = float(np.mean(ras_values)) if ras_values else 0.0
    dvr = float(np.mean(dvr_flags)) if dvr_flags else 0.0
    return {
        "RAS": ras,
        "DVR": dvr,
        "BRAS": bras_score(ras, dvr, alpha=alpha),
        "N_eval": len(indices),
    }
