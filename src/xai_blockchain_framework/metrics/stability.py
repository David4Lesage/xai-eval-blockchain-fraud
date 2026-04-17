"""Module 2: Stability metrics for XAI explanations.

Four complementary measures quantify whether an explanation method produces
consistent outputs across similar inputs and repeated runs:

- :func:`lipschitz_stability` — local Lipschitz constant of the explanation
  function on a neighborhood graph (lower is better).
- :func:`rank_stability_kendall` — mean Kendall τ between feature rankings
  of neighboring instances (higher is better).
- :func:`cov_bootstrap` — coefficient of variation of attributions under
  small input perturbations (lower is better).
- :func:`identity_score` — fraction of repeated runs that produce identical
  outputs for the same input (higher is better).
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import kendalltau
from sklearn.neighbors import NearestNeighbors


def lipschitz_stability(
    X: NDArray[np.float64],
    explanations: NDArray[np.float64],
    indices: list[int] | NDArray[np.int64],
    n_neighbors: int = 5,
) -> float:
    """Local Lipschitz constant of the explanation function.

    For each instance, find its k nearest neighbors in input space and
    compute the ratio ``||phi(x_i) - phi(x_j)|| / ||x_i - x_j||``. Return the
    mean of these ratios across all instances and neighbors.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape ``(n_total, n_features)``.
    explanations : numpy.ndarray
        Attribution matrix of shape ``(len(indices), n_features)`` aligned
        with ``indices``.
    indices : sequence of int
        Row indices of ``X`` to evaluate.
    n_neighbors : int, default 5
        Number of neighbors (excluding the point itself) used per instance.

    Returns
    -------
    float
        Mean Lipschitz ratio. Lower is better. Raw values can span several
        orders of magnitude; use :func:`~xai_blockchain_framework.utils.normalization.log_normalize`
        for display and composite scoring.
    """
    indices = np.asarray(indices)
    X_sub = X[indices]
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(X_sub)

    ratios: list[float] = []
    for i in range(len(indices)):
        dists, nbrs = nn.kneighbors(X_sub[i: i + 1])
        for dist, j in zip(dists[0, 1:], nbrs[0, 1:]):
            if dist <= 1e-10:
                continue
            expl_diff = float(np.linalg.norm(explanations[i] - explanations[j]))
            ratios.append(expl_diff / float(dist))

    return float(np.mean(ratios)) if ratios else 0.0


def rank_stability_kendall(explanations: NDArray[np.float64]) -> float:
    """Mean Kendall τ between feature rankings of neighboring instances.

    Computes the pairwise Kendall τ correlation between consecutive rows of
    ``explanations`` (after ranking features by absolute value). Consecutive
    rows are assumed to be ordered so that neighbors in input space are also
    neighbors in the array; for independent samples, shuffle beforehand is
    not required because we average over all consecutive pairs.

    Parameters
    ----------
    explanations : numpy.ndarray
        Attribution matrix of shape ``(n_instances, n_features)``.

    Returns
    -------
    float
        Mean τ in ``[-1, 1]``. Higher is better.
    """
    if len(explanations) < 2:
        return 0.0
    taus: list[float] = []
    for i in range(len(explanations) - 1):
        r1 = np.argsort(-np.abs(explanations[i]))
        r2 = np.argsort(-np.abs(explanations[i + 1]))
        tau, _ = kendalltau(r1, r2)
        if not np.isnan(tau):
            taus.append(float(tau))
    return float(np.mean(taus)) if taus else 0.0


def cov_bootstrap(
    explain_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    n_bootstrap: int = 5,
    perturbation_scale: float = 0.01,
    rng: np.random.Generator | None = None,
) -> float:
    """Coefficient of variation under small Gaussian perturbations.

    Re-runs the explanation function on ``n_bootstrap`` noisy copies of
    ``x`` and measures the mean coefficient of variation (std / |mean|)
    across features.

    Parameters
    ----------
    explain_fn : callable
        Function that, given an instance, returns an attribution vector.
    x : numpy.ndarray
        Input instance of shape ``(n_features,)``.
    n_bootstrap : int, default 5
        Number of perturbed runs.
    perturbation_scale : float, default 0.01
        Gaussian noise standard deviation.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    float
        Mean CoV. Lower is better. Raw values may be large when features
        have near-zero means, so log-normalize for display.
    """
    rng = rng if rng is not None else np.random.default_rng()
    runs: list[NDArray[np.float64]] = []
    for _ in range(n_bootstrap):
        noise = rng.normal(0.0, perturbation_scale, size=x.shape)
        runs.append(explain_fn(x + noise))
    arr = np.array(runs)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    cov = np.where(np.abs(mean) > 1e-10, std / np.abs(mean), 0.0)
    return float(np.mean(cov))


def identity_score(
    explain_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    n_runs: int = 5,
    rtol: float = 1e-5,
) -> float:
    """Fraction of repeated runs that return the same output.

    Deterministic methods (e.g. TreeSHAP) return 1.0. Stochastic methods
    (e.g. LIME) return values closer to ``1/n_runs``.

    Parameters
    ----------
    explain_fn : callable
        Function that, given an instance, returns an attribution vector.
    x : numpy.ndarray
        Input instance.
    n_runs : int, default 5
        Number of runs to compare.
    rtol : float, default 1e-5
        Relative tolerance for ``numpy.allclose``.

    Returns
    -------
    float
        Value in ``[1 / n_runs, 1]``. Higher is better.
    """
    runs = [explain_fn(x) for _ in range(n_runs)]
    ref = runs[0]
    matches = sum(1 for r in runs[1:] if np.allclose(r, ref, rtol=rtol))
    return (matches + 1) / n_runs


def evaluate_stability(
    X: NDArray[np.float64],
    explanations: NDArray[np.float64],
    indices: list[int] | NDArray[np.int64],
    explain_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    n_cov_sample: int = 20,
    n_bootstrap: int = 5,
    perturbation_scale: float = 0.01,
    n_identity_runs: int = 5,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Run all four stability metrics and return a dictionary.

    Returns keys: ``Lipschitz``, ``Kendall_tau``, ``CoV_Bootstrap``,
    ``Identity``.
    """
    indices = np.asarray(indices)
    lip = lipschitz_stability(X, explanations, indices)
    kendall = rank_stability_kendall(explanations)

    covs, ids = [], []
    for i in range(min(n_cov_sample, len(indices))):
        x = X[indices[i]]
        covs.append(
            cov_bootstrap(
                explain_fn, x, n_bootstrap=n_bootstrap,
                perturbation_scale=perturbation_scale, rng=rng,
            )
        )
        ids.append(identity_score(explain_fn, x, n_runs=n_identity_runs))

    return {
        "Lipschitz": lip,
        "Kendall_tau": kendall,
        "CoV_Bootstrap": float(np.mean(covs)),
        "Identity": float(np.mean(ids)),
    }
