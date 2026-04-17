"""Module 4: Metrics for evaluating LLM-agent decisions.

These functions are applied to the outputs of multiple LLM agents across
three experimental conditions (C1 = raw features only, C2 = C1 + model
prediction + XAI explanation, C3 = C2 + explanation-quality scores).
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import cohen_kappa_score


def decision_accuracy(decisions: Iterable[str], ground_truth: Iterable[int]) -> float:
    """Fraction of agent decisions that match the ground truth.

    ``decisions`` must contain the strings ``"fraud"`` or ``"legitimate"``.
    ``ground_truth`` is encoded as 1 (fraud) or 0 (legitimate).
    """
    decs = [1 if str(d).lower() == "fraud" else 0 for d in decisions]
    labels = [int(g) for g in ground_truth]
    if not decs:
        return 0.0
    correct = sum(1 for d, g in zip(decs, labels) if d == g)
    return correct / len(decs)


def expected_calibration_error(
    confidences: Iterable[float],
    correctness: Iterable[int],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) with equal-width confidence bins.

    Compares the agent's stated confidence against its empirical accuracy
    within each bin. A value of 0.0 indicates perfect calibration.

    Parameters
    ----------
    confidences : iterable of float
        Agent confidences in ``[0, 1]``.
    correctness : iterable of int
        Binary correctness indicators (1 if the decision matches ground
        truth, else 0).
    n_bins : int, default 10
        Number of bins to partition the confidence axis.

    Returns
    -------
    float
        ECE in ``[0, 1]``. Lower is better.
    """
    conf = np.asarray(list(confidences), dtype=np.float64)
    acc = np.asarray(list(correctness), dtype=np.float64)
    n = len(conf)
    if n == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        bin_acc = float(acc[mask].mean())
        bin_conf = float(conf[mask].mean())
        total += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(total)


def explanation_utilization(
    reasoning: str,
    attributions: NDArray[np.float64],
    feature_names: list[str],
    k: int = 5,
) -> float:
    """Fraction of the top-k XAI features mentioned in the agent's reasoning.

    Simple case-insensitive substring check: a feature counts as mentioned
    if its name appears anywhere in the reasoning text.

    Parameters
    ----------
    reasoning : str
        The ``reasoning`` field extracted from the agent's JSON response.
    attributions : numpy.ndarray
        Attribution vector of shape ``(n_features,)``.
    feature_names : list of str
        Display names for each feature (same ordering as ``attributions``).
    k : int, default 5
        Number of top features to check.

    Returns
    -------
    float
        Value in ``[0, 1]``. Higher means the agent referenced the
        explanation more thoroughly.
    """
    if not reasoning:
        return 0.0
    top = np.argsort(-np.abs(attributions))[:k]
    low = reasoning.lower()
    count = sum(1 for fi in top if fi < len(feature_names) and feature_names[fi].lower() in low)
    return count / k


def cohen_kappa_pair(decisions_a: Iterable[str], decisions_b: Iterable[str]) -> float:
    """Cohen's κ between two agents' binary decisions on the same instances.

    ``decisions_a`` and ``decisions_b`` must have the same length and contain
    the strings ``"fraud"`` or ``"legitimate"``.
    """
    a = [1 if str(d).lower() == "fraud" else 0 for d in decisions_a]
    b = [1 if str(d).lower() == "fraud" else 0 for d in decisions_b]
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b))


def mean_inter_agent_kappa(
    agent_decisions: dict[str, list[str]],
) -> float:
    """Mean pairwise Cohen's κ across all agents.

    Parameters
    ----------
    agent_decisions : dict
        Mapping from agent name to their list of decisions. All lists must
        have the same length.

    Returns
    -------
    float
        Mean κ. Returns NaN if fewer than two agents are provided.
    """
    names = list(agent_decisions.keys())
    if len(names) < 2:
        return float("nan")
    kappas: list[float] = []
    for a, b in combinations(names, 2):
        kappas.append(cohen_kappa_pair(agent_decisions[a], agent_decisions[b]))
    valid = [k for k in kappas if not np.isnan(k)]
    return float(np.mean(valid)) if valid else float("nan")
