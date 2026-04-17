"""Blockchain domain rules for the Elliptic Bitcoin dataset.

The Elliptic dataset contains 166 anonymized features organized by position:

- ``0 - 19``  : transaction structure (inputs / outputs)
- ``20 - 49`` : amount-related features (BTC values, fees)
- ``50 - 69`` : temporal features (timing, frequency)
- ``70 - 93`` : auxiliary metadata (contradictory features)
- ``94 - 165``: aggregated neighborhood statistics

Feature names are not publicly available, so the rules are defined on
position ranges rather than names. This is a known methodological limitation
discussed in the paper.

The function :func:`elliptic_rules` returns a tuple
``(relevant_features, contradictory_features)`` for a given feature vector.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ELLIPTIC_STRUCTURAL = set(range(0, 20))
ELLIPTIC_AMOUNT = set(range(20, 50))
ELLIPTIC_TEMPORAL = set(range(50, 70))
ELLIPTIC_NEIGHBORHOOD = set(range(94, 166))

ELLIPTIC_DOMAIN_FEATURES = (
    ELLIPTIC_STRUCTURAL | ELLIPTIC_AMOUNT | ELLIPTIC_TEMPORAL | ELLIPTIC_NEIGHBORHOOD
)

ELLIPTIC_CONTRA_FEATURES = set(range(70, 94))

NEIGHBORHOOD_ACTIVATION_THRESHOLD = 2.0


def elliptic_rules(
    features: NDArray[np.float64],
    n_features: int,
) -> tuple[set[int], set[int]]:
    """Return ``(relevant, contradictory)`` feature indices for one instance.

    Rules applied
    -------------
    - **R1 Structural anomaly** (always active): fraudulent transactions
      tend to have atypical input/output structures, so features 0–19 are
      always considered domain-relevant.
    - **R2 Amount anomaly** (always active): features 20–49 cover amounts
      and fees, known fraud signals.
    - **R3 Temporal anomaly** (always active): features 50–69 cover timing
      patterns (unusual hours, bursts).
    - **R4 Neighborhood anomaly** (conditional): aggregated neighborhood
      features 94–165 are included only when at least one of them has an
      absolute value above the activation threshold of 2σ after
      normalization, indicating the transaction sits in an unusual local
      subgraph.
    - **Contradictory**: features 70–93 are auxiliary metadata that is
      rarely informative about fraud; their presence in the top-k
      explanation is a domain violation.

    Parameters
    ----------
    features : numpy.ndarray
        Feature vector of shape ``(n_features,)``.
    n_features : int
        Total feature count. Used only to cap neighborhood indices when a
        reduced feature set is being used in ablations.

    Returns
    -------
    tuple of set
        ``(relevant_features, contradictory_features)`` as sets of indices.
    """
    relevant: set[int] = set()
    relevant |= ELLIPTIC_STRUCTURAL
    relevant |= ELLIPTIC_AMOUNT
    relevant |= ELLIPTIC_TEMPORAL

    # Neighborhood rule: activate only when a neighborhood feature is extreme.
    neigh_indices = [i for i in ELLIPTIC_NEIGHBORHOOD if i < n_features]
    if neigh_indices and np.max(np.abs(features[neigh_indices])) > NEIGHBORHOOD_ACTIVATION_THRESHOLD:
        relevant |= set(neigh_indices)

    contradictory = {i for i in ELLIPTIC_CONTRA_FEATURES if i < n_features}
    relevant = {i for i in relevant if i < n_features}
    return relevant, contradictory


def elliptic_feature_label(feature_idx: int) -> str:
    """Human-readable semantic label for an Elliptic feature index.

    Used in LLM prompts to make anonymized Elliptic features interpretable
    for agents that refuse to classify based on raw ``f12, f13, ...``
    feature names.
    """
    if feature_idx < 20:
        return f"tx_structure_f{feature_idx}"
    if feature_idx < 50:
        return f"tx_amount_f{feature_idx}"
    if feature_idx < 70:
        return f"tx_temporal_f{feature_idx}"
    if feature_idx < 94:
        return f"tx_auxiliary_f{feature_idx}"
    return f"neighbor_agg_f{feature_idx}"
