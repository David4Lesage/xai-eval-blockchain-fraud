"""Blockchain domain rules for the Ethereum Fraud Detection dataset.

Unlike Elliptic, the Ethereum dataset uses named features, so the rules can
reference behavior patterns directly (ERC20 volume, transaction frequency,
address diversity, contract interactions).

Feature groups (45 features total):

- ``0, 1, 2``                     : temporal (frequency, intervals)
- ``3, 4, 17, 18, 19, 39``        : volume (total Ether sent/received)
- ``5, 14, 15, 16, 20, 44``       : contract interactions
- ``6, 7, 25, 26``                : address diversity (mixing signal)
- ``8 - 13``                      : Ether amount statistics
- ``21 - 38``                     : ERC20 token activity
- ``40 - 44``                     : engineered ratios

All indices are drawn from the domain set; no feature is marked as
contradictory by default (the dataset is already pre-selected for fraud
relevance).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ETH_TEMPORAL = {0, 1, 2}
ETH_VOLUME = {3, 4, 17, 18, 19, 39}
ETH_AMOUNT = {8, 9, 10, 11, 12, 13}
ETH_CONTRACT = {5, 14, 15, 16, 20, 44}
ETH_DIVERSITY = {6, 7, 25, 26}
ETH_ERC20 = set(range(21, 39))
ETH_RATIOS = {40, 41, 42, 43, 44}

ETHEREUM_DOMAIN_FEATURES = (
    ETH_TEMPORAL | ETH_VOLUME | ETH_AMOUNT | ETH_CONTRACT
    | ETH_DIVERSITY | ETH_ERC20 | ETH_RATIOS
)

ETHEREUM_CONTRA_FEATURES: set[int] = set()

# Activation thresholds (values above these, after standard normalization,
# flag the associated behavioral pattern).
TOKEN_FRAUD_THRESHOLD = 2.0
TEMPORAL_THRESHOLD = -1.0
MIXING_THRESHOLD = 2.0
VOLUME_THRESHOLD = 2.0
CONTRACT_THRESHOLD = 1.0


def ethereum_rules(
    features: NDArray[np.float64],
    n_features: int,
) -> tuple[set[int], set[int]]:
    """Return ``(relevant, contradictory)`` feature indices for one instance.

    Rules applied
    -------------
    - **R1 Token fraud pattern**: active when ``Total ERC20 Tnx`` (index 21)
      exceeds 2σ. Flags the 18 ERC20 features as relevant.
    - **R2 Temporal anomaly**: active when ``Avg min between sent tnx``
      (index 0) is more than 1σ below the mean (very high frequency).
      Flags temporal features.
    - **R3 Mixing / distribution**: active when ``Unique sent to addresses``
      (index 7) exceeds 2σ. Flags diversity features.
    - **R4 Volume anomaly**: active when ``Total Ether sent`` (index 18)
      exceeds 2σ. Flags volume and amount features.
    - **R5 Contract interactions**: active when ``Created contracts``
      (index 5) exceeds 1σ. Flags contract features.
    - **Fallback**: if no rule activates, include the base set of volume,
      temporal, and amount features so that the top-k explanation can still
      meaningfully intersect with the domain.

    Parameters
    ----------
    features : numpy.ndarray
        Feature vector of shape ``(n_features,)``.
    n_features : int
        Total feature count.

    Returns
    -------
    tuple of set
        ``(relevant_features, contradictory_features)``.
    """
    relevant: set[int] = set()

    def _value(idx: int) -> float:
        return float(features[idx]) if idx < n_features else 0.0

    # R1: Token fraud
    if _value(21) > TOKEN_FRAUD_THRESHOLD:
        relevant |= ETH_ERC20

    # R2: Temporal anomaly
    if _value(0) < TEMPORAL_THRESHOLD:
        relevant |= ETH_TEMPORAL

    # R3: Mixing
    if _value(7) > MIXING_THRESHOLD:
        relevant |= ETH_DIVERSITY

    # R4: Volume
    if _value(18) > VOLUME_THRESHOLD:
        relevant |= ETH_VOLUME | ETH_AMOUNT

    # R5: Contracts
    if _value(5) > CONTRACT_THRESHOLD:
        relevant |= ETH_CONTRACT

    # Fallback
    if not relevant:
        relevant = ETH_VOLUME | ETH_TEMPORAL | ETH_AMOUNT

    relevant = {i for i in relevant if i < n_features}
    contradictory = {i for i in ETHEREUM_CONTRA_FEATURES if i < n_features}
    return relevant, contradictory
