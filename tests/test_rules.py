"""Tests for the domain rules used by BRAS."""

from __future__ import annotations

import numpy as np

from xai_blockchain_framework.rules.elliptic_rules import (
    ELLIPTIC_AMOUNT,
    ELLIPTIC_CONTRA_FEATURES,
    ELLIPTIC_NEIGHBORHOOD,
    ELLIPTIC_STRUCTURAL,
    ELLIPTIC_TEMPORAL,
    elliptic_rules,
)
from xai_blockchain_framework.rules.ethereum_rules import (
    ETH_CONTRACT,
    ETH_DIVERSITY,
    ETH_ERC20,
    ETH_TEMPORAL,
    ETH_VOLUME,
    ethereum_rules,
)


class TestEllipticRules:
    def test_always_relevant_groups_are_included(self) -> None:
        x = np.zeros(166)
        relevant, contra = elliptic_rules(x, n_features=166)
        assert ELLIPTIC_STRUCTURAL.issubset(relevant)
        assert ELLIPTIC_AMOUNT.issubset(relevant)
        assert ELLIPTIC_TEMPORAL.issubset(relevant)

    def test_contradictory_features_are_marked(self) -> None:
        x = np.zeros(166)
        _, contra = elliptic_rules(x, n_features=166)
        assert contra == ELLIPTIC_CONTRA_FEATURES

    def test_neighborhood_activates_on_extreme_value(self) -> None:
        x = np.zeros(166)
        x[100] = 3.0  # extreme neighborhood feature
        relevant, _ = elliptic_rules(x, n_features=166)
        assert ELLIPTIC_NEIGHBORHOOD.issubset(relevant)

    def test_neighborhood_inactive_when_normal(self) -> None:
        x = np.zeros(166)
        relevant, _ = elliptic_rules(x, n_features=166)
        # No neighborhood features should be in relevant when all values small
        assert ELLIPTIC_NEIGHBORHOOD.isdisjoint(relevant)

    def test_respects_reduced_feature_count(self) -> None:
        x = np.zeros(50)
        relevant, contra = elliptic_rules(x, n_features=50)
        assert all(i < 50 for i in relevant)
        assert all(i < 50 for i in contra)


class TestEthereumRules:
    def test_fallback_when_no_rule_active(self) -> None:
        x = np.zeros(45)
        relevant, _ = ethereum_rules(x, n_features=45)
        assert relevant  # non-empty fallback
        assert ETH_TEMPORAL.issubset(relevant) or ETH_VOLUME.issubset(relevant)

    def test_token_fraud_activates_erc20(self) -> None:
        x = np.zeros(45)
        x[21] = 3.0
        relevant, _ = ethereum_rules(x, n_features=45)
        assert ETH_ERC20.issubset(relevant)

    def test_temporal_activates_on_low_interval(self) -> None:
        x = np.zeros(45)
        x[0] = -2.0
        relevant, _ = ethereum_rules(x, n_features=45)
        assert ETH_TEMPORAL.issubset(relevant)

    def test_mixing_activates_on_many_unique_addresses(self) -> None:
        x = np.zeros(45)
        x[7] = 3.0
        relevant, _ = ethereum_rules(x, n_features=45)
        assert ETH_DIVERSITY.issubset(relevant)

    def test_contract_activates_on_many_contracts(self) -> None:
        x = np.zeros(45)
        x[5] = 2.0
        relevant, _ = ethereum_rules(x, n_features=45)
        assert ETH_CONTRACT.issubset(relevant)
