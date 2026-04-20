"""Domain-specific rule functions and feature groups for the BRAS metric."""

from xai_blockchain_framework.rules.elliptic_rules import (
    ELLIPTIC_AMOUNT,
    ELLIPTIC_CONTRA_FEATURES,
    ELLIPTIC_DOMAIN_FEATURES,
    ELLIPTIC_NEIGHBORHOOD,
    ELLIPTIC_STRUCTURAL,
    ELLIPTIC_TEMPORAL,
    elliptic_feature_label,
    elliptic_rules,
)
from xai_blockchain_framework.rules.ethereum_rules import (
    ETH_AMOUNT,
    ETH_CONTRACT,
    ETH_DIVERSITY,
    ETH_ERC20,
    ETH_RATIOS,
    ETH_TEMPORAL,
    ETH_VOLUME,
    ETHEREUM_CONTRA_FEATURES,
    ETHEREUM_DOMAIN_FEATURES,
    ethereum_rules,
)

__all__ = [
    # Elliptic
    "elliptic_rules",
    "elliptic_feature_label",
    "ELLIPTIC_STRUCTURAL",
    "ELLIPTIC_AMOUNT",
    "ELLIPTIC_TEMPORAL",
    "ELLIPTIC_NEIGHBORHOOD",
    "ELLIPTIC_DOMAIN_FEATURES",
    "ELLIPTIC_CONTRA_FEATURES",
    # Ethereum
    "ethereum_rules",
    "ETH_TEMPORAL",
    "ETH_VOLUME",
    "ETH_AMOUNT",
    "ETH_CONTRACT",
    "ETH_DIVERSITY",
    "ETH_ERC20",
    "ETH_RATIOS",
    "ETHEREUM_DOMAIN_FEATURES",
    "ETHEREUM_CONTRA_FEATURES",
]
