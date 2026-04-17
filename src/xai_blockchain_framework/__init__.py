"""XAI Evaluation Framework for Blockchain Fraud Detection.

A reproducible, multi-dimensional framework for evaluating explainable AI (XAI)
methods applied to blockchain fraud detection, structured around four
complementary modules: fidelity, stability, blockchain rule alignment, and
LLM-agent validation.
"""

from xai_blockchain_framework.config import (
    CONFIG,
    DEFAULT_MODELS,
    PATHS,
    set_seed,
)

__version__ = "0.1.0"

__all__ = [
    "CONFIG",
    "DEFAULT_MODELS",
    "PATHS",
    "set_seed",
    "__version__",
]
