"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed-seed random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def small_attributions(rng: np.random.Generator) -> np.ndarray:
    """Tiny attribution matrix (5 instances, 10 features)."""
    return rng.normal(size=(5, 10))


@pytest.fixture
def small_features(rng: np.random.Generator) -> np.ndarray:
    """Tiny feature matrix (5 instances, 10 features)."""
    return rng.normal(size=(5, 10))
