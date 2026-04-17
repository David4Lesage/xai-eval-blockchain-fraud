"""Tests for the normalization utilities."""

from __future__ import annotations

import numpy as np
import pytest

from xai_blockchain_framework.utils.normalization import (
    log_normalize,
    min_max_normalize,
)


class TestMinMaxNormalize:
    def test_maps_min_to_zero_and_max_to_one(self) -> None:
        out = min_max_normalize([0.0, 5.0, 10.0])
        assert np.allclose(out, [0.0, 0.5, 1.0])

    def test_lower_better_inverts(self) -> None:
        out = min_max_normalize([0.0, 5.0, 10.0], lower_better=True)
        assert np.allclose(out, [1.0, 0.5, 0.0])

    def test_constant_input_returns_half(self) -> None:
        out = min_max_normalize([3.0, 3.0, 3.0])
        assert np.allclose(out, [0.5, 0.5, 0.5])


class TestLogNormalize:
    def test_basic_range(self) -> None:
        out = log_normalize([0.0, 9.0, 99.0, 999.0])
        # log10(1+x) is [0, 1, 2, 3] -> min-max -> [0, 1/3, 2/3, 1]
        assert np.allclose(out, [0.0, 1 / 3, 2 / 3, 1.0], atol=1e-6)

    def test_lower_better_inverts(self) -> None:
        out = log_normalize([0.0, 9.0, 99.0, 999.0], lower_better=True)
        assert np.allclose(out, [1.0, 2 / 3, 1 / 3, 0.0], atol=1e-6)

    def test_handles_aberrant_ranges(self) -> None:
        # Simulates the Lipschitz case: [0.01, 0.42, 62, 122779, 493846]
        values = [0.01, 0.42, 62.0, 122_779.0, 493_846.0]
        out = log_normalize(values, lower_better=True)
        # Smallest value maps to 1.0, largest to 0.0.
        assert out[0] == pytest.approx(1.0, abs=1e-6)
        assert out[-1] == pytest.approx(0.0, abs=1e-6)
        # Monotonic.
        assert all(out[i] > out[i + 1] for i in range(len(out) - 1))

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            log_normalize([-1.0, 0.0, 1.0])
