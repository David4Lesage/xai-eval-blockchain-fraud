"""Tests for the BRAS metric family."""

from __future__ import annotations

import numpy as np

from xai_blockchain_framework.metrics.bras import (
    bras_score,
    domain_violation_rate,
    evaluate_bras,
    rule_alignment_score,
    top_k_indices,
)


class TestTopKIndices:
    def test_selects_largest_magnitudes(self) -> None:
        attr = np.array([0.1, -0.9, 0.3, 0.5, -0.05])
        assert top_k_indices(attr, k=2) == {1, 3}

    def test_empty_k(self) -> None:
        attr = np.array([0.1, 0.2])
        assert top_k_indices(attr, k=0) == set()


class TestRuleAlignmentScore:
    def test_full_alignment(self) -> None:
        attr = np.array([0.0, 1.0, 0.0, 0.8, 0.0])
        relevant = {1, 3}
        assert rule_alignment_score(attr, relevant, k=2) == 1.0

    def test_no_alignment(self) -> None:
        attr = np.array([1.0, 0.0, 0.0, 0.8, 0.0])
        relevant = {1, 2}
        assert rule_alignment_score(attr, relevant, k=2) == 0.0

    def test_partial_alignment(self) -> None:
        attr = np.array([0.0, 1.0, 0.0, 0.8, 0.0])
        relevant = {1, 2}
        assert rule_alignment_score(attr, relevant, k=2) == 0.5


class TestDomainViolationRate:
    def test_no_violation(self) -> None:
        attr = np.array([0.9, 0.0, 0.8])
        contradictory = {1}
        assert domain_violation_rate(attr, contradictory, k=2) == 0

    def test_violation_present(self) -> None:
        attr = np.array([0.9, 0.7, 0.0])
        contradictory = {1}
        assert domain_violation_rate(attr, contradictory, k=2) == 1

    def test_empty_contradictory(self) -> None:
        attr = np.array([0.9, 0.7, 0.0])
        assert domain_violation_rate(attr, set(), k=2) == 0


class TestBrasScore:
    def test_perfect(self) -> None:
        assert bras_score(1.0, 0.0) == 1.0

    def test_worst(self) -> None:
        assert bras_score(0.0, 1.0) == 0.0

    def test_alpha_weight(self) -> None:
        # alpha=0 -> BRAS = 1 - DVR
        assert bras_score(1.0, 0.5, alpha=0.0) == 0.5
        # alpha=1 -> BRAS = RAS
        assert bras_score(0.25, 0.0, alpha=1.0) == 0.25


class TestEvaluateBras:
    def test_perfect_alignment(self) -> None:
        X = np.zeros((3, 5))
        attrs = np.array([
            [0.0, 1.0, 0.0, 0.9, 0.0],
            [0.0, 1.0, 0.0, 0.9, 0.0],
            [0.0, 1.0, 0.0, 0.9, 0.0],
        ])

        def rule(x: np.ndarray, n: int) -> tuple[set[int], set[int]]:
            return {1, 3}, set()

        result = evaluate_bras(X, attrs, indices=[0, 1, 2], rule_fn=rule, k=2, alpha=0.5)
        assert result["RAS"] == 1.0
        assert result["DVR"] == 0.0
        assert result["BRAS"] == 1.0
        assert result["N_eval"] == 3
