"""Tests for LLM agent metrics."""

from __future__ import annotations

from xai_blockchain_framework.metrics.llm import (
    cohen_kappa_pair,
    decision_accuracy,
    expected_calibration_error,
    explanation_utilization,
)


class TestDecisionAccuracy:
    def test_all_correct(self) -> None:
        decisions = ["fraud", "legitimate", "fraud"]
        truth = [1, 0, 1]
        assert decision_accuracy(decisions, truth) == 1.0

    def test_all_wrong(self) -> None:
        decisions = ["fraud", "legitimate"]
        truth = [0, 1]
        assert decision_accuracy(decisions, truth) == 0.0

    def test_empty(self) -> None:
        assert decision_accuracy([], []) == 0.0


class TestExpectedCalibrationError:
    def test_perfectly_calibrated(self) -> None:
        # Confidence exactly matches accuracy bin by bin.
        conf = [0.9] * 10
        correct = [1] * 9 + [0]  # 90% accuracy
        ece = expected_calibration_error(conf, correct, n_bins=10)
        assert ece < 0.05

    def test_miscalibrated_overconfident(self) -> None:
        conf = [0.95] * 10
        correct = [0] * 5 + [1] * 5  # 50% accuracy
        ece = expected_calibration_error(conf, correct, n_bins=10)
        assert ece > 0.3


class TestExplanationUtilization:
    def test_all_features_mentioned(self) -> None:
        import numpy as np
        attr = np.array([0.0, 1.0, 0.0, 0.9, 0.0])
        features = ["f0", "f1", "f2", "f3", "f4"]
        reasoning = "The key drivers are f1 and f3 which push toward fraud."
        assert explanation_utilization(reasoning, attr, features, k=2) == 1.0

    def test_no_feature_mentioned(self) -> None:
        import numpy as np
        attr = np.array([0.0, 1.0, 0.0, 0.9, 0.0])
        features = ["alpha", "beta", "gamma", "delta", "epsilon"]
        reasoning = "This transaction is suspicious."
        assert explanation_utilization(reasoning, attr, features, k=2) == 0.0


class TestCohenKappaPair:
    def test_perfect_agreement(self) -> None:
        a = ["fraud", "legitimate", "fraud", "legitimate"]
        b = ["fraud", "legitimate", "fraud", "legitimate"]
        assert cohen_kappa_pair(a, b) == 1.0

    def test_perfect_disagreement(self) -> None:
        a = ["fraud", "fraud", "legitimate", "legitimate"]
        b = ["legitimate", "legitimate", "fraud", "fraud"]
        assert cohen_kappa_pair(a, b) == -1.0
