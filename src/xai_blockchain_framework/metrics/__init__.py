"""Evaluation metrics for the four framework modules."""

from xai_blockchain_framework.metrics.bras import (
    bras_score,
    domain_violation_rate,
    evaluate_bras,
    rule_alignment_score,
    top_k_indices,
)
from xai_blockchain_framework.metrics.fidelity import (
    comprehensiveness,
    evaluate_fidelity,
    infidelity,
    sufficiency,
)
from xai_blockchain_framework.metrics.gnn_fidelity import (
    gnn_comprehensiveness,
    gnn_infidelity,
    gnn_sufficiency,
)
from xai_blockchain_framework.metrics.llm import (
    cohen_kappa_pair,
    decision_accuracy,
    expected_calibration_error,
    explanation_utilization,
    mean_inter_agent_kappa,
)
from xai_blockchain_framework.metrics.stability import (
    cov_bootstrap,
    evaluate_stability,
    identity_score,
    lipschitz_stability,
    rank_stability_kendall,
)

__all__ = [
    # Fidelity
    "comprehensiveness",
    "sufficiency",
    "infidelity",
    "evaluate_fidelity",
    "gnn_comprehensiveness",
    "gnn_sufficiency",
    "gnn_infidelity",
    # Stability
    "lipschitz_stability",
    "rank_stability_kendall",
    "cov_bootstrap",
    "identity_score",
    "evaluate_stability",
    # BRAS
    "top_k_indices",
    "rule_alignment_score",
    "domain_violation_rate",
    "bras_score",
    "evaluate_bras",
    # LLM
    "decision_accuracy",
    "expected_calibration_error",
    "explanation_utilization",
    "cohen_kappa_pair",
    "mean_inter_agent_kappa",
]
