"""Prompt templates for Module 4 LLM-agent evaluation.

Three conditions of progressively richer information:

- **C1** — raw features only, no model probability, no explanation.
- **C2** — C1 plus the model's fraud probability and the top-k XAI
  attributions (feature name, value, and direction).
- **C3** — C2 plus the nine explanation-quality scores from Modules 1-3
  (Comprehensiveness, Sufficiency, Infidelity, Lipschitz-normalized,
  Kendall τ, Identity, RAS, DVR, BRAS).

The C3 prompt is designed to be **informative, not prescriptive**: it does
not instruct the agent how to weight the scores. This avoids confounding
the experiment with author biases.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from xai_blockchain_framework.rules.elliptic_rules import elliptic_feature_label


SYSTEM_PROMPT = """You are an expert blockchain fraud analyst.
You will be given information about a blockchain transaction and must decide whether it is fraudulent or legitimate.

Analyze ALL provided information carefully. Respond ONLY with this JSON (no other text):
{
    "decision": "fraud" or "legitimate",
    "confidence": <float 0.0 to 1.0>,
    "reasoning": "<your analysis explaining WHY, referencing specific evidence>",
    "explanation": "<REQUIRED for C3 only: a human-readable summary naming (1) which top features most influenced your decision and why, (2) which quality scores from Modules 1-3 affected your confidence, (3) a plain-language conclusion accessible to a non-technical compliance officer. Omit this field for C1/C2.>"
}"""


_DATASET_CONTEXT = {
    "Elliptic": (
        "This is a Bitcoin transaction from the Elliptic dataset. "
        "Feature groups: tx_structure (inputs/outputs), tx_amount (BTC values and fees), "
        "tx_temporal (timing and frequency), tx_auxiliary (metadata with limited fraud "
        "relevance), neighbor_agg (aggregated statistics of neighboring transactions)."
    ),
    "Ethereum": (
        "This is an Ethereum address from the Ethereum Fraud Detection dataset. "
        "Features describe transaction patterns, ERC20 token activity, address "
        "diversity, contract interactions, and Ether balances."
    ),
}

_QUALITY_SCORE_LABELS: list[tuple[str, str, bool]] = [
    ("Comprehensiveness", "Comprehensiveness (do top features matter for the prediction?)", True),
    ("Sufficiency", "Sufficiency (can top features alone reproduce the prediction?)", True),
    ("Infidelity", "Infidelity (discrepancy between explanation and actual model behavior)", False),
    ("Lipschitz_norm", "Lipschitz stability, normalized (sensitivity to small input changes)", True),
    ("Kendall_tau", "Rank Stability (consistency of feature rankings across similar inputs)", True),
    ("Identity", "Determinism (same input always produces same explanation)", True),
    ("RAS", "Rule Alignment (overlap with known blockchain fraud patterns)", True),
    ("DVR", "Domain Violation Rate (fraction of top features contradicting the domain)", False),
    ("BRAS", "Overall Domain Coherence (combined alignment and no violations)", True),
]


def _feature_label(feature_idx: int, feature_names: list[str] | None, dataset: str) -> str:
    """Return a human-readable label for a feature index."""
    if feature_names and feature_idx < len(feature_names):
        return feature_names[feature_idx]
    if dataset == "Elliptic":
        return elliptic_feature_label(feature_idx)
    return f"f{feature_idx}"


def build_prompts(
    dataset: str,
    x: NDArray[np.float64],
    model_name: str,
    model_fraud_probability: float,
    attributions: NDArray[np.float64],
    feature_names: list[str] | None,
    quality_scores: dict[str, float] | None,
    top_k: int = 5,
) -> dict[str, str]:
    """Build the three user prompts (C1, C2, C3) for one transaction.

    Parameters
    ----------
    dataset : str
        ``"Elliptic"`` or ``"Ethereum"``.
    x : numpy.ndarray
        Feature vector of shape ``(n_features,)``.
    model_name : str
        Display name of the ML model (e.g. ``"Random Forest"``).
    model_fraud_probability : float
        Model's predicted fraud probability in ``[0, 1]``.
    attributions : numpy.ndarray
        Attribution vector for this instance, shape ``(n_features,)``.
    feature_names : list of str, optional
        Feature display names. If None, anonymized labels are used.
    quality_scores : dict of str to float, optional
        Module 1/2/3 scores keyed by the strings in ``_QUALITY_SCORE_LABELS``.
        Required for C3.
    top_k : int, default 5
        Number of top features to include in the XAI explanation.

    Returns
    -------
    dict
        Keys ``"C1"``, ``"C2"``, ``"C3"`` mapping to prompt strings.
    """
    context = _DATASET_CONTEXT.get(dataset, f"This is a transaction from the {dataset} dataset.")

    # ----- C1: raw features -----
    magnitudes = np.argsort(-np.abs(x))
    c1_lines = []
    for fi in magnitudes[:10]:
        label = _feature_label(int(fi), feature_names, dataset)
        c1_lines.append(f"  {label} = {float(x[fi]):.4f}")

    c1 = (
        f"TRANSACTION ANALYSIS - {dataset}\n"
        f"{context}\n\n"
        f"Transaction features (top 10 by magnitude):\n"
        + "\n".join(c1_lines)
        + "\n\nBased on these raw features, classify this transaction as fraud or legitimate."
    )

    # ----- C2: + model prediction + XAI -----
    top_attr_idx = np.argsort(-np.abs(attributions))[:top_k]
    xai_lines = []
    for rank, fi in enumerate(top_attr_idx, start=1):
        label = _feature_label(int(fi), feature_names, dataset)
        w = float(attributions[fi])
        direction = "pushes toward FRAUD" if w > 0 else "pushes toward LEGITIMATE"
        xai_lines.append(
            f"  {rank}. {label} = {float(x[fi]):.4f}  "
            f"attribution: {w:+.4f} ({direction})"
        )

    c2 = (
        f"TRANSACTION ANALYSIS - {dataset}\n"
        f"{context}\n\n"
        f"ML Model prediction ({model_name}): "
        f"{model_fraud_probability:.1%} probability of fraud.\n\n"
        f"XAI explanation, top-{top_k} most influential features:\n"
        + "\n".join(xai_lines)
        + "\n\nUse both the model's prediction AND the feature explanations to make your decision."
    )

    # ----- C3: + quality scores -----
    c3 = c2
    if quality_scores:
        score_lines = []
        for key, label, higher_better in _QUALITY_SCORE_LABELS:
            if key in quality_scores:
                hint = "higher is better" if higher_better else "lower is better"
                score_lines.append(
                    f"  {label}: {float(quality_scores[key]):.4f}  ({hint})"
                )
        c3 = (
            f"TRANSACTION ANALYSIS - {dataset}\n"
            f"{context}\n\n"
            f"ML Model prediction ({model_name}): "
            f"{model_fraud_probability:.1%} probability of fraud.\n\n"
            f"XAI explanation, top-{top_k} most influential features:\n"
            + "\n".join(xai_lines)
            + "\n\nEXPLANATION QUALITY ASSESSMENT (from Modules 1-3):\n"
            + "These scores measure how reliable the above explanation is.\n"
            + "\n".join(score_lines)
            + "\n\nConsider the explanation quality scores when weighing the feature "
            + "attributions. In your response, include an 'explanation' field naming "
            + "the top features that drove your decision and the quality scores that "
            + "influenced your confidence. Classify as fraud or legitimate."
        )

    return {"C1": c1, "C2": c2, "C3": c3}
