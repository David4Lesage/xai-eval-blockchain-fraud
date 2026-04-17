# Methodology

The framework evaluates XAI methods along four complementary dimensions.
Each module answers a different question about the quality of an
explanation.

## Module 1 — Fidelity

**Question**: does the explanation faithfully reflect the model's decision?

Three metrics are computed for each instance and averaged:

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **Comprehensiveness@k** | Drop in predicted fraud probability when the top-k features are zeroed out. | Higher is better |
| **Sufficiency@k** | Drop in predicted fraud probability when only the top-k features are kept. | Lower is better |
| **Infidelity** | Mean squared error between the attribution-predicted change and the model's actual change under small Gaussian perturbations. | Lower is better |

## Module 2 — Stability

**Question**: do similar inputs and repeated runs produce similar
explanations?

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **Lipschitz** | Mean ratio of explanation difference over input distance among k-nearest neighbors. | Lower is better |
| **Kendall τ** | Mean Kendall τ between feature rankings of consecutive instances. | Higher is better |
| **CoV Bootstrap** | Coefficient of variation of attributions under small input perturbations. | Lower is better |
| **Identity** | Fraction of repeated runs that return identical attributions for the same input. | Higher is better |

Raw Lipschitz and CoV values can span many orders of magnitude. The
framework reports both the raw value and two normalized variants:

- `Lipschitz_log` = `log10(1 + Lipschitz)` for readability.
- `Lipschitz_norm` = min-max normalized in `[0, 1]` after log transform,
  inverted so that 1.0 corresponds to the most stable configuration.

The same treatment is applied to CoV.

## Module 3 — BRAS (Blockchain Rule Alignment Score)

**Question**: is the explanation consistent with blockchain domain rules?

For each instance, domain-specific rule functions determine which features
are relevant and which are contradictory. The metric is:

- **RAS@k**: fraction of the top-k features that appear in the
  relevant-feature set. Higher is better (perfect alignment is 1.0).
- **DVR@k**: rate of instances whose top-k includes at least one
  contradictory feature. Lower is better.
- **BRAS** = `0.5 * RAS + 0.5 * (1 - DVR)`.

### Rule definitions

**Elliptic Bitcoin** (positions; features are anonymized):

- R1 (always active) — structural features (0–19)
- R2 (always active) — amount features (20–49)
- R3 (always active) — temporal features (50–69)
- R4 (active when any neighborhood feature exceeds 2σ) — neighborhood
  aggregates (94–165)
- Contradictory set — auxiliary metadata (70–93)

**Ethereum** (named features; rules conditional on input values):

- R1 — ERC20 fraud pattern (activates when `Total ERC20 Tnx > 2σ`)
- R2 — temporal anomaly (very high sending frequency)
- R3 — mixing (many unique destination addresses)
- R4 — volume anomaly (very high Ether sent)
- R5 — contract interactions (many created contracts)

## Module 4 — LLM-Agent validation

**Question**: does the explanation improve a downstream analyst's
decision?

Three frontier LLMs (Claude Opus 4.7, Gemini 3.1 Pro, GPT 5.4, accessed
via OpenRouter) classify each transaction under three conditions of
progressively richer information:

- **C1** — raw features only. No model probability, no XAI. This is the
  control baseline.
- **C2** — C1 plus the model's fraud probability and the top-k XAI
  attributions (feature name, value, direction).
- **C3** — C2 plus the nine explanation-quality scores from Modules 1–3
  (Comprehensiveness, Sufficiency, Infidelity, Lipschitz normalized,
  Kendall τ, Identity, RAS, DVR, BRAS). Provided as factual information
  without any instruction on how to use them.

The four metrics computed per (dataset, condition, agent) cell are:

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **Decision Accuracy (DA)** | Fraction of decisions matching the ground truth. | Higher is better |
| **Expected Calibration Error (ECE)** | Gap between stated confidence and empirical accuracy, equal-width bins. | Lower is better |
| **Explanation Utilization (EU)** | Fraction of the top-k XAI features referenced in the agent's free-text reasoning. | Higher is better |
| **Cohen's κ** | Pairwise agreement between agents on identical transactions. | Higher is better |

### ML baseline validation

The framework includes an independent check in notebook 09: a plain
Random Forest classifier is trained on the same feature vectors supplied
to the LLM agents, and its decisions are compared pairwise to each agent's
decisions via Cohen's κ. High agreement (κ ≈ 0.93 in our experiments)
supports the claim that LLM agents behave as coherent, non-arbitrary
decision-makers and not as random oracles.

## Hypotheses tested

1. **H1**: DA(C2) > DA(C1). XAI improves decisions.
2. **H2**: DA(C3) > DA(C2). Module 1–3 scores further improve decisions.
3. **H3**: ECE(C3) < ECE(C1). Richer information improves calibration.
4. **H4**: κ(C3) > κ(C1). Richer information increases inter-agent
   agreement.
5. **H5**: DA(high-BRAS config) > DA(low-BRAS config) in C2/C3. A
   better-aligned explanation yields better decisions.

See the paper for the empirical outcomes of each hypothesis.
