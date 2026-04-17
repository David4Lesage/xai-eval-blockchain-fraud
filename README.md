# XAI Evaluation Framework for Blockchain Fraud Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: Research Code](https://img.shields.io/badge/status-research-orange)](#)

A reproducible, multi-dimensional framework for evaluating explainable AI (XAI)
methods applied to blockchain fraud detection. The framework complements
predictive metrics (F1, AUC) with a structured evaluation across **four
complementary dimensions**: fidelity, stability, domain-rule alignment, and
decision utility.

This repository accompanies the paper *Beyond Predictive Metrics: A
Multi-Dimensional Framework for XAI Evaluation in Blockchain Fraud Detection*
(2026, under review).

---

## Why this framework?

A systematic review of 49 recent studies shows that over **80% of works** on
XAI for blockchain fraud detection use the model's F1 or AUC as a proxy for
explanation quality. This is methodologically flawed: a highly accurate model
can still produce unfaithful, unstable, or domain-inconsistent explanations.

This framework addresses that gap with four evaluation modules:

| Module | Question answered | Metrics |
|--------|-------------------|---------|
| **M1. Fidelity** | Does the explanation faithfully reflect the model's reasoning? | Comprehensiveness, Sufficiency, Infidelity |
| **M2. Stability** | Do similar inputs produce similar explanations? | Lipschitz, Kendall τ, CoV, Identity |
| **M3. BRAS** *(novel)* | Is the explanation consistent with blockchain domain knowledge? | Rule Alignment Score, Domain Violation Rate, BRAS |
| **M4. LLM Agents** | Does the explanation improve decisions for a downstream analyst? | Decision Accuracy, ECE, Explanation Utilization, Cohen's κ |

**Headline result.** Providing XAI explanations to LLM agents raises
inter-agent agreement from κ=0.01 (chance level) to κ=0.95 (near-perfect
agreement) and decision accuracy from 0.53 to 0.87. A Machine Learning
baseline trained on the same features achieves κ=0.93 agreement with the LLM
agents, independently validating the framework.

---

## Repository layout

```
xai-eval-blockchain-fraud/
├── data/                   Raw and processed datasets (git-ignored)
├── docs/                   Methodology, installation, reproduction guide
├── experiments/            Results (CSVs) and figures (PNGs), git-ignored
├── models/saved/           Trained model checkpoints, git-ignored
├── notebooks/              Reproducible experiment notebooks (00 to 09)
├── src/xai_blockchain_framework/
│                           Python library (metrics, models, XAI, LLM)
└── tests/                  Unit tests
```

See [docs/methodology.md](docs/methodology.md) for a detailed description of
each module and [docs/reproducing_paper.md](docs/reproducing_paper.md) for the
exact notebook execution order.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/xai-eval-blockchain-fraud.git
cd xai-eval-blockchain-fraud

python -m venv .venv
# On Linux / macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 2. Configure secrets

Module 4 calls three frontier LLMs (Claude Opus 4.7, Gemini 3.1 Pro, GPT 5.4)
through the [OpenRouter](https://openrouter.ai) API. You need a single API
key.

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY
```

Modules 1, 2, and 3 do not require any API key and run entirely locally.

### 3. Run the notebooks

Execute the notebooks in order. Each notebook saves its outputs to
`experiments/results/` and `experiments/figures/`.

```
notebooks/00_setup_and_data_download.ipynb     Download Elliptic + Ethereum
notebooks/01a_eda_elliptic.ipynb               Exploratory data analysis
notebooks/01b_eda_ethereum.ipynb               Exploratory data analysis
notebooks/02a_baselines_ml.ipynb               Train RF + LightGBM
notebooks/02b_baselines_gnn.ipynb              Train Temporal GCN + GraphSAGE
notebooks/03a_xai_shap_lime.ipynb              Compute SHAP and LIME
notebooks/03b_xai_gnn_explainers.ipynb         Compute GNNExplainer, IG, GraphLIME
notebooks/04_module1_fidelity.ipynb            Module 1: Fidelity
notebooks/05_module2_stability.ipynb           Module 2: Stability
notebooks/06_module3_bras.ipynb                Module 3: BRAS
notebooks/07_exp_class_imbalance.ipynb         Class imbalance ablation
notebooks/08_module4_llm_agents.ipynb          Module 4: LLM agent evaluation
notebooks/09_module4_ml_baseline.ipynb         ML baseline validation
```

Total runtime: approximately 6 hours on a modern laptop (CPU only), plus the
Module 4 notebook which takes around 3 hours and costs roughly 15 to 20 USD
in OpenRouter API usage.

---

## Datasets

Two public datasets are used:

- **Elliptic Bitcoin** ([Kaggle link](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)):
  203,769 Bitcoin transactions with 166 anonymized features and a temporal
  graph of 234,355 edges. Fraud ratio: 9.76%.
- **Ethereum Fraud Detection** ([Kaggle link](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)):
  9,841 Ethereum addresses with 45 named features (transaction counts, ERC20
  activity, balances). Fraud ratio: 22.14%.

The `notebooks/00_setup_and_data_download.ipynb` notebook guides you through
obtaining both datasets. See [data/README.md](data/README.md) for manual
download instructions.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{oumbefokou2026beyond,
  title        = {Beyond Predictive Metrics: A Multi-Dimensional Framework for
                  XAI Evaluation in Blockchain Fraud Detection},
  author       = {Oumbe Fokou, David Le Sage and Salhab, Wissam and Jaafar, Fehmi},
  year         = {2026},
  howpublished = {\url{https://github.com/David4Lesage/xai-eval-blockchain-fraud}},
  note         = {Preprint, under review}
}
```

A machine-readable citation is also available in [CITATION.cff](CITATION.cff).

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request.

---

## License

Released under the [MIT License](LICENSE). The Elliptic and Ethereum datasets
retain their original licenses, which must be respected.

---

## Acknowledgments

This work is supported by the Université du Québec à Chicoutimi (UQAC).
We thank the open-source maintainers of SHAP, LIME, PyTorch Geometric, and
OpenRouter whose tools underpin this framework.
