# Reproducing the paper

Execute the notebooks in numerical order. Each notebook is independent
in its inputs (reads from `data/processed/`, `experiments/results/`, and
`models/saved/`) and idempotent in its outputs.

## Execution order and runtime estimates

| Order | Notebook | Runtime (CPU) | Notes |
|-------|----------|---------------|-------|
| 1 | `00_setup_and_data_download.ipynb` | ~15 min | Dataset download, depends on network |
| 2 | `01a_eda_elliptic.ipynb` | ~2 min | EDA figures |
| 3 | `01b_eda_ethereum.ipynb` | ~2 min | EDA figures |
| 4 | `02a_baselines_ml.ipynb` | ~20 min | Trains 4 ML models, saves checkpoints |
| 5 | `02b_baselines_gnn.ipynb` | ~45 min | Trains 2 GNNs, GPU speeds this up |
| 6 | `03a_xai_shap_lime.ipynb` | ~25 min | SHAP + LIME for 4 ML configs |
| 7 | `03b_xai_gnn_explainers.ipynb` | ~30 min | IG, GraphLIME, GNNExplainer |
| 8 | `04_module1_fidelity.ipynb` | ~10 min | Fidelity CSV |
| 9 | `05_module2_stability.ipynb` | ~20 min | Stability CSV with log-normalized columns |
| 10 | `06_module3_bras.ipynb` | ~5 min | BRAS CSV |
| 11 | `07_exp_class_imbalance.ipynb` | ~15 min | Class imbalance ablation |
| 12 | `08_module4_llm_agents.ipynb` | ~3 h | API calls, costs 10–20 USD |
| 13 | `09_module4_ml_baseline.ipynb` | ~5 min | ML baseline vs LLM agreement |

Total wall-clock time: roughly 6 hours of local computation plus 2–4
hours for the LLM notebook depending on API latency and rate limits.

## Prerequisite checks

Before running notebook 08, verify:

- [ ] `OPENROUTER_API_KEY` is set in `.env`.
- [ ] You have a budget alert configured on OpenRouter (the notebook can
      issue up to ~3,000 API calls).
- [ ] Notebooks 00–07 have produced their expected CSV and NPY outputs in
      `experiments/results/`.

## Expected artifacts

After successful execution, `experiments/results/` contains:

```
ml_baselines_results.csv
gnn_baselines_results.csv
xai_eval_indices_{elliptic,ethereum}.npy
shap_{randomforest,lightgbm}_{elliptic,ethereum}.npy
lime_{randomforest,lightgbm}_{elliptic,ethereum}.npy
gnn_{ig,graphlime,gnnexplainer}_{temporalgcn,graphsage}.npy
gnn_eval_node_indices.npy
module1_fidelity_results.csv
module2_stability_results.csv
module3_bras_results.csv
exp_imbalance_results.csv
module4_llm_raw_responses.csv
module4_llm_summary.csv
module4_ml_baseline_results.csv
```

And `experiments/figures/` contains the EDA plots and any plotting
results produced along the way.

## Sanity checks

- `module3_bras_results.csv` should contain at least one row with
  `BRAS >= 0.9` (typically Random Forest with LIME on Elliptic).
- `module4_llm_summary.csv` should show DA(C2) > DA(C1) for all agents on
  both datasets.
- `module4_ml_baseline_results.csv` should show `Kappa_ML_LLM > 0.85` in
  condition C2 for most (dataset, agent) pairs.

Any significant deviation from these patterns suggests an environment or
configuration issue and should be investigated before citing results.
