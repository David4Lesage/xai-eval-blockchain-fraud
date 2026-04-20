"""Model training utilities for classical ML and graph neural networks."""

from xai_blockchain_framework.models.ml import (
    compute_metrics,
    evaluate_ml,
    find_optimal_threshold,
    load_ml_model,
    make_fraud_predict_fn,
    save_ml_model,
    train_lightgbm,
    train_random_forest,
)
from xai_blockchain_framework.models.gnn import (
    GraphSAGE,
    GraphSAGEModel,
    TemporalGCN,
    build_edge_index,
    eval_gnn,
    get_device,
    train_gnn,
)

__all__ = [
    # Classical ML
    "train_random_forest",
    "train_lightgbm",
    "evaluate_ml",
    "compute_metrics",
    "find_optimal_threshold",
    "save_ml_model",
    "load_ml_model",
    "make_fraud_predict_fn",
    # Graph Neural Networks
    "TemporalGCN",
    "GraphSAGEModel",
    "GraphSAGE",
    "train_gnn",
    "eval_gnn",
    "get_device",
    "build_edge_index",
]
