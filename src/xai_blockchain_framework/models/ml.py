"""Training and evaluation of classical ML baselines (Random Forest, LightGBM).

These functions wrap the standard scikit-learn / LightGBM APIs to:

- keep training reproducible (seed plumbing),
- provide a consistent interface for balanced and unbalanced variants,
- compute a full set of classification metrics used throughout the paper,
- find the threshold that maximizes the F1 score on a validation split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xai_blockchain_framework.config import CONFIG


@dataclass
class ClassificationReport:
    """Container for the metrics used to compare classifiers."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    mcc: float
    false_positive_rate: float
    threshold: float

    def as_dict(self) -> dict[str, float]:
        return {
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1": self.f1,
            "ROC-AUC": self.roc_auc,
            "PR-AUC": self.pr_auc,
            "MCC": self.mcc,
            "FPR": self.false_positive_rate,
            "Threshold": self.threshold,
        }


def train_random_forest(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    n_estimators: int = 200,
    max_depth: int | None = None,
    class_weight: str | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Train a Random Forest classifier with reproducible seeding.

    Parameters
    ----------
    X_train, y_train : numpy.ndarray
        Training features and binary labels.
    n_estimators : int, default 200
        Number of trees.
    max_depth : int, optional
        Maximum depth. None means fully grown trees.
    class_weight : str, optional
        ``"balanced"`` to reweight minority class, None for standard training.
    seed : int, optional
        Random seed. Defaults to ``CONFIG.random_seed``.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`sklearn.ensemble.RandomForestClassifier`.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=seed if seed is not None else CONFIG.random_seed,
        n_jobs=-1,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    class_weight: str | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> LGBMClassifier:
    """Train a LightGBM classifier with reproducible seeding."""
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        class_weight=class_weight,
        random_state=seed if seed is not None else CONFIG.random_seed,
        n_jobs=-1,
        verbose=-1,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def find_optimal_threshold(
    y_true: NDArray[np.int64],
    probs: NDArray[np.float64],
    grid_step: float = 0.01,
) -> tuple[float, float]:
    """Grid-search the decision threshold that maximizes F1.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary ground-truth labels.
    probs : numpy.ndarray
        Predicted fraud probabilities.
    grid_step : float, default 0.01
        Step size of the threshold grid over ``(0, 1)``.

    Returns
    -------
    tuple
        ``(best_threshold, best_f1)``.
    """
    thresholds = np.arange(grid_step, 1.0, grid_step)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def evaluate_ml(
    model: RandomForestClassifier | LGBMClassifier,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    threshold: float | None = None,
) -> ClassificationReport:
    """Compute the standard classification metrics for a trained model.

    If ``threshold`` is None, the threshold that maximizes F1 on the given
    split is used and returned in the report.
    """
    probs = model.predict_proba(X)[:, 1]
    if threshold is None:
        threshold, _ = find_optimal_threshold(y, probs)
    pred = (probs >= threshold).astype(int)

    tn, fp, _, _ = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return ClassificationReport(
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y, probs)),
        pr_auc=float(average_precision_score(y, probs)),
        mcc=float(matthews_corrcoef(y, pred)),
        false_positive_rate=fpr,
        threshold=float(threshold),
    )


def save_ml_model(model: RandomForestClassifier | LGBMClassifier, path: Path | str) -> Path:
    """Serialize a fitted model with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  wrote {path.name}")
    return path


def load_ml_model(path: Path | str) -> RandomForestClassifier | LGBMClassifier:
    """Load a joblib-serialized model."""
    path = Path(path)
    model = joblib.load(path)
    print(f"  read  {path.name}")
    return model
