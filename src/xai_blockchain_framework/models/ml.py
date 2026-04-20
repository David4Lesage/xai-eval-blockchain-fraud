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
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
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
) -> tuple[float, float]:
    """Return the decision threshold that maximizes F1.

    Thresholds are drawn from the scikit-learn precision-recall curve, which
    gives one candidate per distinct predicted probability (rather than a
    fixed grid). This matches the reference research implementation and is
    generally more precise than a 0.01 step grid.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary ground-truth labels.
    probs : numpy.ndarray
        Predicted fraud probabilities.

    Returns
    -------
    tuple of float
        ``(best_threshold, best_f1)``.
    """
    pr, rc, thresholds = precision_recall_curve(y_true, probs)
    denom = pr + rc
    f1 = np.where(denom > 0, 2 * pr * rc / denom, 0.0)
    # precision_recall_curve returns len(thresholds) == len(pr) - 1.
    idx = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    return float(thresholds[idx]), float(f1[idx])


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


def compute_metrics(
    y_true: NDArray[np.int64],
    y_proba: NDArray[np.float64],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the standard binary-classification metrics as a plain dict.

    Mirrors the dict format used throughout the research notebooks. Prefer
    :func:`evaluate_ml` when you also need an automatic threshold search.

    Parameters
    ----------
    y_true : numpy.ndarray
        Binary ground-truth labels.
    y_proba : numpy.ndarray
        Predicted fraud probabilities.
    threshold : float, default 0.5
        Decision threshold on ``y_proba``.

    Returns
    -------
    dict
        Keys: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, FPR,
        Threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_proba)),
        "PR-AUC": float(average_precision_score(y_true, y_proba)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "FPR": fpr,
        "Threshold": float(threshold),
    }


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


def make_fraud_predict_fn(
    model: RandomForestClassifier | LGBMClassifier,
    n_features: int,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Wrap ``model.predict_proba`` to return fraud probability only.

    The wrapper also re-attaches column names to numpy input arrays so that
    scikit-learn and LightGBM do not emit ``UserWarning`` about missing
    feature names on every call. It returns an array of shape ``(n,)`` with
    the probability of the positive class, which is the signature expected by
    every metric in :mod:`xai_blockchain_framework.metrics`.
    """
    feat_names = [f"f{i}" for i in range(n_features)]
    try:
        feat_names = list(model.feature_names_in_)
    except AttributeError:
        try:
            feat_names = list(model.feature_name_)
        except AttributeError:
            pass

    def predict(X: NDArray[np.float64]) -> NDArray[np.float64]:
        df = pd.DataFrame(X, columns=feat_names)
        return model.predict_proba(df)[:, 1]

    return predict
