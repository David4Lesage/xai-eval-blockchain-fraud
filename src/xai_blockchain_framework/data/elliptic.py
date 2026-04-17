"""Loader and preprocessing for the Elliptic Bitcoin dataset.

The raw dataset consists of three CSV files:

- ``elliptic_txs_classes.csv`` with columns ``txId, class`` where
  ``class`` is ``1`` (illicit), ``2`` (licit), or ``unknown``.
- ``elliptic_txs_features.csv`` with 166 columns: ``txId``, ``time_step``,
  and 165 anonymized numerical features (``feat_1`` through ``feat_165``).
- ``elliptic_txs_edgelist.csv`` with columns ``txId1, txId2`` describing
  the transaction graph.

See https://www.kaggle.com/datasets/ellipticco/elliptic-data-set for the
original distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from xai_blockchain_framework.config import PATHS

ELLIPTIC_FILES = {
    "classes": "elliptic_txs_classes.csv",
    "features": "elliptic_txs_features.csv",
    "edges": "elliptic_txs_edgelist.csv",
}


@dataclass
class EllipticData:
    """Container for the Elliptic dataset in a uniform shape."""

    features: pd.DataFrame
    labels: pd.Series
    time_steps: pd.Series
    edges: pd.DataFrame
    tx_ids: pd.Series

    @property
    def n_features(self) -> int:
        return self.features.shape[1]


def _resolve_dir(data_dir: Path | str | None) -> Path:
    if data_dir is None:
        return PATHS.data_raw / "elliptic_bitcoin_dataset"
    return Path(data_dir)


def load_elliptic(data_dir: Path | str | None = None) -> EllipticData:
    """Load the three Elliptic CSVs and return them aligned.

    Parameters
    ----------
    data_dir : path, optional
        Directory containing the three CSV files. Defaults to
        ``data/raw/elliptic_bitcoin_dataset/``.

    Returns
    -------
    EllipticData
        Features, labels, time steps, edges, and ordered transaction ids.

    Raises
    ------
    FileNotFoundError
        If any of the three files is missing.
    """
    directory = _resolve_dir(data_dir)
    paths = {key: directory / name for key, name in ELLIPTIC_FILES.items()}
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Elliptic {key} file at {path}. "
                "Run notebook 00_setup_and_data_download.ipynb first."
            )

    # Features: first column is tx_id, second is time_step, rest are features.
    features_raw = pd.read_csv(paths["features"], header=None)
    features_raw.columns = (
        ["txId", "time_step"] + [f"feat_{i}" for i in range(1, features_raw.shape[1] - 1)]
    )
    tx_ids = features_raw["txId"].copy()
    time_steps = features_raw["time_step"].astype(int).copy()
    features = features_raw.drop(columns=["txId", "time_step"]).astype(float)

    # Classes.
    classes_raw = pd.read_csv(paths["classes"])
    classes_raw["txId"] = classes_raw["txId"].astype(features_raw["txId"].dtype)
    merged = features_raw[["txId"]].merge(classes_raw, on="txId", how="left")
    labels = merged["class"].copy()

    # Edges.
    edges = pd.read_csv(paths["edges"])

    return EllipticData(
        features=features,
        labels=labels,
        time_steps=time_steps,
        edges=edges,
        tx_ids=tx_ids,
    )


def preprocess_elliptic(
    data: EllipticData,
    drop_unknown: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert the raw dataset to numpy arrays ready for modeling.

    Parameters
    ----------
    data : EllipticData
        Output of :func:`load_elliptic`.
    drop_unknown : bool, default True
        Drop rows whose class is ``"unknown"``. Set to False to keep all
        nodes for semi-supervised / transductive GNN training.
    normalize : bool, default True
        Apply per-feature z-score standardization using the training subset
        statistics. Returns raw features when False.

    Returns
    -------
    X : numpy.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    y : numpy.ndarray
        Binary label array of shape ``(n_samples,)`` with ``1`` for fraud
        and ``0`` for legitimate. Unknown rows (when kept) carry ``-1``.
    time_steps : numpy.ndarray
        Time step per row, shape ``(n_samples,)``.
    """
    features = data.features.copy()
    labels = data.labels.copy()
    time_steps = data.time_steps.copy()

    if drop_unknown:
        mask = labels.isin(["1", "2", 1, 2])
        features = features[mask].reset_index(drop=True)
        labels = labels[mask].reset_index(drop=True)
        time_steps = time_steps[mask].reset_index(drop=True)

    # Map classes to binary: '1' (illicit) -> 1, '2' (licit) -> 0, unknown -> -1
    def to_binary(v: object) -> int:
        s = str(v).strip()
        if s == "1":
            return 1
        if s == "2":
            return 0
        return -1

    y = labels.apply(to_binary).to_numpy(dtype=np.int64)
    X = features.to_numpy(dtype=np.float64)
    ts = time_steps.to_numpy(dtype=np.int64)

    if normalize:
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True)
        sigma[sigma < 1e-8] = 1.0
        X = (X - mu) / sigma

    return X, y, ts
