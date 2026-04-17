"""Loader and preprocessing for the Ethereum Fraud Detection dataset.

The raw dataset is a single CSV with 49 columns: an index, an address, 45
numerical features (transaction counts, Ether volumes, ERC20 activity,
etc.), a flag column, and a target column. The target is named
``FLAG`` with values ``0`` (legit) or ``1`` (fraudulent).

See https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset
for the original distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from xai_blockchain_framework.config import PATHS

ETHEREUM_FILE = "Ethereum_Fraud_Detection.csv"

# Columns that must be dropped because they are identifiers or non-numeric.
_NON_FEATURE_COLUMNS = {"Unnamed: 0", "Index", "Address", "FLAG"}


@dataclass
class EthereumData:
    """Container for the Ethereum Fraud Detection dataset."""

    features: pd.DataFrame
    labels: pd.Series
    addresses: pd.Series

    @property
    def n_features(self) -> int:
        return self.features.shape[1]


def _resolve_path(data_path: Path | str | None) -> Path:
    if data_path is None:
        return PATHS.data_raw / ETHEREUM_FILE
    return Path(data_path)


def load_ethereum(data_path: Path | str | None = None) -> EthereumData:
    """Load the Ethereum Fraud Detection CSV.

    Parameters
    ----------
    data_path : path, optional
        Path to ``Ethereum_Fraud_Detection.csv``. Defaults to
        ``data/raw/Ethereum_Fraud_Detection.csv``.

    Returns
    -------
    EthereumData

    Raises
    ------
    FileNotFoundError
        If the CSV is not present.
    """
    path = _resolve_path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Ethereum dataset at {path}. "
            "Run notebook 00_setup_and_data_download.ipynb first."
        )

    df = pd.read_csv(path)
    addresses = df["Address"].copy() if "Address" in df.columns else pd.Series(
        [f"addr_{i}" for i in range(len(df))]
    )
    labels = df["FLAG"].astype(int).copy()

    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLUMNS]
    features = df[feature_cols].copy()

    # Coerce remaining non-numeric columns to numeric, dropping those that
    # cannot be converted (typically categorical string columns).
    for col in features.columns:
        if not pd.api.types.is_numeric_dtype(features[col]):
            features[col] = pd.to_numeric(features[col], errors="coerce")
    # Drop columns that are still entirely NaN after coercion and rows with
    # any remaining NaN in the surviving numeric columns.
    features = features.dropna(axis=1, how="all")
    features = features.fillna(0.0)

    return EthereumData(features=features, labels=labels, addresses=addresses)


def preprocess_ethereum(
    data: EthereumData,
    fit_scaler_on: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Convert the dataset to numpy arrays with standard-scaled features.

    Parameters
    ----------
    data : EthereumData
        Output of :func:`load_ethereum`.
    fit_scaler_on : numpy.ndarray, optional
        Pre-existing training split to fit the scaler. When None, the
        scaler is fit on the full dataset (acceptable for unsupervised
        analyses but not for rigorous ML; split first then pass the
        training subset).

    Returns
    -------
    X : numpy.ndarray
        Standardized feature matrix.
    y : numpy.ndarray
        Binary label array.
    scaler : sklearn.preprocessing.StandardScaler
        The fitted scaler, useful for inverse transforms and inference.
    """
    X_raw = data.features.to_numpy(dtype=np.float64)
    y = data.labels.to_numpy(dtype=np.int64)

    scaler = StandardScaler()
    if fit_scaler_on is None:
        X = scaler.fit_transform(X_raw)
    else:
        scaler.fit(fit_scaler_on)
        X = scaler.transform(X_raw)

    return X, y, scaler
