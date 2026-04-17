"""Thin I/O wrappers for the file formats used by the framework.

These helpers do no extra work beyond what pandas / numpy provide, but they
centralize the path handling and log a short message so that notebooks
produce a clear trace of what was read or written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Save a DataFrame to CSV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"  wrote {path.name}  ({len(df)} rows)")
    return path


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    path = Path(path)
    df = pd.read_csv(path, **kwargs)
    print(f"  read  {path.name}  ({len(df)} rows)")
    return df


def save_npy(array: np.ndarray, path: str | Path) -> Path:
    """Save a NumPy array to ``.npy``, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    print(f"  wrote {path.name}  shape={array.shape}")
    return path


def load_npy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from ``.npy``."""
    path = Path(path)
    array = np.load(path, allow_pickle=False)
    print(f"  read  {path.name}  shape={array.shape}")
    return array
