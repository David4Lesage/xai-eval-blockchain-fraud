"""Instructions and helpers for obtaining the raw datasets.

The two datasets are distributed on Kaggle under non-redistributable terms,
so this framework does not bundle them. This module provides print
instructions rather than automated downloads to keep licensing clean.

Programmatic download is possible via the Kaggle API when the user has a
local ``~/.kaggle/kaggle.json`` credential file.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from xai_blockchain_framework.config import PATHS


ELLIPTIC_KAGGLE_SLUG = "ellipticco/elliptic-data-set"
ETHEREUM_KAGGLE_SLUG = "vagifa/ethereum-frauddetection-dataset"


def print_download_instructions() -> None:
    """Print a manual download guide."""
    print("=" * 78)
    print("Dataset download instructions")
    print("=" * 78)
    print()
    print("1. Elliptic Bitcoin dataset")
    print(f"   Source: https://www.kaggle.com/datasets/{ELLIPTIC_KAGGLE_SLUG}")
    print("   Place the three CSV files in:")
    print(f"     {PATHS.data_raw / 'elliptic_bitcoin_dataset/'}")
    print("     - elliptic_txs_classes.csv")
    print("     - elliptic_txs_features.csv")
    print("     - elliptic_txs_edgelist.csv")
    print()
    print("2. Ethereum Fraud Detection dataset")
    print(f"   Source: https://www.kaggle.com/datasets/{ETHEREUM_KAGGLE_SLUG}")
    print("   Place the CSV file at:")
    print(f"     {PATHS.data_raw / 'Ethereum_Fraud_Detection.csv'}")
    print()
    print("Alternatively, use the Kaggle CLI:")
    print("  pip install kaggle")
    print("  # add ~/.kaggle/kaggle.json with your API credentials")
    print(f"  kaggle datasets download -d {ELLIPTIC_KAGGLE_SLUG} -p data/raw/ --unzip")
    print(f"  kaggle datasets download -d {ETHEREUM_KAGGLE_SLUG} -p data/raw/ --unzip")
    print("=" * 78)


def try_kaggle_download() -> bool:
    """Attempt a programmatic download using the Kaggle CLI.

    Returns True if both datasets are fetched successfully, False otherwise.
    Prints actionable diagnostics when credentials are missing.
    """
    target = PATHS.data_raw
    target.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Kaggle CLI not available. Falling back to manual instructions.")
        print_download_instructions()
        return False

    ok = True
    for slug in (ELLIPTIC_KAGGLE_SLUG, ETHEREUM_KAGGLE_SLUG):
        print(f"Downloading {slug} ...")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(result.stderr)
            ok = False
    return ok


def check_datasets_present() -> dict[str, bool]:
    """Check whether both datasets are already downloaded."""
    elliptic_dir = PATHS.data_raw / "elliptic_bitcoin_dataset"
    elliptic_files = [
        "elliptic_txs_classes.csv",
        "elliptic_txs_features.csv",
        "elliptic_txs_edgelist.csv",
    ]
    elliptic_ok = all((elliptic_dir / f).exists() for f in elliptic_files)
    ethereum_ok = (PATHS.data_raw / "Ethereum_Fraud_Detection.csv").exists()
    return {"elliptic": elliptic_ok, "ethereum": ethereum_ok}
