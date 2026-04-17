# Datasets

Two public datasets are used by the framework. Neither is redistributed
here, in accordance with their original licenses.

## Elliptic Bitcoin

- **Source**: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- **Size**: 203,769 Bitcoin transactions, 166 anonymized features each,
  234,355 edges.
- **Fraud ratio**: 9.76%.
- **Files** (must be placed in `data/raw/elliptic_bitcoin_dataset/`):
  - `elliptic_txs_classes.csv`
  - `elliptic_txs_features.csv`
  - `elliptic_txs_edgelist.csv`

## Ethereum Fraud Detection

- **Source**: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset
- **Size**: 9,841 addresses, 45 named features.
- **Fraud ratio**: 22.14%.
- **File** (must be placed in `data/raw/`):
  - `Ethereum_Fraud_Detection.csv`

## Automatic download

With the Kaggle CLI installed and credentials placed in
`~/.kaggle/kaggle.json`, you can download both datasets with:

```bash
pip install kaggle

kaggle datasets download -d ellipticco/elliptic-data-set \
    -p data/raw/ --unzip

kaggle datasets download -d vagifa/ethereum-frauddetection-dataset \
    -p data/raw/ --unzip
```

The notebook `notebooks/00_setup_and_data_download.ipynb` wraps this and
reports missing files.

## Licenses and ethical considerations

Both datasets are released for research use. Redistributing the raw data is
not allowed; users must agree to Kaggle's dataset terms before downloading.
The Elliptic dataset is released without public feature names for
confidentiality; the Ethereum dataset uses public behavioral metrics.

Any downstream use of a trained model on real wallet addresses should
comply with anti-money-laundering regulations applicable in the operator's
jurisdiction and with privacy laws regarding pseudonymous identifiers.
