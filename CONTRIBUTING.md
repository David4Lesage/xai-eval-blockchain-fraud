# Contributing

Thank you for your interest in contributing to this project. The framework is
primarily a research artifact, but contributions that improve reproducibility,
extend support for new datasets or XAI methods, or fix bugs are welcome.

## Ways to contribute

- **Report a bug** by opening an issue with a minimal reproducible example.
- **Propose an enhancement** (new metric, new explainer wrapper, new dataset
  loader) by opening an issue first to discuss the design.
- **Improve documentation** by opening a pull request against the `docs/`
  folder.
- **Add unit tests** for existing metric functions.

## Development setup

```bash
git clone https://github.com/<your-username>/xai-eval-blockchain-fraud.git
cd xai-eval-blockchain-fraud

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .[dev]
```

## Style

- Code follows [PEP 8](https://peps.python.org/pep-0008/) with a line length
  of 100 characters.
- Public functions have NumPy-style docstrings.
- All code and comments are in English.
- Commit messages follow the imperative mood (e.g. "Add Lipschitz log
  normalization" rather than "Added Lipschitz log normalization").

## Tests

Run the test suite before submitting a pull request:

```bash
pytest tests/ -v
```

New metric functions must have at least one unit test that verifies the
expected behavior on a small synthetic input.

## Adding a new XAI method

1. Add a wrapper class in `src/xai_blockchain_framework/xai/` that exposes
   `.explain(model, X, indices)` returning an array of shape
   `(n_instances, n_features)`.
2. Register the method in `notebooks/03a_xai_shap_lime.ipynb` or
   `notebooks/03b_xai_gnn_explainers.ipynb` as appropriate.
3. Ensure the method runs on both datasets (Elliptic, Ethereum).
4. Add a unit test with a toy model.

## Adding a new dataset

1. Add a loader in `src/xai_blockchain_framework/data/`.
2. Define the domain rules for the new dataset in
   `src/xai_blockchain_framework/rules/`.
3. Add a new EDA notebook numbered after the existing ones.

## Pull request process

1. Fork the repository and create a feature branch.
2. Make your changes with tests and documentation.
3. Ensure the test suite passes.
4. Open a pull request describing the motivation and the changes.

## Code of conduct

Be respectful and constructive. Technical disagreements are welcome when
framed around the work, not the person.
