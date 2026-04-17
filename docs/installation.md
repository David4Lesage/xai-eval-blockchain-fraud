# Installation

## Requirements

- Python 3.11 or newer
- About 4 GB of free RAM and 2 GB of disk space (datasets and checkpoints
  included)
- A modern CPU is sufficient; a CUDA-capable GPU accelerates notebook 02b
  (GNN training) but is not required
- An OpenRouter API key for Module 4 (notebook 08). Modules 1–3 work
  without any network access once data is downloaded.

## Step-by-step setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/xai-eval-blockchain-fraud.git
cd xai-eval-blockchain-fraud
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The `pip install -e .` step registers the `xai_blockchain_framework`
package so that notebooks and tests can `import xai_blockchain_framework`
from anywhere.

For development extras (linting, testing, Jupyter kernel):

```bash
pip install -e .[dev]
```

### 4. Configure API access

Module 4 uses three frontier LLMs through OpenRouter. Copy the template:

```bash
cp .env.example .env
```

Edit `.env` and set `OPENROUTER_API_KEY` to your key. Optionally override
the model identifiers:

```
OPENROUTER_API_KEY=sk-or-v1-...
MODEL_OPUS=anthropic/claude-opus-4.7
MODEL_GEMINI=google/gemini-3.1-pro
MODEL_GPT=openai/gpt-5.4
```

### 5. Download datasets

See [data/README.md](../data/README.md). The notebook
`notebooks/00_setup_and_data_download.ipynb` guides you through the
process.

### 6. Verify the installation

```bash
pytest tests/ -v
```

All tests should pass. If any test fails, it typically indicates a
dependency version mismatch; verify your `pip list` against
`requirements.txt`.

## Troubleshooting

- **`ImportError: No module named xai_blockchain_framework`** — make sure
  you ran `pip install -e .` from the repository root.
- **PyTorch Geometric install errors** — `torch-geometric` requires the
  matching `torch` version. See the
  [official install matrix](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
- **CUDA not detected** — set `TORCH_DEVICE=cpu` in `.env` to force CPU
  execution.
- **OpenRouter rate limiting** — increase `LLM_RATE_LIMIT_SLEEP` in `.env`.
