"""Global configuration: paths, model identifiers, and reproducibility helpers.

All paths are resolved relative to the repository root, which is inferred from
the location of this file. Values that depend on the runtime environment (API
keys, chosen model IDs) are loaded from a ``.env`` file when present.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = _THIS_FILE.parents[2]


@dataclass(frozen=True)
class Paths:
    """Filesystem paths used throughout the framework."""

    repo_root: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    data_raw: Path = REPO_ROOT / "data" / "raw"
    data_processed: Path = REPO_ROOT / "data" / "processed"
    models_dir: Path = REPO_ROOT / "models" / "saved"
    experiments_dir: Path = REPO_ROOT / "experiments"
    results_dir: Path = REPO_ROOT / "experiments" / "results"
    figures_dir: Path = REPO_ROOT / "experiments" / "figures"
    notebooks_dir: Path = REPO_ROOT / "notebooks"

    def ensure_exists(self) -> None:
        """Create all output directories if they do not yet exist."""
        for attr_name in (
            "data_raw",
            "data_processed",
            "models_dir",
            "results_dir",
            "figures_dir",
        ):
            getattr(self, attr_name).mkdir(parents=True, exist_ok=True)


PATHS = Paths()
PATHS.ensure_exists()


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

load_dotenv(REPO_ROOT / ".env", override=False)


def _get_env(key: str, default: str) -> str:
    """Read an environment variable with a default fallback."""
    value = os.getenv(key)
    return value if value is not None and value.strip() else default


# ---------------------------------------------------------------------------
# LLM models (resolved at import time, can be overridden by .env)
# ---------------------------------------------------------------------------

DEFAULT_MODELS: dict[str, str] = {
    "opus": _get_env("MODEL_OPUS", "anthropic/claude-opus-4.7"),
    "gemini": _get_env("MODEL_GEMINI", "google/gemini-3.1-pro-preview"),
    "gpt": _get_env("MODEL_GPT", "openai/gpt-5.4"),
}

AGENT_DISPLAY_NAMES: dict[str, str] = {
    "opus": "Claude Opus 4.7",
    "gemini": "Gemini 3.1 Pro",
    "gpt": "GPT 5.4",
}


# ---------------------------------------------------------------------------
# Global configuration object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """Runtime configuration: reproducibility seeds and LLM parameters."""

    random_seed: int = int(_get_env("RANDOM_SEED", "42"))
    torch_device: str = _get_env("TORCH_DEVICE", "")
    llm_temperature: float = float(_get_env("LLM_TEMPERATURE", "0.7"))
    llm_max_tokens: int = int(_get_env("LLM_MAX_TOKENS", "800"))
    llm_retries: int = int(_get_env("LLM_RETRIES", "3"))
    llm_rate_limit_sleep: float = float(_get_env("LLM_RATE_LIMIT_SLEEP", "1.0"))
    openrouter_api_key: str = _get_env("OPENROUTER_API_KEY", "")
    models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))

    def has_llm_key(self) -> bool:
        """Return True if an OpenRouter API key is configured."""
        key = self.openrouter_api_key
        return bool(key) and key != "your_openrouter_api_key_here"


CONFIG = Config()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int | None = None) -> int:
    """Fix random seeds for ``random``, ``numpy``, and (optionally) ``torch``.

    Parameters
    ----------
    seed : int, optional
        Seed value. If None, uses ``CONFIG.random_seed``.

    Returns
    -------
    int
        The seed that was applied.
    """
    seed = seed if seed is not None else CONFIG.random_seed
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
