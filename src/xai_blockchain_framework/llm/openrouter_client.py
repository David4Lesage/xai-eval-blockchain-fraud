"""Unified LLM client using the OpenRouter API.

OpenRouter implements the OpenAI chat completion interface and proxies to
multiple providers (Anthropic, Google, OpenAI, Meta, etc.) with a single
API key. This simplifies the framework significantly compared to
maintaining three separate SDKs.

Usage
-----
>>> from xai_blockchain_framework.llm import OpenRouterClient
>>> client = OpenRouterClient()
>>> response = client.call(
...     model="anthropic/claude-opus-4.7",
...     system="You are a helpful assistant.",
...     user="Say hello.",
... )
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import APIError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from xai_blockchain_framework.config import CONFIG


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMResponse:
    """Minimal structured response from an LLM call."""

    content: str
    model: str
    usage: dict[str, Any] | None = None


class OpenRouterClient:
    """Thin wrapper around the OpenAI SDK configured for OpenRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        key = api_key if api_key is not None else CONFIG.openrouter_api_key
        if not key or key == "your_openrouter_api_key_here":
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured. "
                "Copy .env.example to .env and set your key."
            )
        headers = default_headers or {
            "HTTP-Referer": "https://github.com/david-lesage/xai-eval-blockchain-fraud",
            "X-Title": "XAI Blockchain Evaluation Framework",
        }
        self._client = OpenAI(
            api_key=key,
            base_url=base_url,
            default_headers=headers,
        )

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def call(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request and return the textual response.

        Parameters
        ----------
        model : str
            OpenRouter model identifier (e.g. ``"anthropic/claude-opus-4.7"``).
        system : str
            System prompt.
        user : str
            User prompt.
        temperature : float, optional
            Sampling temperature. Defaults to ``CONFIG.llm_temperature``.
        max_tokens : int, optional
            Maximum tokens in the response. Defaults to ``CONFIG.llm_max_tokens``.
        response_format : dict, optional
            Override response format (e.g. ``{"type": "json_object"}``).

        Returns
        -------
        LLMResponse
            Dataclass with ``content``, ``model`` and ``usage`` fields.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature if temperature is not None else CONFIG.llm_temperature,
            "max_tokens": max_tokens if max_tokens is not None else CONFIG.llm_max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)
        content = (response.choices[0].message.content or "").strip()
        usage = response.usage.model_dump() if response.usage else None
        return LLMResponse(content=content, model=model, usage=usage)


def call_agent(
    agent_id: str,
    system: str,
    user: str,
    client: OpenRouterClient | None = None,
    rate_limit_sleep: float | None = None,
) -> LLMResponse:
    """Convenience wrapper for calling an agent by its short identifier.

    Parameters
    ----------
    agent_id : str
        Short identifier: ``"opus"``, ``"gemini"``, or ``"gpt"``. Resolved to
        a full model name via :data:`xai_blockchain_framework.config.CONFIG.models`.
    system : str
        System prompt.
    user : str
        User prompt.
    client : OpenRouterClient, optional
        Pre-constructed client. If None, a new one is created.
    rate_limit_sleep : float, optional
        Sleep after the call in seconds. Defaults to
        ``CONFIG.llm_rate_limit_sleep``.

    Returns
    -------
    LLMResponse
    """
    model = CONFIG.models.get(agent_id)
    if model is None:
        raise KeyError(f"Unknown agent identifier: {agent_id!r}. Known: {list(CONFIG.models)}")
    client = client if client is not None else OpenRouterClient()
    response = client.call(
        model=model,
        system=system,
        user=user,
        response_format={"type": "json_object"},
    )
    sleep = rate_limit_sleep if rate_limit_sleep is not None else CONFIG.llm_rate_limit_sleep
    if sleep > 0:
        time.sleep(sleep)
    return response
