"""Parse LLM JSON responses and extract decision, confidence, reasoning, explanation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ParsedResponse:
    """Structured output from an LLM agent's JSON response."""

    decision: str | None
    confidence: float | None
    reasoning: str
    explanation: str

    @property
    def is_valid(self) -> bool:
        return self.decision in ("fraud", "legitimate") and self.confidence is not None


_MARKDOWN_CODEBLOCK = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_DECISION_RE = re.compile(r'"decision"\s*:\s*"(fraud|legitimate)"', re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*([\d.]+)')
_REASONING_RE = re.compile(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_EXPLANATION_RE = re.compile(r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def parse_response(raw: str | None) -> ParsedResponse:
    """Parse an LLM response into a :class:`ParsedResponse`.

    The function is robust to:

    - responses wrapped in Markdown code fences (```json ... ```),
    - responses with leading/trailing whitespace,
    - malformed JSON (falls back to regex extraction),
    - missing optional fields.

    Parameters
    ----------
    raw : str or None
        The raw text returned by the LLM.

    Returns
    -------
    ParsedResponse
        A dataclass with ``decision``, ``confidence``, ``reasoning`` and
        ``explanation`` fields. If parsing fails entirely, all fields are
        set to ``None`` or empty strings and ``is_valid`` returns False.
    """
    if not raw:
        return ParsedResponse(decision=None, confidence=None, reasoning="", explanation="")

    cleaned = _MARKDOWN_CODEBLOCK.sub("", raw.strip()).strip()

    # Primary path: parse as JSON.
    try:
        data = json.loads(cleaned)
        decision = _normalize_decision(data.get("decision"))
        confidence = _normalize_confidence(data.get("confidence"))
        reasoning = str(data.get("reasoning", ""))
        explanation = str(data.get("explanation", ""))
        return ParsedResponse(decision, confidence, reasoning, explanation)
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: extract fields with regex.
    decision_match = _DECISION_RE.search(cleaned)
    confidence_match = _CONFIDENCE_RE.search(cleaned)
    reasoning_match = _REASONING_RE.search(cleaned)
    explanation_match = _EXPLANATION_RE.search(cleaned)

    if decision_match is None:
        return ParsedResponse(None, None, "", "")

    return ParsedResponse(
        decision=decision_match.group(1).lower(),
        confidence=_normalize_confidence(confidence_match.group(1)) if confidence_match else 0.5,
        reasoning=reasoning_match.group(1) if reasoning_match else "",
        explanation=explanation_match.group(1) if explanation_match else "",
    )


def _normalize_decision(value: object) -> str | None:
    """Normalize a decision value to ``"fraud"``, ``"legitimate"``, or None."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("fraud", "legitimate"):
        return text
    if "fraud" in text:
        return "fraud"
    if "legit" in text:
        return "legitimate"
    return None


def _normalize_confidence(value: object) -> float | None:
    """Clamp a confidence value to ``[0, 1]``, or return None if invalid."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, v))
