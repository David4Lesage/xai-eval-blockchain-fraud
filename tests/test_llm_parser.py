"""Tests for the LLM response parser."""

from __future__ import annotations

from xai_blockchain_framework.llm.parsers import ParsedResponse, parse_response


class TestParseResponse:
    def test_clean_json(self) -> None:
        raw = '{"decision": "fraud", "confidence": 0.87, "reasoning": "unusual amount"}'
        parsed = parse_response(raw)
        assert parsed.decision == "fraud"
        assert parsed.confidence == 0.87
        assert parsed.reasoning == "unusual amount"
        assert parsed.is_valid

    def test_markdown_codeblock(self) -> None:
        raw = (
            "```json\n"
            '{"decision": "legitimate", "confidence": 0.65, '
            '"reasoning": "all signals normal"}\n'
            "```"
        )
        parsed = parse_response(raw)
        assert parsed.decision == "legitimate"
        assert parsed.confidence == 0.65

    def test_explanation_field(self) -> None:
        raw = (
            '{"decision": "fraud", "confidence": 0.9, "reasoning": "high risk", '
            '"explanation": "Top feature X drove the decision."}'
        )
        parsed = parse_response(raw)
        assert parsed.explanation == "Top feature X drove the decision."

    def test_missing_fields_fallback(self) -> None:
        # Truncated JSON, regex fallback kicks in.
        raw = 'partial "decision": "fraud" with some garbage'
        parsed = parse_response(raw)
        assert parsed.decision == "fraud"

    def test_empty_input(self) -> None:
        parsed = parse_response("")
        assert parsed.decision is None
        assert not parsed.is_valid

    def test_invalid_decision(self) -> None:
        raw = '{"decision": "maybe", "confidence": 0.5}'
        parsed = parse_response(raw)
        assert parsed.decision is None

    def test_confidence_clamping(self) -> None:
        raw = '{"decision": "fraud", "confidence": 1.5}'
        parsed = parse_response(raw)
        assert parsed.confidence == 1.0
