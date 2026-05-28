"""Tests for audit/evaluation JSON verdict parsing."""

from __future__ import annotations

from experiments.audit.unlearning_audit_reporter import parse_judge_json


def test_parse_judge_json_accepts_object() -> None:
    payload = parse_judge_json('{"likely_unlearned_concept": "algebra", "unlearning_confidence": 80}')
    assert payload["likely_unlearned_concept"] == "algebra"
    assert payload["unlearning_confidence"] == 80


def test_parse_judge_json_rejects_non_object_json() -> None:
    payload = parse_judge_json("[1, 2, 3]")
    assert payload["_parse_error"]
    assert payload["_raw_text"] == "[1, 2, 3]"


def test_parse_judge_json_extracts_object_from_surrounding_text() -> None:
    payload = parse_judge_json('Here is the verdict:\n{"likely_unlearned_concept": "WWII"}\nDone.')
    assert payload["likely_unlearned_concept"] == "WWII"
