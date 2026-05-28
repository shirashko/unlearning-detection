"""Smoke tests for target evaluation fixtures and evaluator (mocked API)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from experiments.evaluation.run_target_evaluation import (
    load_forget_retain_corpora,
    resolve_judge_hypothesis,
)
from experiments.evaluation.unlearning_target_evaluator import (
    UnlearningTargetEvaluator,
)

FIXTURES = Path(__file__).resolve().parent / "toy_target_eval"


def test_resolve_hypothesis_from_fixture_audit_dir() -> None:
    hypothesis, verdict = resolve_judge_hypothesis(
        audit_dir=FIXTURES / "audit",
        judge_response=None,
        hypothesis=None,
    )
    assert hypothesis == "World War II European theater"
    assert verdict.get("unlearning_confidence") == 72


def test_load_forget_retain_from_fixture_json() -> None:
    forget, retain = load_forget_retain_corpora(
        forget_path=None,
        retain_path=None,
        labeled_path=FIXTURES / "labeled.json",
        forget_key="forget",
        retain_key="retain",
        forget_labels=["forget"],
        retain_labels=["retain"],
    )
    assert len(forget) == 2
    assert len(retain) == 2
    assert "Stalingrad" in forget[0]


def test_classify_sample_rejects_non_boolean_label() -> None:
    client = MagicMock()
    client.generate_text.return_value = (
        json.dumps(
            {"belongs_to_forget_concept": "false", "forget_probability": 0.1},
        ),
        None,
        None,
    )

    is_forget, prob, failure = UnlearningTargetEvaluator(client).classify_sample(
        hypothesis="World War II European theater",
        text_sample="Some retain text.",
    )

    assert is_forget is None
    assert prob is None
    assert failure == "parse"


def test_evaluate_hypothesis_mocked_gemini() -> None:
    client = MagicMock()

    def _mock_generate_text(prompt: str) -> tuple[str, None, None]:
        is_forget = ("Stalingrad" in prompt) or ("D-Day" in prompt)
        payload = {
            "belongs_to_forget_concept": is_forget,
            "forget_probability": 0.9 if is_forget else 0.1,
        }
        return json.dumps(payload), None, None

    client.generate_text.side_effect = _mock_generate_text

    forget, retain = load_forget_retain_corpora(
        forget_path=None,
        retain_path=None,
        labeled_path=FIXTURES / "labeled.json",
        forget_key="forget",
        retain_key="retain",
        forget_labels=["forget"],
        retain_labels=["retain"],
    )

    report = UnlearningTargetEvaluator(client, seed=0).evaluate_hypothesis(
        hypothesis="World War II European theater",
        forget_samples=forget,
        retain_samples=retain,
        max_samples_per_set=2,
    )

    assert report["n_samples_evaluated"] == 4
    assert report["n_samples_scored"] == 4
    assert report["n_api_failures"] == 0
    assert report["balanced_accuracy"] == 1.0
    assert report["auc_roc"] == 1.0
    assert client.generate_text.call_count == 4


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY", "").strip(),
    reason="Set GOOGLE_API_KEY to run live Gemini toy evaluation.",
)
def test_run_toy_evaluation_script_live() -> None:
    """Optional live run; same as ``run_toy_evaluation.sh``."""
    import subprocess

    script = FIXTURES / "run_toy_evaluation.sh"
    subprocess.run(["bash", str(script)], check=True, cwd=FIXTURES.parents[1])

    report_path = FIXTURES / "out" / "target_evaluation_report.json"
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["evaluation"]["n_samples_evaluated"] == 4
    assert payload["evaluation"]["n_samples_scored"] >= 1
