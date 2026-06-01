"""Tests for judge-only rerun helpers."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.audit.judge_runner import patch_judge_outputs


def test_patch_judge_outputs_updates_summary_and_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "audit"
    out_dir.mkdir()
    summary = {
        "meta": {"judge_skipped": True},
        "judge_verdict": {},
        "judge_error": "429 RESOURCE_EXHAUSTED",
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    verdict = {
        "likely_unlearned_concept": "golf",
        "unlearning_confidence": "high",
    }
    patch_judge_outputs(out_dir, verdict, None, update_summary=True)

    updated = json.loads((out_dir / "audit_summary.json").read_text(encoding="utf-8"))
    assert updated["judge_verdict"] == verdict
    assert updated["judge_error"] is None
    assert updated["meta"]["judge_skipped"] is False
    assert json.loads((out_dir / "judge_response.json").read_text(encoding="utf-8")) == verdict
