"""Gemini judge invocation and output patching for SNMF unlearning audits."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from experiments.audit.config import JudgeConfig
from experiments.audit.unlearning_audit_reporter import UnlearningAuditReporter
from llm_utils.gemini_client import GeminiClient


def invoke_gemini_audit_judge(
    judge_cfg: JudgeConfig,
    judge_prompt: str,
    out_dir: Path,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Evaluate the packaged audit prompt with the configured Gemini judge.

    Do not call when ``judge_cfg.skip_judge`` is true.

    Returns ``(judge_verdict, judge_error)``.

    - judge_verdict is {} only when the client cannot be constructed
      (e.g. missing API key).
    - On HTTP/empty-response or JSON parse failure, judge_verdict may be a
      non-empty dict with _parse_error (and optionally _raw_text).
    - On success, judge_verdict is the parsed verdict JSON and
      judge_error is None.
    """
    judge_verdict: Dict[str, Any] = {}
    judge_error: Optional[str] = None

    try:
        client = GeminiClient(
            model=judge_cfg.judge_model,
            temperature=judge_cfg.judge_temperature,
            max_output_tokens=judge_cfg.judge_max_output_tokens,
            api_key_env=judge_cfg.judge_api_key_env,
        )
        reporter = UnlearningAuditReporter(client)
    except ValueError as e:
        judge_error = str(e)
        logger.warning("Judge call failed: %s", judge_error)
        (out_dir / "judge_response_raw.txt").write_text("", encoding="utf-8")
        return judge_verdict, judge_error

    logger.info("Calling judge model: %s", client.model)

    judge_verdict, raw_text, judge_error, finish_reason = reporter.run_prompt(
        judge_prompt,
    )
    (out_dir / "judge_response_raw.txt").write_text(raw_text or "", encoding="utf-8")
    if finish_reason:
        lvl = logger.warning if finish_reason == "MAX_TOKENS" else logger.info
        lvl("Judge finished with reason: %s", finish_reason)
    if judge_error:
        logger.warning("Judge call failed: %s", judge_error)
    elif judge_verdict.get("_parse_error"):
        judge_error = str(judge_verdict.get("_parse_error", "parse error"))
        logger.warning("Judge parse failed: %s", judge_error)
    else:
        logger.info(
            "Judge verdict: confidence=%s | concept=%r",
            judge_verdict.get("unlearning_confidence"),
            judge_verdict.get("likely_unlearned_concept"),
        )
    return judge_verdict, judge_error


def patch_judge_outputs(
    out_dir: Path,
    judge_verdict: Dict[str, Any],
    judge_error: Optional[str],
    *,
    update_summary: bool = True,
) -> None:
    """Write ``judge_response.json`` and optionally patch ``audit_summary.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if (
        judge_verdict
        and not judge_verdict.get("_parse_error")
    ):
        (out_dir / "judge_response.json").write_text(
            json.dumps(judge_verdict, indent=2),
            encoding="utf-8",
        )

    if not update_summary:
        return

    summary_path = out_dir / "audit_summary.json"
    if not summary_path.is_file():
        logging.getLogger(__name__).info(
            "No audit_summary.json at %s; skipping summary patch.", summary_path,
        )
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["judge_verdict"] = judge_verdict
    summary["judge_error"] = judge_error
    meta = summary.get("meta")
    if isinstance(meta, dict):
        meta["judge_skipped"] = False
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
