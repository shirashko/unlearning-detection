"""Re-run the Gemini judge for an existing audit output directory."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from experiments.audit.config import (
    AuditConfig,
    JudgeConfig,
    apply_argparse_namespace_overrides,
    audit_config_to_nested_dict,
    load_audit_config_yaml,
)
from experiments.audit.general_unlearning_audit import setup_logger
from experiments.audit.judge_runner import invoke_gemini_audit_judge, patch_judge_outputs

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


def _resolve_output_dir(cfg: AuditConfig, cli_output_dir: Optional[str]) -> Path:
    if cli_output_dir:
        return Path(cli_output_dir).expanduser().resolve()
    if cfg.output_dir.strip():
        return Path(cfg.output_dir).expanduser().resolve()
    raise ValueError("Provide --output-dir or a YAML --config with output_dir.")


def _resolve_judge_prompt_path(out_dir: Path, prompt_path: Optional[str]) -> Path:
    if prompt_path:
        path = Path(prompt_path).expanduser().resolve()
    else:
        path = out_dir / "judge_prompt.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Judge prompt not found: {path}")
    return path


def rerun_audit_judge(
    *,
    out_dir: Path,
    judge_cfg: JudgeConfig,
    judge_prompt_path: Optional[Path] = None,
    update_summary: bool = True,
) -> int:
    """Call the audit judge and refresh judge artifacts under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)
    logger.info("=" * 60)
    logger.info("RERUN AUDIT JUDGE  (%s)", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)
    logger.info("Output dir: %s", out_dir)
    logger.info(
        "Judge config:\n%s",
        json.dumps(audit_config_to_nested_dict(AuditConfig(judge=judge_cfg))["judge"], indent=2),
    )

    prompt_path = _resolve_judge_prompt_path(out_dir, str(judge_prompt_path) if judge_prompt_path else None)
    judge_prompt = prompt_path.read_text(encoding="utf-8")
    logger.info("Loaded judge prompt (%d chars) from %s", len(judge_prompt), prompt_path)

    if judge_cfg.skip_judge:
        logger.error("skip_judge is true; refusing to call the judge.")
        return 1

    judge_verdict, judge_error = invoke_gemini_audit_judge(
        judge_cfg, judge_prompt, out_dir, logger,
    )
    patch_judge_outputs(
        out_dir,
        judge_verdict,
        judge_error,
        update_summary=update_summary,
    )

    if judge_error:
        logger.error("Judge rerun failed: %s", judge_error)
        return 1
    if judge_verdict.get("_parse_error"):
        logger.error("Judge response could not be parsed.")
        return 1

    logger.info("Judge rerun complete. Outputs under: %s", out_dir)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Re-run the Gemini judge for an existing audit output directory "
            "using judge_prompt.txt and the same judge pipeline as general_unlearning_audit.py."
        ),
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML file (uses output_dir and judge section).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Audit output directory containing judge_prompt.txt.",
    )
    p.add_argument(
        "--judge-prompt",
        type=str,
        default=None,
        help="Override path to judge prompt (default: <output-dir>/judge_prompt.txt).",
    )
    p.add_argument(
        "--no-update-summary",
        action="store_true",
        help="Do not patch judge fields in audit_summary.json.",
    )
    p.add_argument("--judge-model", type=str, default=argparse.SUPPRESS)
    p.add_argument("--judge-temperature", type=float, default=argparse.SUPPRESS)
    p.add_argument("--judge-max-output-tokens", type=int, default=argparse.SUPPRESS)
    p.add_argument("--judge-api-key-env", type=str, default=argparse.SUPPRESS)
    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_audit_config_yaml(args.config) if args.config else AuditConfig()
    apply_argparse_namespace_overrides(cfg, args, skip_names=frozenset({"config", "output_dir", "judge_prompt", "no_update_summary"}))

    try:
        out_dir = _resolve_output_dir(cfg, args.output_dir)
        prompt_path = Path(args.judge_prompt).expanduser().resolve() if args.judge_prompt else None
    except (ValueError, FileNotFoundError) as e:
        build_arg_parser().error(str(e))

    code = rerun_audit_judge(
        out_dir=out_dir,
        judge_cfg=cfg.judge,
        judge_prompt_path=prompt_path,
        update_summary=not args.no_update_summary,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
