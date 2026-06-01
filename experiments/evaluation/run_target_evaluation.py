#!/usr/bin/env python3
"""
Blind validation of an audit judge hypothesis against labeled forget/retain text.

Reads ``likely_unlearned_concept`` from a prior audit run (``judge_response.json`` or
``audit_summary.json`` under ``--audit-dir``) and scores it.

Example::

    cd /path/to/unlearning-detection
    python3 experiments/evaluation/run_target_evaluation.py \\
        --audit-dir outputs/audit/my_run \\
        --labeled-data data/forget_retain_eval.json \\
        --output-dir outputs/audit/my_run/target_evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

from data_utils.concept_dataset import SupervisedConceptDataset
from experiments.audit.unlearning_audit_reporter import parse_judge_json
from experiments.evaluation.unlearning_target_evaluator import (
    TEXT_SAMPLE_PLACEHOLDER,
    UnlearningTargetEvaluator,
)
from llm_utils.gemini_client import GeminiClient

load_dotenv()

logger = logging.getLogger(__name__)


def setup_logger(output_dir: Path) -> logging.Logger:
    """Mirror audit logging: stdout + ``run.log`` under the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    return root


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_texts(items: Sequence[Any]) -> List[str]:
    texts: List[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            texts.append(item.strip())
        elif isinstance(item, dict):
            text = item.get("prompt") or item.get("text")
            if text is not None and str(text).strip():
                texts.append(str(text).strip())
    return texts


def load_text_corpus(path: Path) -> List[str]:
    """Load a flat list of strings from JSON (list or dict-of-lists)."""
    raw = _load_json(path)
    if isinstance(raw, list):
        return _normalize_texts(raw)
    if isinstance(raw, dict):
        texts: List[str] = []
        for value in raw.values():
            if isinstance(value, list):
                texts.extend(_normalize_texts(value))
        return texts
    raise ValueError(
        f"Unsupported JSON in {path}: expected list or dict-of-lists, "
        f"got {type(raw).__name__}."
    )


def load_forget_retain_from_dict_json(
    path: Path,
    *,
    forget_key: str,
    retain_key: str,
) -> Tuple[List[str], List[str]]:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Labeled data {path} must be a JSON object with "
            f"'{forget_key}' and '{retain_key}' lists when not using CSV columns."
        )
    forget_key_l = forget_key.lower()
    retain_key_l = retain_key.lower()
    forget_samples: List[str] = []
    retain_samples: List[str] = []
    for key, value in raw.items():
        if not isinstance(value, list):
            continue
        texts = _normalize_texts(value)
        key_l = str(key).lower()
        if key_l == forget_key_l:
            forget_samples.extend(texts)
        elif key_l == retain_key_l:
            retain_samples.extend(texts)
    if forget_samples and retain_samples:
        return forget_samples, retain_samples
    raise ValueError(
        f"Could not find both '{forget_key}' and '{retain_key}' lists in {path}. "
        f"Got {len(forget_samples)} forget and {len(retain_samples)} retain samples."
    )


def load_forget_retain_from_supervised(
    path: Path,
    *,
    forget_labels: Sequence[str],
    retain_labels: Sequence[str],
) -> Tuple[List[str], List[str]]:
    dataset = SupervisedConceptDataset(str(path))
    forget_set = {label.lower() for label in forget_labels}
    retain_set = {label.lower() for label in retain_labels}
    forget_samples: List[str] = []
    retain_samples: List[str] = []
    for prompt, label in dataset.data:
        label_l = str(label).lower()
        if label_l in forget_set:
            forget_samples.append(prompt)
        elif label_l in retain_set:
            retain_samples.append(prompt)
    if not forget_samples or not retain_samples:
        raise ValueError(
            f"No samples matched forget labels {forget_labels} / retain labels "
            f"{retain_labels} in {path}."
        )
    return forget_samples, retain_samples


def load_forget_retain_corpora(
    *,
    forget_path: Optional[Path],
    retain_path: Optional[Path],
    labeled_path: Optional[Path],
    forget_key: str,
    retain_key: str,
    forget_labels: Sequence[str],
    retain_labels: Sequence[str],
) -> Tuple[List[str], List[str]]:
    if forget_path and retain_path:
        forget_samples = load_text_corpus(forget_path)
        retain_samples = load_text_corpus(retain_path)
        if not forget_samples or not retain_samples:
            raise ValueError("Forget and retain corpora must both be non-empty.")
        return forget_samples, retain_samples

    if labeled_path is None:
        raise ValueError(
            "Provide either --forget-data and --retain-data, or --labeled-data."
        )

    if labeled_path.suffix.lower() == ".json":
        try:
            return load_forget_retain_from_dict_json(
                labeled_path, forget_key=forget_key, retain_key=retain_key,
            )
        except json.JSONDecodeError:
            raise
        except ValueError:
            pass

    return load_forget_retain_from_supervised(
        labeled_path,
        forget_labels=forget_labels,
        retain_labels=retain_labels,
    )


def _verdict_from_payload(payload: Any, *, source_path: Path) -> Dict[str, Any]:
    if source_path.name == "audit_summary.json" and isinstance(payload, dict):
        verdict = payload.get("judge_verdict") or {}
        if payload.get("judge_error"):
            logger.warning("Audit summary records judge_error: %s", payload["judge_error"])
        return verdict if isinstance(verdict, dict) else {}
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unexpected judge payload type in {source_path}.")


def _extract_concept_from_raw_text(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(
        r'"likely_unlearned_concept"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        flags=re.DOTALL,
    )
    if not match:
        return None
    concept = json.loads(f'"{match.group(1)}"')
    if not str(concept).strip():
        return None
    recovered: Dict[str, Any] = {"likely_unlearned_concept": str(concept).strip()}
    conf = re.search(r'"unlearning_confidence"\s*:\s*(\d+)', text)
    if conf:
        recovered["unlearning_confidence"] = int(conf.group(1))
    return recovered


def _recover_parsed_verdict(verdict: Dict[str, Any], audit_dir: Optional[Path]) -> Dict[str, Any]:
    if verdict.get("likely_unlearned_concept") and not verdict.get("_parse_error"):
        return verdict

    raw_text = verdict.get("_raw_text")
    if isinstance(raw_text, str) and raw_text.strip():
        recovered = parse_judge_json(raw_text)
        if recovered.get("likely_unlearned_concept") and not recovered.get("_parse_error"):
            return recovered
        partial = _extract_concept_from_raw_text(raw_text)
        if partial:
            return partial

    if audit_dir is not None:
        raw_path = audit_dir / "judge_response_raw.txt"
        if raw_path.is_file():
            raw = raw_path.read_text(encoding="utf-8")
            recovered = parse_judge_json(raw)
            if recovered.get("likely_unlearned_concept") and not recovered.get("_parse_error"):
                return recovered
            partial = _extract_concept_from_raw_text(raw)
            if partial:
                return partial

    return verdict


def resolve_judge_hypothesis(
    *,
    audit_dir: Optional[Path],
    judge_response: Optional[Path],
    hypothesis: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    """Return ``(hypothesis_text, source_verdict_dict)``."""
    if hypothesis and hypothesis.strip():
        return hypothesis.strip(), {}

    verdict_path = judge_response
    if verdict_path is None and audit_dir is not None:
        for candidate in (
            audit_dir / "judge_response.json",
            audit_dir / "judge_response_raw.txt",
            audit_dir / "audit_summary.json",
        ):
            if candidate.is_file():
                verdict_path = candidate
                break

    if verdict_path is None or not verdict_path.is_file():
        raise FileNotFoundError(
            "No hypothesis provided. Pass --hypothesis, --judge-response, or "
            "--audit-dir containing judge_response.json, judge_response_raw.txt, "
            "or audit_summary.json."
        )

    payload = _load_json(verdict_path) if verdict_path.suffix == ".json" else None
    if payload is None:
        verdict = parse_judge_json(verdict_path.read_text(encoding="utf-8"))
    else:
        verdict = _verdict_from_payload(payload, source_path=verdict_path)

    verdict = _recover_parsed_verdict(verdict, audit_dir)

    if verdict.get("_parse_error"):
        raise ValueError(
            f"Judge verdict in {verdict_path} failed to parse; cannot evaluate."
        )

    concept = verdict.get("likely_unlearned_concept")
    if concept is None or not str(concept).strip():
        raise ValueError(
            f"Judge verdict in {verdict_path} has no 'likely_unlearned_concept'."
        )

    return str(concept).strip(), verdict


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate an audit judge hypothesis (likely_unlearned_concept) with "
            "blind forget/retain classification metrics."
        ),
    )
    src = p.add_argument_group("hypothesis source (one required unless --hypothesis)")
    src.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="Audit output directory (reads judge_response.json or audit_summary.json).",
    )
    src.add_argument(
        "--judge-response",
        type=Path,
        default=None,
        help="Path to judge_response.json (overrides --audit-dir file lookup).",
    )
    src.add_argument(
        "--hypothesis",
        type=str,
        default=None,
        help="Override hypothesis text (skips judge file).",
    )

    data = p.add_argument_group("evaluation corpora")
    data.add_argument(
        "--labeled-data",
        type=Path,
        default=None,
        help=(
            "JSON with forget/retain keys, or CSV/JSON with prompt+label columns "
            "(see --forget-labels / --retain-labels)."
        ),
    )
    data.add_argument("--forget-data", type=Path, default=None, help="JSON list of forget texts.")
    data.add_argument("--retain-data", type=Path, default=None, help="JSON list of retain texts.")
    data.add_argument(
        "--forget-key",
        type=str,
        default="forget",
        help="Key for forget list in dict JSON (default: forget).",
    )
    data.add_argument(
        "--retain-key",
        type=str,
        default="retain",
        help="Key for retain list in dict JSON (default: retain).",
    )
    data.add_argument(
        "--forget-labels",
        type=str,
        default="forget,positive,1",
        help="Comma-separated labels treated as forget class in supervised files.",
    )
    data.add_argument(
        "--retain-labels",
        type=str,
        default="retain,negative,0",
        help="Comma-separated labels treated as retain class in supervised files.",
    )

    run = p.add_argument_group("run")
    run.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write target_evaluation_report.json and run.log here (default: <audit-dir>/target_evaluation).",
    )
    run.add_argument("--eval-model", type=str, default="gemini-2.5-flash")
    run.add_argument("--eval-temperature", type=float, default=0.0)
    run.add_argument("--eval-max-output-tokens", type=int, default=1024)
    run.add_argument("--eval-api-key-env", type=str, default="GOOGLE_API_KEY")
    run.add_argument("--max-samples-per-set", type=int, default=25)
    run.add_argument("--seed", type=int, default=42)
    return p


def _split_labels(spec: str) -> List[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    audit_dir = args.audit_dir.resolve() if args.audit_dir else None
    judge_response = args.judge_response.resolve() if args.judge_response else None

    out_dir = args.output_dir
    if out_dir is None:
        if audit_dir is None:
            raise ValueError("--output-dir is required when --audit-dir is not set.")
        out_dir = audit_dir / "target_evaluation"
    out_dir = out_dir.resolve()
    setup_logger(out_dir)

    logger.info("=" * 60)
    logger.info(
        "UNLEARNING TARGET EVALUATION  (%s)",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    logger.info("=" * 60)

    hypothesis, source_verdict = resolve_judge_hypothesis(
        audit_dir=audit_dir,
        judge_response=judge_response,
        hypothesis=args.hypothesis,
    )
    logger.info("Hypothesis under test: %r", hypothesis)
    if source_verdict:
        logger.info(
            "Source judge confidence=%s",
            source_verdict.get("unlearning_confidence"),
        )

    forget_labels = _split_labels(args.forget_labels)
    retain_labels = _split_labels(args.retain_labels)
    forget_samples, retain_samples = load_forget_retain_corpora(
        forget_path=args.forget_data.resolve() if args.forget_data else None,
        retain_path=args.retain_data.resolve() if args.retain_data else None,
        labeled_path=args.labeled_data.resolve() if args.labeled_data else None,
        forget_key=args.forget_key,
        retain_key=args.retain_key,
        forget_labels=forget_labels,
        retain_labels=retain_labels,
    )
    logger.info(
        "Loaded %d forget and %d retain evaluation texts.",
        len(forget_samples),
        len(retain_samples),
    )

    client = GeminiClient(
        model=args.eval_model,
        temperature=args.eval_temperature,
        max_output_tokens=args.eval_max_output_tokens,
        api_key_env=args.eval_api_key_env,
    )
    evaluator = UnlearningTargetEvaluator(client, seed=args.seed)
    metrics = evaluator.evaluate_hypothesis(
        hypothesis=hypothesis,
        forget_samples=forget_samples,
        retain_samples=retain_samples,
        max_samples_per_set=args.max_samples_per_set,
    )

    prompt_template = metrics.pop("_classification_prompt_template", None)

    report: Dict[str, Any] = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "audit_dir": str(audit_dir) if audit_dir else None,
            "judge_response": str(judge_response) if judge_response else None,
            "classification_prompt_template": prompt_template,
            "text_sample_placeholder": TEXT_SAMPLE_PLACEHOLDER,
            "eval_model": args.eval_model,
            "eval_temperature": args.eval_temperature,
            "eval_max_output_tokens": args.eval_max_output_tokens,
            "max_samples_per_set": args.max_samples_per_set,
            "seed": args.seed,
            "n_forget_pool": len(forget_samples),
            "n_retain_pool": len(retain_samples),
        },
        "source_judge_verdict": source_verdict or None,
        "evaluation": metrics,
    }

    report_path = out_dir / "target_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote evaluation report to %s", report_path)
    logger.info(
        "Results: balanced_accuracy=%s auc_roc=%s scored=%d/%d "
        "(api_failures=%d parse_failures=%d)",
        metrics.get("balanced_accuracy"),
        metrics.get("auc_roc"),
        metrics.get("n_samples_scored"),
        metrics.get("n_samples_evaluated"),
        metrics.get("n_api_failures"),
        metrics.get("n_parse_failures"),
    )
    return report


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
