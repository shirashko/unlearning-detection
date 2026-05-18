"""Serialization schemas and exporters for SNMF unlearning audit reports (label-free)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from experiments.audit.config import AuditConfig, audit_config_to_nested_dict


class AuditMetadata(BaseModel):
    """Execution context and hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    base_model_path: str
    candidate_model_path: str
    snmf_dir: str
    data_path: str
    layers: List[int]
    n_prompts: int
    max_prompts: int
    ridge_lambda: float
    rank_by: str
    top_k_global: int
    top_k_per_layer: int
    contexts_per_feature: int
    context_window: int
    vocab_lens_top_k: int
    skip_vocab_lens: bool
    lens_center_unembed: bool
    lens_mask_special_tokens: bool
    vocab_lens_aggregate_top_k: int
    lens_delta_weighted: bool
    context_rare_top_n: int
    skip_context_rare_words: bool
    context_rare_zipf_cutoff: float
    context_rare_min_len: int
    context_rare_effective_top_n: int
    wordfreq_available: bool
    mode: str
    judge_model: str
    judge_temperature: float
    judge_max_output_tokens: int
    judge_skipped: bool
    audit_config: Dict[str, Any]


class AuditSummaryReport(BaseModel):
    """Full label-free audit summary written to ``audit_summary.json``."""

    model_config = ConfigDict(extra="forbid")

    meta: AuditMetadata
    per_layer_summary: List[Dict[str, Any]]
    global_top_features: List[Dict[str, Any]]
    per_layer_aggregate_vocab: List[Dict[str, Any]]
    global_aggregate_vocab: Optional[Dict[str, Any]] = None
    judge_verdict: Dict[str, Any] = Field(default_factory=dict)
    judge_error: Optional[str] = None

    def export_all(
        self,
        out_dir: Path,
        *,
        include_standalone_judge: bool = True,
    ) -> None:
        """Write summary JSON, per-layer CSV, and optionally ``judge_response.json``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "audit_summary.json").write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if self.per_layer_summary:
            pd.DataFrame(self.per_layer_summary).to_csv(
                out_dir / "audit_summary_per_layer.csv",
                index=False,
            )
        if (
            include_standalone_judge
            and self.judge_verdict
            and not self.judge_verdict.get("_parse_error")
        ):
            with open(out_dir / "judge_response.json", "w", encoding="utf-8") as f:
                json.dump(self.judge_verdict, f, indent=2)


def audit_metadata_from_run(
    cfg: AuditConfig,
    *,
    snmf_dir: Path,
    layers: List[int],
    n_prompts: int,
    want_rare_top_n: int,
    wordfreq_available: bool,
) -> AuditMetadata:
    """Build validated metadata from the live run (paths, effective flags, full config snapshot)."""
    return AuditMetadata(
        base_model_path=cfg.base_model_path,
        candidate_model_path=cfg.candidate_model_path,
        snmf_dir=str(snmf_dir),
        data_path=cfg.data_path,
        layers=layers,
        n_prompts=n_prompts,
        max_prompts=cfg.runtime.max_prompts,
        ridge_lambda=cfg.snmf.ridge_lambda,
        rank_by=cfg.snmf.rank_by,
        top_k_global=cfg.snmf.top_k_global,
        top_k_per_layer=cfg.snmf.top_k_per_layer,
        contexts_per_feature=cfg.snmf.contexts_per_feature,
        context_window=cfg.snmf.context_window,
        vocab_lens_top_k=cfg.lens.vocab_lens_top_k,
        skip_vocab_lens=bool(cfg.lens.skip_vocab_lens),
        lens_center_unembed=bool(cfg.lens.lens_center_unembed),
        lens_mask_special_tokens=bool(cfg.lens.lens_mask_special_tokens),
        vocab_lens_aggregate_top_k=cfg.lens.vocab_lens_aggregate_top_k,
        lens_delta_weighted=bool(cfg.lens.lens_delta_weighted),
        context_rare_top_n=int(cfg.rare.context_rare_top_n),
        skip_context_rare_words=bool(cfg.rare.skip_context_rare_words),
        context_rare_zipf_cutoff=float(cfg.rare.context_rare_zipf_cutoff),
        context_rare_min_len=int(cfg.rare.context_rare_min_len),
        context_rare_effective_top_n=int(want_rare_top_n),
        wordfreq_available=bool(wordfreq_available),
        mode=cfg.snmf.mode,
        judge_model=cfg.judge.judge_model,
        judge_temperature=cfg.judge.judge_temperature,
        judge_max_output_tokens=cfg.judge.judge_max_output_tokens,
        judge_skipped=bool(cfg.judge.skip_judge),
        audit_config=audit_config_to_nested_dict(cfg),
    )


def build_audit_summary_report(
    cfg: AuditConfig,
    *,
    snmf_dir: Path,
    layers: List[int],
    n_prompts: int,
    want_rare_top_n: int,
    wordfreq_available: bool,
    per_layer_summary: List[Dict[str, Any]],
    global_top_features: List[Dict[str, Any]],
    per_layer_aggregate_vocab: List[Dict[str, Any]],
    global_aggregate_vocab: Optional[Dict[str, Any]],
    judge_verdict: Dict[str, Any],
    judge_error: Optional[str],
) -> AuditSummaryReport:
    return AuditSummaryReport(
        meta=audit_metadata_from_run(
            cfg,
            snmf_dir=snmf_dir,
            layers=layers,
            n_prompts=n_prompts,
            want_rare_top_n=want_rare_top_n,
            wordfreq_available=wordfreq_available,
        ),
        per_layer_summary=per_layer_summary,
        global_top_features=global_top_features,
        per_layer_aggregate_vocab=per_layer_aggregate_vocab,
        global_aggregate_vocab=global_aggregate_vocab,
        judge_verdict=judge_verdict,
        judge_error=judge_error,
    )
