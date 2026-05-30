"""Display precision for audit metrics (matches log / JSON formatting)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

AUDIT_METRIC_DECIMALS = 4
AUDIT_LOGIT_DECIMALS = 2
AUDIT_RESIDUAL_NORM_DECIMALS = 3

LATENT_METRIC_KEYS = (
    "mean_Y_base",
    "mean_Y_candidate",
    "rel_delta",
    "abs_rel_delta",
)

PER_LAYER_SUMMARY_METRIC_KEYS = (
    "residual_base",
    "residual_candidate",
    "residual_delta",
    "rel_delta_mean",
    "rel_delta_max",
    "rel_delta_min",
)


def round_audit_metric(value: float) -> float:
    return float(round(value, AUDIT_METRIC_DECIMALS))


def round_audit_logit(value: float) -> float:
    return float(round(value, AUDIT_LOGIT_DECIMALS))


def round_audit_residual_norm(value: float) -> float:
    return float(round(value, AUDIT_RESIDUAL_NORM_DECIMALS))


def round_rare_word_metrics(rec: Dict[str, Any]) -> None:
    if "zipf" in rec:
        rec["zipf"] = round_audit_metric(float(rec["zipf"]))
    if "score" in rec:
        rec["score"] = round_audit_metric(float(rec["score"]))


def round_latent_feature_record(rec: Dict[str, Any]) -> None:
    """Round metrics on a top-feature / per-latent export record in place."""
    for key in LATENT_METRIC_KEYS:
        if key in rec:
            rec[key] = round_audit_metric(float(rec[key]))
    for ctx in rec.get("top_contexts") or []:
        if isinstance(ctx, dict) and "activation" in ctx:
            ctx["activation"] = round_audit_metric(float(ctx["activation"]))
    for tok in rec.get("top_vocab_base") or []:
        if isinstance(tok, dict) and "logit" in tok:
            tok["logit"] = round_audit_logit(float(tok["logit"]))
    for word in rec.get("rare_context_words") or []:
        if isinstance(word, dict):
            round_rare_word_metrics(word)


def round_vocab_lens_block(block: Dict[str, Any]) -> None:
    if "residual_norm" in block:
        block["residual_norm"] = round_audit_residual_norm(float(block["residual_norm"]))
    for tok in block.get("tokens") or []:
        if isinstance(tok, dict) and "logit" in tok:
            tok["logit"] = round_audit_logit(float(tok["logit"]))


def round_per_layer_summary_row(row: Dict[str, Any]) -> None:
    for key in PER_LAYER_SUMMARY_METRIC_KEYS:
        if key in row:
            row[key] = round_audit_metric(float(row[key]))


def round_audit_summary_for_export(
    *,
    per_layer_summary: List[Dict[str, Any]],
    global_top_features: List[Dict[str, Any]],
    per_layer_aggregate_vocab: List[Dict[str, Any]],
    global_aggregate_vocab: Optional[Dict[str, Any]],
) -> None:
    """Normalize float precision in summary artifacts before JSON/CSV export."""
    for row in per_layer_summary:
        round_per_layer_summary_row(row)
    for rec in global_top_features:
        round_latent_feature_record(rec)
    for block in per_layer_aggregate_vocab:
        round_vocab_lens_block(block)
        for word in block.get("words") or []:
            if isinstance(word, dict):
                round_rare_word_metrics(word)
    if global_aggregate_vocab is not None:
        round_vocab_lens_block(global_aggregate_vocab)
        for word in global_aggregate_vocab.get("words") or []:
            if isinstance(word, dict):
                round_rare_word_metrics(word)
