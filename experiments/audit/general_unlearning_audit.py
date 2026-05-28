"""General (label-free) SNMF unlearning audit.

 See `docs/general_unlearning_audit.md` for pipeline, outputs, CLI examples, and dataset format.

"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from experiments.audit.config import (
    AuditConfig,
    JudgeConfig,
    audit_config_to_nested_dict,
    parse_args_to_config,
)
from experiments.audit.context_windows import _sample_id_to_spans
from experiments.audit.unlearning_audit_reporter import UnlearningAuditReporter
from llm_utils.gemini_client import GeminiClient
from experiments.audit.core.layer_auditor import LayerAuditor
from experiments.audit.core.projection import SubspaceProjector
from experiments.audit.core.rankers import global_top_features
from experiments.audit.summary_report import build_audit_summary_report
from experiments.audit.logit_lens import LogitLens
from experiments.audit.text_processing import (
    HAS_WORDFREQ as _HAS_WORDFREQ,
    extract_context_strings as _extract_context_strings,
    rare_word_ranking_from_contexts as _rare_word_ranking_from_contexts,
    top_contexts_for_latent as _top_contexts_for_latent,
)
from llm_utils.local_activation_generator import LocalActivationGenerator
from llm_utils.model_utils import load_local_model
from llm_utils.utils import resolve_device, set_seed, sorted_numeric_layer_dirs

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(output_dir: Path) -> logging.Logger:
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Data loading (label-free)
# ---------------------------------------------------------------------------

def _load_prompts(data_path: str, max_prompts: int, seed: int) -> List[str]:
    """Load a flat list of prompts; supports list or dict-of-list JSON."""
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        prompts = [str(x) for x in raw if isinstance(x, str) and x.strip()]
    elif isinstance(raw, dict):
        prompts = []
        for v in raw.values():
            if isinstance(v, list):
                prompts.extend(str(x) for x in v if isinstance(x, str) and x.strip())
    else:
        raise ValueError(
            f"Unsupported JSON structure in {data_path}: expected list or dict, "
            f"got {type(raw).__name__}."
        )
    if not prompts:
        raise ValueError(f"No usable prompts found in {data_path}.")

    rng = np.random.default_rng(seed)
    if max_prompts > 0 and len(prompts) > max_prompts:
        idx = rng.choice(np.arange(len(prompts)), size=max_prompts, replace=False)
        idx.sort()
        prompts = [prompts[int(i)] for i in idx]
    return prompts


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def _collect_activations(
    model_path: str,
    prompts: List[str],
    layers: List[int],
    *,
    mode: str,
    batch_size: int,
    device: str,
    return_tokenizer: bool = False,
    return_logit_lens: bool = False,
    lens_center_unembed: bool = True,
    lens_mask_special_tokens: bool = True,
) -> Tuple[List[torch.Tensor], List[int], List[int], Optional[Any], Optional[LogitLens]]:
    """Load model, hook MLPs, return acts + (token_ids, sample_ids), then unload.

    If ``return_logit_lens`` is set, also returns a ``LogitLens`` snapshot
    built from the loaded model (cheap to build, but pins ~lm_head bytes of
    extra CPU memory until released).
    """
    logging.info(f"Loading model from {model_path} on {device} for activation collection.")
    local = load_local_model(model_path, device=device)
    tokenizer = local.tokenizer if (return_tokenizer or return_logit_lens) else None
    lens: Optional[LogitLens] = None
    try:
        gen = LocalActivationGenerator(local, data_device="cpu", mode=mode)
        acts, token_ids, sample_ids = gen.generate_activations(
            prompts=prompts, layers=layers, batch_size=batch_size
        )
        if return_logit_lens:
            logging.info("Extracting logit-lens snapshot from base model "
                         "(final_norm + lm_head + per-layer down_proj). "
                         f"center_unembed={lens_center_unembed} "
                         f"mask_special_tokens={lens_mask_special_tokens}")
            lens = LogitLens(
                local.model, layers,
                tokenizer=local.tokenizer,
                center_unembed=lens_center_unembed,
                mask_special_tokens=lens_mask_special_tokens,
            )
            if lens.special_token_ids:
                logging.info(f"Lens will mask {len(lens.special_token_ids)} "
                             f"special / unused token ids before topk.")
    finally:
        del local
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Caller may not have asked for the tokenizer even though we needed it for
    # the lens; honor the original request.
    return acts, token_ids, sample_ids, (tokenizer if return_tokenizer else None), lens


# ---------------------------------------------------------------------------
# Per-layer audit
# ---------------------------------------------------------------------------

def _audit_one_layer(
    layer_idx: int,
    layer_dir: Path,
    A_base: torch.Tensor,
    A_cand: torch.Tensor,
    token_ids: List[int],
    sample_ids: List[int],
    tokenizer: Any,
    *,
    cfg: AuditConfig,
    out_dir: Path,
    lens: Optional[LogitLens] = None,
    vocab_lens_top_k: int = 0,
    aggregate_top_k: int = 0,
    rare_words_top_n: int = 0,
) -> Optional[Dict[str, Any]]:
    factors_path = layer_dir / "snmf_factors.pt"
    if not factors_path.exists():
        logging.warning(f"Layer {layer_idx}: missing {factors_path}; skipping.")
        return None
    ckpt = torch.load(factors_path, map_location="cpu", weights_only=False)
    if ckpt.get("mode", "mlp_intermediate") != "mlp_intermediate":
        raise ValueError(
            f"layer {layer_idx}: SNMF mode {ckpt.get('mode')!r} != 'mlp_intermediate'."
        )
    Z = ckpt["F"].float().cpu()  # (d_mlp, K)
    ridge_lambda = cfg.snmf.ridge_lambda
    contexts_per_feature = cfg.snmf.contexts_per_feature
    context_window = cfg.snmf.context_window
    aggregate_delta_weighted = cfg.lens.lens_delta_weighted
    rare_words_zipf_cutoff = cfg.rare.context_rare_zipf_cutoff
    rare_words_min_len = cfg.rare.context_rare_min_len
    logging.info(f"Layer {layer_idx}: Z shape={tuple(Z.shape)}; A_base shape={tuple(A_base.shape)}")

    projector = SubspaceProjector(ridge_lambda)
    auditor = LayerAuditor(cfg, projector)
    core = auditor.audit_layer(layer_idx, Z, A_base, A_cand, sample_ids)
    Y_base = core["Y_base"]
    top_idx_by_delta = core["top_indices"]
    rank_field = cfg.snmf.rank_by
    weight_vec = getattr(core["metrics"], rank_field)
    partial_payload, rows = auditor.build_layer_payload_numeric(core, layer_idx)
    per_latent = partial_payload["per_latent"]

    spans_for_ctx = _sample_id_to_spans(np.asarray(sample_ids))

    top_records: List[Dict[str, Any]] = []
    # Logit-lens for the same per-layer top set (uses M_base's down_proj +
    # final norm + lm_head, snapshotted before the model was discarded).
    F_cpu = ckpt["F"].float().cpu() if lens is not None else None
    top_vocab_per_latent: Dict[int, List[Dict[str, Any]]] = {}
    if lens is not None and vocab_lens_top_k > 0:
        top_vocab_per_latent = lens.project_latents(
            F=F_cpu,
            layer=layer_idx,
            latent_indices=top_idx_by_delta.tolist(),
            top_k=vocab_lens_top_k,
            tokenizer=tokenizer,
        )

    # Aggregate logit-lens: project the (delta-weighted) sum of this layer's
    # top-decreased SNMF columns through W_down, then logit-lens. This
    # surfaces the SHARED token signal across the top features, which is
    # often more interpretable than any individual feature when individual
    # lens results look noisy.
    top_vocab_sum: Optional[Dict[str, Any]] = None
    if (lens is not None and aggregate_top_k > 0
            and len(top_idx_by_delta) > 0):
        idx_list = [int(i) for i in top_idx_by_delta.tolist()]
        weights = (
            [float(weight_vec[i]) for i in idx_list]
            if aggregate_delta_weighted else None
        )
        agg_residual = lens.feature_residual(
            F=F_cpu, layer=layer_idx,
            latent_indices=idx_list, weights=weights,
        )
        agg_tokens = lens.topk_from_residual(
            agg_residual, top_k=aggregate_top_k, tokenizer=tokenizer,
        )
        top_vocab_sum = {
            "n_features_summed": len(idx_list),
            "delta_weighted": bool(aggregate_delta_weighted),
            "residual_norm": float(agg_residual.norm().item()),
            "tokens": agg_tokens,
        }

    # Compute contexts only for the per-layer top set (cheap, but still O(K) is wasteful).
    layer_context_strings: List[str] = []
    for i in top_idx_by_delta.tolist():
        ctxs = _top_contexts_for_latent(
            Y_base, token_ids, sample_ids, spans_for_ctx, tokenizer,
            latent_idx=int(i),
            n_contexts=contexts_per_feature,
            context_window=context_window,
        )
        rec = dict(per_latent[int(i)])
        rec["top_contexts"] = ctxs
        if int(i) in top_vocab_per_latent:
            rec["top_vocab_base"] = top_vocab_per_latent[int(i)]
        # Per-feature rare-word ranking over this latent's top contexts.
        ctx_strings = _extract_context_strings(ctxs)
        layer_context_strings.extend(ctx_strings)
        if rare_words_top_n > 0 and ctx_strings:
            rec["rare_context_words"] = _rare_word_ranking_from_contexts(
                ctx_strings,
                top_n=rare_words_top_n,
                zipf_cutoff=rare_words_zipf_cutoff,
                min_word_len=rare_words_min_len,
            )
        top_records.append(rec)

    # Per-layer aggregate rare-word ranking: union of all top features' contexts.
    rare_context_words_layer: List[Dict[str, Any]] = []
    if rare_words_top_n > 0 and layer_context_strings:
        rare_context_words_layer = _rare_word_ranking_from_contexts(
            layer_context_strings,
            top_n=rare_words_top_n,
            zipf_cutoff=rare_words_zipf_cutoff,
            min_word_len=rare_words_min_len,
        )

    layer_dir_out = out_dir / f"layer_{layer_idx}"
    layer_dir_out.mkdir(parents=True, exist_ok=True)
    layer_payload: Dict[str, Any] = {
        **partial_payload,
        "top_decreased_latents": top_records,
    }
    if top_vocab_sum is not None:
        layer_payload["top_vocab_base_sum"] = top_vocab_sum
    if rare_context_words_layer:
        layer_payload["rare_context_words_layer"] = {
            "n_contexts": int(len(layer_context_strings)),
            "n_features_pooled": int(len(top_idx_by_delta)),
            "zipf_cutoff": float(rare_words_zipf_cutoff),
            "min_word_len": int(rare_words_min_len),
            "words": rare_context_words_layer,
        }
    with open(layer_dir_out / "audit.json", "w", encoding="utf-8") as f:
        json.dump(layer_payload, f, indent=2)

    pd.DataFrame(rows).to_csv(layer_dir_out / "audit_features.csv", index=False)

    logging.info(
        "Layer %d: residual_base=%.4f residual_candidate=%.4f (residual_delta=%+.4f) | "
        "rel_delta: mean=%+.4f max=%+.4f min=%+.4f | "
        "abs_rel_delta: mean=%.4f max=%.4f min=%.4f",
        layer_idx,
        core["residuals"]["base"],
        core["residuals"]["candidate"],
        core["residuals"]["delta"],
        float(core["metrics"].rel_delta.mean()),
        float(core["metrics"].rel_delta.max()),
        float(core["metrics"].rel_delta.min()),
        float(core["metrics"].abs_rel_delta.mean()),
        float(core["metrics"].abs_rel_delta.max()),
        float(core["metrics"].abs_rel_delta.min()),
    )
    return layer_payload


# ---------------------------------------------------------------------------
# General audit pipeline (helpers for run_audit function)
# ---------------------------------------------------------------------------

def _build_per_layer_metrics_rows(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in payloads:
        rel_stats = p.get("rel_delta_stats", {}) or {}
        rows.append({
            "layer": p["layer"],
            "K": p["K"],
            "n_prompts": p["n_prompts"],
            "residual_base": p["reconstruction_residual_relative"]["base"],
            "residual_candidate": p["reconstruction_residual_relative"]["candidate"],
            "residual_delta": p["reconstruction_residual_relative"]["delta"],
            "rel_delta_mean": float(rel_stats.get("mean", 0.0)),
            "rel_delta_max": float(rel_stats.get("max", 0.0)),
            "rel_delta_min": float(rel_stats.get("min", 0.0)),
        })
    return rows


def _resolve_audit_layer_plan(
    cfg: AuditConfig, logger: logging.Logger,
) -> Tuple[Path, List[Tuple[int, Path]], List[int]]:
    """Resolve SNMF output directory plus layer directories and indices to audit."""
    snmf_dir = Path(cfg.snmf_dir).resolve()
    layer_pairs = sorted_numeric_layer_dirs(snmf_dir)
    wanted = cfg.layers_wanted()
    if wanted is not None:
        layer_pairs = [(i, p) for i, p in layer_pairs if i in wanted]
        missing = wanted - {i for i, _ in layer_pairs}
        if missing:
            logger.warning(
                f"Requested layers not found in {snmf_dir}: {sorted(missing)}",
            )
    if not layer_pairs:
        raise RuntimeError(f"No layer_* dirs to audit under {snmf_dir}")
    layers = [i for i, _ in layer_pairs]
    logger.info(f"Auditing layers: {layers}")
    return snmf_dir, layer_pairs, layers


def _is_vocab_logit_lens_enabled(cfg: AuditConfig) -> bool:
    return (not cfg.lens.skip_vocab_lens) and (cfg.lens.vocab_lens_top_k > 0)


def _resolve_context_rare_word_top_n(
    cfg: AuditConfig, logger: logging.Logger,
) -> int:
    """Effective rare-context word count: config value, or zero if disabled or ``wordfreq`` is missing."""
    if cfg.rare.skip_context_rare_words:
        return 0
    n = max(0, int(cfg.rare.context_rare_top_n))
    if n > 0 and not _HAS_WORDFREQ:
        logger.warning(
            "wordfreq is not installed; rare-context-word ranking will be skipped. "
            "`pip install wordfreq` to enable it (or pass --skip-context-rare-words "
            "to silence this warning).",
        )
        return 0
    return n


def _collect_aligned_dual_model_activations(
    cfg: AuditConfig,
    prompts: List[str],
    layers: List[int],
    device: str,
    vocab_lens_enabled: bool,
    logger: logging.Logger,
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[int],
    List[int],
    Any,
    Optional[LogitLens],
]:
    """Collect base- and candidate-model activations on identical prompt order; verify token alignment."""
    logger.info("=== Collecting BASE activations ===")
    acts_base, token_ids_base, sample_ids_base, tokenizer, lens = _collect_activations(
        cfg.base_model_path,
        prompts,
        layers,
        mode=cfg.snmf.mode,
        batch_size=cfg.runtime.batch_size,
        device=device,
        return_tokenizer=True,
        return_logit_lens=vocab_lens_enabled,
        lens_center_unembed=cfg.lens.lens_center_unembed,
        lens_mask_special_tokens=cfg.lens.lens_mask_special_tokens,
    )
    if lens is not None:
        logger.info(
            f"Logit-lens snapshot ready ({len(lens.down_proj)} layers); "
            f"moving to device={device} for projection.",
        )
        lens.to(device)

    logger.info("=== Collecting CANDIDATE activations ===")
    acts_cand, token_ids_cand, sample_ids_cand, _, _ = _collect_activations(
        cfg.candidate_model_path,
        prompts,
        layers,
        mode=cfg.snmf.mode,
        batch_size=cfg.runtime.batch_size,
        device=device,
        return_tokenizer=False,
        return_logit_lens=False,
    )
    if sample_ids_base != sample_ids_cand or token_ids_base != token_ids_cand:
        raise RuntimeError(
            "Token streams from base vs candidate model do not match "
            "(different tokenizers / padding?). The audit requires identical "
            "token alignment between the two models.",
        )
    if tokenizer is None:
        raise RuntimeError(
            "Base model loader did not return a tokenizer; cannot decode contexts for audit.",
        )
    return acts_base, acts_cand, token_ids_base, sample_ids_base, tokenizer, lens


def _execute_per_layer_audits(
    cfg: AuditConfig,
    out_dir: Path,
    layer_pairs: List[Tuple[int, Path]],
    acts_base: List[torch.Tensor],
    acts_cand: List[torch.Tensor],
    token_ids_base: List[int],
    sample_ids_base: List[int],
    tokenizer: Any,
    lens: Optional[LogitLens],
    vocab_lens_enabled: bool,
    context_rare_word_top_n: int,
) -> List[Dict[str, Any]]:
    layer_payloads: List[Dict[str, Any]] = []
    for li, (layer_idx, layer_dir) in enumerate(layer_pairs):
        A_base = acts_base[li]
        A_cand = acts_cand[li]
        if A_base.shape != A_cand.shape:
            raise RuntimeError(
                f"layer {layer_idx}: A_base shape {tuple(A_base.shape)} != "
                f"A_candidate shape {tuple(A_cand.shape)}.",
            )
        payload = _audit_one_layer(
            layer_idx,
            layer_dir,
            A_base,
            A_cand,
            token_ids_base,
            sample_ids_base,
            tokenizer,
            cfg=cfg,
            out_dir=out_dir,
            lens=lens,
            vocab_lens_top_k=(cfg.lens.vocab_lens_top_k if vocab_lens_enabled else 0),
            aggregate_top_k=(
                cfg.lens.vocab_lens_aggregate_top_k if vocab_lens_enabled else 0
            ),
            rare_words_top_n=context_rare_word_top_n,
        )
        if payload is not None:
            layer_payloads.append(payload)

    if not layer_payloads:
        raise RuntimeError(
            "No layers were successfully audited (no snmf_factors.pt found?).",
        )
    return layer_payloads


def _build_cross_layer_logit_lens_aggregate(
    lens: Optional[LogitLens],
    vocab_lens_enabled: bool,
    cfg: AuditConfig,
    layer_pairs: List[Tuple[int, Path]],
    global_top: List[Dict[str, Any]],
    tokenizer: Optional[Any],
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    """Sum residuals for globally ranked features across layers, then run logit lens once."""
    if not (
        lens is not None
        and vocab_lens_enabled
        and cfg.lens.vocab_lens_aggregate_top_k > 0
        and global_top
    ):
        return None
    logit_lens = lens

    layer_dir_by_idx: Dict[int, Path] = {int(li): p for li, p in layer_pairs}
    by_layer: Dict[int, List[Tuple[int, float]]] = {}
    rank_field = cfg.snmf.rank_by
    for rec in global_top:
        L = int(rec["layer"])
        i = int(rec["latent_idx"])
        w = float(rec[rank_field])
        by_layer.setdefault(L, []).append((i, w))

    r_global: Optional[torch.Tensor] = None
    actual_layers_summed: set[int] = set()
    for L, entries in by_layer.items():
        ldir = layer_dir_by_idx.get(L)
        if ldir is None:
            logger.warning(f"Global aggregate: no layer dir for layer {L}; skipping.")
            continue
        ckpt_L = torch.load(
            ldir / "snmf_factors.pt",
            map_location="cpu",
            weights_only=False,
        )
        F_L = ckpt_L["F"].float().cpu()
        indices = [i for i, _ in entries]
        weights = ([w for _, w in entries] if cfg.lens.lens_delta_weighted else None)
        r_L = logit_lens.feature_residual(F_L, L, indices, weights)
        r_global = r_L if r_global is None else r_global + r_L
        actual_layers_summed.add(L)

    if r_global is None:
        return None

    agg_tokens = logit_lens.topk_from_residual(
        r_global,
        top_k=cfg.lens.vocab_lens_aggregate_top_k,
        tokenizer=tokenizer,
    )
    payload: Dict[str, Any] = {
        "n_features_summed": int(len(global_top)),
        "n_layers_spanned": int(len(actual_layers_summed)),
        "delta_weighted": bool(cfg.lens.lens_delta_weighted),
        "rank_by": cfg.snmf.rank_by,
        "residual_norm": float(r_global.norm().item()),
        "tokens": agg_tokens,
    }
    logger.info(
        "Global aggregate logit-lens: summed %d features across %d layers; "
        "residual norm=%.3f, top token=%r (%.2f).",
        len(global_top),
        len(actual_layers_summed),
        payload["residual_norm"],
        agg_tokens[0]["token"] if agg_tokens else None,
        agg_tokens[0]["logit"] if agg_tokens else float("nan"),
    )
    return payload


def _assemble_judge_prompt_context(
    layer_payloads: List[Dict[str, Any]],
    global_top: List[Dict[str, Any]],
    context_rare_word_top_n: int,
    cfg: AuditConfig,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    """Structured vocabulary and rare-token statistics for inclusion in the judge prompt."""
    per_layer_aggregate_vocab_lens: List[Dict[str, Any]] = []
    for p in layer_payloads:
        agg = p.get("top_vocab_base_sum")
        if agg is not None:
            per_layer_aggregate_vocab_lens.append({"layer": p["layer"], **agg})

    per_layer_rare_word_blocks: List[Dict[str, Any]] = []
    for p in layer_payloads:
        rw_layer = p.get("rare_context_words_layer")
        if rw_layer and rw_layer.get("words"):
            per_layer_rare_word_blocks.append({"layer": p["layer"], **rw_layer})

    global_rare_word_block: Optional[Dict[str, Any]] = None
    if context_rare_word_top_n > 0 and global_top:
        all_global_ctx: List[str] = []
        for rec in global_top:
            all_global_ctx.extend(
                _extract_context_strings(rec.get("top_contexts") or []),
            )
        g_ranked = _rare_word_ranking_from_contexts(
            all_global_ctx,
            top_n=context_rare_word_top_n,
            zipf_cutoff=float(cfg.rare.context_rare_zipf_cutoff),
            min_word_len=int(cfg.rare.context_rare_min_len),
        )
        if g_ranked:
            global_rare_word_block = {
                "n_features_pooled": int(len(global_top)),
                "n_contexts": int(len(all_global_ctx)),
                "zipf_cutoff": float(cfg.rare.context_rare_zipf_cutoff),
                "words": g_ranked,
            }

    return (
        per_layer_aggregate_vocab_lens,
        per_layer_rare_word_blocks,
        global_rare_word_block,
    )


def _invoke_gemini_audit_judge(
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
        logger.warning(f"Judge call failed: {judge_error}")
        (out_dir / "judge_response_raw.txt").write_text("", encoding="utf-8")
        return judge_verdict, judge_error

    logger.info(f"Calling judge model: {client.model}")

    judge_verdict, raw_text, judge_error, finish_reason = reporter.run_prompt(
        judge_prompt,
    )
    (out_dir / "judge_response_raw.txt").write_text(raw_text or "", encoding="utf-8")
    if finish_reason:
        lvl = logger.warning if finish_reason == "MAX_TOKENS" else logger.info
        lvl("Judge finished with reason: %s", finish_reason)
    if judge_error:
        logger.warning(f"Judge call failed: {judge_error}")
    elif judge_verdict.get("_parse_error"):
        judge_error = str(judge_verdict.get("_parse_error", "parse error"))
        logger.warning(f"Judge parse failed: {judge_error}")
    else:
        logger.info(
            "Judge verdict: confidence=%s | concept=%r",
            judge_verdict.get("unlearning_confidence"),
            judge_verdict.get("likely_unlearned_concept"),
        )
    return judge_verdict, judge_error


def _emit_audit_run_summary_logs(
    logger: logging.Logger,
    *,
    cfg: AuditConfig,
    out_dir: Path,
    per_layer_summary: List[Dict[str, Any]],
    global_top: List[Dict[str, Any]],
    judge_verdict: Dict[str, Any],
) -> None:
    """Structured logging for headline metrics and the LLM verdict (if parsed)."""
    logger.info("=== Audit headline (per-layer) ===")
    for row in per_layer_summary:
        logger.info(
            "L%02d | residual base=%.4f -> candidate=%.4f (residual_delta=%+.4f) | "
            "rel_delta_max=%+.4f rel_delta_mean=%+.4f",
            row["layer"],
            row["residual_base"],
            row["residual_candidate"],
            row["residual_delta"],
            row["rel_delta_max"],
            row["rel_delta_mean"],
        )
    logger.info(
        "Top-%d most-changed features (global, by %s):",
        cfg.snmf.top_k_global,
        cfg.snmf.rank_by,
    )
    for rec in global_top:
        logger.info(
            "  L%d.lat%d  rel_delta=%+.4f  abs_rel_delta=%.4f",
            rec["layer"],
            rec["latent_idx"],
            rec.get("rel_delta", 0.0),
            rec.get("abs_rel_delta", 0.0),
        )
    if judge_verdict and not judge_verdict.get("_parse_error"):
        logger.info("=== Judge verdict ===")
        logger.info(
            "confidence=%s  concept=%r",
            judge_verdict.get("unlearning_confidence"),
            judge_verdict.get("likely_unlearned_concept"),
        )
        if judge_verdict.get("concept_rationale"):
            logger.info("rationale: %s", judge_verdict["concept_rationale"])
    logger.info("Done. Outputs under: %s", out_dir)


def run_audit(cfg: AuditConfig) -> None:
    """Execute the full general SNMF audit pipeline from a structured config."""
    set_seed(cfg.runtime.seed)
    out_dir = Path(cfg.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)
    logger.info("=" * 60)
    logger.info(
        "GENERAL SNMF UNLEARNING AUDIT  "
        f"({datetime.now():%Y-%m-%d %H:%M:%S})",
    )
    logger.info("=" * 60)
    logger.info(
        "Audit config:\n"
        f"{json.dumps(audit_config_to_nested_dict(cfg), indent=2)}",
    )

    snmf_dir, layer_pairs, layers = _resolve_audit_layer_plan(cfg, logger)
    prompts = _load_prompts(
        cfg.data_path, cfg.runtime.max_prompts, cfg.runtime.seed,
    )
    logger.info(f"Loaded {len(prompts)} audit prompts from {cfg.data_path}.")

    device = resolve_device(cfg.runtime.device)
    logger.info(f"Resolved compute device: {device}")

    vocab_lens_enabled = _is_vocab_logit_lens_enabled(cfg)
    context_rare_word_top_n = _resolve_context_rare_word_top_n(cfg, logger)

    acts_base, acts_cand, token_ids_base, sample_ids_base, tokenizer, lens = (
        _collect_aligned_dual_model_activations(
            cfg, prompts, layers, device, vocab_lens_enabled, logger,
        )
    )

    layer_payloads = _execute_per_layer_audits(
        cfg,
        out_dir,
        layer_pairs,
        acts_base,
        acts_cand,
        token_ids_base,
        sample_ids_base,
        tokenizer,
        lens,
        vocab_lens_enabled,
        context_rare_word_top_n,
    )

    global_top = global_top_features(
        layer_payloads, cfg.snmf.rank_by, cfg.snmf.top_k_global,
    )
    per_layer_summary = _build_per_layer_metrics_rows(layer_payloads)

    global_aggregate_vocab = _build_cross_layer_logit_lens_aggregate(
        lens,
        vocab_lens_enabled,
        cfg,
        layer_pairs,
        global_top,
        tokenizer,
        logger,
    )
    (
        per_layer_aggregate_vocab,
        judge_per_layer_rare_words,
        judge_global_rare_words,
    ) = _assemble_judge_prompt_context(
        layer_payloads,
        global_top,
        context_rare_word_top_n,
        cfg,
    )

    judge_prompt = UnlearningAuditReporter.build_audit_report_prompt(
        n_prompts=int(len(prompts)),
        layers=layers,
        rank_by=cfg.snmf.rank_by,
        global_top=global_top,
        per_layer_summary=per_layer_summary,
        per_layer_aggregate_vocab=per_layer_aggregate_vocab,
        global_aggregate_vocab=global_aggregate_vocab,
        per_layer_rare_words=(judge_per_layer_rare_words or None),
        global_rare_words=judge_global_rare_words,
    )
    (out_dir / "judge_prompt.txt").write_text(judge_prompt, encoding="utf-8")
    logger.info(
        "Wrote judge prompt (%d chars) to %s",
        len(judge_prompt),
        out_dir / "judge_prompt.txt",
    )

    judge_verdict: Dict[str, Any] = {}
    judge_error: Optional[str] = None
    if cfg.judge.skip_judge:
        logger.info("--skip-judge set; not calling the LLM judge.")
        judge_error = "skipped (--skip-judge)"
    else:
        judge_verdict, judge_error = _invoke_gemini_audit_judge(
            cfg.judge, judge_prompt, out_dir, logger,
        )

    report = build_audit_summary_report(
        cfg,
        snmf_dir=snmf_dir,
        layers=layers,
        n_prompts=int(len(prompts)),
        want_rare_top_n=context_rare_word_top_n,
        wordfreq_available=_HAS_WORDFREQ,
        per_layer_summary=per_layer_summary,
        global_top_features=global_top,
        per_layer_aggregate_vocab=per_layer_aggregate_vocab,
        global_aggregate_vocab=global_aggregate_vocab,
        judge_verdict=judge_verdict,
        judge_error=judge_error,
    )
    report.export_all(out_dir)

    _emit_audit_run_summary_logs(
        logger,
        cfg=cfg,
        out_dir=out_dir,
        per_layer_summary=per_layer_summary,
        global_top=global_top,
        judge_verdict=judge_verdict,
    )


def main() -> None:
    cfg = parse_args_to_config()
    run_audit(cfg)


if __name__ == "__main__":
    main()
