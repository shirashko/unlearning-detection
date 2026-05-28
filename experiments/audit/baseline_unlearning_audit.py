"""Raw MLP neuron activation baseline for label-free unlearning audit.

Mirrors ``general_unlearning_audit.py`` but skips SNMF: each MLP intermediate
dimension is its own "feature" (identity basis). Peak statistics, ranking,
context windows, rare-context words, aggregate logit lens, and the Gemini
judge flow match the SNMF pipeline where applicable.

**Activation cache format** (``torch.save`` dict, loaded with ``weights_only=False``):

- ``acts_base``: ``list[torch.Tensor]``, each ``(n_tokens, d_mlp)``
- ``acts_cand``: same length and shapes as ``acts_base``
- ``token_ids``: ``list[int]`` (length ``n_tokens``)
- ``sample_ids``: ``list[int]`` (length ``n_tokens``, aligned prompts)
- ``layers`` (optional): ``list[int]`` layer index per tensor; if omitted,
  uses ``0 .. len(acts_base)-1``.

No transformer weights are required to load activations; logit-lens still
needs base-model ``W_down``, ``final_norm``, and ``lm_head`` (loaded briefly
via ``load_local_model`` unless vocab lens is skipped).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from experiments.audit.config import (
    AuditConfig,
    apply_argparse_namespace_overrides,
    audit_config_to_nested_dict,
    build_arg_parser,
    load_audit_config_yaml,
)
from experiments.audit.context_windows import _sample_id_to_spans
from experiments.audit.core.layer_auditor import LayerAuditor
from experiments.audit.core.projection import SubspaceProjector, per_prompt_peaks
from experiments.audit.core.rankers import (
    REL_DELTA_EPS,
    RankerFactory,
    compute_mean_peak_metrics,
    global_top_features,
)
from experiments.audit.general_unlearning_audit import (
    _assemble_judge_prompt_context,
    _build_per_layer_metrics_rows,
    _collect_activations,
    _collect_aligned_dual_model_activations,
    _invoke_gemini_audit_judge,
    _is_vocab_logit_lens_enabled,
    _load_prompts,
    _resolve_context_rare_word_top_n,
    _emit_audit_run_summary_logs,
    setup_logger,
)
from experiments.audit.unlearning_audit_reporter import UnlearningAuditReporter
from experiments.audit.logit_lens import LogitLens
from experiments.audit.summary_report import build_audit_summary_report
from experiments.audit.text_processing import (
    HAS_WORDFREQ as _HAS_WORDFREQ,
    extract_context_strings as _extract_context_strings,
    rare_word_ranking_from_contexts as _rare_word_ranking_from_contexts,
    top_contexts_for_latent as _top_contexts_for_latent,
)
from llm_utils.model_utils import load_local_model
from llm_utils.utils import resolve_device, set_seed

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


def validate_baseline_audit_config(
    cfg: AuditConfig,
    *,
    activations_cache: Optional[str],
) -> None:
    """Validate paths for baseline audit (cache and/or live collection)."""
    missing = [
        name for name, val in (
            ("base_model_path", cfg.base_model_path),
            ("output_dir", cfg.output_dir),
        ) if not str(val).strip()
    ]
    if missing:
        raise ValueError("Missing required config fields: " + ", ".join(missing))

    if cfg.snmf.rank_by not in ("rel_delta", "abs_rel_delta"):
        raise ValueError(f"Invalid rank_by={cfg.snmf.rank_by!r}")
    if cfg.snmf.mode != "mlp_intermediate":
        raise ValueError(
            f"baseline audit expects snmf.mode='mlp_intermediate', got {cfg.snmf.mode!r}",
        )

    out_dir = Path(cfg.output_dir).expanduser()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Cannot create output_dir {cfg.output_dir!r}: {e}") from e

    cache_path = str(activations_cache or "").strip()
    if cache_path:
        p = Path(cache_path).expanduser()
        if not p.is_file():
            raise ValueError(f"--activations-cache is not a file: {cache_path!r}")
    else:
        for name, val in (
            ("candidate_model_path", cfg.candidate_model_path),
            ("data_path", cfg.data_path),
        ):
            if not str(val).strip():
                raise ValueError(
                    f"Without --activations-cache, {name} is required.",
                )
        data_file = Path(cfg.data_path).expanduser()
        if not data_file.is_file():
            raise ValueError(f"data_path is not an existing file: {cfg.data_path!r}")
        for label, mp in (
            ("base_model_path", cfg.base_model_path),
            ("candidate_model_path", cfg.candidate_model_path),
        ):
            mps = str(mp).strip()
            if Path(mps).expanduser().is_absolute() or mps.startswith(("./", "../")):
                if not Path(mps).expanduser().is_dir():
                    raise ValueError(
                        f"{label} must be an existing local directory: {mps!r}",
                    )


def load_activations_cache(path: str | Path) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[int],
    List[int],
    List[int],
]:
    """Load aligned base/candidate activation lists and token metadata from disk."""
    path = Path(path).expanduser().resolve()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(f"Activation cache must be a dict, got {type(blob).__name__}")

    def _req(key: str) -> Any:
        if key not in blob:
            raise ValueError(f"Activation cache missing key {key!r} in {path}")
        return blob[key]

    acts_b = _req("acts_base")
    acts_c = _req("acts_cand")
    token_ids = _req("token_ids")
    sample_ids = _req("sample_ids")

    if not isinstance(acts_b, list) or not isinstance(acts_c, list):
        raise ValueError("acts_base and acts_cand must be lists of tensors")
    if len(acts_b) != len(acts_c):
        raise ValueError(
            f"acts_base ({len(acts_b)}) and acts_cand ({len(acts_c)}) length mismatch",
        )
    layers_raw = blob.get("layers")
    if layers_raw is None:
        layers_list = list(range(len(acts_b)))
    else:
        layers_list = [int(x) for x in layers_raw]
        if len(layers_list) != len(acts_b):
            raise ValueError(
                f"layers length {len(layers_list)} != len(acts_base) {len(acts_b)}",
            )

    tok = [int(x) for x in token_ids]
    sid = [int(x) for x in sample_ids]
    if len(tok) != len(sid):
        raise ValueError("token_ids and sample_ids length mismatch")

    for i, (tb, tc) in enumerate(zip(acts_b, acts_c)):
        if not isinstance(tb, torch.Tensor):
            tb = torch.as_tensor(tb)
        if not isinstance(tc, torch.Tensor):
            tc = torch.as_tensor(tc)
        acts_b[i] = tb.float().cpu()
        acts_c[i] = tc.float().cpu()
        if acts_b[i].shape != acts_c[i].shape:
            raise ValueError(
                f"Layer list idx {i}: acts_base shape {tuple(acts_b[i].shape)} != "
                f"acts_cand {tuple(acts_c[i].shape)}",
            )
        if acts_b[i].ndim != 2:
            raise ValueError(f"Layer list idx {i}: expected 2D activations")
        if acts_b[i].shape[0] != len(tok):
            raise ValueError(
                f"Layer list idx {i}: n_tokens {acts_b[i].shape[0]} "
                f"!= len(token_ids) {len(tok)}",
            )

    return acts_b, acts_c, tok, sid, layers_list


@torch.inference_mode()
def _project_raw_neurons_vocab_lens(
    lens: LogitLens,
    layer: int,
    neuron_indices: Sequence[int],
    top_k: int,
    tokenizer: Any,
) -> Dict[int, List[Dict[str, Any]]]:
    """Tokens-most-promoted for raw neuron ``i``: ``lm_head ∘ norm ∘ W_down[:, i]``."""
    if top_k <= 0 or not neuron_indices:
        return {}
    device = lens._device
    W_down = lens.down_proj[int(layer)].to(device)
    out: Dict[int, List[Dict[str, Any]]] = {}
    for i in neuron_indices:
        r = W_down[:, int(i)].to(device)
        out[int(i)] = lens.topk_from_residual(r, top_k, tokenizer)
    return out


@torch.inference_mode()
def _feature_residual_raw_neurons(
    lens: LogitLens,
    layer: int,
    neuron_indices: Sequence[int],
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Residual direction ``W_down @ v`` with ``v`` supported on ``neuron_indices``."""
    if not neuron_indices:
        raise ValueError("neuron_indices must be non-empty")
    device = lens._device
    W_down = lens.down_proj[int(layer)].to(device)
    d_model = W_down.shape[0]
    accum = torch.zeros(d_model, device=device, dtype=W_down.dtype)
    idx_list = [int(i) for i in neuron_indices]
    if weights is None:
        for i in idx_list:
            accum = accum + W_down[:, i]
    else:
        if len(weights) != len(idx_list):
            raise ValueError(
                f"weights length {len(weights)} != neuron_indices length {len(idx_list)}",
            )
        w_t = torch.tensor(list(weights), device=device, dtype=W_down.dtype)
        for j, i in enumerate(idx_list):
            accum = accum + w_t[j] * W_down[:, i]
    return accum


def _build_cross_layer_logit_lens_aggregate_raw(
    lens: Optional[LogitLens],
    vocab_lens_enabled: bool,
    cfg: AuditConfig,
    global_top: List[Dict[str, Any]],
    tokenizer: Optional[Any],
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    """Sum ``W_down[:, neuron]`` (optionally weighted) across global top neurons, then lens once."""
    if not (
        lens is not None
        and vocab_lens_enabled
        and cfg.lens.vocab_lens_aggregate_top_k > 0
        and global_top
    ):
        return None
    rank_field = cfg.snmf.rank_by
    by_layer: Dict[int, List[Tuple[int, float]]] = {}
    for rec in global_top:
        L = int(rec["layer"])
        neuron_i = int(rec["latent_idx"])
        w = float(rec[rank_field])
        by_layer.setdefault(L, []).append((neuron_i, w))

    r_global: Optional[torch.Tensor] = None
    actual_layers: set[int] = set()
    for L, entries in by_layer.items():
        if int(L) not in lens.down_proj:
            logger.warning(f"Global aggregate: layer {L} missing from logit lens; skip.")
            continue
        idx_list = [i for i, _ in entries]
        wts = ([w for _, w in entries] if cfg.lens.lens_delta_weighted else None)
        r_L = _feature_residual_raw_neurons(lens, L, idx_list, wts)
        r_global = r_L if r_global is None else r_global + r_L
        actual_layers.add(int(L))

    if r_global is None:
        return None

    agg_tokens = lens.topk_from_residual(
        r_global,
        top_k=cfg.lens.vocab_lens_aggregate_top_k,
        tokenizer=tokenizer,
    )
    payload: Dict[str, Any] = {
        "n_features_summed": int(len(global_top)),
        "n_layers_spanned": int(len(actual_layers)),
        "delta_weighted": bool(cfg.lens.lens_delta_weighted),
        "rank_by": cfg.snmf.rank_by,
        "residual_norm": float(r_global.norm().item()),
        "tokens": agg_tokens,
    }
    logger.info(
        "Global aggregate logit-lens (raw neurons): summed %d neurons across %d layers; "
        "residual norm=%.3f.",
        len(global_top),
        len(actual_layers),
        payload["residual_norm"],
    )
    return payload


def _audit_one_layer_raw(
    layer_idx: int,
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
) -> Dict[str, Any]:
    """Per-layer identity-basis audit (one feature per MLP neuron)."""
    if A_base.shape != A_cand.shape:
        raise RuntimeError(
            f"layer {layer_idx}: A_base shape {tuple(A_base.shape)} != "
            f"A_candidate shape {tuple(A_cand.shape)}.",
        )
    d_mlp = int(A_base.shape[1])
    K = d_mlp

    # Y: (d_mlp, n_tokens) — identity "projection"; peak stats match raw neuron peaks.
    Y_base = A_base.T.contiguous()
    Y_cand = A_cand.T.contiguous()

    Y_base_max, sample_ids_list = per_prompt_peaks(Y_base, sample_ids)
    Y_cand_max, _ = per_prompt_peaks(Y_cand, sample_ids)

    mean_base = Y_base_max.mean(axis=0)
    mean_cand = Y_cand_max.mean(axis=0)
    m = compute_mean_peak_metrics(mean_base, mean_cand, eps=REL_DELTA_EPS)
    ranker = RankerFactory.get_ranker(cfg.snmf.rank_by)
    scores = ranker.ranking_vector(m)

    n_keep = max(1, min(cfg.snmf.top_k_per_layer, K))
    top_indices = np.argpartition(-scores, n_keep - 1)[:n_keep]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    core = {
        "layer_idx": layer_idx,
        "K": K,
        "Y_base": Y_base,
        "sample_ids_list": sample_ids_list,
        "residuals": {"base": 0.0, "candidate": 0.0, "delta": 0.0},
        "mean_base": mean_base,
        "mean_candidate": mean_cand,
        "metrics": m,
        "scores": scores,
        "top_indices": top_indices,
    }
    # Reuse CSV/JSON latent table builder (ridge_lambda field unused for identity).
    auditor = LayerAuditor(cfg, SubspaceProjector(cfg.snmf.ridge_lambda))
    partial_payload, rows = auditor.build_layer_payload_numeric(core, layer_idx)
    partial_payload["feature_basis"] = "raw_mlp_identity"
    partial_payload["ridge_lambda"] = 0.0

    per_latent = partial_payload["per_latent"]
    spans_for_ctx = _sample_id_to_spans(np.asarray(sample_ids))
    contexts_per_feature = cfg.snmf.contexts_per_feature
    context_window = cfg.snmf.context_window
    aggregate_delta_weighted = cfg.lens.lens_delta_weighted
    rare_words_zipf_cutoff = cfg.rare.context_rare_zipf_cutoff
    rare_words_min_len = cfg.rare.context_rare_min_len
    rank_field = cfg.snmf.rank_by
    weight_vec = getattr(m, rank_field)

    top_records: List[Dict[str, Any]] = []
    top_vocab_per_neuron: Dict[int, List[Dict[str, Any]]] = {}
    if lens is not None and vocab_lens_top_k > 0:
        top_vocab_per_neuron = _project_raw_neurons_vocab_lens(
            lens,
            layer_idx,
            top_indices.tolist(),
            vocab_lens_top_k,
            tokenizer,
        )

    top_vocab_sum: Optional[Dict[str, Any]] = None
    if lens is not None and aggregate_top_k > 0 and len(top_indices) > 0:
        idx_list = [int(i) for i in top_indices.tolist()]
        weights = (
            [float(weight_vec[i]) for i in idx_list]
            if aggregate_delta_weighted
            else None
        )
        agg_residual = _feature_residual_raw_neurons(
            lens, layer_idx, idx_list, weights,
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

    layer_context_strings: List[str] = []
    for i in top_indices.tolist():
        ctxs = _top_contexts_for_latent(
            Y_base, token_ids, sample_ids, spans_for_ctx, tokenizer,
            latent_idx=int(i),
            n_contexts=contexts_per_feature,
            context_window=context_window,
        )
        rec = dict(per_latent[int(i)])
        rec["top_contexts"] = ctxs
        if int(i) in top_vocab_per_neuron:
            rec["top_vocab_base"] = top_vocab_per_neuron[int(i)]
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
            "n_features_pooled": int(len(top_indices)),
            "zipf_cutoff": float(rare_words_zipf_cutoff),
            "min_word_len": int(rare_words_min_len),
            "words": rare_context_words_layer,
        }
    with open(layer_dir_out / "audit.json", "w", encoding="utf-8") as f:
        json.dump(layer_payload, f, indent=2)

    pd.DataFrame(rows).to_csv(layer_dir_out / "audit_features.csv", index=False)

    logging.info(
        "Layer %d (raw neurons): K=%d | identity residual=0 | "
        "rel_delta: mean=%+.4f max=%+.4f min=%+.4f | "
        "abs_rel_delta: mean=%.4f max=%.4f min=%.4f",
        layer_idx,
        K,
        float(m.rel_delta.mean()),
        float(m.rel_delta.max()),
        float(m.rel_delta.min()),
        float(m.abs_rel_delta.mean()),
        float(m.abs_rel_delta.max()),
        float(m.abs_rel_delta.min()),
    )
    return layer_payload


def _filter_layers_from_cfg(
    layers: List[int],
    cfg: AuditConfig,
    logger: logging.Logger,
) -> List[int]:
    wanted = cfg.layers_wanted()
    if wanted is None:
        return layers
    out = [L for L in layers if L in wanted]
    missing = wanted - set(out)
    if missing:
        logger.warning(
            "Requested layers not present in activation bundle: %s",
            sorted(missing),
        )
    if not out:
        raise RuntimeError("No layers left after applying --layers filter.")
    return out


def _build_logit_lens_from_base(
    cfg: AuditConfig,
    layers: List[int],
    device: str,
    logger: logging.Logger,
) -> Tuple[Optional[LogitLens], Any]:
    """Load base model briefly to snapshot norm, lm_head, and per-layer W_down."""
    logger.info(
        "Loading base model for logit-lens snapshot: %s",
        cfg.base_model_path,
    )
    local = load_local_model(cfg.base_model_path, device=device)
    tokenizer = local.tokenizer
    try:
        lens = LogitLens(
            local.model,
            layers,
            tokenizer=tokenizer,
            center_unembed=cfg.lens.lens_center_unembed,
            mask_special_tokens=cfg.lens.lens_mask_special_tokens,
        )
    finally:
        del local
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return lens, tokenizer


def run_baseline_audit(
    cfg: AuditConfig,
    *,
    activations_cache: Optional[str] = None,
) -> None:
    """Run raw-neuron baseline audit from cache and/or live activation collection."""
    set_seed(cfg.runtime.seed)
    out_dir = Path(cfg.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)
    logger.info("=" * 60)
    logger.info(
        "RAW MLP NEURON BASELINE UNLEARNING AUDIT  "
        f"({datetime.now():%Y-%m-%d %H:%M:%S})",
    )
    logger.info("=" * 60)

    meta_extra = {
        "activations_cache": activations_cache or None,
        "audit_family": "baseline_raw_mlp_neurons",
    }
    logger.info(
        "Audit config:\n%s\nmeta:\n%s",
        json.dumps(audit_config_to_nested_dict(cfg), indent=2),
        json.dumps(meta_extra, indent=2),
    )

    device = resolve_device(cfg.runtime.device)
    logger.info("Resolved compute device: %s", device)

    vocab_lens_enabled = _is_vocab_logit_lens_enabled(cfg)
    context_rare_word_top_n = _resolve_context_rare_word_top_n(cfg, logger)

    cache_key = str(activations_cache or "").strip()
    lens: Optional[LogitLens] = None
    tokenizer: Any = None

    if cache_key:
        acts_base, acts_cand, token_ids, sample_ids, layers_from_cache = (
            load_activations_cache(cache_key)
        )
        layers = _filter_layers_from_cfg(layers_from_cache, cfg, logger)
        # Align tensor lists with filtered layer indices
        layer_to_pos = {L: i for i, L in enumerate(layers_from_cache)}
        indices = [layer_to_pos[L] for L in layers]
        acts_base = [acts_base[i] for i in indices]
        acts_cand = [acts_cand[i] for i in indices]

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        n_prompts_meta = len({int(s) for s in sample_ids})
        prompts = None  # type: ignore[assignment]

        if vocab_lens_enabled:
            lens, tok2 = _build_logit_lens_from_base(cfg, layers, device, logger)
            tokenizer = tok2
            lens.to(device)
    else:
        layers_spec = cfg.layers_wanted()
        if layers_spec is None:
            raise ValueError(
                "Live collection requires --layers (or config layers:) "
                "to know which layers to hook.",
            )
        layers = sorted(layers_spec)
        prompts = _load_prompts(
            cfg.data_path, cfg.runtime.max_prompts, cfg.runtime.seed,
        )
        logger.info("Loaded %d audit prompts from %s.", len(prompts), cfg.data_path)

        acts_base, acts_cand, token_ids, sample_ids, tokenizer, lens = (
            _collect_aligned_dual_model_activations(
                cfg, prompts, layers, device, vocab_lens_enabled, logger,
            )
        )
        n_prompts_meta = len(prompts)

    if tokenizer is None:
        raise RuntimeError("Tokenizer missing; cannot decode context windows.")

    layer_payloads: List[Dict[str, Any]] = []
    for li, layer_idx in enumerate(layers):
        A_base = acts_base[li]
        A_cand = acts_cand[li]
        payload = _audit_one_layer_raw(
            layer_idx,
            A_base,
            A_cand,
            token_ids,
            sample_ids,
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
        layer_payloads.append(payload)

    global_top = global_top_features(
        layer_payloads, cfg.snmf.rank_by, cfg.snmf.top_k_global,
    )
    per_layer_summary = _build_per_layer_metrics_rows(layer_payloads)

    global_aggregate_vocab = _build_cross_layer_logit_lens_aggregate_raw(
        lens,
        vocab_lens_enabled,
        cfg,
        global_top,
        tokenizer,
        logger,
    )

    per_layer_aggregate_vocab: List[Dict[str, Any]] = []
    for p in layer_payloads:
        agg = p.get("top_vocab_base_sum")
        if agg is not None:
            per_layer_aggregate_vocab.append({"layer": p["layer"], **agg})

    (
        _pl_vocab_dup,
        judge_per_layer_rare_words,
        judge_global_rare_words,
    ) = _assemble_judge_prompt_context(
        layer_payloads,
        global_top,
        context_rare_word_top_n,
        cfg,
    )
    del _pl_vocab_dup

    snmf_dir_label = (
        str(Path(cache_key).resolve()) if cache_key else "live_activation_collection"
    )

    judge_prompt = UnlearningAuditReporter.build_audit_report_prompt(
        n_prompts=int(n_prompts_meta),
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
        snmf_dir=Path(snmf_dir_label),
        layers=layers,
        n_prompts=int(n_prompts_meta),
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


def parse_baseline_args(argv: Optional[List[str]] = None) -> Tuple[AuditConfig, Optional[str]]:
    p = build_arg_parser()
    p.description = "Raw MLP neuron baseline unlearning audit (no SNMF)."
    p.add_argument(
        "--activations-cache",
        type=str,
        default=None,
        help="Path to a .pt dict with acts_base, acts_cand, token_ids, sample_ids[, layers].",
    )
    ns = p.parse_args(argv)
    cfg = load_audit_config_yaml(ns.config) if ns.config else AuditConfig()
    apply_argparse_namespace_overrides(cfg, ns)
    cache = getattr(ns, "activations_cache", None)
    try:
        validate_baseline_audit_config(cfg, activations_cache=cache)
    except ValueError as e:
        p.error(str(e))
    return cfg, cache


def main() -> None:
    cfg, cache = parse_baseline_args()
    run_baseline_audit(cfg, activations_cache=cache)


if __name__ == "__main__":
    main()
