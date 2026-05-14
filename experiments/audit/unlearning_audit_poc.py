"""
SNMF-based unlearning audit (initial check for "Intrinsic Auditing via SNMF").

Implements Sections 3.2 + 3.3 of the proposed methodology:

  3.2  Audit Transform
       For each layer, take the SNMF feature basis Z = F (d_mlp x K) trained on
       the BASE model's mlp_intermediate activations, and compute coefficients
       on the SAME prompts using:
         (a) base activations          A_base
         (b) unlearned activations     A_unlearned
       Coefficients are recovered by ridge least squares against the basis
       (an "audit projection"; we do NOT re-train SNMF on the unlearned model):
         Y = (Z^T Z + lambda I)^{-1} Z^T A^T          shape (K, n_tokens)
       Per-prompt peak  Y_max[p, i] = max over tokens of Y[i, span(p)]
       (matches the geometry used in feature_analysis_supervised_wmdp_bio.json).

       Per-feature erasure metrics:
         delta_i             = E[Y_base_max,i] - E[Y_unlearned_max,i]
         delta_forget,i      = E_forget[Y_base_max,i] - E_forget[Y_unlearned_max,i]
         delta_retain,i      = E_retain[Y_base_max,i] - E_retain[Y_unlearned_max,i]
         rel_delta_i         = delta_i / (E[Y_base_max,i] + eps)         # fractional drop
         rel_delta_forget,i  = delta_forget,i / (E_forget[Y_base_max,i] + eps)
         rel_delta_retain,i  = delta_retain,i / (E_retain[Y_base_max,i] + eps)
       ``rel_delta`` captures "surgical" unlearning of niche features that were
       already small on M_base: a feature that goes from 0.05 -> 0
       (rel_delta=1.0) is ranked above one that drops from 5.0 -> 4.0
       (rel_delta=0.2) even though the latter has a larger absolute delta.

  3.3  Activation traces vs weight traces (deep vs superficial)
       - "Activation trace" = drop in coefficient magnitudes (delta).
       - "Weight trace"     = inability of the BASE basis Z to reconstruct
         A_unlearned. Reported per layer as relative Frobenius residual:
           res_base      = || A_base      - Z Y_base      ||_F^2 / || A_base      ||_F^2
           res_unlearned = || A_unlearned - Z Y_unlearned ||_F^2 / || A_unlearned ||_F^2

  Logit-lens (output-side interpretation, M_base only)
       For each top-erased forget latent we also project the SNMF column
       through M_base's W_down + final_norm + lm_head:
         r_i      = W_down_L @ F_L[:, i]               # in residual space
         logits_i = lm_head( final_norm(r_i) )         # in vocab space
       The top ``--vocab-lens-top-k`` tokens of logits_i tell us what M_base
       would WRITE into the residual stream when this feature fires, which
       complements the INPUT-side ``top_tokens`` (top-activating contexts
       from the supervised JSON). We additionally compute a per-layer
       aggregate by summing the layer's top-erased forget columns (optionally
       weighted by their delta / rel_delta) and projecting that sum, and a
       single global aggregate across all layers' top forget features. The
       unembedding is mean-centered (Mu & Viswanath "all-but-the-top") and
       special / unused / reserved tokens are masked so the topk reflects the
       feature's content direction.

The script reads a WMDP-bio supervised JSON next to the SNMF factors so each
latent gets a role_label (bio_forget_lean / retain_lean / weak_mixed / ...).
Roles are recomputed on the fly at --role-assignment-threshold from
group_means / log_ratios, exactly the way create_forget_ablated_model.py does
it; this means you can ask the audit at a different threshold than the one
baked into the JSON.

Outputs (under --output-dir):
  layer_<i>/audit.json              # per-latent profile (role, deltas,
                                    # rel_deltas, top_vocab_base on top-K,
                                    # top_vocab_base_sum aggregate)
  layer_<i>/audit_features.csv      # same data, flat
  audit_summary.json                # per-layer aggregates + global rankings
                                    # + per_layer_aggregate_vocab
                                    # + global_aggregate_vocab
  delta_by_role.png                 # boxplot of delta on forget prompts per role
  delta_top_features.png            # bar chart of top-erased forget latents
                                    # (ranked by --rank-by)

Run:
  python experiments/audit/unlearning_audit.py \
      --base-model-path  /path/to/gemma-2-2b \
      --unlearned-model-path /path/to/.../final_model \
      --snmf-dir         outputs/wmdp/results_data_part1_gemma2_2b \
      --data-path        data/bio_data_part1.json \
      --layers           10-18 \
      --output-dir       outputs/wmdp/audit_iter1_rmu_bio_part1 \
      --rank-by          rel_delta_forget \
      --role-assignment-threshold 0.22 \
      --max-per-group    150
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dotenv import load_dotenv

from create_forget_ablated_model import (
    _recompute_role_labels_by_basis,
    _role_labels_by_basis_from_profile,
)
from data_utils.concept_dataset import SupervisedConceptDataset
from experiments.audit.logit_lens import LogitLens
from experiments.train.train import parse_int_list
from llm_utils.local_activation_generator import LocalActivationGenerator
from llm_utils.model_utils import load_local_model
from llm_utils.utils import resolve_device, set_seed, sorted_numeric_layer_dirs
from supervised_analysis import _sample_id_to_spans

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


BIO_FORGET_LABEL = "bio_forget"
BIO_RETAIN_LABEL = "bio_retain"
NEUTRAL_LABEL = "neutral"
SUPERVISED_JSON_DEFAULT = "feature_analysis_supervised_wmdp_bio.json"

# Floor for rel_delta denominators so features that are ~0 on M_base don't
# blow up to +/-inf; also makes rel_delta well-defined when the numerator is
# exactly zero.
REL_DELTA_EPS: float = 1e-9


# Ranking metric -> (numerator field, denominator-style flag) is implicit; we
# pre-compute every metric per latent and just sort by the chosen field.
RANK_BY_FIELDS: Tuple[str, ...] = (
    "rel_delta_forget",
    "abs_rel_delta_forget",
    "delta_forget",
    "abs_delta_forget",
)


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SNMF-based unlearning audit (initial-check pipeline).",
    )
    p.add_argument("--base-model-path", type=str, required=True,
                   help="Path to M_base (the unlearning baseline model).")
    p.add_argument("--unlearned-model-path", type=str, required=True,
                   help="Path to M_unlearned (the model whose unlearning we audit).")
    p.add_argument("--snmf-dir", type=str, required=True,
                   help="SNMF train output dir for M_base (contains layer_*/snmf_factors.pt + "
                        "feature_analysis_supervised_wmdp_bio.json per layer).")
    p.add_argument("--data-path", type=str, required=True,
                   help="WMDP-bio supervised JSON dataset (bio_forget / bio_retain / neutral).")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Where to write audit results.")
    p.add_argument("--layers", type=str, default=None,
                   help="Layers to audit (e.g. '10-18'). Default: every layer_* in --snmf-dir.")
    p.add_argument("--mode", type=str, default="mlp_intermediate",
                   choices=["mlp_intermediate"],
                   help="Activation hook mode. Must match what F was trained on. SNMF basis "
                        "vectors only live in the down_proj input space, so this is fixed.")
    p.add_argument("--max-per-group", type=int, default=200,
                   help="Cap per label group (bio_forget / bio_retain / neutral).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ridge-lambda", type=float, default=1e-4,
        help="Tikhonov on Z^T Z when solving for coefficients (numerical stability).",
    )
    p.add_argument(
        "--role-assignment-threshold", type=float, default=0.22,
        help="Min |log_forget_vs_retain| margin for bio_forget_lean vs retain_lean. Recomputed "
             "on the fly from each profile's group_means / group_counts / log_ratios "
             "(same logic as create_forget_ablated_model.py, threshold knob).",
    )
    p.add_argument(
        "--role-label-basis", type=str, default="pooled",
        choices=["pooled", "neutral", "bio_retain"],
        help="Which retain basis to use when grouping latents into roles for delta "
             "summaries / plots.",
    )
    p.add_argument(
        "--supervised-json-filename", type=str, default=SUPERVISED_JSON_DEFAULT,
        help="Per-layer supervised JSON filename (default: feature_analysis_supervised_wmdp_bio.json).",
    )
    p.add_argument(
        "--top-k-report", type=int, default=20,
        help="How many top-erased forget latents to surface in the global summary "
             "(and in each layer's per-layer aggregate logit-lens).",
    )
    p.add_argument(
        "--rank-by", type=str, default="rel_delta_forget",
        choices=list(RANK_BY_FIELDS),
        help="Metric used to rank bio_forget_lean latents by 'how much was erased'. "
             "'rel_delta_forget' (default) = fractional drop on forget prompts "
             "(delta_forget / (E_forget[Y_base] + eps)); 'abs_rel_delta_forget' = "
             "magnitude of fractional change; 'delta_forget' = raw signed decrease; "
             "'abs_delta_forget' = magnitude of raw decrease. rel_delta_forget "
             "surfaces 'surgical' unlearning of niche features whose base activation "
             "was already small.",
    )
    # Logit-lens knobs (mirrors general_unlearning_audit.py).
    p.add_argument(
        "--vocab-lens-top-k", type=int, default=15,
        help="How many top vocab tokens to logit-lens per surfaced latent "
             "via M_base's W_down + final_norm + lm_head. 0 disables.",
    )
    p.add_argument(
        "--skip-vocab-lens", action="store_true",
        help="Skip the logit-lens vocab projection step entirely "
             "(cheaper: avoids snapshotting lm_head from M_base).",
    )
    p.add_argument(
        "--no-lens-center-unembed", dest="lens_center_unembed",
        action="store_false",
        help="Disable mean-centering of the unembedding before topk. "
             "Centering is on by default; turn it off to inspect the raw "
             "logit-lens output.",
    )
    p.set_defaults(lens_center_unembed=True)
    p.add_argument(
        "--no-lens-mask-special-tokens", dest="lens_mask_special_tokens",
        action="store_false",
        help="Disable masking of special / unused / reserved tokens before topk. "
             "Masking is on by default; turn it off if you specifically want to "
             "see whether <bos> etc. show up.",
    )
    p.set_defaults(lens_mask_special_tokens=True)
    p.add_argument(
        "--vocab-lens-aggregate-top-k", type=int, default=20,
        help="How many top vocab tokens to logit-lens for the SUM of top-erased "
             "forget features (per-layer aggregate over the layer's top, plus a "
             "single global aggregate across layers). 0 disables.",
    )
    p.add_argument(
        "--lens-delta-weighted", dest="lens_delta_weighted",
        action="store_true",
        help="Weight each feature by its (rank_by) score when summing for the "
             "aggregate logit-lens. Default is a uniform sum; turn this on if you "
             "want the aggregate biased toward larger-erasure features.",
    )
    p.set_defaults(lens_delta_weighted=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------

def _load_and_balance_dataset(
    data_path: str, max_per_group: int, seed: int
) -> Tuple[List[str], List[str]]:
    """
    Read the WMDP-bio supervised dataset, then take up to ``max_per_group`` prompts
    per label, deterministic under ``seed``. Returns (prompts, labels) parallel lists.
    """
    ds = SupervisedConceptDataset(data_path)
    prompts_all, labels_all = ds.get_data()
    rng = np.random.default_rng(seed)

    by_label: Dict[str, List[int]] = {}
    for i, lab in enumerate(labels_all):
        by_label.setdefault(lab, []).append(i)

    keep_idx: List[int] = []
    for lab, idxs in by_label.items():
        if max_per_group > 0 and len(idxs) > max_per_group:
            chosen = rng.choice(np.asarray(idxs), size=max_per_group, replace=False)
            keep_idx.extend(int(x) for x in chosen.tolist())
        else:
            keep_idx.extend(idxs)
    keep_idx.sort()

    prompts = [prompts_all[i] for i in keep_idx]
    labels = [labels_all[i] for i in keep_idx]
    return prompts, labels


# ---------------------------------------------------------------------------
# Activation collection + projection
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
    """Load model, hook MLPs, return activations per layer + token/sample ids, then unload.

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
    return acts, token_ids, sample_ids, (tokenizer if return_tokenizer else None), lens


def _project_onto_basis(
    A: torch.Tensor,
    Z: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    """
    Audit projection: solve  Z Y ~= A.T  in the ridge sense.

    A : (n_tokens, d_mlp) activations from some model on the audit prompts.
    Z : (d_mlp, K)        SNMF feature basis from M_base (saved as F in snmf_factors.pt).

    Returns Y of shape (K, n_tokens) so that Z @ Y ~= A.T.
    Y is unconstrained (we do NOT enforce non-negativity here -- this is a
    diagnostic projection on a fixed basis, not a re-fit of SNMF).
    """
    if A.ndim != 2 or Z.ndim != 2:
        raise ValueError(f"A and Z must be 2D; got A={tuple(A.shape)} Z={tuple(Z.shape)}")
    d_mlp_a, K = Z.shape
    if A.shape[1] != d_mlp_a:
        raise ValueError(
            f"A has d_mlp={A.shape[1]} but Z expects d_mlp={d_mlp_a}; check --mode and --snmf-dir."
        )
    Z64 = Z.to(dtype=torch.float64)
    A64 = A.to(dtype=torch.float64)
    G = Z64.T @ Z64 + ridge_lambda * torch.eye(K, dtype=torch.float64)
    rhs = Z64.T @ A64.T
    Y = torch.linalg.solve(G, rhs)
    return Y.to(dtype=torch.float32)


def _per_prompt_peaks(
    Y: torch.Tensor, sample_ids: Sequence[int]
) -> Tuple[np.ndarray, List[int]]:
    """
    Reduce token-level coefficients to per-prompt peaks (max over tokens), matching
    the geometry of feature_analysis_supervised_wmdp_bio.json.

    Y           : (K, n_tokens)
    sample_ids  : len n_tokens, parallel to Y's columns

    Returns (Y_max, sample_ids_list) where:
      Y_max         : (n_prompts, K) np.float64
      sample_ids_list : sorted unique sample ids (length n_prompts)
    """
    sample_ids_arr = np.asarray(sample_ids)
    spans = _sample_id_to_spans(sample_ids_arr)
    sample_ids_list = list(spans.keys())
    Y_np = Y.detach().cpu().numpy().astype(np.float64, copy=False)
    K, n_tokens = Y_np.shape
    n_prompts = len(sample_ids_list)
    Y_max = np.empty((n_prompts, K), dtype=np.float64)
    for i, sid in enumerate(sample_ids_list):
        s, e = spans[sid]
        seg = Y_np[:, s:e]
        Y_max[i, :] = seg.max(axis=1)
    return Y_max, sample_ids_list


def _frob_relative_residual(A: torch.Tensor, Z: torch.Tensor, Y: torch.Tensor) -> float:
    """|| A - (Z Y)^T ||_F^2 / || A ||_F^2 . Returns a scalar float."""
    A64 = A.to(dtype=torch.float64)
    Z64 = Z.to(dtype=torch.float64)
    Y64 = Y.to(dtype=torch.float64)
    recon = (Z64 @ Y64).T
    diff = A64 - recon
    num = float((diff * diff).sum().item())
    den = float((A64 * A64).sum().item()) + 1e-12
    return num / den


# ---------------------------------------------------------------------------
# Role labels per latent (re-uses create_forget_ablated_model logic)
# ---------------------------------------------------------------------------

def _load_role_labels(
    layer_dir: Path,
    *,
    supervised_json_filename: str,
    role_assignment_threshold: float,
    basis: str,
) -> Optional[Dict[int, str]]:
    """
    Return {latent_idx: role_label_for_basis}. None if the supervised JSON is missing.
    Roles are recomputed on the fly at the requested threshold (matches the ablation script).
    """
    p = layer_dir / supervised_json_filename
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[int, str] = {}
    for k, prof in raw.items():
        if not isinstance(prof, dict):
            continue
        try:
            i = int(k)
        except ValueError:
            continue
        labels_by_basis = _recompute_role_labels_by_basis(prof, role_assignment_threshold)
        if basis in labels_by_basis:
            out[i] = labels_by_basis[basis]
        else:
            # Fall back to whatever is already stored (e.g. legacy arithmetic JSON).
            out[i] = _role_labels_by_basis_from_profile(prof).get(basis, "unknown")
    return out


def _load_top_tokens_per_latent(
    layer_dir: Path, supervised_json_filename: str
) -> Dict[int, List[str]]:
    """Pull top_positive_activation_contexts strings (small k) so the audit report can quote them."""
    p = layer_dir / supervised_json_filename
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[int, List[str]] = {}
    for k, prof in raw.items():
        if not isinstance(prof, dict):
            continue
        try:
            i = int(k)
        except ValueError:
            continue
        ctxs = prof.get("top_positive_activation_contexts") or []
        out[i] = [str(c) for c in ctxs[:5]]
    return out


# ---------------------------------------------------------------------------
# Per-layer audit
# ---------------------------------------------------------------------------

def _rank_score_from_rec(rec: Dict[str, Any], rank_by: str) -> float:
    """Read the ``rank_by`` field from a per-latent rec; None becomes -inf."""
    v = rec.get(rank_by)
    if v is None:
        return float("-inf")
    return float(v)


def _audit_one_layer(
    layer_idx: int,
    layer_dir: Path,
    A_base: torch.Tensor,
    A_unlearned: torch.Tensor,
    sample_ids: List[int],
    labels: List[str],
    tokenizer: Any,
    *,
    ridge_lambda: float,
    role_assignment_threshold: float,
    role_basis: str,
    supervised_json_filename: str,
    out_dir: Path,
    rank_by: str,
    top_k_report: int,
    lens: Optional[LogitLens] = None,
    vocab_lens_top_k: int = 0,
    aggregate_top_k: int = 0,
    aggregate_delta_weighted: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run the audit for a single layer. Returns a per-layer summary dict (or None
    if this layer has no SNMF factors saved).
    """
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
    K = int(Z.shape[1])
    logging.info(f"Layer {layer_idx}: Z shape={tuple(Z.shape)}; A_base shape={tuple(A_base.shape)}")

    Y_base = _project_onto_basis(A_base, Z, ridge_lambda=ridge_lambda)
    Y_unl = _project_onto_basis(A_unlearned, Z, ridge_lambda=ridge_lambda)

    res_base = _frob_relative_residual(A_base, Z, Y_base)
    res_unl = _frob_relative_residual(A_unlearned, Z, Y_unl)

    # Per-prompt peak.
    Y_base_max, sample_ids_list = _per_prompt_peaks(Y_base, sample_ids)
    Y_unl_max, _ = _per_prompt_peaks(Y_unl, sample_ids)

    sample_labels = np.asarray([labels[sid] for sid in sample_ids_list])
    is_forget = sample_labels == BIO_FORGET_LABEL
    is_bio_retain = sample_labels == BIO_RETAIN_LABEL
    is_neutral = sample_labels == NEUTRAL_LABEL
    is_pooled_retain = is_bio_retain | is_neutral

    def _means(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not mask.any():
            empty = np.full(K, np.nan, dtype=np.float64)
            return empty, empty
        return Y_base_max[mask].mean(axis=0), Y_unl_max[mask].mean(axis=0)

    mean_base_all, mean_unl_all = Y_base_max.mean(axis=0), Y_unl_max.mean(axis=0)
    mean_base_forget, mean_unl_forget = _means(is_forget)
    mean_base_retain, mean_unl_retain = _means(is_pooled_retain)
    mean_base_bio_retain, mean_unl_bio_retain = _means(is_bio_retain)
    mean_base_neutral, mean_unl_neutral = _means(is_neutral)

    delta_all = mean_base_all - mean_unl_all
    delta_forget = mean_base_forget - mean_unl_forget
    delta_retain = mean_base_retain - mean_unl_retain

    # Fractional decrease vs M_base, floored to avoid blow-up on near-zero
    # base activations. NaN propagation from _means is preserved.
    rel_delta_all = delta_all / (mean_base_all + REL_DELTA_EPS)
    rel_delta_forget = delta_forget / (mean_base_forget + REL_DELTA_EPS)
    rel_delta_retain = delta_retain / (mean_base_retain + REL_DELTA_EPS)

    role_labels = _load_role_labels(
        layer_dir,
        supervised_json_filename=supervised_json_filename,
        role_assignment_threshold=role_assignment_threshold,
        basis=role_basis,
    ) or {}
    top_tokens_per_latent = _load_top_tokens_per_latent(layer_dir, supervised_json_filename)

    def _opt_float(x: float) -> Optional[float]:
        return None if np.isnan(x) else float(x)

    per_latent: Dict[int, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for i in range(K):
        role = role_labels.get(i, "unknown")
        d_f = delta_forget[i]
        d_r = delta_retain[i]
        rd_a = rel_delta_all[i]
        rd_f = rel_delta_forget[i]
        rd_r = rel_delta_retain[i]
        rec = {
            "latent_idx": i,
            "role_label": role,
            "mean_Y_base": float(mean_base_all[i]),
            "mean_Y_unlearned": float(mean_unl_all[i]),
            "delta": float(delta_all[i]),
            "abs_delta": float(abs(delta_all[i])),
            "rel_delta": float(rel_delta_all[i]),
            "abs_rel_delta": float(abs(rel_delta_all[i])),
            "delta_forget": _opt_float(d_f),
            "abs_delta_forget": _opt_float(abs(d_f)),
            "rel_delta_forget": _opt_float(rd_f),
            "abs_rel_delta_forget": _opt_float(abs(rd_f)),
            "delta_retain": _opt_float(d_r),
            "abs_delta_retain": _opt_float(abs(d_r)),
            "rel_delta_retain": _opt_float(rd_r),
            "abs_rel_delta_retain": _opt_float(abs(rd_r)),
            "mean_Y_base_forget": _opt_float(mean_base_forget[i]),
            "mean_Y_unlearned_forget": _opt_float(mean_unl_forget[i]),
            "mean_Y_base_bio_retain": _opt_float(mean_base_bio_retain[i]),
            "mean_Y_unlearned_bio_retain": _opt_float(mean_unl_bio_retain[i]),
            "mean_Y_base_neutral": _opt_float(mean_base_neutral[i]),
            "mean_Y_unlearned_neutral": _opt_float(mean_unl_neutral[i]),
            # Backwards-compat keys (older audit JSONs used these names).
            "delta_forget_prompts": _opt_float(d_f),
            "delta_retain_prompts": _opt_float(d_r),
        }
        per_latent[i] = rec
        row = dict(rec)
        row["layer"] = layer_idx
        row["top_tokens"] = " | ".join(top_tokens_per_latent.get(i, [])[:3])
        rows.append(row)

    # Per-layer summary by role.
    role_summary: Dict[str, Dict[str, Any]] = {}
    role_to_idx: Dict[str, List[int]] = {}
    for i, rec in per_latent.items():
        role_to_idx.setdefault(rec["role_label"], []).append(i)
    for role, idxs in role_to_idx.items():
        d_all = np.array([per_latent[i]["delta"] for i in idxs], dtype=np.float64)
        rd_all = np.array([per_latent[i]["rel_delta"] for i in idxs], dtype=np.float64)
        d_forget = np.array(
            [per_latent[i]["delta_forget"] for i in idxs
             if per_latent[i]["delta_forget"] is not None],
            dtype=np.float64,
        )
        rd_forget = np.array(
            [per_latent[i]["rel_delta_forget"] for i in idxs
             if per_latent[i]["rel_delta_forget"] is not None],
            dtype=np.float64,
        )
        d_retain = np.array(
            [per_latent[i]["delta_retain"] for i in idxs
             if per_latent[i]["delta_retain"] is not None],
            dtype=np.float64,
        )
        rd_retain = np.array(
            [per_latent[i]["rel_delta_retain"] for i in idxs
             if per_latent[i]["rel_delta_retain"] is not None],
            dtype=np.float64,
        )

        def _stats(a: np.ndarray) -> Dict[str, float]:
            if a.size == 0:
                return {"n": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
            return {
                "n": int(a.size),
                "mean": float(np.mean(a)),
                "median": float(np.median(a)),
                "std": float(np.std(a)),
            }

        role_summary[role] = {
            "delta_all": _stats(d_all),
            "rel_delta_all": _stats(rd_all),
            "delta_forget": _stats(d_forget),
            "rel_delta_forget": _stats(rd_forget),
            "delta_retain": _stats(d_retain),
            "rel_delta_retain": _stats(rd_retain),
            # Backwards-compat:
            "delta_forget_prompts": _stats(d_forget),
            "delta_retain_prompts": _stats(d_retain),
        }

    # Ranking: top-erased bio_forget_lean latents by --rank-by.
    forget_lean = [i for i, rec in per_latent.items() if rec["role_label"] == "bio_forget_lean"]
    forget_ranked = sorted(
        forget_lean,
        key=lambda i: _rank_score_from_rec(per_latent[i], rank_by),
        reverse=True,
    )

    # Logit-lens: top vocab tokens for each top-erased forget latent.
    top_vocab_per_latent: Dict[int, List[Dict[str, Any]]] = {}
    if lens is not None and vocab_lens_top_k > 0 and forget_ranked:
        top_ids_for_lens = forget_ranked[:top_k_report]
        top_vocab_per_latent = lens.project_latents(
            F=Z,
            layer=layer_idx,
            latent_indices=top_ids_for_lens,
            top_k=vocab_lens_top_k,
            tokenizer=tokenizer,
        )

    # Per-layer aggregate logit-lens over the top forget latents.
    top_vocab_sum: Optional[Dict[str, Any]] = None
    if (lens is not None and aggregate_top_k > 0 and forget_ranked):
        agg_indices = forget_ranked[:top_k_report]
        weights_for_agg = None
        if aggregate_delta_weighted:
            weights_for_agg = [
                _rank_score_from_rec(per_latent[int(i)], rank_by) for i in agg_indices
            ]
            # If every weight is -inf (e.g. no forget prompts), fall back to uniform.
            if not any(np.isfinite(w) and w != 0.0 for w in weights_for_agg):
                weights_for_agg = None
            else:
                # Sanitize -inf to 0 so the sum is well-defined.
                weights_for_agg = [
                    (0.0 if (not np.isfinite(w)) else float(w)) for w in weights_for_agg
                ]
        agg_residual = lens.feature_residual(
            F=Z, layer=layer_idx, latent_indices=agg_indices, weights=weights_for_agg,
        )
        agg_tokens = lens.topk_from_residual(
            agg_residual, top_k=aggregate_top_k, tokenizer=tokenizer,
        )
        top_vocab_sum = {
            "n_features_summed": int(len(agg_indices)),
            "rank_by": rank_by,
            "delta_weighted": bool(aggregate_delta_weighted and weights_for_agg is not None),
            "residual_norm": float(agg_residual.norm().item()),
            "tokens": agg_tokens,
        }

    layer_dir_out = out_dir / f"layer_{layer_idx}"
    layer_dir_out.mkdir(parents=True, exist_ok=True)

    # Build top-erased records (now enriched with top_vocab_base).
    top_erased_records: List[Dict[str, Any]] = []
    for i in forget_ranked[:50]:
        rec = dict(per_latent[i])
        rec["top_tokens"] = top_tokens_per_latent.get(i, [])[:5]
        if i in top_vocab_per_latent:
            rec["top_vocab_base"] = top_vocab_per_latent[i]
        top_erased_records.append(rec)

    layer_payload: Dict[str, Any] = {
        "layer": layer_idx,
        "K": K,
        "n_prompts": int(len(sample_ids_list)),
        "n_forget_prompts": int(is_forget.sum()),
        "n_bio_retain_prompts": int(is_bio_retain.sum()),
        "n_neutral_prompts": int(is_neutral.sum()),
        "ridge_lambda": ridge_lambda,
        "role_basis": role_basis,
        "role_assignment_threshold": role_assignment_threshold,
        "rank_by": rank_by,
        "reconstruction_residual_relative": {
            "base": res_base,
            "unlearned": res_unl,
            "delta": res_unl - res_base,
        },
        "rel_delta_eps": REL_DELTA_EPS,
        "role_summary": role_summary,
        "top_erased_forget_latents": top_erased_records,
        "per_latent": per_latent,
    }
    if top_vocab_sum is not None:
        layer_payload["top_vocab_base_sum"] = top_vocab_sum
    with open(layer_dir_out / "audit.json", "w", encoding="utf-8") as f:
        json.dump(layer_payload, f, indent=2)

    df_layer = pd.DataFrame(rows)
    df_layer.to_csv(layer_dir_out / "audit_features.csv", index=False)

    fg_stats = role_summary.get("bio_forget_lean", {})
    rt_stats = role_summary.get("retain_lean", {})
    logging.info(
        "Layer %d: residual_base=%.4f residual_unlearned=%.4f (delta=%.4f) | "
        "forget-lean: n=%d mean_delta_forget=%.4f mean_rel_delta_forget=%.4f | "
        "retain-lean: n=%d mean_delta_retain=%.4f mean_rel_delta_retain=%.4f",
        layer_idx,
        res_base,
        res_unl,
        res_unl - res_base,
        fg_stats.get("delta_forget", {}).get("n", 0),
        fg_stats.get("delta_forget", {}).get("mean", float("nan")),
        fg_stats.get("rel_delta_forget", {}).get("mean", float("nan")),
        rt_stats.get("delta_retain", {}).get("n", 0),
        rt_stats.get("delta_retain", {}).get("mean", float("nan")),
        rt_stats.get("rel_delta_retain", {}).get("mean", float("nan")),
    )
    return layer_payload


# ---------------------------------------------------------------------------
# Aggregate plots / reporting
# ---------------------------------------------------------------------------

ROLE_PLOT_ORDER = ["bio_forget_lean", "retain_lean", "weak_mixed", "low_signal",
                   "insufficient_groups", "unknown"]
ROLE_PLOT_PALETTE = {
    "bio_forget_lean": "#d62728",
    "retain_lean": "#1f77b4",
    "weak_mixed": "#7f7f7f",
    "low_signal": "#bcbd22",
    "insufficient_groups": "#aaaaaa",
    "unknown": "#000000",
}


def _plot_delta_by_role(
    out_dir: Path,
    layer_payloads: List[Dict[str, Any]],
    *,
    metric: str = "delta_forget",
    filename: str = "delta_by_role.png",
    ylabel: Optional[str] = None,
) -> None:
    """Boxplot of per-latent ``metric`` grouped by role, per layer."""
    rows: List[Dict[str, Any]] = []
    for payload in layer_payloads:
        for i, rec in payload["per_latent"].items():
            v = rec.get(metric)
            if v is None:
                continue
            rows.append({"layer": payload["layer"], "role": rec["role_label"],
                         "value": float(v)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["role"] = pd.Categorical(df["role"], categories=ROLE_PLOT_ORDER, ordered=True)
    fig, ax = plt.subplots(figsize=(max(10, 0.6 * df["layer"].nunique() + 6), 6))
    sns.boxplot(
        data=df, x="layer", y="value", hue="role", ax=ax,
        showfliers=False, palette=ROLE_PLOT_PALETTE,
    )
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.5)
    ax.set_title(f"Audit: {metric} per role per layer (higher => more erased).")
    ax.set_ylabel(ylabel or metric)
    plt.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {out_path}")


def _plot_top_features(
    out_dir: Path,
    layer_payloads: List[Dict[str, Any]],
    *,
    top_k: int,
    rank_by: str,
    filename: str = "delta_top_features.png",
) -> None:
    """Bar chart of the top-K bio_forget_lean latents ranked by ``rank_by``."""
    items: List[Dict[str, Any]] = []
    for payload in layer_payloads:
        layer = payload["layer"]
        for rec in payload["top_erased_forget_latents"]:
            v = rec.get(rank_by)
            if v is None:
                continue
            items.append({
                "label": f"L{layer}.{rec['latent_idx']}",
                "value": float(v),
                "layer": layer,
            })
    if not items:
        return
    items.sort(key=lambda r: r["value"], reverse=True)
    items = items[:top_k]
    df = pd.DataFrame(items)
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(df) + 4), 6))
    sns.barplot(data=df, x="label", y="value", hue="layer", dodge=False, ax=ax)
    ax.set_title(f"Top-{top_k} most-erased bio_forget_lean latents (ranked by {rank_by})")
    ax.set_ylabel(rank_by)
    ax.set_xlabel("layer.latent")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {out_path}")


def _global_summary(
    layer_payloads: List[Dict[str, Any]], *, top_k: int, rank_by: str,
) -> Dict[str, Any]:
    """Aggregate role summaries + global ranking of erased forget latents."""
    forget_items: List[Dict[str, Any]] = []
    for payload in layer_payloads:
        for rec in payload["top_erased_forget_latents"]:
            v = rec.get(rank_by)
            if v is None:
                continue
            forget_items.append({
                "layer": payload["layer"],
                "latent_idx": rec["latent_idx"],
                "delta": rec.get("delta"),
                "rel_delta": rec.get("rel_delta"),
                "delta_forget": rec.get("delta_forget"),
                "rel_delta_forget": rec.get("rel_delta_forget"),
                "delta_retain": rec.get("delta_retain"),
                "rel_delta_retain": rec.get("rel_delta_retain"),
                "top_tokens": rec.get("top_tokens", [])[:5],
                "top_vocab_base": rec.get("top_vocab_base", []),
            })
    forget_items.sort(key=lambda r: float(r[rank_by]), reverse=True)

    # Per-layer compact view.
    per_layer = []
    for payload in layer_payloads:
        rs = payload["role_summary"]
        fg = rs.get("bio_forget_lean", {})
        rt = rs.get("retain_lean", {})
        per_layer.append({
            "layer": payload["layer"],
            "n_forget_lean": fg.get("delta_all", {}).get("n", 0),
            "n_retain_lean": rt.get("delta_all", {}).get("n", 0),
            "mean_delta_forget_on_forget_lean": fg.get("delta_forget", {}).get("mean", float("nan")),
            "mean_rel_delta_forget_on_forget_lean": fg.get("rel_delta_forget", {}).get("mean", float("nan")),
            "mean_delta_retain_on_retain_lean": rt.get("delta_retain", {}).get("mean", float("nan")),
            "mean_rel_delta_retain_on_retain_lean": rt.get("rel_delta_retain", {}).get("mean", float("nan")),
            "residual_base": payload["reconstruction_residual_relative"]["base"],
            "residual_unlearned": payload["reconstruction_residual_relative"]["unlearned"],
            "residual_delta": payload["reconstruction_residual_relative"]["delta"],
        })
    return {
        "per_layer": per_layer,
        "top_erased_forget_latents_global": forget_items[:top_k],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)
    logger.info("=" * 60)
    logger.info(f"SNMF UNLEARNING AUDIT  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    logger.info("=" * 60)
    logger.info(f"Args: {json.dumps(vars(args), indent=2)}")

    snmf_dir = Path(args.snmf_dir).resolve()
    layer_pairs = sorted_numeric_layer_dirs(snmf_dir)
    if args.layers:
        wanted = set(parse_int_list(args.layers))
        layer_pairs = [(i, p) for i, p in layer_pairs if i in wanted]
        missing = wanted - {i for i, _ in layer_pairs}
        if missing:
            logger.warning(f"Requested layers not found in {snmf_dir}: {sorted(missing)}")
    if not layer_pairs:
        raise RuntimeError(f"No layer_* dirs to audit under {snmf_dir}")
    layers = [i for i, _ in layer_pairs]
    logger.info(f"Auditing layers: {layers}")

    prompts, labels = _load_and_balance_dataset(args.data_path, args.max_per_group, args.seed)
    counts = pd.Series(labels).value_counts().to_dict()
    logger.info(f"Audit prompts: total={len(prompts)} per-label={counts}")
    if BIO_FORGET_LABEL not in counts:
        raise RuntimeError(
            f"No prompts with label {BIO_FORGET_LABEL!r} in {args.data_path}; "
            "delta_forget cannot be computed."
        )

    device = resolve_device(args.device)
    logger.info(f"Resolved compute device: {device}")

    want_lens = (not args.skip_vocab_lens) and (args.vocab_lens_top_k > 0)

    # --- Activations from BOTH models on the SAME prompt order ---
    logger.info("=== Collecting BASE activations ===")
    acts_base, token_ids_base, sample_ids_base, tokenizer, lens = _collect_activations(
        args.base_model_path, prompts, layers,
        mode=args.mode, batch_size=args.batch_size, device=device,
        return_tokenizer=True,
        return_logit_lens=want_lens,
        lens_center_unembed=args.lens_center_unembed,
        lens_mask_special_tokens=args.lens_mask_special_tokens,
    )
    if lens is not None:
        logger.info(f"Logit-lens snapshot ready ({len(lens.down_proj)} layers); "
                    f"moving to device={device} for projection.")
        lens.to(device)

    logger.info("=== Collecting UNLEARNED activations ===")
    acts_unl, token_ids_unl, sample_ids_unl, _, _ = _collect_activations(
        args.unlearned_model_path, prompts, layers,
        mode=args.mode, batch_size=args.batch_size, device=device,
        return_tokenizer=False,
        return_logit_lens=False,
    )
    if sample_ids_base != sample_ids_unl or token_ids_base != token_ids_unl:
        raise RuntimeError(
            "Token streams from base vs unlearned model do not match "
            "(different tokenizers / padding?). The audit requires identical token alignment."
        )

    # Per-layer audit.
    layer_payloads: List[Dict[str, Any]] = []
    for li, (layer_idx, layer_dir) in enumerate(layer_pairs):
        A_base = acts_base[li]
        A_unl = acts_unl[li]
        if A_base.shape != A_unl.shape:
            raise RuntimeError(
                f"layer {layer_idx}: A_base shape {tuple(A_base.shape)} != "
                f"A_unlearned shape {tuple(A_unl.shape)}."
            )
        payload = _audit_one_layer(
            layer_idx, layer_dir, A_base, A_unl,
            sample_ids_base, labels, tokenizer,
            ridge_lambda=args.ridge_lambda,
            role_assignment_threshold=args.role_assignment_threshold,
            role_basis=args.role_label_basis,
            supervised_json_filename=args.supervised_json_filename,
            out_dir=out_dir,
            rank_by=args.rank_by,
            top_k_report=args.top_k_report,
            lens=lens,
            vocab_lens_top_k=(args.vocab_lens_top_k if want_lens else 0),
            aggregate_top_k=(args.vocab_lens_aggregate_top_k if want_lens else 0),
            aggregate_delta_weighted=args.lens_delta_weighted,
        )
        if payload is not None:
            layer_payloads.append(payload)

    if not layer_payloads:
        raise RuntimeError("No layers were successfully audited (no snmf_factors.pt found?).")

    summary = _global_summary(layer_payloads, top_k=args.top_k_report, rank_by=args.rank_by)

    # Per-layer aggregate vocab (pulled out of layer payloads for easier access
    # in audit_summary.json).
    per_layer_aggregate_vocab: List[Dict[str, Any]] = []
    for p in layer_payloads:
        agg = p.get("top_vocab_base_sum")
        if agg is not None:
            per_layer_aggregate_vocab.append({"layer": p["layer"], **agg})

    # Global aggregate logit-lens: project each (layer, latent) in the global
    # top through its layer's W_down, sum residuals, then logit-lens once.
    global_aggregate_vocab: Optional[Dict[str, Any]] = None
    if (lens is not None and want_lens
            and args.vocab_lens_aggregate_top_k > 0
            and summary["top_erased_forget_latents_global"]):
        layer_dir_by_idx: Dict[int, Path] = {int(li): pth for li, pth in layer_pairs}
        # Group global top entries by layer so we only load each layer's F once.
        by_layer: Dict[int, List[Tuple[int, float]]] = {}
        for rec in summary["top_erased_forget_latents_global"]:
            L = int(rec["layer"])
            i = int(rec["latent_idx"])
            score_val = rec.get(args.rank_by)
            s = float(score_val) if score_val is not None else 0.0
            by_layer.setdefault(L, []).append((i, s))

        r_global: Optional[torch.Tensor] = None
        for L, entries in by_layer.items():
            ldir = layer_dir_by_idx.get(L)
            if ldir is None:
                logger.warning(f"Global aggregate: no layer dir for layer {L}; skipping.")
                continue
            ckpt_L = torch.load(ldir / "snmf_factors.pt", map_location="cpu",
                                weights_only=False)
            F_L = ckpt_L["F"].float().cpu()
            indices = [i for i, _ in entries]
            weights = ([s for _, s in entries]
                       if args.lens_delta_weighted else None)
            r_L = lens.feature_residual(F_L, L, indices, weights)
            r_global = r_L if r_global is None else r_global + r_L
        if r_global is not None:
            agg_tokens = lens.topk_from_residual(
                r_global, top_k=args.vocab_lens_aggregate_top_k, tokenizer=tokenizer,
            )
            global_aggregate_vocab = {
                "n_features_summed": int(len(summary["top_erased_forget_latents_global"])),
                "n_layers_spanned": int(len(by_layer)),
                "rank_by": args.rank_by,
                "delta_weighted": bool(args.lens_delta_weighted),
                "residual_norm": float(r_global.norm().item()),
                "tokens": agg_tokens,
            }
            logger.info(
                "Global aggregate logit-lens: summed %d features across %d layers; "
                "residual norm=%.3f, top token=%r (%.2f).",
                global_aggregate_vocab["n_features_summed"],
                global_aggregate_vocab["n_layers_spanned"],
                global_aggregate_vocab["residual_norm"],
                agg_tokens[0]["token"] if agg_tokens else None,
                agg_tokens[0]["logit"] if agg_tokens else float("nan"),
            )

    summary["per_layer_aggregate_vocab"] = per_layer_aggregate_vocab
    summary["global_aggregate_vocab"] = global_aggregate_vocab
    summary["meta"] = {
        "base_model_path": args.base_model_path,
        "unlearned_model_path": args.unlearned_model_path,
        "snmf_dir": str(snmf_dir),
        "data_path": args.data_path,
        "layers": layers,
        "max_per_group": args.max_per_group,
        "ridge_lambda": args.ridge_lambda,
        "rank_by": args.rank_by,
        "role_assignment_threshold": args.role_assignment_threshold,
        "role_label_basis": args.role_label_basis,
        "n_prompts": int(len(prompts)),
        "labels_counts": counts,
        "mode": args.mode,
        "vocab_lens_top_k": args.vocab_lens_top_k,
        "vocab_lens_aggregate_top_k": args.vocab_lens_aggregate_top_k,
        "skip_vocab_lens": bool(args.skip_vocab_lens),
        "lens_center_unembed": bool(args.lens_center_unembed),
        "lens_mask_special_tokens": bool(args.lens_mask_special_tokens),
        "lens_delta_weighted": bool(args.lens_delta_weighted),
    }
    with open(out_dir / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(summary["per_layer"]).to_csv(out_dir / "audit_summary_per_layer.csv", index=False)

    # Plots: keep the existing delta_* plots and add rel_delta_* variants so
    # both signals are visible side by side.
    _plot_delta_by_role(
        out_dir, layer_payloads,
        metric="delta_forget",
        filename="delta_by_role.png",
        ylabel="delta_forget = E[Y_base] - E[Y_unlearned] on forget prompts",
    )
    _plot_delta_by_role(
        out_dir, layer_payloads,
        metric="rel_delta_forget",
        filename="rel_delta_by_role.png",
        ylabel="rel_delta_forget = delta_forget / (E[Y_base] + eps)",
    )
    _plot_top_features(
        out_dir, layer_payloads,
        top_k=args.top_k_report,
        rank_by="delta_forget",
        filename="delta_top_features.png",
    )
    _plot_top_features(
        out_dir, layer_payloads,
        top_k=args.top_k_report,
        rank_by="rel_delta_forget",
        filename="rel_delta_top_features.png",
    )

    logger.info("=== Audit headline (per-layer) ===")
    for row in summary["per_layer"]:
        logger.info(
            "L%02d | n_forget_lean=%d  mean_delta_forget=%+.4f  mean_rel_delta_forget=%+.4f | "
            "n_retain_lean=%d  mean_delta_retain=%+.4f  mean_rel_delta_retain=%+.4f | "
            "residual base=%.4f -> unlearned=%.4f (delta=%+.4f)",
            row["layer"],
            row["n_forget_lean"],
            row["mean_delta_forget_on_forget_lean"],
            row["mean_rel_delta_forget_on_forget_lean"],
            row["n_retain_lean"],
            row["mean_delta_retain_on_retain_lean"],
            row["mean_rel_delta_retain_on_retain_lean"],
            row["residual_base"],
            row["residual_unlearned"],
            row["residual_delta"],
        )
    logger.info(f"Top-{args.top_k_report} erased forget latents (global, ranked by {args.rank_by}):")
    for rec in summary["top_erased_forget_latents_global"]:
        logger.info(
            "  L%d.lat%d  delta_forget=%+.4f  rel_delta_forget=%+.4f  delta_retain=%s  rel_delta_retain=%s",
            rec["layer"], rec["latent_idx"],
            rec["delta_forget"] if rec["delta_forget"] is not None else float("nan"),
            rec["rel_delta_forget"] if rec["rel_delta_forget"] is not None else float("nan"),
            ("%+.4f" % rec["delta_retain"]) if rec["delta_retain"] is not None else "n/a",
            ("%+.4f" % rec["rel_delta_retain"]) if rec["rel_delta_retain"] is not None else "n/a",
        )
    if global_aggregate_vocab and global_aggregate_vocab.get("tokens"):
        tok_str = "  ".join(
            f"{json.dumps(t['token'])} ({t['logit']:+.2f})"
            for t in global_aggregate_vocab["tokens"][:10]
        )
        logger.info("Global aggregate logit-lens top tokens: %s", tok_str)
    logger.info("Done. Outputs under: %s", out_dir)


if __name__ == "__main__":
    main()
