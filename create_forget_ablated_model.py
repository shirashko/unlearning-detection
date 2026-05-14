"""
Build a new HF checkpoint whose MLP removes SNMF forget directions (dual-sided where applicable).

Expected pipeline (matches the repo shell scripts):

  1. ``scripts/wmdp/train_snmf.sh`` → writes ``outputs/snmf_train_results/layer_*/snmf_factors.pt``
     with ``mode=mlp_intermediate`` (required: ``F`` lives in the same space as
     ``mlp.down_proj`` input).
  2. Analysis → per-layer supervised JSON (e.g. ``feature_analysis_supervised.json`` or
     ``feature_analysis_supervised_wmdp_bio.json``). WMDP-bio files include ``role_labels_by_basis``
     (pooled / neutral / bio_retain); use ``--role-label-bases`` + ``--role-basis-combine`` to decide
     which bases must agree before a latent counts as a forget direction. If those flags are omitted,
     the legacy top-level ``role_label`` field is used (matches older arithmetic runs).
  3. This script → reads ``F`` + role selection, applies projections on MLP weights, saves a new model.

Use the **same** ``--model-path`` as training/analysis and the **same** ``--results-dir``
as ``--output-dir`` / ``RESULTS_DIR`` (default: ``outputs/snmf_train_results``).

Mathematical setup (matches the cited write-up):
  - Each SNMF column z_i ∈ ℝ^{d_mlp} is a direction in the post-activation (neuron) space.
  - **Output side (down_proj):** y = W_V x with x ∈ ℝ^{d_mlp}. Remove forget span from x before W_V:
        W_V^{new} = W_V @ P_perp.
  - **Input / gate side (up_proj, gate_proj in Gemma-2):** these map ℝ^{d_model} → ℝ^{d_mlp}
    with weight W of shape (d_mlp, d_model). Output lives in the same space as z, so remove the
    span from the *output* of these layers:
        W^{new} = P_perp @ W.

For multiple forget features {z_1,…,z_k}, use the projector onto their span:
    P_span = Z (Z^T Z + λ I)^{-1} Z^T,   Z = [z_1 | … | z_k],
    P_perp = I - s P_span   (s = ``--span-projection-scale``; on-span scaling is (1-s)—s=1 removes
    the span component; s>1 over-subtracts and can flip that component; see ``--span-projection-scale`` help).

This edits weights only (no runtime hooks). Layers without ``gate_proj`` / ``up_proj`` only get
``down_proj`` edits.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import torch

from evaluation.eveluate_model import run_standalone_eval
from llm_utils.model_utils import load_local_model
from llm_utils.utils import resolve_device, sorted_numeric_layer_dirs

EVAL_ENG_VALID_FILE = "/home/morg/students/rashkovits/Localized-UNDO/datasets/pretrain/valid_eng.jsonl"
EVAL_MAX_LENGTH = 256
CACHE_DIR = "./cache"

# WMDP-bio supervised JSON uses role_labels_by_basis (pooled / neutral / bio_retain).
ROLE_LABEL_BASIS_CHOICES = frozenset({"pooled", "neutral", "bio_retain"})

# basis -> (log_ratio key in profile["log_ratios"], group key in profile["group_{means,counts}"]).
# Mirrors wmdp_bio_supervised_analysis.analyze_features_supervised_wmdp_bio.
_BASIS_TO_STATS_KEYS: Dict[str, tuple[str, str]] = {
    "pooled": ("log_forget_vs_pooled_retain", "pooled_retain"),
    "neutral": ("log_forget_vs_neutral", "neutral"),
    "bio_retain": ("log_forget_vs_bio_retain", "bio_retain"),
}

# Latent-selection strategies used by --selection-mode.
#   log_ratio  : legacy unary log-ratio role_labels_by_basis (with --role-label-bases + --role-basis-combine).
#   probe_topk : top-K latents per layer by signed L1-logistic probe weight (positive = forget-predictive).
#                Requires per-layer probe_weights_wmdp_bio.json (see wmdp_bio_probe_snmf_results.py).
#   intersect  : intersection of the two — a latent must pass BOTH the log-ratio rule AND be in the probe top-K.
#                This preserves the specificity filter of AND(bio_retain, neutral) while adding the multivariate
#                joint-decoding signal.
SELECTION_MODE_CHOICES = frozenset({"log_ratio", "probe_topk", "intersect"})


def _gc_and_empty_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_standalone_eval_for_args(model_path: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Single place for kwargs passed to evaluation/eveluate_model.run_standalone_eval."""
    return run_standalone_eval(
        model_path,
        eval_mode=args.eval_mode,
        large_eval=args.eval_large,
        no_mmlu=args.eval_no_mmlu,
        wmdp_include_path=args.eval_wmdp_include_path,
        wmdp_task_name=args.eval_wmdp_task_name,
        device=args.eval_device,
        batch_size=args.eval_batch_size,
        max_length=EVAL_MAX_LENGTH,
        cache_dir=CACHE_DIR,
        dataset_cache_dir=CACHE_DIR,
        eng_valid_file=EVAL_ENG_VALID_FILE,
    )


def _role_labels_by_basis_from_profile(profile: Dict[str, Any]) -> Dict[str, str]:
    """
    Normalize to a dict basis -> role string.
    Legacy JSON (arithmetic): only ``role_label`` → treated as pooled.
    WMDP-bio JSON: ``role_labels_by_basis``; ``role_label`` is pooled-compatible alias.
    """
    by_basis = profile.get("role_labels_by_basis")
    if isinstance(by_basis, dict) and by_basis:
        out: Dict[str, str] = {str(k): str(v) for k, v in by_basis.items()}
        if "pooled" not in out and profile.get("role_label") is not None:
            out["pooled"] = str(profile["role_label"])
        return out
    rl = str(profile.get("role_label", "unknown"))
    return {"pooled": rl}


def _assign_role_label_bio(
    log_forget_vs_retain: float,
    mean_forget: float,
    mean_retain: float,
    n_forget: int,
    n_retain: int,
    min_log_ratio: float,
) -> str:
    """Mirror of wmdp_bio_supervised_analysis._assign_role_label_bio (kept local to avoid import)."""
    if n_forget == 0 or n_retain == 0:
        return "insufficient_groups"
    total = mean_forget + mean_retain
    if total < 1e-9:
        return "low_signal"
    if log_forget_vs_retain >= min_log_ratio:
        return "bio_forget_lean"
    if log_forget_vs_retain <= -min_log_ratio:
        return "retain_lean"
    return "weak_mixed"


def _recompute_role_labels_by_basis(
    profile: Dict[str, Any],
    threshold: float,
) -> Dict[str, str]:
    """
    Recompute the per-basis role label from the raw stats stored in the WMDP-bio supervised JSON
    (group_means, group_counts, log_ratios), using ``threshold`` instead of the value used at
    analysis time. Falls back to ``insufficient_groups`` when the required fields are missing.
    """
    log_ratios = profile.get("log_ratios") or {}
    means = profile.get("group_means") or {}
    counts = profile.get("group_counts") or {}
    mean_forget = float(means.get("bio_forget", 0.0))
    n_forget = int(counts.get("bio_forget", 0))

    out: Dict[str, str] = {}
    for basis, (lr_key, grp_key) in _BASIS_TO_STATS_KEYS.items():
        log_fr = log_ratios.get(lr_key)
        n_r = int(counts.get(grp_key, 0))
        mean_r = float(means.get(grp_key, 0.0))
        if log_fr is None:
            out[basis] = "insufficient_groups"
            continue
        out[basis] = _assign_role_label_bio(
            float(log_fr), mean_forget, mean_r, n_forget, n_r, threshold
        )
    return out


def _latent_matches_forget_roles(
    profile: Dict[str, Any],
    forget_roles: Set[str],
    role_label_bases: Optional[Sequence[str]],
    role_basis_combine: str,
    role_assignment_threshold: Optional[float] = None,
) -> bool:
    """
    If ``role_label_bases`` is None or empty, match legacy top-level ``role_label`` only.
    Otherwise require forget-role membership per listed basis, combined with all/any.
    If ``role_assignment_threshold`` is given, labels are recomputed from the raw stats in
    ``profile`` (log_ratios / group_means / group_counts) rather than read from the stored
    ``role_labels_by_basis`` (which used the threshold fixed at analysis time).
    """
    if not role_label_bases:
        return str(profile.get("role_label", "unknown")) in forget_roles

    if role_assignment_threshold is not None:
        labels = _recompute_role_labels_by_basis(profile, role_assignment_threshold)
    else:
        labels = _role_labels_by_basis_from_profile(profile)
    checks: List[bool] = []
    for b in role_label_bases:
        if b not in ROLE_LABEL_BASIS_CHOICES:
            raise ValueError(
                f"Unknown role-label basis {b!r}; expected one of {sorted(ROLE_LABEL_BASIS_CHOICES)}"
            )
        if b not in labels:
            raise KeyError(
                f"Supervised profile missing basis {b!r} in role_labels_by_basis "
                f"(have keys: {sorted(labels.keys())}). Re-run WMDP-bio analysis or use fewer bases."
            )
        checks.append(labels[b] in forget_roles)

    if role_basis_combine == "all":
        return all(checks)
    if role_basis_combine == "any":
        return any(checks)
    raise ValueError(f"role_basis_combine must be 'all' or 'any', got {role_basis_combine!r}")


def _load_supervised_profiles(layer_dir: Path, supervised_json_filename: str) -> Dict[int, Dict[str, Any]]:
    path = layer_dir / supervised_json_filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run analysis on --results-dir first so each layer has supervised "
            f"role entries (e.g. feature_analysis_supervised_wmdp_bio.json)."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items() if isinstance(v, dict)}


def _load_probe_weights(layer_dir: Path, probe_weights_filename: str) -> Optional[Dict[int, float]]:
    """
    Load per-latent L1-logistic probe weights from ``layer_dir/<probe_weights_filename>``.

    Returns {latent_index: weight} or None when the file is missing (probe step wasn't run
    for this layer). Missing probe files should prevent ``probe_topk`` / ``intersect`` modes
    from being used silently — callers raise on None.
    """
    path = layer_dir / probe_weights_filename
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    weights = doc.get("weights_by_latent") or {}
    return {int(k): float(v) for k, v in weights.items()}


def _probe_top_k_cols(weights: Dict[int, float], top_k: int, k_all: int) -> List[int]:
    """
    Return up to ``top_k`` latent indices with the highest *positive* probe weight.

    A latent with w_i > 0 is predictive of ``bio_forget`` (when features are standardized,
    which the probe script enforces). Indices out of range [0, k_all) are dropped. Ties are
    broken by latent index ascending (deterministic).
    """
    cand = [
        (i, w) for i, w in weights.items()
        if 0 <= i < k_all and w > 0.0
    ]
    cand.sort(key=lambda t: (-t[1], t[0]))
    return sorted(i for i, _ in cand[: max(0, int(top_k))])


def _select_forget_cols(
    layer_dir: Path,
    selection_mode: str,
    *,
    forget_roles: Set[str],
    supervised_json_filename: str,
    role_label_bases: Optional[Sequence[str]],
    role_basis_combine: str,
    role_assignment_threshold: Optional[float],
    probe_weights_filename: str,
    probe_top_k: int,
    k_all: int,
) -> List[int]:
    """
    Pick the per-layer forget-column indices, dispatching on ``selection_mode``.

    - ``log_ratio`` : classic per-basis ``role_labels_by_basis`` rule (see _latent_matches_forget_roles).
    - ``probe_topk``: top-K latents by positive probe weight; ignores role labels entirely.
    - ``intersect`` : intersection of the two — latent must satisfy BOTH, preserving the
                      AND(bio_retain, neutral) specificity filter while adding the
                      multivariate signal of the probe.
    """
    if selection_mode not in SELECTION_MODE_CHOICES:
        raise ValueError(
            f"selection_mode={selection_mode!r} not in {sorted(SELECTION_MODE_CHOICES)}"
        )

    if selection_mode in ("log_ratio", "intersect"):
        profiles = _load_supervised_profiles(layer_dir, supervised_json_filename)
        log_ratio_cols = sorted(
            i
            for i, prof in profiles.items()
            if 0 <= i < k_all
            and _latent_matches_forget_roles(
                prof,
                forget_roles,
                role_label_bases,
                role_basis_combine,
                role_assignment_threshold=role_assignment_threshold,
            )
        )
    else:
        log_ratio_cols = []

    if selection_mode == "log_ratio":
        return log_ratio_cols

    weights = _load_probe_weights(layer_dir, probe_weights_filename)
    if weights is None:
        raise FileNotFoundError(
            f"selection_mode={selection_mode!r} requires {probe_weights_filename} in "
            f"{layer_dir}. Run wmdp_bio_probe_snmf_results.py on this SNMF dir first."
        )
    probe_cols = _probe_top_k_cols(weights, top_k=probe_top_k, k_all=k_all)

    if selection_mode == "probe_topk":
        return probe_cols
    # intersect
    probe_set = set(probe_cols)
    return sorted(i for i in log_ratio_cols if i in probe_set)


def _forget_feature_matrix(
    layer_dir: Path,
    forget_roles: Set[str],
    supervised_json_filename: str,
    role_label_bases: Optional[Sequence[str]],
    role_basis_combine: str,
    role_assignment_threshold: Optional[float] = None,
    *,
    selection_mode: str = "log_ratio",
    probe_weights_filename: str = "probe_weights_wmdp_bio.json",
    probe_top_k: int = 5,
) -> torch.Tensor | None:
    """Returns Z of shape (d_mlp, k) with columns z_i from F, or None if nothing to remove."""
    ckpt_path = layer_dir / "snmf_factors.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mode = ckpt.get("mode", "mlp_intermediate")
    if mode != "mlp_intermediate":
        raise ValueError(
            f"{ckpt_path}: checkpoint mode is {mode!r}, but this script only applies to "
            f"mlp_intermediate (F must match down_proj input). Re-train with train_snmf.py "
            f"--mode mlp_intermediate (see scripts/wmdp/train_snmf.sh)."
        )
    F = ckpt["F"].float().cpu()
    if F.ndim != 2:
        raise ValueError(f"Unexpected F shape in {ckpt_path}: {tuple(F.shape)}")

    k_all = F.shape[1]
    forget_cols = _select_forget_cols(
        layer_dir,
        selection_mode=selection_mode,
        forget_roles=forget_roles,
        supervised_json_filename=supervised_json_filename,
        role_label_bases=role_label_bases,
        role_basis_combine=role_basis_combine,
        role_assignment_threshold=role_assignment_threshold,
        probe_weights_filename=probe_weights_filename,
        probe_top_k=probe_top_k,
        k_all=k_all,
    )
    if not forget_cols:
        return None
    Z = F[:, forget_cols].contiguous()
    return Z


def orthogonal_projector_complement(
    Z: torch.Tensor,
    ridge_lambda: float,
    *,
    span_projection_scale: float = 1.0,
) -> torch.Tensor:
    """
    P_perp = I - s · Z (Z^T Z + λ I)^{-1} Z^T  ∈ ℝ^{d×d}, with Z ∈ ℝ^{d×k},
    where s = span_projection_scale (coefficient on P_span).

    For s=1 this is the usual orthogonal complement projector. For k=1 and s=1 this equals
    I - z z^T / (||z||^2 + λ) (≈ paper formula when λ=0).
    """
    d, k = Z.shape
    device, dtype = Z.device, Z.dtype
    I_d = torch.eye(d, device=device, dtype=dtype)
    if k == 0:
        return I_d
    g = Z.T @ Z + ridge_lambda * torch.eye(k, device=device, dtype=dtype)
    inv = torch.linalg.solve(g, torch.eye(k, device=device, dtype=dtype))
    p_span = Z @ inv @ Z.T
    return I_d - float(span_projection_scale) * p_span


def _summarize_eval(d: Dict[str, Any]) -> Dict[str, float]:
    """Keep scalar metrics for a compact table."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out


def _print_eval_comparison(before: Dict[str, Any], after: Dict[str, Any]) -> None:
    keys = sorted(set(before.keys()) | set(after.keys()))
    acc_keys = [k for k in keys if "acc" in k]
    other_keys = [k for k in keys if k not in acc_keys]
    print("\n=== Evaluation comparison (evaluation/eveluate_model.py) ===")
    for k in acc_keys + other_keys:
        b = before.get(k)
        a = after.get(k)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            delta = a - b
            print(f"  {k}: before={b:.6f}  after={a:.6f}  delta={delta:+.6f}")
        else:
            print(f"  {k}: before={b!r}  after={a!r}")


def _random_direction_matrix(
    d_mlp: int,
    n_dirs: int,
    *,
    seed: int,
    layer_idx: int,
) -> torch.Tensor:
    """
    Build a random orthonormal-basis subset Z in R^(d_mlp x n_dirs).
    """
    if n_dirs <= 0:
        return torch.empty((d_mlp, 0), dtype=torch.float32)
    if n_dirs > d_mlp:
        raise ValueError(f"Requested n_dirs={n_dirs} but d_mlp={d_mlp}.")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(layer_idx))
    rnd = torch.randn((d_mlp, n_dirs), generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(rnd, mode="reduced")
    return q.to(dtype=torch.float32, device="cpu")


def _apply_ablation_to_model(
    model_path: str,
    results_dir: Path,
    forget_roles: Set[str],
    supervised_json_filename: str,
    ridge_lambda: float,
    device: str,
    random_baseline: bool,
    random_seed: int,
    *,
    role_label_bases: Optional[Sequence[str]] = None,
    role_basis_combine: str = "all",
    role_assignment_threshold: Optional[float] = None,
    span_projection_scale: float = 1.0,
    down_proj_only: bool = False,
    selection_mode: str = "log_ratio",
    probe_weights_filename: str = "probe_weights_wmdp_bio.json",
    probe_top_k: int = 5,
) -> tuple[object, Dict[str, object]]:
    """
    Load model, apply either learned-direction or random-direction ablation.
    Returns (local_model_wrapper, metadata).
    """
    local = load_local_model(model_path, device=device)
    model = local.model
    base = getattr(model, "model", model)
    d_mlp = local.d_mlp

    bases_norm: Optional[List[str]] = None
    if role_label_bases is not None:
        bases_norm = [str(b).strip() for b in role_label_bases if str(b).strip()]
        if not bases_norm:
            bases_norm = None

    meta: Dict[str, object] = {
        "model_path": model_path,
        "results_dir": str(results_dir),
        "forget_roles": sorted(forget_roles),
        "role_label_bases": bases_norm,
        "role_basis_combine": role_basis_combine if bases_norm else None,
        "role_assignment_threshold": (
            float(role_assignment_threshold)
            if role_assignment_threshold is not None
            else None
        ),
        "supervised_json_filename": supervised_json_filename,
        "ridge_lambda": ridge_lambda,
        "span_projection_scale": float(span_projection_scale),
        "ablation_type": "random_matched_count" if random_baseline else "learned_forget_directions",
        "random_seed": int(random_seed) if random_baseline else None,
        "down_proj_only": bool(down_proj_only),
        "selection_mode": str(selection_mode),
        "probe_weights_filename": probe_weights_filename if selection_mode != "log_ratio" else None,
        "probe_top_k": int(probe_top_k) if selection_mode != "log_ratio" else None,
        "layers": [],
    }

    for _layer_num, layer_dir in sorted_numeric_layer_dirs(results_dir):
        Z_learned = _forget_feature_matrix(
            layer_dir,
            forget_roles,
            supervised_json_filename,
            role_label_bases=bases_norm,
            role_basis_combine=role_basis_combine,
            role_assignment_threshold=role_assignment_threshold,
            selection_mode=selection_mode,
            probe_weights_filename=probe_weights_filename,
            probe_top_k=probe_top_k,
        )
        if Z_learned is None:
            continue

        layer_idx = int(layer_dir.name.split("_")[-1])
        if Z_learned.shape[0] != d_mlp:
            raise ValueError(
                f"Layer {layer_idx}: F rows {Z_learned.shape[0]} != model d_mlp {d_mlp}. "
                "Train SNMF with the same architecture / mlp_intermediate as this model."
            )
        n_forget = int(Z_learned.shape[1])
        Z = (
            _random_direction_matrix(d_mlp, n_forget, seed=random_seed, layer_idx=layer_idx)
            if random_baseline
            else Z_learned
        )

        mlp = base.layers[layer_idx].mlp
        w_down = mlp.down_proj.weight.data  # (d_model, d_mlp)
        dtype = w_down.dtype
        dev = w_down.device

        # Projector on CPU (float64): avoids CUDA on GPUs where PyTorch has no kernels (e.g. sm_61).
        z_cpu = Z.to(device="cpu", dtype=torch.float64)
        p_perp_cpu = orthogonal_projector_complement(
            z_cpu,
            ridge_lambda=ridge_lambda,
            span_projection_scale=span_projection_scale,
        )
        p_perp = p_perp_cpu.to(device=dev, dtype=dtype)

        with torch.no_grad():
            # W_V^{new} = W_V @ P_perp  (remove forget subspace from down_proj *input*)
            w_down.copy_(torch.mm(w_down, p_perp))
            if not down_proj_only:
                for name in ("gate_proj", "up_proj"):
                    lin = getattr(mlp, name, None)
                    if lin is None:
                        continue
                    w_in = lin.weight.data  # (d_mlp, d_model)
                    if w_in.shape[0] != p_perp.shape[0]:
                        raise ValueError(
                            f"Layer {layer_idx}: {name}.weight.shape[0]={w_in.shape[0]} != "
                            f"d_mlp={p_perp.shape[0]}; cannot apply P_perp @ W."
                        )
                    # y' = P_perp @ (W @ x + b)  =>  W' = P_perp @ W, b' = P_perp @ b
                    w_in.copy_(torch.mm(p_perp, w_in))
                    if lin.bias is not None:
                        b = lin.bias.data
                        lin.bias.data.copy_(torch.mv(p_perp, b))

        layer_meta = {
            "layer": layer_idx,
            "n_forget_columns": n_forget,
            "d_mlp": int(d_mlp),
            "dual_sided": not down_proj_only,
        }
        if random_baseline:
            layer_meta["random_seed"] = int(random_seed) + int(layer_idx)
        meta["layers"].append(layer_meta)
        side_msg = "W_down @ P_perp only" if down_proj_only else "P_perp @ W_gate/up + W_down @ P_perp"
        if random_baseline:
            print(
                f"Layer {layer_idx}: removed span of {n_forget} RANDOM direction(s) "
                f"(matched to forget count) with span_projection_scale={span_projection_scale} "
                f"via {side_msg}."
            )
        else:
            print(
                f"Layer {layer_idx}: removed span of {n_forget} forget SNMF column(s) "
                f"with span_projection_scale={span_projection_scale} via {side_msg}."
            )

    if not meta["layers"]:
        raise RuntimeError(
            f"No forget features found under {results_dir} for roles={sorted(forget_roles)}"
            + (
                f" (bases={bases_norm!r}, combine={role_basis_combine!r})."
                if bases_norm
                else " (using top-level role_label only)."
            )
        )
    return local, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove SNMF forget directions: W_down <- W_down @ P_perp; "
        "W_up,W_gate <- P_perp @ W when present.",
        epilog=(
            "Typical use after running train_snmf.py and analyze_snmf_results.py, using the same relevant parameters. "
        ),
    )
    p.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Same model used in train_snmf.py and analyze_snmf_results.py.",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="SNMF output dir with layer_*/. subdirs in it that contain snmf_factors.pt and feature_analysis_supervised.json.",
    )
    p.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Directory for saving the ablated model. This directory will be created if it does not exist.",
    )
    p.add_argument(
        "--save-path-random",
        type=str,
        default="",
        help="Optional directory for random-direction ablated model. "
        "Default: <save-path>_random_baseline.",
    )
    p.add_argument(
        "--forget-roles",
        type=str,
        nargs="+",
        default=["mult_forget", "div_forget", "forget_mixed"],
    )
    p.add_argument(
        "--role-label-bases",
        type=str,
        nargs="*",
        default=None,
        metavar="BASIS",
        help=(
            "WMDP-bio only: which supervised bases to read from each latent's role_labels_by_basis "
            "(pooled | neutral | bio_retain). Example: --role-label-bases pooled bio_retain "
            "--role-basis-combine all targets latents labeled forget on every listed basis. "
            "Omit this flag to use legacy top-level role_label only (arithmetic JSONs)."
        ),
    )
    p.add_argument(
        "--role-basis-combine",
        type=str,
        default="all",
        choices=["all", "any"],
        help="How to combine --role-label-bases: 'all' (AND) or 'any' (OR). Ignored when bases omitted.",
    )
    p.add_argument(
        "--role-assignment-threshold",
        type=float,
        default=None,
        help=(
            "Optional WMDP-bio threshold (min |log_forget_vs_retain|) used to recompute per-basis "
            "role labels on-the-fly from the raw stats in each supervised JSON profile "
            "(log_ratios / group_means / group_counts). When omitted, the stored "
            "role_labels_by_basis (baked at analysis time, default threshold 0.15) is used. "
            "Only affects selection when --role-label-bases is also set."
        ),
    )
    p.add_argument(
        "--supervised-json-filename",
        type=str,
        default="feature_analysis_supervised.json",
        help=(
            "Per-layer supervised analysis JSON file name inside each layer_* folder "
            "(e.g. feature_analysis_supervised_wmdp_bio.json)."
        ),
    )
    p.add_argument(
        "--ridge-lambda",
        type=float,
        default=1e-6,
        help="Tikhonov on Z^T Z when building the span projector (stability).",
    )
    p.add_argument(
        "--span-projection-scale",
        type=float,
        default=1.0,
        help=(
            "Coefficient s on P_span in P_perp = I - s·P_span. On vectors in the forget span, "
            "this acts as scaling by (1-s): s=1 removes that component entirely (orthogonal "
            "complement); s<1 leaves a residual (softer removal); s>1 over-subtracts—(1-s) is "
            "negative, so the span component is flipped and scaled in magnitude (e.g. s=2 gives "
            "the opposite direction with equal norm). Default 1.0."
        ),
    )
    p.add_argument(
        "--random-baseline",
        action="store_true",
        help="Also run matched-count random-direction ablation baseline.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=1234,
        help="Seed for reproducible random-direction baseline.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for loading the model during weight edit (auto: cuda if usable, else cpu; "
        "matches analyze_snmf_results / utils.resolve_device for old GPUs).",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Do not run evaluation/eveluate_model.py before/after ablation.",
    )
    p.add_argument(
        "--skip-pre-eval",
        action="store_true",
        help="Skip only the baseline (pre-ablation) eval of the original model; still run "
        "the post-ablation eval(s). Useful when the original model's metrics are already known.",
    )
    p.add_argument(
        "--eval-device",
        type=str,
        default="auto",
        help="Device for standalone eval (passed through to eveluate_model: auto|cuda|cpu).",
    )
    p.add_argument(
        "--eval-mode",
        type=str,
        default="arithmetic",
        choices=["arithmetic", "wmdp_bio", "wmdp_cyber", "both_wmdp", "wmdp_bio_categorized"],
        help="Evaluation mode passed to run_standalone_eval / eveluate_model.py.",
    )
    p.add_argument(
        "--eval-large",
        action="store_true",
        help="Use larger/full evaluation limits for WMDP/MMLU tasks.",
    )
    p.add_argument(
        "--eval-no-mmlu",
        action="store_true",
        help="For single-domain WMDP eval modes, skip MMLU.",
    )
    p.add_argument(
        "--eval-wmdp-include-path",
        type=str,
        default="",
        help="Path to lm-eval task YAML directory (used with eval-mode=wmdp_bio_categorized).",
    )
    p.add_argument(
        "--eval-wmdp-task-name",
        type=str,
        default="wmdp_bio_robust",
        help="Task/group name for eval-mode=wmdp_bio_categorized.",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Batch size for CE leg of arithmetic eval.",
    )
    p.add_argument(
        "--down-proj-only",
        action="store_true",
        help="Only ablate down_proj (W_V @ P_perp). Skip gate_proj/up_proj for ablations that "
        "match the old single-sided behavior.",
    )
    p.add_argument(
        "--selection-mode",
        type=str,
        default="log_ratio",
        choices=sorted(SELECTION_MODE_CHOICES),
        help=(
            "How to pick forget columns per layer: "
            "'log_ratio' = classic role_labels_by_basis rule (existing behavior); "
            "'probe_topk' = top-K latents by positive L1-logistic probe weight "
            "(requires running wmdp_bio_probe_snmf_results.py first to produce "
            "layer_*/probe_weights_wmdp_bio.json); "
            "'intersect' = log_ratio ∩ probe_topk (preserves specificity filters, "
            "adds multivariate decoding signal)."
        ),
    )
    p.add_argument(
        "--probe-weights-filename",
        type=str,
        default="probe_weights_wmdp_bio.json",
        help="Per-layer probe-weights JSON used by --selection-mode probe_topk/intersect.",
    )
    p.add_argument(
        "--probe-top-k",
        type=int,
        default=5,
        help="Per-layer K for --selection-mode probe_topk/intersect (positive weights only).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    forget_roles = set(args.forget_roles)
    if args.role_label_bases is None:
        role_label_bases: Optional[List[str]] = None
    else:
        role_label_bases = [str(b).strip() for b in args.role_label_bases if str(b).strip()] or None

    results_before: Dict[str, Any] | None = None
    results_after: Dict[str, Any] | None = None

    if not args.skip_eval and not args.skip_pre_eval:
        print("\n=== Baseline eval (original model, before ablation) ===")
        results_before = _run_standalone_eval_for_args(args.model_path, args)
        _gc_and_empty_cuda()
    elif not args.skip_eval and args.skip_pre_eval:
        print("\n=== Skipping baseline eval (--skip-pre-eval set); post-ablation eval will still run ===")

    ablation_device = resolve_device(args.device)
    print(f"Ablation model load device (after resolve): {ablation_device}")
    local, meta = _apply_ablation_to_model(
        model_path=args.model_path,
        results_dir=results_dir,
        forget_roles=forget_roles,
        supervised_json_filename=args.supervised_json_filename,
        ridge_lambda=args.ridge_lambda,
        device=ablation_device,
        random_baseline=False,
        random_seed=args.random_seed,
        role_label_bases=role_label_bases,
        role_basis_combine=args.role_basis_combine,
        role_assignment_threshold=args.role_assignment_threshold,
        span_projection_scale=args.span_projection_scale,
        down_proj_only=args.down_proj_only,
        selection_mode=args.selection_mode,
        probe_weights_filename=args.probe_weights_filename,
        probe_top_k=args.probe_top_k,
    )

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    local.model.save_pretrained(save_path)
    local.tokenizer.save_pretrained(save_path)
    print(f"Saved edited model and tokenizer to {save_path}")

    metadata_out = save_path / "forget_ablation_metadata.json"
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {metadata_out}")

    # Drop ablation model from memory before loading again for eval.
    del local
    _gc_and_empty_cuda()

    if not args.skip_eval:
        print("\n=== Post-ablation eval (saved checkpoint) ===")
        results_after = _run_standalone_eval_for_args(str(save_path), args)
        if results_before is not None:
            _print_eval_comparison(results_before, results_after)
        else:
            print("\n=== Post-ablation eval (no baseline; --skip-pre-eval was set) ===")
            _print_eval_comparison({}, results_after)

        eval_out = save_path / "ablation_eval_comparison.json"
        eval_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline_model_path": args.model_path,
            "ablated_model_path": str(save_path),
            "before": _summarize_eval(results_before) if results_before is not None else None,
            "after": _summarize_eval(results_after),
        }
        with open(eval_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote eval comparison JSON to {eval_out}")

    if args.random_baseline:
        print("\n=== Random baseline ablation (matched direction count) ===")
        local_rand, rand_meta = _apply_ablation_to_model(
            model_path=args.model_path,
            results_dir=results_dir,
            forget_roles=forget_roles,
            supervised_json_filename=args.supervised_json_filename,
            ridge_lambda=args.ridge_lambda,
            device=ablation_device,
            random_baseline=True,
            random_seed=args.random_seed,
            role_label_bases=role_label_bases,
            role_basis_combine=args.role_basis_combine,
            role_assignment_threshold=args.role_assignment_threshold,
            span_projection_scale=args.span_projection_scale,
            down_proj_only=args.down_proj_only,
            selection_mode=args.selection_mode,
            probe_weights_filename=args.probe_weights_filename,
            probe_top_k=args.probe_top_k,
        )
        save_path_random = (
            Path(args.save_path_random)
            if args.save_path_random
            else Path(f"{args.save_path}_random_baseline")
        )
        save_path_random.mkdir(parents=True, exist_ok=True)
        local_rand.model.save_pretrained(save_path_random)
        local_rand.tokenizer.save_pretrained(save_path_random)
        print(f"Saved random-baseline model and tokenizer to {save_path_random}")

        del local_rand
        _gc_and_empty_cuda()

        if not args.skip_eval:
            print("\n=== Post-random-baseline eval (saved checkpoint) ===")
            results_random = _run_standalone_eval_for_args(str(save_path_random), args)
            assert results_after is not None
            print("\n--- Learned-direction ablation vs random baseline ---")
            _print_eval_comparison(results_after, results_random)
            if results_before is not None:
                print("\n--- Original baseline vs random baseline ---")
                _print_eval_comparison(results_before, results_random)
            else:
                print("\n--- Skipping 'original baseline vs random baseline' (--skip-pre-eval was set) ---")

            payload["random_baseline"] = {
                "random_seed": int(args.random_seed),
                "ablated_model_path": str(save_path_random),
                "metadata": rand_meta,
                "after": _summarize_eval(results_random),
            }
            with open(eval_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Updated eval comparison JSON with random baseline at {eval_out}")


if __name__ == "__main__":
    main()
