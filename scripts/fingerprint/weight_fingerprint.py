"""Weight-space unlearning fingerprinting.

For a base model and a set of unlearned-candidate models (typically SNMF /
MaxEnt / RMU wmdp-bio runs), extract per-layer per-matrix metrics from the
attention K/V/Q/O and MLP gate/up/down projections, then write CSV / NPZ
artefacts and a battery of matplotlib plots so we can eyeball which signals
work as a *fingerprint* of unlearning.

Why these metrics?
------------------
Two complementary views:

1. Intrinsic shape of each weight matrix W:
     - effective rank (entropy of the squared-singular-value distribution)
     - stable rank, participation ratio, hard rank for 99% energy
     - spectral norm, Frobenius norm
   Hypothesis: aggressive forgetting that drops sub-spaces should reduce the
   effective rank of K/V (the matrices that read the residual stream and
   produce keys/values used by attention) at the targeted layers.

2. The *update* W_candidate - W_base. This is dramatically more sensitive
   than absolute weight values because every method nudges the same
   initialization in its own characteristic way:
     - Frobenius norm of the delta (how much was changed)
     - cosine similarity between W and W_base (drift direction)
     - effective rank of the delta (low-rank => RMU/LoRA-like surgical
       update; high-rank => diffuse fine-tuning / MaxEnt)
     - relative delta = ||delta||_F / ||W_base||_F

The same module can be re-pointed at retain-tuned or relearned checkpoints
in the same Gemma-2 family.

The script streams one weight tensor at a time via safetensors, so peak
memory stays at ~ one matrix per process; SVDs run on GPU when available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - plotting is best-effort
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

from safetensors import safe_open


# --------------------------------------------------------------------------
# Matrix groups we extract
# --------------------------------------------------------------------------
ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ALL_PROJECTIONS = ATTN_PROJECTIONS + MLP_PROJECTIONS


def _attn_key(layer: int, name: str) -> str:
    return f"model.layers.{layer}.self_attn.{name}.weight"


def _mlp_key(layer: int, name: str) -> str:
    return f"model.layers.{layer}.mlp.{name}.weight"


def _weight_key(layer: int, name: str) -> str:
    if name in ATTN_PROJECTIONS:
        return _attn_key(layer, name)
    if name in MLP_PROJECTIONS:
        return _mlp_key(layer, name)
    raise KeyError(f"Unknown projection name: {name}")


# --------------------------------------------------------------------------
# Lightweight safetensors-backed weight loader
# --------------------------------------------------------------------------
@dataclass
class SafeTensorsModel:
    """Resolves a model directory to a list of safetensors shards and the
    weight-name -> shard index from `model.safetensors.index.json` (if any),
    so we can pull single tensors on demand without instantiating the full
    transformers model."""

    name: str
    path: Path
    shards: List[Path]
    name_to_file: Dict[str, Path]

    @classmethod
    def from_path(cls, name: str, model_path: str) -> "SafeTensorsModel":
        root = Path(model_path)
        if not root.exists():
            raise FileNotFoundError(f"Model path not found: {root}")
        # Some unlearned-model layouts use a `final_model/` subdir.
        if not any(root.glob("*.safetensors")) and (root / "final_model").is_dir():
            root = root / "final_model"

        index_file = root / "model.safetensors.index.json"
        if index_file.is_file():
            with index_file.open("r") as fh:
                index = json.load(fh)
            weight_map = index.get("weight_map", {})
            name_to_file = {k: root / v for k, v in weight_map.items()}
            shards = sorted({p for p in name_to_file.values()})
        else:
            shards = sorted(root.glob("*.safetensors"))
            if not shards:
                raise FileNotFoundError(f"No safetensors shards in {root}")
            name_to_file = {}
            for shard in shards:
                with safe_open(str(shard), framework="pt") as fh:
                    for key in fh.keys():
                        name_to_file[key] = shard
        return cls(name=name, path=root, shards=shards, name_to_file=name_to_file)

    def has(self, key: str) -> bool:
        return key in self.name_to_file

    def load(self, key: str, device: str = "cpu") -> torch.Tensor:
        shard = self.name_to_file.get(key)
        if shard is None:
            raise KeyError(f"{key} not found in {self.path}")
        with safe_open(str(shard), framework="pt", device=device) as fh:
            return fh.get_tensor(key)


# --------------------------------------------------------------------------
# Numeric core: SVD + per-matrix metrics
# --------------------------------------------------------------------------
def _to_float32(t: torch.Tensor) -> torch.Tensor:
    if t.dtype != torch.float32:
        t = t.to(torch.float32)
    return t


def singular_values(weight: torch.Tensor, device: str) -> np.ndarray:
    """Return singular values (sorted descending) as a numpy float32 array."""
    w = _to_float32(weight).to(device)
    # torch.linalg.svdvals is cheaper than full svd and is plenty for our
    # diagnostics; we only need the singular values themselves.
    s = torch.linalg.svdvals(w)
    return s.detach().cpu().to(torch.float32).numpy()


def spectrum_metrics(s: np.ndarray) -> Dict[str, float]:
    """Compute scalar spectral / energy metrics from a singular-value vector."""
    s = np.asarray(s, dtype=np.float64)
    s = np.clip(s, 0.0, None)
    s2 = s * s
    total = float(s2.sum())
    smax = float(s.max()) if s.size else 0.0
    smax2 = smax * smax

    if total <= 0.0 or smax2 <= 0.0:
        return {
            "spectral_norm": 0.0,
            "frobenius_norm": 0.0,
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "participation_ratio": 0.0,
            "hard_rank_99": 0.0,
            "top1_energy_ratio": 0.0,
            "num_singular_values": float(s.size),
        }

    p = s2 / total
    nonzero = p > 0
    entropy = -float((p[nonzero] * np.log(p[nonzero])).sum())
    eff_rank = float(math.exp(entropy))

    stable_rank = float(total / smax2)
    participation_ratio = float((total ** 2) / float((s2 * s2).sum() + 1e-30))

    cumulative = np.cumsum(s2) / total
    hard_rank_99 = int(np.searchsorted(cumulative, 0.99) + 1)

    return {
        "spectral_norm": smax,
        "frobenius_norm": float(math.sqrt(total)),
        "effective_rank": eff_rank,
        "stable_rank": stable_rank,
        "participation_ratio": participation_ratio,
        "hard_rank_99": float(hard_rank_99),
        "top1_energy_ratio": float(s2[0] / total),
        "num_singular_values": float(s.size),
    }


def weight_distribution_metrics(weight: torch.Tensor, zero_tol: float = 1e-6) -> Dict[str, float]:
    """Cheap entry-wise statistics that don't need SVD."""
    w = _to_float32(weight).detach()
    flat = w.flatten()
    n = flat.numel()
    if n == 0:
        return {"mean_abs": 0.0, "std": 0.0, "kurtosis": 0.0, "near_zero_frac": 0.0}

    abs_w = flat.abs()
    mean_abs = float(abs_w.mean())
    std = float(flat.std(unbiased=False))
    # Use central moments to compute excess kurtosis (so a Gaussian sits at 0).
    centered = flat - flat.mean()
    var = float((centered ** 2).mean())
    if var > 0:
        m4 = float((centered ** 4).mean())
        kurtosis = m4 / (var ** 2) - 3.0
    else:
        kurtosis = 0.0
    near_zero_frac = float((abs_w < zero_tol).float().mean())
    return {
        "mean_abs": mean_abs,
        "std": std,
        "kurtosis": kurtosis,
        "near_zero_frac": near_zero_frac,
    }


def delta_metrics(
    weight: torch.Tensor,
    base_weight: torch.Tensor,
    base_frob: float,
    device: str,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Compare W vs W_base. Returns (scalar_dict, singular_values_of_delta)."""
    w = _to_float32(weight).to(device)
    wb = _to_float32(base_weight).to(device)
    delta = w - wb
    s_delta = torch.linalg.svdvals(delta).detach().cpu().to(torch.float32).numpy()
    spectrum = spectrum_metrics(s_delta)

    flat_w = w.flatten()
    flat_b = wb.flatten()
    denom = (flat_w.norm() * flat_b.norm()).clamp_min(1e-30)
    cosine = float((flat_w @ flat_b) / denom)
    # clamp away fp32 round-off when the vectors are essentially identical
    cosine = max(-1.0, min(1.0, cosine))

    rel = float(spectrum["frobenius_norm"] / (base_frob + 1e-30))
    out = {
        "delta_spectral_norm": spectrum["spectral_norm"],
        "delta_frobenius_norm": spectrum["frobenius_norm"],
        "delta_effective_rank": spectrum["effective_rank"],
        "delta_stable_rank": spectrum["stable_rank"],
        "delta_participation_ratio": spectrum["participation_ratio"],
        "delta_hard_rank_99": spectrum["hard_rank_99"],
        "delta_top1_energy_ratio": spectrum["top1_energy_ratio"],
        "delta_rel_frobenius": rel,
        "cosine_to_base": cosine,
    }
    return out, s_delta


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _parse_layers(spec: str, num_layers: int) -> List[int]:
    """Accept "all", "0-25", "0,3,5", "10-18,20", etc."""
    spec = spec.strip()
    if spec.lower() in ("all", "*", ""):
        return list(range(num_layers))
    out: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", chunk)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(chunk))
    out = sorted(set(i for i in out if 0 <= i < num_layers))
    return out


def _parse_models(spec: Sequence[str]) -> List[Tuple[str, str]]:
    """Each entry is "label=path"."""
    out: List[Tuple[str, str]] = []
    for raw in spec:
        if "=" not in raw:
            raise ValueError(f"--candidate-model needs label=path form, got {raw!r}")
        label, path = raw.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Bad --candidate-model entry: {raw!r}")
        out.append((label, path))
    return out


def _autodetect_num_layers(base: SafeTensorsModel) -> int:
    pat = re.compile(r"^model\.layers\.(\d+)\.")
    layers = set()
    for key in base.name_to_file.keys():
        m = pat.match(key)
        if m:
            layers.add(int(m.group(1)))
    if not layers:
        raise RuntimeError("Could not autodetect layer count from base model.")
    return max(layers) + 1


def analyze(
    base_path: str,
    candidates: Sequence[Tuple[str, str]],
    output_dir: Path,
    layers_spec: str,
    projections: Sequence[str],
    device: str,
    spectra_layers: Sequence[int],
    save_full_spectra: bool,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    base = SafeTensorsModel.from_path("base", base_path)
    num_layers = _autodetect_num_layers(base)
    layers = _parse_layers(layers_spec, num_layers)
    print(f"[info] base model at {base.path}, layers={num_layers}, "
          f"analysing layers={layers}")

    cand_models: List[SafeTensorsModel] = []
    for label, path in candidates:
        m = SafeTensorsModel.from_path(label, path)
        cand_models.append(m)
        print(f"[info] candidate {label!r} -> {m.path}")

    all_models: List[SafeTensorsModel] = [base] + cand_models
    model_labels = [m.name for m in all_models]

    rows: List[Dict[str, object]] = []
    # spectra: model -> proj -> layer -> np.ndarray
    weight_spectra: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {
        m: {p: {} for p in projections} for m in model_labels
    }
    delta_spectra: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {
        m.name: {p: {} for p in projections} for m in cand_models
    }

    for layer in layers:
        for proj in projections:
            key = _weight_key(layer, proj)
            if not base.has(key):
                print(f"[warn] missing {key} in base; skipping")
                continue

            base_w = base.load(key)
            base_s = singular_values(base_w, device=device)
            base_spec = spectrum_metrics(base_s)
            base_dist = weight_distribution_metrics(base_w)
            base_frob = base_spec["frobenius_norm"]
            weight_spectra["base"][proj][layer] = base_s

            base_row = {
                "model": "base",
                "layer": layer,
                "projection": proj,
                **base_spec,
                **base_dist,
                "delta_spectral_norm": 0.0,
                "delta_frobenius_norm": 0.0,
                "delta_effective_rank": 0.0,
                "delta_stable_rank": 0.0,
                "delta_participation_ratio": 0.0,
                "delta_hard_rank_99": 0.0,
                "delta_top1_energy_ratio": 0.0,
                "delta_rel_frobenius": 0.0,
                "cosine_to_base": 1.0,
            }
            rows.append(base_row)

            for cand in cand_models:
                if not cand.has(key):
                    print(f"[warn] missing {key} in {cand.name}; skipping")
                    continue
                w = cand.load(key)
                s = singular_values(w, device=device)
                spec = spectrum_metrics(s)
                dist = weight_distribution_metrics(w)
                d_scalars, s_delta = delta_metrics(w, base_w, base_frob, device=device)
                weight_spectra[cand.name][proj][layer] = s
                delta_spectra[cand.name][proj][layer] = s_delta

                row = {
                    "model": cand.name,
                    "layer": layer,
                    "projection": proj,
                    **spec,
                    **dist,
                    **d_scalars,
                }
                rows.append(row)

                # free explicitly to keep host RAM bounded
                del w
            del base_w
        print(f"[info] finished layer {layer}/{layers[-1]}")

    df = pd.DataFrame(rows)
    csv_path = output_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[info] wrote {csv_path} ({len(df)} rows)")

    if save_full_spectra:
        npz_payload: Dict[str, np.ndarray] = {}
        for model_name, by_proj in weight_spectra.items():
            for proj, by_layer in by_proj.items():
                for layer, s in by_layer.items():
                    npz_payload[f"W::{model_name}::{proj}::layer_{layer}"] = s.astype(np.float32)
        for model_name, by_proj in delta_spectra.items():
            for proj, by_layer in by_proj.items():
                for layer, s in by_layer.items():
                    npz_payload[f"D::{model_name}::{proj}::layer_{layer}"] = s.astype(np.float32)
        npz_path = output_dir / "singular_values.npz"
        np.savez_compressed(npz_path, **npz_payload)
        print(f"[info] wrote {npz_path} ({len(npz_payload)} arrays)")

    if plt is not None:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        make_plots(
            df=df,
            weight_spectra=weight_spectra,
            delta_spectra=delta_spectra,
            projections=list(projections),
            model_labels=model_labels,
            cand_labels=[c.name for c in cand_models],
            layers=layers,
            spectra_layers=list(spectra_layers) or layers[:: max(1, len(layers) // 4)],
            plot_dir=plot_dir,
        )
        print(f"[info] wrote plots to {plot_dir}")
    else:
        print(f"[warn] matplotlib not available, skipping plots ({_MATPLOTLIB_IMPORT_ERROR})")

    summary = {
        "base": str(base.path),
        "candidates": {c.name: str(c.path) for c in cand_models},
        "num_layers": num_layers,
        "layers": layers,
        "projections": list(projections),
        "metrics_csv": str(csv_path),
    }
    with (output_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
_DEFAULT_COLORS = {
    "base": "#444444",
    "snmf": "#1f77b4",
    "maxent": "#d62728",
    "rmu": "#2ca02c",
}


def _color_for(label: str, fallback_cycle: Sequence[str], idx: int) -> str:
    key = label.lower()
    for k, v in _DEFAULT_COLORS.items():
        if k in key:
            return v
    return fallback_cycle[idx % len(fallback_cycle)]


def _fig_grid(projections: Sequence[str]):
    n = len(projections)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)
    return fig, axes, nrows, ncols


def make_plots(
    df: pd.DataFrame,
    weight_spectra: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    delta_spectra: Dict[str, Dict[str, Dict[int, np.ndarray]]],
    projections: List[str],
    model_labels: List[str],
    cand_labels: List[str],
    layers: List[int],
    spectra_layers: List[int],
    plot_dir: Path,
) -> None:
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map = {lbl: _color_for(lbl, cycle, i) for i, lbl in enumerate(model_labels)}

    # ---- 1. Effective rank vs layer, faceted by projection. -------------
    metrics_for_curve = [
        ("effective_rank", "Effective rank (entropy of σ²)"),
        ("stable_rank", "Stable rank = ‖W‖_F² / ‖W‖_2²"),
        ("hard_rank_99", "99% energy hard rank"),
    ]
    for metric, title_metric in metrics_for_curve:
        fig, axes, nrows, ncols = _fig_grid(projections)
        for ax, proj in zip(axes.flat, projections):
            sub = df[df["projection"] == proj]
            for lbl in model_labels:
                line = sub[sub["model"] == lbl].sort_values("layer")
                if line.empty:
                    continue
                ax.plot(line["layer"], line[metric], marker="o", lw=1.4,
                        ms=3, color=color_map[lbl], label=lbl)
            ax.set_title(proj)
            ax.set_xlabel("layer")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
        for ax in axes.flat[len(projections):]:
            ax.set_visible(False)
        axes[0, 0].legend(loc="best", fontsize=8)
        fig.suptitle(title_metric)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = plot_dir / f"01_{metric}_vs_layer.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)

    # ---- 2. Delta-Frobenius heatmap per projection. ---------------------
    if cand_labels:
        for proj in projections:
            sub = df[(df["projection"] == proj) & (df["model"].isin(cand_labels))]
            if sub.empty:
                continue
            pivot = sub.pivot(index="model", columns="layer", values="delta_rel_frobenius")
            pivot = pivot.reindex(cand_labels)
            fig, ax = plt.subplots(figsize=(0.45 * len(layers) + 2, 1.0 * len(cand_labels) + 1.5))
            im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns, rotation=0, fontsize=8)
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("layer")
            ax.set_title(f"‖ΔW‖_F / ‖W_base‖_F   ({proj})")
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            fig.tight_layout()
            fig.savefig(plot_dir / f"02_delta_relfrob_{proj}.png", dpi=140)
            plt.close(fig)

        # ---- 3. Cosine-to-base heatmap per projection. ------------------
        for proj in projections:
            sub = df[(df["projection"] == proj) & (df["model"].isin(cand_labels))]
            if sub.empty:
                continue
            pivot = sub.pivot(index="model", columns="layer", values="cosine_to_base")
            pivot = pivot.reindex(cand_labels)
            fig, ax = plt.subplots(figsize=(0.45 * len(layers) + 2, 1.0 * len(cand_labels) + 1.5))
            im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns, rotation=0, fontsize=8)
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("layer")
            ax.set_title(f"cos(W, W_base)   ({proj})")
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            fig.tight_layout()
            fig.savefig(plot_dir / f"03_cosine_to_base_{proj}.png", dpi=140)
            plt.close(fig)

        # ---- 4. Delta effective rank vs layer. --------------------------
        fig, axes, _, _ = _fig_grid(projections)
        for ax, proj in zip(axes.flat, projections):
            sub = df[(df["projection"] == proj) & (df["model"].isin(cand_labels))]
            for lbl in cand_labels:
                line = sub[sub["model"] == lbl].sort_values("layer")
                if line.empty:
                    continue
                ax.plot(line["layer"], line["delta_effective_rank"], marker="o", lw=1.4,
                        ms=3, color=color_map[lbl], label=lbl)
            ax.set_title(proj)
            ax.set_xlabel("layer")
            ax.set_ylabel("effective rank of ΔW")
            ax.grid(True, alpha=0.3)
        for ax in axes.flat[len(projections):]:
            ax.set_visible(False)
        axes[0, 0].legend(loc="best", fontsize=8)
        fig.suptitle("Effective rank of update W - W_base  (low ⇒ surgical/LoRA-like)")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(plot_dir / "04_delta_effective_rank.png", dpi=140)
        plt.close(fig)

        # ---- 5. Scatter: Δ frobenius vs Δ effective rank. ---------------
        for proj in projections:
            sub = df[(df["projection"] == proj) & (df["model"].isin(cand_labels))]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            for lbl in cand_labels:
                pts = sub[sub["model"] == lbl]
                if pts.empty:
                    continue
                ax.scatter(pts["delta_rel_frobenius"], pts["delta_effective_rank"],
                           s=24, color=color_map[lbl], label=lbl, alpha=0.85,
                           edgecolor="white", linewidth=0.5)
            ax.set_xlabel("relative ‖ΔW‖_F")
            ax.set_ylabel("effective rank of ΔW")
            ax.set_title(f"Update size vs spread  ({proj})")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plot_dir / f"05_delta_size_vs_rank_{proj}.png", dpi=140)
            plt.close(fig)

    # ---- 6. Singular-value spectra of W and ΔW at chosen layers. -------
    spectra_layers = [L for L in spectra_layers if L in layers]
    for proj in projections:
        if not spectra_layers:
            break
        ncols = min(4, len(spectra_layers))
        nrows = math.ceil(len(spectra_layers) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                                 squeeze=False)
        for ax, layer in zip(axes.flat, spectra_layers):
            for lbl in model_labels:
                s = weight_spectra.get(lbl, {}).get(proj, {}).get(layer)
                if s is None or len(s) == 0:
                    continue
                ax.semilogy(np.arange(1, len(s) + 1), s / max(s[0], 1e-30),
                            color=color_map[lbl], lw=1.2, label=lbl)
            ax.set_title(f"{proj}  layer {layer}")
            ax.set_xlabel("index")
            ax.set_ylabel("σ / σ₁ (log)")
            ax.grid(True, which="both", alpha=0.3)
        for ax in axes.flat[len(spectra_layers):]:
            ax.set_visible(False)
        axes[0, 0].legend(loc="best", fontsize=8)
        fig.suptitle(f"Normalized singular-value spectrum of W  ({proj})")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(plot_dir / f"06_W_spectrum_{proj}.png", dpi=140)
        plt.close(fig)

        # spectrum of the *update*
        if not cand_labels:
            continue
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                                 squeeze=False)
        for ax, layer in zip(axes.flat, spectra_layers):
            for lbl in cand_labels:
                s = delta_spectra.get(lbl, {}).get(proj, {}).get(layer)
                if s is None or len(s) == 0:
                    continue
                norm = max(s[0], 1e-30)
                ax.semilogy(np.arange(1, len(s) + 1), s / norm,
                            color=color_map[lbl], lw=1.2, label=lbl)
            ax.set_title(f"{proj}  layer {layer}")
            ax.set_xlabel("index")
            ax.set_ylabel("σ(ΔW) / σ₁ (log)")
            ax.grid(True, which="both", alpha=0.3)
        for ax in axes.flat[len(spectra_layers):]:
            ax.set_visible(False)
        axes[0, 0].legend(loc="best", fontsize=8)
        fig.suptitle(f"Normalized singular-value spectrum of ΔW  ({proj})")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(plot_dir / f"07_delta_spectrum_{proj}.png", dpi=140)
        plt.close(fig)

    # ---- 8. 2D PCA scatter of per-layer fingerprint vectors. -----------
    # Build a fingerprint vector per (candidate, layer) by concatenating the
    # delta / cosine / rank scalars across projections, then project to 2D.
    if cand_labels:
        feat_cols = [
            "delta_rel_frobenius", "delta_effective_rank",
            "delta_stable_rank", "delta_top1_energy_ratio",
            "cosine_to_base", "effective_rank", "stable_rank",
        ]
        feat_cols = [c for c in feat_cols if c in df.columns]
        cand_df = df[df["model"].isin(cand_labels)].copy()
        # one feature row per (model, layer): wide over projections
        wide = cand_df.pivot_table(index=["model", "layer"], columns="projection",
                                   values=feat_cols)
        wide = wide.dropna(axis=1, how="all").fillna(0.0)
        X = wide.values.astype(np.float64)
        if X.shape[0] >= 3 and X.shape[1] >= 2:
            Xc = X - X.mean(axis=0, keepdims=True)
            std = Xc.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            Xc = Xc / std
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            coords = U[:, :2] * S[:2]
            fig, ax = plt.subplots(figsize=(6, 5))
            labels = wide.index.get_level_values("model").to_list()
            layer_ids = wide.index.get_level_values("layer").to_list()
            for lbl in cand_labels:
                mask = [m == lbl for m in labels]
                if not any(mask):
                    continue
                ax.scatter(coords[mask, 0], coords[mask, 1], s=36,
                           color=color_map[lbl], label=lbl,
                           edgecolor="white", linewidth=0.6, alpha=0.9)
                for (xi, yi, li) in zip(coords[mask, 0], coords[mask, 1],
                                        [l for l, m in zip(layer_ids, mask) if m]):
                    ax.annotate(str(li), (xi, yi), fontsize=6, color="black",
                                xytext=(2, 2), textcoords="offset points")
            ax.set_xlabel(f"PC1 ({S[0] ** 2 / (S ** 2).sum():.1%})")
            ax.set_ylabel(f"PC2 ({S[1] ** 2 / (S ** 2).sum():.1%})")
            ax.set_title("PCA of per-layer fingerprint vectors")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plot_dir / "08_pca_fingerprint.png", dpi=140)
            plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-model-path", required=True,
                   help="HF-format directory containing the base checkpoint.")
    p.add_argument("--candidate-model", action="append", default=[], required=True,
                   help='Repeatable. Format: "label=path". E.g. "snmf=/path/to/snmf".')
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layers", default="all",
                   help='Layer spec: "all", "10-18", "0,3,5", "0-5,10".')
    p.add_argument("--projections", nargs="+", default=list(ALL_PROJECTIONS),
                   choices=list(ALL_PROJECTIONS),
                   help="Which weight matrices per layer to analyse.")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--spectra-layers", default="",
                   help="Comma-separated layers to plot full SVD spectra for. "
                        "Default: ~4 evenly-spaced layers.")
    p.add_argument("--no-save-spectra-npz", action="store_true",
                   help="Skip the (potentially large) singular_values.npz dump.")
    args = p.parse_args(argv)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU")
        device = "cpu"
    print(f"[info] device={device}")

    candidates = _parse_models(args.candidate_model)
    spectra_layers = [int(x) for x in args.spectra_layers.split(",") if x.strip()]
    output_dir = Path(args.output_dir)

    analyze(
        base_path=args.base_model_path,
        candidates=candidates,
        output_dir=output_dir,
        layers_spec=args.layers,
        projections=args.projections,
        device=device,
        spectra_layers=spectra_layers,
        save_full_spectra=not args.no_save_spectra_npz,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
