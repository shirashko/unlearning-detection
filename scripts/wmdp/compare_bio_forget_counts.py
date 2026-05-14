#!/usr/bin/env python3
"""
Compare the number of `bio_forget_lean` SNMF latents across iterations of
WMDP-bio forgetting pipelines, so we can see whether each iteration has fewer
forget-specific features left to ablate (i.e. whether forgetting is actually
reducing the bio-forget subspace).

Inputs per iteration: an SNMF results directory produced by `train_snmf.py` +
`wmdp_bio_analyze_snmf_results.py`. Required files under that dir:
  - analysis_summary_wmdp_bio.json              (per-layer counts @ analysis thr)
  - layer_<i>/feature_analysis_supervised_wmdp_bio.json (per-latent raw stats)

Each iteration can be pinned to a different "analysis threshold" (usually 0.05)
and to a different "ablation threshold" used at create_forget_ablated_model.py
time. This tool always reports BOTH:
  1. Stored counts read directly from analysis_summary (their own analysis thr).
  2. Recomputed counts at a SHARED --comparison-threshold, for apples-to-apples
     comparison across iterations — this is the one you want for trend analysis.
  3. The AND combination of two bases at --comparison-threshold (by default
     bio_retain AND neutral — this is the selection rule used by iter-2 top-up
     and iter-3).

Outputs (in --output-dir):
  - bio_forget_counts.csv      (long-format: iteration, layer, basis, n_*, ...)
  - bio_forget_counts_summary.md   (one compact table per iteration + a cross-
                                    iteration rollup — easy to paste into notes)
  - bio_forget_counts_by_layer.png (optional plot if matplotlib is available)

Usage (CLI):

    python scripts/wmdp/compare_bio_forget_counts.py \
      --iteration iter1:outputs/wmdp/results_data_part1_gemma2_2b \
      --iteration iter2:outputs/wmdp/results_data_part2_gemma2_2b_iter2_thr022_down_proj_only \
      --iteration iter3:outputs/wmdp/results_data_part3_gemma2_2b_iter3_thr018_both_up_down \
      --comparison-threshold 0.30 \
      --output-dir outputs/wmdp/iteration_comparison

Notes:
  - "iteration" here means "SNMF basis trained on top of some checkpoint". The
    order you pass iterations is the one used in the output tables.
  - The comparison threshold does NOT change anything on disk — it just re-reads
    the raw `log_ratios` field of every latent and re-applies the bio_forget_lean
    rule (`log_forget_vs_<basis> >= threshold`).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Map basis -> (log_ratio key, "nice" display name)
BASIS_KEYS: Dict[str, str] = {
    "pooled": "log_forget_vs_pooled_retain",
    "neutral": "log_forget_vs_neutral",
    "bio_retain": "log_forget_vs_bio_retain",
}
# The counts-file group key that pairs each basis with its retain group size.
BASIS_GROUP: Dict[str, str] = {
    "pooled": "pooled_retain",
    "neutral": "neutral",
    "bio_retain": "bio_retain",
}


@dataclass
class IterationSpec:
    name: str
    snmf_dir: Path

    @classmethod
    def parse(cls, raw: str) -> "IterationSpec":
        if ":" not in raw:
            raise ValueError(
                f"--iteration value {raw!r} must be of the form 'name:path/to/snmf_dir'"
            )
        name, path = raw.split(":", 1)
        name = name.strip()
        p = Path(path.strip())
        if not p.is_dir():
            raise FileNotFoundError(f"SNMF dir for iteration {name!r} does not exist: {p}")
        return cls(name=name, snmf_dir=p)


@dataclass
class LayerCounts:
    """Counts of bio_forget_lean latents for one (iteration, layer) cell."""
    layer: int
    n_latents: int                          # total SNMF columns in this layer
    stored_by_basis: Dict[str, int]         # from analysis_summary (their own thr)
    stored_threshold: float                 # that iteration's analysis threshold
    recomputed_by_basis: Dict[str, int]     # from raw log_ratios at compare thr
    recomputed_and: int                     # AND of the two chosen bases at compare thr
    compare_threshold: float
    and_bases: Tuple[str, str]


# ---- Loaders ---------------------------------------------------------------

def _load_analysis_summary(snmf_dir: Path) -> Dict:
    summary_path = snmf_dir / "analysis_summary_wmdp_bio.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing analysis_summary_wmdp_bio.json under {snmf_dir}. "
            "Run wmdp_bio_analyze_snmf_results.py first."
        )
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_per_layer_supervised(snmf_dir: Path, layer: int) -> Optional[Dict[str, Dict]]:
    """Return the per-latent dict for a given layer (or None if missing)."""
    p = snmf_dir / f"layer_{layer}" / "feature_analysis_supervised_wmdp_bio.json"
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


# ---- Core counting ---------------------------------------------------------

def _latent_is_bio_forget_lean_at(
    profile: Dict, basis: str, threshold: float
) -> Optional[bool]:
    """
    Apply the `bio_forget_lean` rule for the requested basis at `threshold`,
    mirroring create_forget_ablated_model._assign_role_label_bio. Returns None
    when the profile lacks the required fields.
    """
    log_ratios = profile.get("log_ratios") or {}
    counts = profile.get("group_counts") or {}
    means = profile.get("group_means") or {}

    key = BASIS_KEYS[basis]
    grp = BASIS_GROUP[basis]

    log_fr = log_ratios.get(key)
    if log_fr is None:
        return None
    n_forget = int(counts.get("bio_forget", 0))
    n_retain = int(counts.get(grp, 0))
    if n_forget == 0 or n_retain == 0:
        return False
    mean_f = float(means.get("bio_forget", 0.0))
    mean_r = float(means.get(grp, 0.0))
    if mean_f + mean_r < 1e-9:
        return False
    return float(log_fr) >= float(threshold)


def _compute_layer_counts(
    snmf_dir: Path,
    layer_record: Dict,
    stored_threshold: float,
    compare_threshold: float,
    and_bases: Tuple[str, str],
) -> LayerCounts:
    layer = int(layer_record["layer"])
    n_latents = int(layer_record.get("features_explored", 0))

    stored_by_basis: Dict[str, int] = {}
    for basis, counts in (layer_record.get("counts_by_role_by_basis") or {}).items():
        if basis in BASIS_KEYS:
            stored_by_basis[basis] = int((counts or {}).get("bio_forget_lean", 0))

    profiles = _load_per_layer_supervised(snmf_dir, layer)
    recomputed_by_basis: Dict[str, int] = {b: 0 for b in BASIS_KEYS}
    and_count = 0
    if profiles:
        b1, b2 = and_bases
        for prof in profiles.values():
            for basis in BASIS_KEYS:
                hit = _latent_is_bio_forget_lean_at(prof, basis, compare_threshold)
                if hit:
                    recomputed_by_basis[basis] += 1
            hit1 = _latent_is_bio_forget_lean_at(prof, b1, compare_threshold)
            hit2 = _latent_is_bio_forget_lean_at(prof, b2, compare_threshold)
            if hit1 and hit2:
                and_count += 1

    return LayerCounts(
        layer=layer,
        n_latents=n_latents,
        stored_by_basis=stored_by_basis,
        stored_threshold=stored_threshold,
        recomputed_by_basis=recomputed_by_basis,
        recomputed_and=and_count,
        compare_threshold=compare_threshold,
        and_bases=and_bases,
    )


def _compute_iteration(
    spec: IterationSpec,
    compare_threshold: float,
    and_bases: Tuple[str, str],
) -> Tuple[Dict, List[LayerCounts]]:
    summary = _load_analysis_summary(spec.snmf_dir)
    stored_thr = float(summary.get("role_assignment_threshold", 0.05))
    per_layer: List[LayerCounts] = []
    for rec in summary.get("per_layer") or []:
        per_layer.append(
            _compute_layer_counts(
                spec.snmf_dir,
                rec,
                stored_threshold=stored_thr,
                compare_threshold=compare_threshold,
                and_bases=and_bases,
            )
        )
    per_layer.sort(key=lambda lc: lc.layer)
    meta = {
        "iteration": spec.name,
        "snmf_dir": str(spec.snmf_dir),
        "stored_threshold": stored_thr,
        "layers_processed": int(summary.get("layers_processed", len(per_layer))),
        "total_features_explored": int(summary.get("total_features_explored", 0)),
    }
    return meta, per_layer


# ---- Output formatters -----------------------------------------------------

def _write_csv(
    out_csv: Path,
    rows: List[Dict],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _iter_rows(
    iter_results: List[Tuple[Dict, List[LayerCounts]]],
    compare_threshold: float,
    and_bases: Tuple[str, str],
) -> List[Dict]:
    rows: List[Dict] = []
    b1, b2 = and_bases
    for meta, per_layer in iter_results:
        for lc in per_layer:
            rows.append(
                {
                    "iteration": meta["iteration"],
                    "snmf_dir": meta["snmf_dir"],
                    "layer": lc.layer,
                    "n_latents": lc.n_latents,
                    "stored_threshold": lc.stored_threshold,
                    "stored_bio_forget_pooled": lc.stored_by_basis.get("pooled", 0),
                    "stored_bio_forget_neutral": lc.stored_by_basis.get("neutral", 0),
                    "stored_bio_forget_bio_retain": lc.stored_by_basis.get("bio_retain", 0),
                    "compare_threshold": compare_threshold,
                    "recomputed_bio_forget_pooled": lc.recomputed_by_basis["pooled"],
                    "recomputed_bio_forget_neutral": lc.recomputed_by_basis["neutral"],
                    "recomputed_bio_forget_bio_retain": lc.recomputed_by_basis["bio_retain"],
                    f"recomputed_AND_{b1}_AND_{b2}": lc.recomputed_and,
                }
            )
    return rows


def _write_markdown(
    out_md: Path,
    iter_results: List[Tuple[Dict, List[LayerCounts]]],
    compare_threshold: float,
    and_bases: Tuple[str, str],
) -> None:
    b1, b2 = and_bases
    lines: List[str] = []
    lines.append("# Bio-forget latent counts across iterations\n")
    lines.append(
        "Numbers are counts of SNMF latents classified as `bio_forget_lean` per layer. "
        "Each iteration's SNMF basis is fit on a different model checkpoint (iter-1 on the "
        "original Gemma, iter-2 on the iter-1 ablated ckpt, iter-3 on the iter-2 top-up ckpt), "
        "so a downward trend across iterations means the forget-specific subspace left in the "
        "model is shrinking, which is what we want.\n"
    )
    lines.append(
        f"`compare_threshold = {compare_threshold}` is applied uniformly to every iteration "
        f"from the raw `log_ratios` fields; the AND column requires "
        f"log_forget_vs_{b1} >= thr AND log_forget_vs_{b2} >= thr (same rule as iter-2 top-up / iter-3).\n"
    )

    # Per-iteration per-layer table
    lines.append("## Per-layer bio_forget_lean counts (recomputed @ compare_threshold)\n")
    header_cols = ["layer", "n_latents"] + [
        f"{m.get('iteration')}_{b}" for m, _ in iter_results for b in ("pooled", "neutral", "bio_retain", f"AND_{b1}_{b2}")
    ]
    # Simpler layout: one block per iteration.
    for meta, per_layer in iter_results:
        lines.append(f"### {meta['iteration']} — {meta['snmf_dir']}\n")
        lines.append(
            f"analysis threshold = {meta['stored_threshold']} · layers = "
            f"{meta['layers_processed']} · total SNMF latents = {meta['total_features_explored']}\n"
        )
        lines.append(
            f"| layer | n_latents | stored@{meta['stored_threshold']}: pooled / neutral / bio_retain | "
            f"@{compare_threshold}: pooled / neutral / bio_retain | AND({b1}∧{b2})@{compare_threshold} |"
        )
        lines.append("|---:|---:|---|---|---:|")
        tot_stored = {k: 0 for k in BASIS_KEYS}
        tot_reco = {k: 0 for k in BASIS_KEYS}
        tot_and = 0
        for lc in per_layer:
            stored_str = (
                f"{lc.stored_by_basis.get('pooled', 0)} / "
                f"{lc.stored_by_basis.get('neutral', 0)} / "
                f"{lc.stored_by_basis.get('bio_retain', 0)}"
            )
            reco_str = (
                f"{lc.recomputed_by_basis['pooled']} / "
                f"{lc.recomputed_by_basis['neutral']} / "
                f"{lc.recomputed_by_basis['bio_retain']}"
            )
            lines.append(
                f"| {lc.layer} | {lc.n_latents} | {stored_str} | {reco_str} | {lc.recomputed_and} |"
            )
            for k in BASIS_KEYS:
                tot_stored[k] += lc.stored_by_basis.get(k, 0)
                tot_reco[k] += lc.recomputed_by_basis[k]
            tot_and += lc.recomputed_and
        tot_stored_str = f"{tot_stored['pooled']} / {tot_stored['neutral']} / {tot_stored['bio_retain']}"
        tot_reco_str = f"{tot_reco['pooled']} / {tot_reco['neutral']} / {tot_reco['bio_retain']}"
        lines.append(
            f"| **TOTAL** | {sum(lc.n_latents for lc in per_layer)} | "
            f"**{tot_stored_str}** | **{tot_reco_str}** | **{tot_and}** |\n"
        )

    # Cross-iteration rollup
    lines.append("## Cross-iteration totals (TL;DR — the trend)\n")
    lines.append(
        f"| iteration | stored@own_thr: pooled / neutral / bio_retain | "
        f"@{compare_threshold}: pooled / neutral / bio_retain | AND({b1}∧{b2})@{compare_threshold} |"
    )
    lines.append("|---|---|---|---:|")
    for meta, per_layer in iter_results:
        ts = {k: 0 for k in BASIS_KEYS}
        tr = {k: 0 for k in BASIS_KEYS}
        ta = 0
        for lc in per_layer:
            for k in BASIS_KEYS:
                ts[k] += lc.stored_by_basis.get(k, 0)
                tr[k] += lc.recomputed_by_basis[k]
            ta += lc.recomputed_and
        lines.append(
            f"| {meta['iteration']} (thr={meta['stored_threshold']}) | "
            f"{ts['pooled']} / {ts['neutral']} / {ts['bio_retain']} | "
            f"{tr['pooled']} / {tr['neutral']} / {tr['bio_retain']} | {ta} |"
        )
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _try_plot(
    out_png: Path,
    iter_results: List[Tuple[Dict, List[LayerCounts]]],
    compare_threshold: float,
    and_bases: Tuple[str, str],
) -> Optional[str]:
    """
    Plot `bio_forget_lean` counts per layer, one line per iteration, at the
    comparison threshold (AND combo — this is the one that matters for ablation
    selection). Returns path string on success, or None (with reason) on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        return f"matplotlib unavailable: {e}"

    b1, b2 = and_bases
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for meta, per_layer in iter_results:
        xs = [lc.layer for lc in per_layer]
        ys_pooled = [lc.recomputed_by_basis["pooled"] for lc in per_layer]
        ys_and = [lc.recomputed_and for lc in per_layer]
        axes[0].plot(xs, ys_pooled, marker="o", label=meta["iteration"])
        axes[1].plot(xs, ys_and, marker="o", label=meta["iteration"])
    axes[0].set_title(f"bio_forget_lean (pooled retain basis) @ thr={compare_threshold}")
    axes[1].set_title(f"bio_forget_lean AND({b1} ∧ {b2}) @ thr={compare_threshold}")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.set_ylabel("# latents")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return str(out_png)


# ---- CLI -------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare bio_forget SNMF latent counts across iterations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--iteration",
        action="append",
        required=True,
        metavar="NAME:PATH",
        help="Iteration spec (repeatable). Example: "
        "'iter3:outputs/wmdp/results_data_part3_gemma2_2b_iter3_thr018_both_up_down'.",
    )
    p.add_argument(
        "--comparison-threshold",
        type=float,
        default=0.30,
        help="Shared threshold used to recompute bio_forget_lean counts "
        "across all iterations (apples-to-apples).",
    )
    p.add_argument(
        "--and-bases",
        default="bio_retain,neutral",
        help="Two bases (comma-separated) combined with AND for the "
        "selection-rule column; matches iter-2 top-up / iter-3 recipe.",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/wmdp/iteration_comparison",
        help="Directory where the CSV / Markdown / PNG are written.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the matplotlib plot (useful for CI / no-display envs).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    and_bases_list = [b.strip() for b in args.and_bases.split(",") if b.strip()]
    if len(and_bases_list) != 2 or any(b not in BASIS_KEYS for b in and_bases_list):
        raise SystemExit(
            f"--and-bases must be exactly two bases chosen from "
            f"{sorted(BASIS_KEYS)}, got {and_bases_list!r}"
        )
    and_bases: Tuple[str, str] = (and_bases_list[0], and_bases_list[1])

    specs = [IterationSpec.parse(x) for x in args.iteration]
    out_dir = Path(args.output_dir)

    iter_results: List[Tuple[Dict, List[LayerCounts]]] = []
    for spec in specs:
        meta, per_layer = _compute_iteration(
            spec,
            compare_threshold=args.comparison_threshold,
            and_bases=and_bases,
        )
        iter_results.append((meta, per_layer))

    rows = _iter_rows(iter_results, args.comparison_threshold, and_bases)
    csv_path = out_dir / "bio_forget_counts.csv"
    _write_csv(csv_path, rows)

    md_path = out_dir / "bio_forget_counts_summary.md"
    _write_markdown(md_path, iter_results, args.comparison_threshold, and_bases)

    png_msg = None
    if not args.no_plot:
        png_msg = _try_plot(
            out_dir / "bio_forget_counts_by_layer.png",
            iter_results,
            args.comparison_threshold,
            and_bases,
        )

    print(f"Wrote CSV:      {csv_path}")
    print(f"Wrote Markdown: {md_path}")
    if png_msg and png_msg.startswith("outputs/") or (png_msg and Path(png_msg).exists()):
        print(f"Wrote PNG:      {png_msg}")
    elif png_msg:
        print(f"Skipped PNG:    {png_msg}")


if __name__ == "__main__":
    main()
