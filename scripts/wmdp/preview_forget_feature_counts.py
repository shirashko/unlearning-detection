"""
Preview how many SNMF latents would be selected for forget-ablation per layer,
under a given ROLE_ASSIGNMENT_THRESHOLD + ROLE_LABEL_BASES + ROLE_BASIS_COMBINE,
without loading the model or editing any weights.

Re-uses the exact selection logic from ``create_forget_ablated_model.py`` so the
numbers match what that script would actually ablate.

Examples (run from repo root):
  python scripts/wmdp/preview_forget_feature_counts.py \
      --results-dir outputs/wmdp/results_data_part1_gemma2_2b \
      --role-label-bases pooled bio_retain \
      --role-basis-combine all \
      --role-assignment-threshold 0.30

  # Sweep a few thresholds at once
  python scripts/wmdp/preview_forget_feature_counts.py \
      --results-dir outputs/wmdp/results_data_part1_gemma2_2b \
      --role-label-bases pooled bio_retain \
      --role-basis-combine all \
      --threshold-sweep 0.15 0.30 0.50 0.69 1.00
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from create_forget_ablated_model import (  # noqa: E402
    _latent_matches_forget_roles,
    _load_supervised_profiles,
)
from llm_utils.utils import sorted_numeric_layer_dirs  # noqa: E402


def _count_selected_for_layer(
    layer_idx: int,
    layer_dir: Path,
    supervised_json_filename: str,
    forget_roles: Set[str],
    role_label_bases: Optional[Sequence[str]],
    role_basis_combine: str,
    role_assignment_threshold: Optional[float],
) -> tuple[int, int]:
    """Returns (n_selected, n_total_profiles) for a single layer."""
    profiles = _load_supervised_profiles(layer_dir, supervised_json_filename)
    n_total = len(profiles)
    n_sel = sum(
        1
        for prof in profiles.values()
        if _latent_matches_forget_roles(
            prof,
            forget_roles,
            role_label_bases,
            role_basis_combine,
            role_assignment_threshold=role_assignment_threshold,
        )
    )
    return n_sel, n_total


def _preview_one_config(
    layers: List[tuple[int, Path]],
    supervised_json_filename: str,
    forget_roles: Set[str],
    role_label_bases: Optional[Sequence[str]],
    role_basis_combine: str,
    threshold: Optional[float],
) -> None:
    bases_label = (
        "legacy role_label"
        if not role_label_bases
        else f"{role_basis_combine.upper()}({', '.join(role_label_bases)})"
    )
    thr_label = "stored" if threshold is None else f"{threshold:.3f}"
    print(f"\n=== bases={bases_label}  threshold={thr_label}  forget_roles={sorted(forget_roles)} ===")
    print(f"{'layer':>6}  {'selected':>9}  {'total':>6}  {'pct':>6}")
    total_sel = 0
    total_all = 0
    for layer_idx, layer_dir in layers:
        try:
            n_sel, n_total = _count_selected_for_layer(
                layer_idx,
                layer_dir,
                supervised_json_filename,
                forget_roles,
                role_label_bases,
                role_basis_combine,
                threshold,
            )
        except FileNotFoundError as err:
            print(f"  layer_{layer_idx}: SKIP ({err})")
            continue
        total_sel += n_sel
        total_all += n_total
        pct = (100.0 * n_sel / n_total) if n_total else 0.0
        print(f"  {layer_idx:>6}  {n_sel:>9}  {n_total:>6}  {pct:>5.1f}%")
    pct_all = (100.0 * total_sel / total_all) if total_all else 0.0
    print(f"  {'TOTAL':>6}  {total_sel:>9}  {total_all:>6}  {pct_all:>5.1f}%")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/wmdp/results_data_part1_gemma2_2b"),
        help="Directory containing layer_*/ with supervised JSONs (same as RESULTS_DIR in the shell script).",
    )
    p.add_argument(
        "--supervised-json-filename",
        default="feature_analysis_supervised_wmdp_bio.json",
    )
    p.add_argument(
        "--forget-roles",
        nargs="+",
        default=["bio_forget_lean"],
        help="Role strings treated as 'forget' (space-separated).",
    )
    p.add_argument(
        "--role-label-bases",
        nargs="*",
        default=["pooled", "bio_retain"],
        choices=["pooled", "neutral", "bio_retain"],
        help="Bases to require (subset of pooled/neutral/bio_retain). Empty = legacy role_label only.",
    )
    p.add_argument(
        "--role-basis-combine",
        choices=["all", "any"],
        default="all",
        help="Combine across bases with AND (all) or OR (any).",
    )
    p.add_argument(
        "--role-assignment-threshold",
        type=float,
        default=None,
        help="Override threshold on |log_forget_vs_retain|. Omit to use labels stored in JSON.",
    )
    p.add_argument(
        "--threshold-sweep",
        type=float,
        nargs="+",
        default=None,
        help="If given, ignore --role-assignment-threshold and print one table per threshold in this list.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.exists():
        raise SystemExit(f"Results dir not found: {results_dir}")

    layers = sorted_numeric_layer_dirs(results_dir)
    if not layers:
        raise SystemExit(f"No layer_* subdirectories under {results_dir}")

    print(f"Results dir: {results_dir}")
    print(f"Supervised JSON filename: {args.supervised_json_filename}")
    print(f"Layers found: {len(layers)} (layer_{layers[0][0]} .. layer_{layers[-1][0]})")

    forget_roles: Set[str] = set(args.forget_roles)
    bases: Optional[List[str]] = list(args.role_label_bases) if args.role_label_bases else None

    thresholds: List[Optional[float]]
    if args.threshold_sweep is not None:
        thresholds = [float(t) for t in args.threshold_sweep]
    else:
        thresholds = [args.role_assignment_threshold]

    for thr in thresholds:
        _preview_one_config(
            layers,
            args.supervised_json_filename,
            forget_roles,
            bases,
            args.role_basis_combine,
            thr,
        )


if __name__ == "__main__":
    main()
