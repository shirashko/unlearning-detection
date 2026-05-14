"""
Analyze SNMF checkpoints trained on WMDP-bio supervision (e.g. data/bio_data.json:
bio_forget vs retain buckets). Unary per-latent log-ratios.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
import torch

from llm_utils.model_utils import load_local_model
from llm_utils.utils import (
    resolve_absolute_path,
    resolve_device,
    set_seed,
    sorted_numeric_layer_dirs,
    verify_checkpoint_data_path,
)
from wmdp_bio_supervised_analysis import (
    RETAIN_BASIS_BIO_RETAIN,
    RETAIN_BASIS_CHOICES,
    RETAIN_BASIS_NEUTRAL,
    RETAIN_BASIS_POOLED,
    ROLE_LABEL_MEANINGS,
    ROLE_LABEL_ORDER,
    analyze_features_supervised_wmdp_bio,
    plot_layer_wmdp_bio_trends,
)

ANALYSIS_OVERVIEW = (
    "Each SNMF column represents one latent feature. The analysis profiles each latent by "
    "calculating a mathematical score (log-ratio) that measures how much its activation "
    "differs between 'dangerous' (bio_forget) and 'safe' (retain) data. "
    "These log-ratios compare the bio_forget group against three distinct benchmarks:\n"
    "1. Pooled Retain: A combination of neutral and bio-retain data.\n"
    "2. Neutral Only: General domain data (e.g., Wikipedia).\n"
    "3. Bio-Retain Only: Safe biological data.\n\n"
    "Role labels are assigned independently for each benchmark (pooled, neutral, bio_retain), "
    "so every latent gets a per-basis role map. Finally, the script aggregates counts by basis "
    "per layer and globally, letting downstream analysis choose which basis to use."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SNMF results (WMDP-bio / bio_data.json).")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Folder with layer_X/snmf_factors.pt from train_snmf.py",
    )
    parser.add_argument(
        "--role-assignment-threshold",
        type=float,
        required=True,
        metavar="LOG_RATIO",
        help="Minimum |log(mean_forget/mean_retain_side)| margin for role assignment.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--activation-context-top-n",
        type=int,
        default=10,
        help="Per latent: max/min activation contexts to log in supervised JSON.",
    )
    parser.add_argument(
        "--activation-context-window",
        type=int,
        default=15,
        help="Tokens before/after peak token in each context (same sample only).",
    )
    parser.add_argument(
        "--summary-filename",
        type=str,
        default="analysis_summary_wmdp_bio.json",
        help="Written under --results-dir (global role counts and meanings).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help=(
            "Expected training data path for consistency check against checkpoint "
            "config['data_path'] in each layer."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)
    expected_data_path = resolve_absolute_path(args.data_path)

    print(f"Loading model from {args.model_path}...")
    local_model = load_local_model(args.model_path, device=device)

    per_layer_stats: list[dict] = []
    basis_order = [RETAIN_BASIS_POOLED, RETAIN_BASIS_NEUTRAL, RETAIN_BASIS_BIO_RETAIN]
    global_role_counts_by_basis: dict[str, Counter[str]] = {
        basis: Counter() for basis in basis_order
    }

    for layer_num, layer_folder in sorted_numeric_layer_dirs(results_dir):
        factors_path = layer_folder / "snmf_factors.pt"
        if not factors_path.exists():
            print(f"Skipping layer {layer_num} because snmf_factors.pt is missing.")
            continue

        supervised_path = layer_folder / "feature_analysis_supervised_wmdp_bio.json"
        print(f"\nProcessing {layer_folder.name}...")

        checkpoint = torch.load(factors_path, map_location="cpu", weights_only=False)
        F, G = checkpoint["F"], checkpoint["G"]
        token_ids, sample_ids = checkpoint["token_ids"], checkpoint["sample_ids"]
        labels, mode = checkpoint["labels"], checkpoint.get("mode", "mlp_intermediate")

        verify_checkpoint_data_path(
            checkpoint=checkpoint,
            expected_data_path=expected_data_path,
            layer_num=layer_num,
        )

        supervised_results = analyze_features_supervised_wmdp_bio(
            G,
            labels,
            sample_ids,
            token_ids,
            local_model.tokenizer,
            role_assignment_threshold=args.role_assignment_threshold,
            context_top_n=args.activation_context_top_n,
            context_window=args.activation_context_window,
        )

        with open(supervised_path, "w", encoding="utf-8") as f:
            json.dump(supervised_results, f, indent=2, ensure_ascii=False)

        n_features = len(supervised_results)
        layer_counts_by_basis: dict[str, dict[str, int]] = {}
        for basis in basis_order:
            layer_roles = Counter(
                supervised_results[k]
                .get("role_labels_by_basis", {})
                .get(basis, supervised_results[k].get("role_label", "unknown"))
                for k in supervised_results
            )
            layer_counts_by_basis[basis] = dict(layer_roles)
            global_role_counts_by_basis[basis].update(layer_roles)
        layer_entry: dict = {
            "layer": layer_num,
            "features_explored": n_features,
            "counts_by_role_by_basis": layer_counts_by_basis,
        }
        per_layer_stats.append(layer_entry)
        layer_print_chunks = []
        for basis in basis_order:
            basis_counts = layer_counts_by_basis[basis]
            compact = ", ".join(f"{r}={c}" for r, c in sorted(basis_counts.items()))
            layer_print_chunks.append(f"{basis}: {compact}")
        print(f"  Layer {layer_num}: {n_features} latents | " + " | ".join(layer_print_chunks))

    print("\nGenerating WMDP-bio trend plots (one PNG per retain basis)...")
    for basis in sorted(RETAIN_BASIS_CHOICES):
        try:
            plot_layer_wmdp_bio_trends(str(results_dir), retain_basis=basis)
        except Exception as e:
            print(f"Could not generate plot for basis={basis}: {e}")

    total_features = sum(s["features_explored"] for s in per_layer_stats)
    summary_path = results_dir / args.summary_filename
    ordered_global_by_basis: dict[str, dict[str, int]] = {}
    for basis in basis_order:
        basis_counts = global_role_counts_by_basis[basis]
        ordered_global = {r: basis_counts.get(r, 0) for r in ROLE_LABEL_ORDER}
        for label, c in basis_counts.items():
            if label not in ordered_global:
                ordered_global[label] = c
        ordered_global_by_basis[basis] = ordered_global

    summary_doc = {
        "overview": ANALYSIS_OVERVIEW,
        "pipeline": "wmdp_bio",
        "retain_basis_note": (
            "Per-latent JSON includes role_labels_by_basis for pooled / neutral / bio_retain, "
            "plus log_forget_vs_pooled_retain, log_forget_vs_neutral, "
            "log_forget_vs_bio_retain, and log_bio_retain_vs_neutral when counts allow."
        ),
        "role_assignment_threshold": args.role_assignment_threshold,
        "data_path_verification": {
            "enabled": True,
            "expected_data_path": str(expected_data_path),
        },
        "threshold_note": (
            "Minimum natural-log ratio margin for bio_forget_lean vs retain_lean vs weak_mixed; "
            "see wmdp_bio_supervised_analysis._assign_role_label_bio."
        ),
        "total_features_explored": total_features,
        "layers_processed": len(per_layer_stats),
        "global_counts_by_role_by_basis": ordered_global_by_basis,
        "per_layer": per_layer_stats,
        "role_meanings": ROLE_LABEL_MEANINGS,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_doc, f, indent=2, ensure_ascii=False)

    print("\n--- Global role counts (WMDP-bio, all layers) ---")
    for basis in basis_order:
        print(f"  Basis={basis}")
        basis_counts = global_role_counts_by_basis[basis]
        for r in ROLE_LABEL_ORDER:
            c = basis_counts.get(r, 0)
            if c:
                print(f"    {r}: {c}")
        for r, c in sorted(basis_counts.items()):
            if r not in ROLE_LABEL_ORDER:
                print(f"    {r}: {c}")

    print(f"\nTotal latents profiled: {total_features}")
    print(f"Wrote summary: {summary_path}")
    print(f"\nAnalysis complete. Outputs use *_wmdp_bio.json suffixes under {args.results_dir}")


if __name__ == "__main__":
    main()
