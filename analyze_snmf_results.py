import argparse
import json
import torch
from pathlib import Path

from llm_utils.utils import resolve_device, set_seed, sorted_numeric_layer_dirs
from llm_utils.model_utils import load_local_model
from collections import Counter

from supervised_analysis import (
    ROLE_LABEL_MEANINGS,
    ROLE_LABEL_ORDER,
    analyze_features_supervised,
    plot_layer_concept_trends,
)

ANALYSIS_OVERVIEW = (
    "Each SNMF column is one latent feature. Supervised profiling compares mean peak "
    "activation per prompt across label groups (neutral vs mult vs div), then assigns "
    "a role_label when log-ratios exceed --role-assignment-threshold. "
    "Counts below are how many latents received each label, per layer and in total."
)


def main():
    parser = argparse.ArgumentParser(description="Analyze pre-trained SNMF results.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True, help="Path to folder containing layer_X subfolders")
    parser.add_argument(
        "--role-assignment-threshold",
        type=float,
        default=0.15,
        metavar="LOG_RATIO",
        help="Minimum log-ratio (natural log) to assign a strong role_label vs weak_mixed / neutral_lean.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--activation-context-top-n",
        type=int,
        default=10,
        help="Per latent: log this many max- and min-activation token contexts in supervised JSON.",
    )
    parser.add_argument(
        "--activation-context-window",
        type=int,
        default=15,
        help="Tokens before/after the peak token in each logged context (same sample only).",
    )
    parser.add_argument(
        "--summary-filename",
        type=str,
        default="analysis_summary.json",
        help="Written under --results-dir: feature counts, role breakdown, and role meanings.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"Loading model from {args.model_path}...")
    local_model = load_local_model(args.model_path, device=device)

    per_layer_stats: list[dict] = []
    global_role_counts: Counter[str] = Counter()

    for layer_num, layer_folder in sorted_numeric_layer_dirs(results_dir):
        factors_path = layer_folder / "snmf_factors.pt"

        if not factors_path.exists():
            print(f"Skipping layer {layer_num} because snmf_factors.pt is missing.")
            continue

        output_json_path = layer_folder / "feature_analysis_supervised.json"

        print(f"\nProcessing {layer_folder.name}...")
        checkpoint = torch.load(factors_path, map_location="cpu", weights_only=False)
        F, G = checkpoint['F'], checkpoint['G']
        token_ids, sample_ids = checkpoint['token_ids'], checkpoint['sample_ids']
        labels, mode = checkpoint['labels'], checkpoint.get('mode', 'mlp_intermediate')

        # 1. Base statistical profiling
        supervised_results = analyze_features_supervised(
            G,
            labels,
            sample_ids,
            token_ids,
            local_model.tokenizer,
            role_assignment_threshold=args.role_assignment_threshold,
            context_top_n=args.activation_context_top_n,
            context_window=args.activation_context_window,
        )

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(supervised_results, f, indent=2, ensure_ascii=False)

        n_features = len(supervised_results)
        layer_roles = Counter(
            supervised_results[k].get("role_label", "unknown") for k in supervised_results
        )
        global_role_counts.update(layer_roles)
        per_layer_stats.append(
            {
                "layer": layer_num,
                "features_explored": n_features,
                "counts_by_role": dict(layer_roles),
            }
        )
        print(
            f"  Layer {layer_num}: {n_features} latents | roles: "
            + ", ".join(f"{r}={c}" for r, c in sorted(layer_roles.items()))
        )

    print("\nGenerating model-wide trend plots...")
    try:
        plot_layer_concept_trends(args.results_dir)
    except Exception as e:
        print(f"Could not generate plots: {e}")

    total_features = sum(s["features_explored"] for s in per_layer_stats)
    summary_path = results_dir / args.summary_filename
    ordered_global: dict[str, int] = {r: global_role_counts.get(r, 0) for r in ROLE_LABEL_ORDER}
    for label, c in global_role_counts.items():
        if label not in ordered_global:
            ordered_global[label] = c

    summary_doc = {
        "overview": ANALYSIS_OVERVIEW,
        "role_assignment_threshold": args.role_assignment_threshold,
        "threshold_note": (
            "Minimum natural-log ratio margin for a strong role vs weak_mixed / neutral_lean; "
            "see supervised_analysis._assign_role_label."
        ),
        "total_features_explored": total_features,
        "layers_processed": len(per_layer_stats),
        "global_counts_by_role": ordered_global,
        "per_layer": per_layer_stats,
        "role_meanings": ROLE_LABEL_MEANINGS,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_doc, f, indent=2, ensure_ascii=False)

    print("\n--- Global role counts (all layers) ---")
    for r in ROLE_LABEL_ORDER:
        c = global_role_counts.get(r, 0)
        if c:
            print(f"  {r}: {c}")
    for r, c in sorted(global_role_counts.items()):
        if r not in ROLE_LABEL_ORDER:
            print(f"  {r}: {c}")

    print(f"\nTotal latents profiled: {total_features}")
    print(f"Wrote summary config: {summary_path}")
    print(f"\nAnalysis complete. Files saved in {args.results_dir}")


if __name__ == "__main__":
    main()
