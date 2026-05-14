"""
Per-layer L1-logistic probe on SNMF prompt-max activations for WMDP-bio.

Complements ``wmdp_bio_analyze_snmf_results.py`` (which computes per-latent,
independent log-ratio role labels). The probe is a MULTIVARIATE, joint signal:
it tells you which latents the model's forget-vs-retain signal is *decodable*
from when latents are considered together, with coefficient magnitudes giving
a calibrated per-latent importance — no hand-picked threshold.

Outputs, one JSON per layer, written into ``<results-dir>/layer_<i>/``:

  ``probe_weights_wmdp_bio.json``::

      {
        "metadata": {
          "probe_type": "logistic_l1_liblinear",
          "feature_aggregation": "prompt_max",
          "standardize": true,
          "C_grid": [0.01, 0.1, 1.0, 10.0],
          "C_selected": 1.0,
          "cv_accuracy_mean": 0.87,
          "cv_accuracy_std": 0.02,
          "test_accuracy": 0.85,
          "test_accuracy_baseline_majority": 0.667,
          "test_auroc": 0.93,
          "n_total_prompts_supervised": 1800,
          "n_forget": 600,
          "n_retain": 1200,
          "n_latents": 300,
          "n_nonzero_weights": 34,
          "random_state": 42,
          "layer": <i>
        },
        "weights_by_latent": {"0": 0.0, "1": 0.23, ...},
        "intercept": 0.15,
        "feature_means": {"0": 0.020, ...},
        "feature_stds": {"0": 0.004, ...}
      }

Plus a top-level ``<results-dir>/probe_summary_wmdp_bio.json`` aggregating
per-layer accuracy / sparsity for easy reading.

Usage::

    python wmdp_bio_probe_snmf_results.py \\
      --results-dir outputs/wmdp/results_data_part3_gemma2_2b_iter3_thr018_both_up_down \\
      --data-path data/bio_data_part3.json

No model load is required — the probe operates entirely on pre-computed SNMF
activations stored in ``snmf_factors.pt``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from llm_utils.utils import (
    resolve_absolute_path,
    set_seed,
    sorted_numeric_layer_dirs,
    verify_checkpoint_data_path,
)

# Prompt-label groups follow wmdp_bio_supervised_analysis.py conventions.
# Keep in sync with that module if label names change.
from wmdp_bio_supervised_analysis import (
    BIO_RETAIN_LABELS,
    FORGET_LABELS,
    NEUTRAL_LABELS,
    RETAIN_LABELS,
)

# sklearn is imported lazily inside main() so that importing this module in
# environments without sklearn doesn't fail at import time.


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L1-logistic probe per layer on SNMF activations (WMDP-bio).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", type=str, required=True,
                   help="SNMF output dir containing layer_<i>/snmf_factors.pt.")
    p.add_argument("--data-path", type=str, required=True,
                   help="Path that the SNMF factors were trained on (used for checkpoint data-path verification).")
    p.add_argument("--summary-filename", type=str,
                   default="probe_summary_wmdp_bio.json",
                   help="Top-level per-run summary JSON written under --results-dir.")
    p.add_argument("--weights-filename", type=str,
                   default="probe_weights_wmdp_bio.json",
                   help="Per-layer probe-weights JSON written under each layer_<i>/.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Held-out test fraction (stratified).")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Stratified CV folds used to pick the L1 strength C on the train split.")
    p.add_argument(
        "--c-grid",
        type=str,
        default="0.01,0.1,1.0,10.0",
        help="Comma-separated grid of inverse-regularization strengths for L1 logistic regression.",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="sklearn LogisticRegression max_iter.",
    )
    p.add_argument(
        "--feature-aggregation",
        type=str,
        default="prompt_max",
        choices=["prompt_max", "prompt_mean"],
        help=(
            "How to aggregate per-token SNMF activations into a single vector per prompt. "
            "'prompt_max' matches what the log-ratio analysis keys off of; 'prompt_mean' is "
            "a more-diluted alternative you can try for ablation ranking."
        ),
    )
    return p.parse_args()


def _build_prompt_matrix(
    G: torch.Tensor,
    sample_ids: np.ndarray,
    labels: List[str],
    feature_aggregation: str,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Aggregate per-token activations into per-prompt features for SUPERVISED prompts only.

    Returns
    -------
    X : (n_prompts_sup, n_latents) float64
    y : (n_prompts_sup,) int8 where 1 = bio_forget, 0 = retain (neutral ∪ bio_retain)
    kept_sample_ids : list[int] aligned with X rows
    """
    labels_arr = np.asarray(labels)
    sample_ids_arr = np.asarray(sample_ids, dtype=np.int64)
    G_np = G.detach().cpu().numpy().astype(np.float64, copy=False)

    # Collect contiguous per-prompt spans (sample_ids are contiguous runs).
    uniq_in_order: List[int] = []
    spans: Dict[int, Tuple[int, int]] = {}
    start = 0
    cur = int(sample_ids_arr[0])
    for i in range(1, sample_ids_arr.shape[0]):
        if int(sample_ids_arr[i]) != cur:
            spans[cur] = (start, i)
            uniq_in_order.append(cur)
            cur = int(sample_ids_arr[i])
            start = i
    spans[cur] = (start, sample_ids_arr.shape[0])
    uniq_in_order.append(cur)

    sup_set = FORGET_LABELS | RETAIN_LABELS
    kept_sample_ids: List[int] = []
    y_list: List[int] = []
    X_list: List[np.ndarray] = []
    for sid in uniq_in_order:
        lab = labels_arr[sid]
        if lab not in sup_set:
            continue
        s, e = spans[sid]
        seg = G_np[s:e, :]  # (T_sid, K)
        if feature_aggregation == "prompt_max":
            feat = seg.max(axis=0)
        else:  # prompt_mean
            feat = seg.mean(axis=0)
        X_list.append(feat)
        y_list.append(1 if lab in FORGET_LABELS else 0)
        kept_sample_ids.append(sid)

    X = np.asarray(X_list, dtype=np.float64) if X_list else np.zeros((0, G_np.shape[1]))
    y = np.asarray(y_list, dtype=np.int8)
    return X, y, kept_sample_ids


def _fit_probe(
    X: np.ndarray,
    y: np.ndarray,
    c_grid: List[float],
    cv_folds: int,
    max_iter: int,
    seed: int,
    test_size: float,
) -> Dict[str, Any]:
    """Train L1 logistic probe with stratified CV over C, evaluate on held-out split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    best_C = c_grid[0]
    best_cv_mean = -np.inf
    best_cv_std = 0.0
    cv_table: List[Dict[str, float]] = []
    for C in c_grid:
        accs = []
        for tr_idx, va_idx in skf.split(Xtr, y_train):
            clf = LogisticRegression(
                penalty="l1", solver="liblinear", C=C, max_iter=max_iter, random_state=seed
            )
            clf.fit(Xtr[tr_idx], y_train[tr_idx])
            accs.append(clf.score(Xtr[va_idx], y_train[va_idx]))
        mean = float(np.mean(accs))
        std = float(np.std(accs))
        cv_table.append({"C": float(C), "cv_accuracy_mean": mean, "cv_accuracy_std": std})
        if mean > best_cv_mean:
            best_cv_mean = mean
            best_cv_std = std
            best_C = float(C)

    # Refit on full train split with the selected C.
    final = LogisticRegression(
        penalty="l1", solver="liblinear", C=best_C, max_iter=max_iter, random_state=seed
    )
    final.fit(Xtr, y_train)
    w = final.coef_[0]
    intercept = float(final.intercept_[0])
    test_acc = float(final.score(Xte, y_test))
    # AUROC may fail if test split has single class (rare); guard it.
    try:
        test_auroc = float(roc_auc_score(y_test, final.decision_function(Xte)))
    except ValueError:
        test_auroc = float("nan")
    majority_acc = float(max(np.mean(y_test == 0), np.mean(y_test == 1)))

    return {
        "w": w,
        "intercept": intercept,
        "feature_means": scaler.mean_,
        "feature_stds": scaler.scale_,
        "C_selected": best_C,
        "cv_accuracy_mean": best_cv_mean,
        "cv_accuracy_std": best_cv_std,
        "cv_table": cv_table,
        "test_accuracy": test_acc,
        "test_accuracy_baseline_majority": majority_acc,
        "test_auroc": test_auroc,
        "n_nonzero_weights": int(np.count_nonzero(w)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    c_grid = [float(x) for x in args.c_grid.split(",") if x.strip()]
    if not c_grid:
        raise SystemExit("--c-grid must contain at least one value")

    results_dir = Path(args.results_dir)
    expected_data_path = resolve_absolute_path(args.data_path)

    per_layer_entries: List[Dict[str, Any]] = []
    for layer_num, layer_folder in sorted_numeric_layer_dirs(results_dir):
        factors_path = layer_folder / "snmf_factors.pt"
        if not factors_path.exists():
            print(f"[skip] layer {layer_num}: snmf_factors.pt missing")
            continue

        ckpt = torch.load(factors_path, map_location="cpu", weights_only=False)
        verify_checkpoint_data_path(
            checkpoint=ckpt,
            expected_data_path=expected_data_path,
            layer_num=layer_num,
        )
        G = ckpt["G"]
        if G.ndim != 2:
            print(f"[skip] layer {layer_num}: unexpected G shape {tuple(G.shape)}")
            continue
        sample_ids = np.asarray(ckpt["sample_ids"], dtype=np.int64)
        labels = list(ckpt["labels"])

        X, y, kept = _build_prompt_matrix(
            G=G,
            sample_ids=sample_ids,
            labels=labels,
            feature_aggregation=args.feature_aggregation,
        )
        n_prompts = int(X.shape[0])
        n_forget = int((y == 1).sum())
        n_retain = int((y == 0).sum())
        if n_prompts == 0 or n_forget < args.cv_folds or n_retain < args.cv_folds:
            print(
                f"[skip] layer {layer_num}: insufficient supervised prompts "
                f"(n_prompts={n_prompts}, n_forget={n_forget}, n_retain={n_retain})"
            )
            continue

        result = _fit_probe(
            X=X,
            y=y,
            c_grid=c_grid,
            cv_folds=args.cv_folds,
            max_iter=args.max_iter,
            seed=args.seed,
            test_size=args.test_size,
        )

        n_latents = int(X.shape[1])
        w = result["w"]
        label_counts = Counter(labels[sid] for sid in kept)

        weights_doc: Dict[str, Any] = {
            "metadata": {
                "probe_type": "logistic_l1_liblinear",
                "feature_aggregation": args.feature_aggregation,
                "standardize": True,
                "C_grid": c_grid,
                "C_selected": result["C_selected"],
                "cv_accuracy_mean": round(result["cv_accuracy_mean"], 6),
                "cv_accuracy_std": round(result["cv_accuracy_std"], 6),
                "cv_table": [
                    {
                        "C": r["C"],
                        "cv_accuracy_mean": round(r["cv_accuracy_mean"], 6),
                        "cv_accuracy_std": round(r["cv_accuracy_std"], 6),
                    }
                    for r in result["cv_table"]
                ],
                "test_accuracy": round(result["test_accuracy"], 6),
                "test_accuracy_baseline_majority": round(
                    result["test_accuracy_baseline_majority"], 6
                ),
                "test_auroc": (
                    round(result["test_auroc"], 6)
                    if not np.isnan(result["test_auroc"])
                    else None
                ),
                "n_total_prompts_supervised": n_prompts,
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "n_forget": n_forget,
                "n_retain": n_retain,
                "label_counts": dict(label_counts),
                "n_latents": n_latents,
                "n_nonzero_weights": result["n_nonzero_weights"],
                "random_state": args.seed,
                "layer": layer_num,
                "data_path_verification": {"expected_data_path": str(expected_data_path)},
            },
            "weights_by_latent": {str(i): float(w[i]) for i in range(n_latents)},
            "intercept": float(result["intercept"]),
            "feature_means": {str(i): float(result["feature_means"][i]) for i in range(n_latents)},
            "feature_stds": {str(i): float(result["feature_stds"][i]) for i in range(n_latents)},
        }

        out_path = layer_folder / args.weights_filename
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(weights_doc, f, indent=2)

        top_pos = sorted(
            ((i, float(w[i])) for i in range(n_latents) if w[i] > 0),
            key=lambda t: t[1],
            reverse=True,
        )[:5]
        top_pos_str = ", ".join(f"{i}(w={v:.3f})" for i, v in top_pos) or "(none)"
        print(
            f"Layer {layer_num}: n_prompts={n_prompts} (f={n_forget}/r={n_retain}) | "
            f"C*={result['C_selected']:.3g} | cv_acc={result['cv_accuracy_mean']:.3f}±"
            f"{result['cv_accuracy_std']:.3f} | test_acc={result['test_accuracy']:.3f} "
            f"(maj={result['test_accuracy_baseline_majority']:.3f}) | "
            f"auroc={result['test_auroc']:.3f} | nnz={result['n_nonzero_weights']}/{n_latents} | "
            f"top_pos={top_pos_str}"
        )

        per_layer_entries.append(
            {
                "layer": layer_num,
                "C_selected": result["C_selected"],
                "cv_accuracy_mean": round(result["cv_accuracy_mean"], 6),
                "cv_accuracy_std": round(result["cv_accuracy_std"], 6),
                "test_accuracy": round(result["test_accuracy"], 6),
                "test_accuracy_baseline_majority": round(
                    result["test_accuracy_baseline_majority"], 6
                ),
                "test_auroc": (
                    round(result["test_auroc"], 6)
                    if not np.isnan(result["test_auroc"])
                    else None
                ),
                "n_nonzero_weights": result["n_nonzero_weights"],
                "n_latents": n_latents,
                "n_forget": n_forget,
                "n_retain": n_retain,
                "weights_path": str(out_path),
            }
        )

    summary_doc = {
        "pipeline": "wmdp_bio_probe",
        "results_dir": str(results_dir),
        "weights_filename": args.weights_filename,
        "feature_aggregation": args.feature_aggregation,
        "standardize": True,
        "c_grid": c_grid,
        "cv_folds": args.cv_folds,
        "test_size": args.test_size,
        "seed": args.seed,
        "label_sets": {
            "forget": sorted(FORGET_LABELS),
            "retain_pooled": sorted(RETAIN_LABELS),
            "retain_neutral": sorted(NEUTRAL_LABELS),
            "retain_bio": sorted(BIO_RETAIN_LABELS),
        },
        "per_layer": per_layer_entries,
        "overview": (
            "L1-logistic probes trained per layer on SNMF prompt-max activations "
            "(standardized) to decode bio_forget vs pooled retain. Positive weights "
            "indicate latents whose high activation predicts bio_forget; negative "
            "weights indicate retain-indicative latents. Magnitudes are comparable "
            "across latents within a layer because features are standardized, and L1 "
            "gives a sparse support. Use with create_forget_ablated_model.py "
            "--selection-mode probe_topk or intersect."
        ),
    }
    summary_path = results_dir / args.summary_filename
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_doc, f, indent=2)
    print(f"\nWrote per-layer probe weights under {results_dir}/layer_*/{args.weights_filename}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
