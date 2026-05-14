#!/bin/bash

# Read ablation_eval_comparison.json from each iter-2 top-up sweep variant and print
# a side-by-side table: wmdp_bio / mmlu for learned + random baseline, plus the
# learned-vs-random gap (the specificity metric).
#
# Run after all 3 sbatch jobs from run_iter2_topup_sweep.sh have completed.

set -euo pipefail

REPO_ROOT="/home/morg/students/rashkovits/snmf"
cd "$REPO_ROOT"

SAVE_GROUP="local_models/wmdp/iter2_topup_data_part2_thr018_both_up_down"

# For context: iter-2 base we're iterating on top of.
ITER2_BASE="local_models/wmdp/iter2/data_part2_thr022_down_proj_only/ablation_eval_comparison.json"

python3 - <<PY
import json, os, glob, sys

ROOT = "${SAVE_GROUP}"
variants = [
    ("A", "bio_retain_and_neutral"),
    ("B", "bio_retain_or_neutral"),
    ("C", "pooled_or_bio_retain"),
]
iter2_base = "${ITER2_BASE}"

def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def n_ablated(meta_path):
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        m = json.load(f)
    return sum(l.get("n_forget_columns", 0) for l in m.get("layers", []))

# Reference: iter-2 base (what we're iterating on top of).
ref = load(iter2_base)
if ref and "after" in ref:
    base_bio  = ref["after"].get("wmdp_bio_limit_None_shots_0")
    base_mmlu = ref["after"].get("mmlu_limit_0.4_shots_5")
else:
    base_bio = base_mmlu = None

hdr = f"{'VARIANT':<30} {'N':>4}   {'wmdp_bio':>9} {'Δvs_base':>9}   {'mmlu':>7} {'Δvs_base':>9}   {'rand_bio':>9} {'gap_bio':>8}   {'rand_mmlu':>9} {'gap_mmlu':>9}"
print()
print(f"iter-2 base (reference):  wmdp_bio={base_bio}  mmlu={base_mmlu}")
print()
print(hdr)
print("-" * len(hdr))

for label, tag in variants:
    base_dir = f"{ROOT}/{tag}"
    eval_json = load(os.path.join(base_dir, "ablation_eval_comparison.json"))
    meta_json = os.path.join(base_dir, "forget_ablation_metadata.json")
    if eval_json is None:
        print(f"{label}: {tag:<27}  [ablation_eval_comparison.json not found — job may still be running]")
        continue

    # With SKIP_PRE_EVAL=1 the learned post-ablation numbers are in `after` and the
    # random-baseline numbers live under random_baseline.eval (or random_baseline.after).
    after = eval_json.get("after", {}) or {}
    rb    = eval_json.get("random_baseline", {}) or {}
    rb_eval = rb.get("eval") or rb.get("after") or {}

    learned_bio  = after.get("wmdp_bio_limit_None_shots_0")
    learned_mmlu = after.get("mmlu_limit_0.4_shots_5")
    rand_bio     = rb_eval.get("wmdp_bio_limit_None_shots_0")
    rand_mmlu    = rb_eval.get("mmlu_limit_0.4_shots_5")

    d_bio  = (learned_bio  - base_bio)  if (learned_bio  is not None and base_bio  is not None) else None
    d_mmlu = (learned_mmlu - base_mmlu) if (learned_mmlu is not None and base_mmlu is not None) else None
    gap_bio  = (learned_bio  - rand_bio)  if (learned_bio  is not None and rand_bio  is not None) else None
    gap_mmlu = (learned_mmlu - rand_mmlu) if (learned_mmlu is not None and rand_mmlu is not None) else None

    n = n_ablated(meta_json)

    def fmt(x, w, fmtstr=".4f"):
        return f"{x:{w}{fmtstr}}" if x is not None else f"{'--':>{w}}"

    tagfull = f"{label}: {tag}"
    print(f"{tagfull:<30} "
          f"{(n if n is not None else '--'):>4}   "
          f"{fmt(learned_bio,  9)} {fmt(d_bio,  9,'+.4f')}   "
          f"{fmt(learned_mmlu, 7)} {fmt(d_mmlu, 9,'+.4f')}   "
          f"{fmt(rand_bio,     9)} {fmt(gap_bio,  8,'+.4f')}   "
          f"{fmt(rand_mmlu,    9)} {fmt(gap_mmlu, 9,'+.4f')}")

print()
print("Legend:")
print("  Δvs_base = learned post-ablation minus the iter-2 base (negative = further drop).")
print("  gap_bio / gap_mmlu = learned MINUS random-matched baseline (more negative = more specific targeting).")
print("  The best variant maximizes |gap_mmlu / gap_bio| (close to 0 = random-like collateral)")
print("  while keeping Δvs_base_bio strongly negative.")
PY
