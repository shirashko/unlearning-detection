#!/bin/bash

# Slurm wrapper for scripts/wmdp/compare_bio_forget_counts.py.
# Compares the number of bio_forget_lean SNMF latents across iterations of the
# WMDP-bio forgetting pipeline and writes CSV / Markdown (and, if matplotlib
# is available, a per-layer PNG) under $OUTPUT_DIR.
#
# CPU-only, lightweight (~seconds) — uses the studentkillable partition.
#
# Usage (defaults: canonical iter1 -> iter2 -> iter3 chain, thr=0.30):
#   sbatch scripts/wmdp/run_compare_bio_forget_counts.sh
#
# Overrides (env vars):
#   ITERATIONS="name1:path1 name2:path2 ..."   (space-separated list)
#   COMPARISON_THRESHOLD=0.40
#   AND_BASES="bio_retain,neutral"
#   OUTPUT_DIR=outputs/wmdp/iteration_comparison_thr040
#   NO_PLOT=1                                  (skip the matplotlib plot)
#
# Example — sweep thresholds from the CLI:
#   COMPARISON_THRESHOLD=0.40 \
#   OUTPUT_DIR=outputs/wmdp/iteration_comparison_thr040 \
#     sbatch scripts/wmdp/run_compare_bio_forget_counts.sh

# --- Slurm (CPU-only) ---
#SBATCH --job-name=compare_bio_forget_counts
#SBATCH --output=logs/compare_bio_forget_counts_%j.out
#SBATCH --error=logs/compare_bio_forget_counts_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

# --- Environment ---
source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate snmf_env

# --- Repo ---
REPO_ROOT="${REPO_ROOT:-/home/morg/students/rashkovits/snmf}"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

mkdir -p logs

# --- Config (overridable via env) ---
# Canonical chain: SNMF was re-fit on each iteration's ablated checkpoint.
DEFAULT_ITERATIONS=(
  "iter1:outputs/wmdp/results_data_part1_gemma2_2b"
  "iter2:outputs/wmdp/results_data_part2_gemma2_2b_iter2_thr022_down_proj_only"
  "iter3:outputs/wmdp/results_data_part3_gemma2_2b_iter3_thr018_both_up_down"
)
if [[ -n "${ITERATIONS:-}" ]]; then
  # shellcheck disable=SC2206
  ITERATIONS_ARR=($ITERATIONS)
else
  ITERATIONS_ARR=("${DEFAULT_ITERATIONS[@]}")
fi

COMPARISON_THRESHOLD="${COMPARISON_THRESHOLD:-0.30}"
AND_BASES="${AND_BASES:-bio_retain,neutral}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wmdp/iteration_comparison}"
NO_PLOT="${NO_PLOT:-0}"

ITER_FLAGS=()
for it in "${ITERATIONS_ARR[@]}"; do
  ITER_FLAGS+=(--iteration "$it")
done

EXTRA_FLAGS=()
if [[ "$NO_PLOT" == "1" ]]; then
  EXTRA_FLAGS+=(--no-plot)
fi

echo "================================================================"
echo " Compare bio_forget counts across iterations"
echo " Node:                  ${SLURMD_NODENAME:-local}"
echo " Iterations:            ${ITERATIONS_ARR[*]}"
echo " Comparison threshold:  $COMPARISON_THRESHOLD"
echo " AND bases:             $AND_BASES"
echo " Output dir:            $OUTPUT_DIR"
echo " NO_PLOT:               $NO_PLOT"
echo "================================================================"

python scripts/wmdp/compare_bio_forget_counts.py \
  "${ITER_FLAGS[@]}" \
  --comparison-threshold "$COMPARISON_THRESHOLD" \
  --and-bases "$AND_BASES" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_FLAGS[@]}"

echo
echo "================================================================"
echo " Done. Artifacts under $OUTPUT_DIR:"
echo "   bio_forget_counts.csv"
echo "   bio_forget_counts_summary.md"
echo "   bio_forget_counts_by_layer.png   (if matplotlib is installed)"
echo "================================================================"
