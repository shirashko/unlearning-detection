#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=eval_wmdp_bio
#SBATCH --output=logs/eval_wmdp_bio_%j.out
#SBATCH --error=logs/eval_wmdp_bio_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

set -euo pipefail

# --- Environment Setup ---
source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate snmf_env

# --- Space & Cache Management ---
export HF_HOME="${HF_HOME:-/home/morg/students/rashkovits/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export TMPDIR="${TMPDIR:-$HF_HOME}"

# --- Project Setup ---
REPO_ROOT="/home/morg/students/rashkovits/snmf"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

mkdir -p logs outputs/eval_results "$HF_HOME" cache

# --- Parallelism Optimization ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# --- Config (override by exporting before sbatch/bash) ---
# Default model is the iter-2 + top-up variant-A ckpt (the only iter-2 top-up that kept
# MMLU intact; wmdp_bio=0.4273, mmlu=0.4837 on stock wmdp_bio). Re-eval this ckpt on
# wmdp_bio_robust + MMLU gives the apples-to-apples baseline for iter-3 comparisons.
MODEL_PATH="${MODEL_PATH:-/home/morg/students/rashkovits/snmf/local_models/wmdp/iter2_topup_data_part2_thr018_both_up_down/bio_retain_and_neutral}"
EVAL_MODE="${EVAL_MODE:-wmdp_bio_categorized}"  # wmdp_bio | wmdp_bio_categorized
DEVICE="${DEVICE:-cuda}"
CACHE_DIR="${CACHE_DIR:-./cache}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-./cache}"
# Default to full WMDP-bio test evaluation.
# Set LARGE_EVAL=0 for a quicker subset run.
LARGE_EVAL="${LARGE_EVAL:-1}"   # 1 -> --large-eval (full WMDP-bio)
# Keep MMLU on by default — we always want the preservation metric alongside WMDP-bio.
NO_MMLU="${NO_MMLU:-0}"         # 1 -> --no-mmlu (WMDP-bio only)
WMDP_INCLUDE_PATH="${WMDP_INCLUDE_PATH:-/home/morg/students/rashkovits/snmf/wmdp_bio_categorized_mcqa}"
WMDP_TASK_NAME="${WMDP_TASK_NAME:-wmdp_bio_robust}"  # robust by default; or wmdp_bio_shortcut / wmdp_bio_categorized_mcqa
RESULTS_JSON="${RESULTS_JSON:-outputs/eval_results/wmdp_bio_${SLURM_JOB_ID:-local}.json}"

echo "--------------------------------------------------------"
echo "Starting WMDP-bio evaluation on node: ${SLURMD_NODENAME:-local}"
echo "Model path: $MODEL_PATH"
echo "Eval mode:  $EVAL_MODE"
echo "Large eval: $LARGE_EVAL"
echo "No MMLU:    $NO_MMLU"
echo "Results:    $RESULTS_JSON"
echo "--------------------------------------------------------"

CMD=(
  python3 evaluation/eveluate_model.py
  --model-path "$MODEL_PATH"
  --eval-mode "$EVAL_MODE"
  --device "$DEVICE"
  --cache-dir "$CACHE_DIR"
  --dataset-cache-dir "$DATASET_CACHE_DIR"
  --output-json "$RESULTS_JSON"
)

if [[ "$LARGE_EVAL" == "1" ]]; then
  CMD+=(--large-eval)
fi

if [[ "$NO_MMLU" == "1" ]]; then
  CMD+=(--no-mmlu)
fi

if [[ "$EVAL_MODE" == "wmdp_bio_categorized" ]]; then
  CMD+=(--wmdp-include-path "$WMDP_INCLUDE_PATH")
  CMD+=(--wmdp-task-name "$WMDP_TASK_NAME")
fi

"${CMD[@]}"

echo "--------------------------------------------------------"
echo "WMDP-bio evaluation finished"
echo "--------------------------------------------------------"
