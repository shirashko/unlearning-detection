#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=analyze_wmdp_bio
#SBATCH --output=logs/analyze_wmdp_bio_%j.out
#SBATCH --error=logs/analyze_wmdp_bio_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

# WMDP-bio pipeline: analyze SNMF factors (pair with scripts/wmdp/train_snmf.sh).
#   train_snmf.sh  ->  OUTPUT_DIR/layer_*/snmf_factors.pt
#   this script      ->  wmdp_bio_analyze_snmf_results.py -> *_wmdp_bio.json + SUMMARY_FILE
# Defaults below match train_snmf.sh (MODEL_PATH, DATA_PATH, OUTPUT_DIR, SEED, DEVICE).

set -euo pipefail

# --- Environment Setup (same as train_snmf.sh) ---
source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate /home/morg/students/rashkovits/envs/snmf_env

# --- Space & Cache Management (same pattern as run_evaluate_wmdp_bio.sh) ---
export HF_HOME="${HF_HOME:-/home/morg/students/rashkovits/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export TMPDIR="${TMPDIR:-$HF_HOME}"

# --- Project Setup ---
REPO_ROOT="/home/morg/students/rashkovits/snmf"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

mkdir -p logs "$HF_HOME"

# --- Analysis I/O (defaults aligned with scripts/wmdp/train_snmf.sh) ---
MODEL_PATH="${MODEL_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/gemma-2-2b}"
DATA_PATH="${DATA_PATH:-data/bio_data_part1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wmdp/results_data_part1_gemma2_2b_450_rank}"
RESULTS_DIR="${RESULTS_DIR:-$OUTPUT_DIR}"
SUMMARY_FILE="${SUMMARY_FILE:-analysis_summary_wmdp_bio.json}"
SEED="${SEED:-${ANALYZE_SEED:-42}}"
DEVICE="${DEVICE:-${ANALYZE_DEVICE:-cuda}}"
ROLE_ASSIGNMENT_THRESHOLD="${ROLE_ASSIGNMENT_THRESHOLD:-0.05}"
ACTIVATION_CONTEXT_TOP_N="${ACTIVATION_CONTEXT_TOP_N:-10}"
ACTIVATION_CONTEXT_WINDOW="${ACTIVATION_CONTEXT_WINDOW:-15}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"   # 1 => fail fast if CUDA GPU is not usable

# --- Parallelism Optimization ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# --- GPU Preflight ---
if [[ "$REQUIRE_GPU" == "1" && "$DEVICE" == cuda* ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[run_analyze_wmdp_bio.sh] REQUIRE_GPU=1 but nvidia-smi is unavailable."
    exit 1
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[run_analyze_wmdp_bio.sh] REQUIRE_GPU=1 but no visible NVIDIA GPU."
    exit 1
  fi
  python3 - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("[run_analyze_wmdp_bio.sh] torch.cuda.is_available() is False.")
    sys.exit(1)
major, minor = torch.cuda.get_device_capability(0)
if major < 7:
    print(f"[run_analyze_wmdp_bio.sh] Unsupported CUDA capability sm_{major}{minor}; expected sm_70+.")
    sys.exit(1)
print(f"[run_analyze_wmdp_bio.sh] CUDA ready on {torch.cuda.get_device_name(0)} (sm_{major}{minor}).")
PY
fi

# --- Execute Analysis ---
echo "--------------------------------------------------------"
echo "WMDP-bio SNMF analysis on Node: ${SLURMD_NODENAME:-local}"
echo "Model path: $MODEL_PATH"
echo "Training data file (for traceability): $DATA_PATH"
echo "Results directory (train --output-dir): $RESULTS_DIR"
echo "Per-run summary: $RESULTS_DIR/$SUMMARY_FILE"
echo "Device: $DEVICE  Seed: $SEED"
echo "Role threshold: $ROLE_ASSIGNMENT_THRESHOLD"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Visible GPUs:"
  nvidia-smi -L || true
fi
echo "--------------------------------------------------------"

python wmdp_bio_analyze_snmf_results.py \
  --model-path "$MODEL_PATH" \
  --results-dir "$RESULTS_DIR" \
  --summary-filename "$SUMMARY_FILE" \
  --data-path "$DATA_PATH" \
  --role-assignment-threshold "$ROLE_ASSIGNMENT_THRESHOLD" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --activation-context-top-n "$ACTIVATION_CONTEXT_TOP_N" \
  --activation-context-window "$ACTIVATION_CONTEXT_WINDOW"

echo "--------------------------------------------------------"
echo "WMDP-bio SNMF analysis finished"
echo "Feature counts and role definitions: $RESULTS_DIR/$SUMMARY_FILE"
