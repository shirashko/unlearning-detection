#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=train_snmf_gemma2_2b_it_data_part1_rank_500
#SBATCH --output=logs/train_snmf_gemma2_2b_it_data_part1_rank_500_%j.out
#SBATCH --error=logs/train_snmf_gemma2_2b_it_data_part1_rank_500_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../audit/audit_runner_env.sh
source "${_SCRIPT_DIR}/../audit/audit_runner_env.sh"

# Defaults target the Gemma-2-2b-it setup (HF repo id + HF_HUB_CACHE from audit_runner_env.sh).
MODEL_PATH="${MODEL_PATH:-${DEFAULT_GEMMA_2_2B_MODEL:-google/gemma-2-2b-it}}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/general_data_part1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/gemma_2_2b_it/data_part1_rank_500}"
LAYERS="${LAYERS:-0-25}"  # cover all layers
RANK="${RANK:-500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SNMF_MODE="${SNMF_MODE:-mlp_intermediate}"
SNMF_INIT="${SNMF_INIT:-svd}"
DEVICE="${DEVICE:-cuda}"
SPARSITY="${SPARSITY:-0.01}"
MAX_ITER="${MAX_ITER:-3000}"
SEED="${SEED:-42}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"   # 1 => fail fast if CUDA GPU is not usable
mkdir -p logs "$OUTPUT_DIR" $HF_HOME

# --- Parallelism Optimization ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRIPT_TAG="[${0##*/}]"

# --- GPU Preflight ---
if [[ "$REQUIRE_GPU" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "${SCRIPT_TAG} REQUIRE_GPU=1 but nvidia-smi is unavailable."
    exit 1
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "${SCRIPT_TAG} REQUIRE_GPU=1 but no visible NVIDIA GPU."
    exit 1
  fi
  SCRIPT_TAG="$SCRIPT_TAG" python3 - <<'PY'
import os
import sys
import torch

tag = os.environ.get("SCRIPT_TAG", "[train_snmf]")
if not torch.cuda.is_available():
    print(f"{tag} torch.cuda.is_available() is False.")
    sys.exit(1)
major, minor = torch.cuda.get_device_capability(0)
if major < 7:
    print(f"{tag} Unsupported CUDA capability sm_{major}{minor}; expected sm_70+.")
    sys.exit(1)
print(f"{tag} CUDA ready on {torch.cuda.get_device_name(0)} (sm_{major}{minor}).")
PY
fi

# --- Execute Training ---
echo "--------------------------------------------------------"
echo "Starting SNMF Training on Node: $SLURMD_NODENAME"
echo "Model path: $MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Layers: $LAYERS"
echo "Device: $DEVICE"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Visible GPUs:"
  nvidia-smi -L || true
fi
echo "--------------------------------------------------------"

python experiments/train/train_snmf.py \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --layers "$LAYERS" \
    --rank "$RANK" \
    --mode "$SNMF_MODE" \
    --init "$SNMF_INIT" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --sparsity "$SPARSITY" \
    --max-iter "$MAX_ITER" \
    --seed "$SEED"

echo "--------------------------------------------------------"
echo "SNMF Training Finished"