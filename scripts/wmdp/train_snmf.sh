#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=train_snmf_wmdp_bio
#SBATCH --output=logs/train_wmdp_bio_%j.out
#SBATCH --error=logs/train_wmdp_bio_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

# --- Environment Setup ---
source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate /home/morg/students/rashkovits/envs/snmf_env

# --- Space & Cache Management ---
export HF_HOME="/home/morg/students/rashkovits/hf_cache"
export TORCH_HOME="/home/morg/students/rashkovits/hf_cache/torch"
export TMPDIR="/home/morg/students/rashkovits/hf_cache"

# --- Project Setup ---
cd /home/morg/students/rashkovits/snmf
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Defaults target the WMDP-bio Gemma-2-2b setup.
MODEL_PATH="${MODEL_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/gemma-2-2b}"
DATA_PATH="${DATA_PATH:-data/bio_data_part1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wmdp/results_data_part1_gemma2_2b_450_rank}"
LAYERS="${LAYERS:-0-25}"        # Gemma-2-2b has 26 layers => indices 0..25
RANK="${RANK:-450}"
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

# --- GPU Preflight ---
if [[ "$REQUIRE_GPU" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[train_snmf.sh] REQUIRE_GPU=1 but nvidia-smi is unavailable."
    exit 1
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[train_snmf.sh] REQUIRE_GPU=1 but no visible NVIDIA GPU."
    exit 1
  fi
  python3 - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("[train_snmf.sh] torch.cuda.is_available() is False.")
    sys.exit(1)
major, minor = torch.cuda.get_device_capability(0)
if major < 7:
    print(f"[train_snmf.sh] Unsupported CUDA capability sm_{major}{minor}; expected sm_70+.")
    sys.exit(1)
print(f"[train_snmf.sh] CUDA ready on {torch.cuda.get_device_name(0)} (sm_{major}{minor}).")
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

python train_snmf.py \
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