#!/bin/bash
#SBATCH --job-name=snmf_g03b_general
#SBATCH --output=logs/train_snmf_g03b_general_%j.out
#SBATCH --error=logs/train_snmf_g03b_general_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

# Fit SNMF on the *general* audit prompts (data/general_data_part1.json) using the
# Gemma-2-0.3B arithmetic+English *base* checkpoint. Use this basis with
# scripts/audit/run_general_unlearning_audit_gemma03b_arithmetic.sh.
#
# This repo's arithmetic SNMF/analysis defaults (local_models/gemma-2-0.3B_reference_model)
# used a different prompt set; the label-free audit expects factors trained on the same
# general JSON as the audit.
#
# Example:
#   sbatch scripts/audit/train_snmf_gemma03b_arithmetic_general_data.sh
#   env RANK=200 LAYERS=0-13 sbatch scripts/audit/train_snmf_gemma03b_arithmetic_general_data.sh
#
# Override partition on crowded clusters, e.g.:
#   sbatch --partition=studentkillable scripts/audit/train_snmf_gemma03b_arithmetic_general_data.sh

set -euo pipefail

source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate /home/morg/students/rashkovits/envs/snmf_env \
  || conda activate snmf_env

export HF_HOME="${HF_HOME:-/home/morg/students/rashkovits/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export TMPDIR="${TMPDIR:-$HF_HOME/tmp}"
mkdir -p "$HF_HOME" "$TMPDIR"

REPO_ROOT="/home/morg/students/rashkovits/snmf"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
mkdir -p logs

# HF root with config.json + weights (not the parent run directory).
MODEL_PATH="${MODEL_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/non-wmdp/pretrained_models/gemma-2-0.3B_all_arithmetic+eng/final_model}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/general_data_part1.json}"
RANK="${RANK:-300}"
LAYERS="${LAYERS:-0-13}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/non_wmdp/audit/snmf_gemma03b_arithmetic_eng_general_data_part1_rank${RANK}}"

BATCH_SIZE="${BATCH_SIZE:-8}"
SNMF_MODE="${SNMF_MODE:-mlp_intermediate}"
SNMF_INIT="${SNMF_INIT:-svd}"
DEVICE="${DEVICE:-cuda}"
SPARSITY="${SPARSITY:-0.01}"
MAX_ITER="${MAX_ITER:-3000}"
SEED="${SEED:-42}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
mkdir -p "$OUTPUT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"

if [[ "$REQUIRE_GPU" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[train_snmf_gemma03b_arithmetic_general_data.sh] REQUIRE_GPU=1 but nvidia-smi is unavailable."
    exit 1
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[train_snmf_gemma03b_arithmetic_general_data.sh] REQUIRE_GPU=1 but no visible NVIDIA GPU."
    exit 1
  fi
fi

echo "----------------------------------------------------------------"
echo " SNMF (general audit basis) — Gemma-2-0.3B arithmetic+eng base"
echo " Model:     $MODEL_PATH"
echo " Data:      $DATA_PATH"
echo " Out:       $OUTPUT_DIR"
echo " Layers:    $LAYERS  (0.3B has 14 layers → indices 0–13)"
echo " Rank:      $RANK"
echo "----------------------------------------------------------------"

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

echo "----------------------------------------------------------------"
echo " Done. Point SNMF_DIR (audit wrapper) at:"
echo "   $OUTPUT_DIR"
echo "----------------------------------------------------------------"
