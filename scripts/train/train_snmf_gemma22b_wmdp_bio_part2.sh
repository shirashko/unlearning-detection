#!/bin/bash
#SBATCH --job-name=snmf_general_part2
#SBATCH --output=logs/train_snmf_general_part2_%j.out
#SBATCH --error=logs/train_snmf_general_part2_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL


# Defaults mirror outputs/gemma22b_general_snmf_r300/config.json (Gemma-2-2B, general_data part2, rank 300, layers 0–25, normalize off).
# Gemma-2-2B: num_hidden_layers=26 ⇒ indices 0–25 cover the full stack.
# Example (run from repo root):
#   cd /home/morg/students/rashkovits/unlearning-detection && sbatch scripts/train/train_snmf_gemma22b_wmdp_bio_part2.sh
#   env RANK=400 sbatch scripts/train/train_snmf_gemma22b_wmdp_bio_part2.sh   # ⇒ outputs/gemma22b_general_snmf_r400_part2

set -euo pipefail

source scripts/audit/audit_runner_env.sh

MODEL_PATH="${MODEL_PATH:-${DEFAULT_GEMMA_2_2B_MODEL:-google/gemma-2-2b}}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/general_data_part2.json}"

RANK="${RANK:-300}"
LAYERS="${LAYERS:-0-25}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/gemma22b_general_snmf_r${RANK}_part2}"
mkdir -p "$OUTPUT_DIR"

BATCH_SIZE="${BATCH_SIZE:-8}"
SNMF_MODE="${SNMF_MODE:-mlp_intermediate}"
SNMF_INIT="${SNMF_INIT:-svd}"
DEVICE="${DEVICE:-cuda}"
SPARSITY="${SPARSITY:-0.01}"
MAX_ITER="${MAX_ITER:-3000}"
SEED="${SEED:-42}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"

# Parallel execution thread safety mapping
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"

if [[ "$REQUIRE_GPU" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[-] Error: REQUIRE_GPU=1 but nvidia-smi is unavailable on $(hostname)." >&2
    exit 1
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[-] Error: REQUIRE_GPU=1 but no visible NVIDIA hardware discovered." >&2
    exit 1
  fi
fi

echo "----------------------------------------------------------------"
echo " SNMF Training"
echo " Model:     $MODEL_PATH"
echo " Data:      $DATA_PATH"
echo " Out:       $OUTPUT_DIR"
echo " Layers:    $LAYERS  (Gemma-2-2B: num_hidden_layers=26 → indices 0–25)"
echo " Rank:      $RANK"
echo "----------------------------------------------------------------"

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

echo "----------------------------------------------------------------"
echo " Done. Point SNMF_DIR (audit wrapper) at:"
echo "   $OUTPUT_DIR"
echo "----------------------------------------------------------------"
