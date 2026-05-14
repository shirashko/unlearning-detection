#!/bin/bash

# Create WMDP-bio supervision JSON(s) via data_utils/create_bio_data.py (CPU-only).
# Submit: sbatch scripts/wmdp/run_create_bio_data.sh
# Override: OUTPUT_PATH=... NUM_FILES=5 sbatch scripts/wmdp/run_create_bio_data.sh

# --- Slurm (no GPU; JSON read/write + shuffle) ---
#SBATCH --job-name=create_bio_data
#SBATCH --output=logs/create_bio_data_%j.out
#SBATCH --error=logs/create_bio_data_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# --- Environment ---
source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate snmf_env

export HF_HOME="${HF_HOME:-/home/morg/students/rashkovits/hf_cache}"
export TMPDIR="${TMPDIR:-$HF_HOME}"

# --- Repo ---
REPO_ROOT="${REPO_ROOT:-/home/morg/students/rashkovits/snmf}"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

mkdir -p logs data "$HF_HOME"

# ========== Defaults (override via env before sbatch) ==========
OUTPUT_PATH="${OUTPUT_PATH:-data/bio_data.json}"
NUM_FILES="${NUM_FILES:-5}"
SAMPLES_PER_LABEL="${SAMPLES_PER_LABEL:-600}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-256}"

# Optional: set to override JSONL paths (empty = use Python defaults under Localized-UNDO/datasets)
REMOVE_PATH="${REMOVE_PATH:-}"
RETAIN_PATH="${RETAIN_PATH:-}"
NEUTRAL_PATH="${NEUTRAL_PATH:-}"

CMD=(
  python data_utils/create_bio_data.py
  --output-path "$OUTPUT_PATH"
  --num-files "$NUM_FILES"
  --samples-per-label "$SAMPLES_PER_LABEL"
  --seed "$SEED"
  --max-tokens "$MAX_TOKENS"
)
[[ -n "$REMOVE_PATH" ]] && CMD+=(--remove-path "$REMOVE_PATH")
[[ -n "$RETAIN_PATH" ]] && CMD+=(--retain-path "$RETAIN_PATH")
[[ -n "$NEUTRAL_PATH" ]] && CMD+=(--neutral-path "$NEUTRAL_PATH")

echo "================================================================"
echo " create_bio_data.py | Node: ${SLURMD_NODENAME:-local}"
echo " Repo:            $REPO_ROOT"
echo " Output:          $OUTPUT_PATH"
echo " Num files:       $NUM_FILES"
echo " Samples/label:   $SAMPLES_PER_LABEL"
echo " Seed:            $SEED"
echo " Max tokens:      $MAX_TOKENS"
echo "================================================================"
echo "Running: ${CMD[*]}"
echo "================================================================"

"${CMD[@]}"

echo "Done."
