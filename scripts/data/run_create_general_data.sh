#!/bin/bash

# Create general pretrain JSON(s) via data_utils/create_general_data.py (CPU-only).
# Submit: sbatch scripts/audit/run_create_general_data.sh
# Override: OUTPUT_PATH=... NUM_FILES=5 sbatch scripts/audit/run_create_general_data.sh

# --- Slurm (no GPU; JSON read/write + shuffle) ---
#SBATCH --job-name=create_general_data
#SBATCH --output=logs/create_general_data_%j.out
#SBATCH --error=logs/create_general_data_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# --- Environment ---
source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate snmf_env

export HF_HOME="${HF_HOME:-/home/morg/students/rashkovits/hf_cache}"
export TMPDIR="${TMPDIR:-$HF_HOME}"

# --- Repo ---
REPO_ROOT="${REPO_ROOT:-/home/morg/students/rashkovits/unlearning-detection}"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

mkdir -p logs data "$HF_HOME"

# ========== Defaults (override via env before sbatch) ==========
OUTPUT_PATH="${OUTPUT_PATH:-data/general_data.json}"
NUM_FILES="${NUM_FILES:-2}"
SAMPLES_PER_SOURCE="${SAMPLES_PER_SOURCE:-600}"
SEED="${SEED:-42}"
MAX_TOKENS="${MAX_TOKENS:-256}"

BASE_DATASET_PATH="${BASE_DATASET_PATH:-/home/morg/students/rashkovits/Localized-UNDO/datasets}"
SOURCE1_PATH="${SOURCE1_PATH:-$BASE_DATASET_PATH/pretrain/train_eng.jsonl}"
SOURCE2_PATH="${SOURCE2_PATH:-$BASE_DATASET_PATH/pretrain/train_wikitext.jsonl}"

CMD=(
  python data_utils/create_general_data.py
  --source1-path "$SOURCE1_PATH"
  --source2-path "$SOURCE2_PATH"
  --output-path "$OUTPUT_PATH"
  --num-files "$NUM_FILES"
  --samples-per-source "$SAMPLES_PER_SOURCE"
  --seed "$SEED"
  --max-tokens "$MAX_TOKENS"
)

echo "================================================================"
echo " create_general_data.py | Node: ${SLURMD_NODENAME:-local}"
echo " Repo:            $REPO_ROOT"
echo " Output:          $OUTPUT_PATH"
echo " Source1:         $SOURCE1_PATH"
echo " Source2:         $SOURCE2_PATH"
echo " Num files:       $NUM_FILES"
echo " Samples/source:  $SAMPLES_PER_SOURCE"
echo " Seed:            $SEED"
echo " Max tokens:      $MAX_TOKENS"
echo "================================================================"
echo "Running: ${CMD[*]}"
echo "================================================================"

"${CMD[@]}"

echo "Done."
