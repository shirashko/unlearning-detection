#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=weight_fingerprint
#SBATCH --output=logs/weight_fingerprint_%j.out
#SBATCH --error=logs/weight_fingerprint_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

# Weight-space unlearning fingerprint.
# Compares the base Gemma-2-2b checkpoint against three wmdp-bio unlearned
# variants (SNMF / MaxEnt / RMU) and extracts per-layer per-matrix SVD- and
# delta-based fingerprint metrics, then emits CSV + plots to OUTPUT_DIR.
#
# Override any candidate via env, e.g.
#   env RMU_PATH=.../RMU/bio_lr_1.00e-04_alpha_0.10_seed_42/final_model \
#       sbatch scripts/fingerprint/run_weight_fingerprint.sh
#   env LAYERS=0-25 SPECTRA_LAYERS=3,10,17,24 \
#       bash scripts/fingerprint/run_weight_fingerprint.sh
#   env PROJECTIONS="k_proj v_proj down_proj" \
#       bash scripts/fingerprint/run_weight_fingerprint.sh

set -euo pipefail

# --- Environment Setup ---
if [[ -f /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh ]]; then
  source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
  conda activate /home/morg/students/rashkovits/envs/snmf_env 2>/dev/null \
    || conda activate snmf_env
fi

export HF_HOME="${HF_HOME:-/home/morg/students/rashkovits/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export TMPDIR="${TMPDIR:-$HF_HOME}"

REPO_ROOT="/home/morg/students/rashkovits/unlearning-detection"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
mkdir -p logs "$HF_HOME"

# --- Models ---
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/gemma-2-2b}"
SNMF_PATH="${SNMF_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/snmf_iter3b_thr030_bio_retain_and_neutral}"
MAXENT_PATH="${MAXENT_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/unlearned_models/MaxEnt/bio_lr_2.00e-05_alpha_0.30_seed_42/final_model}"
RMU_PATH="${RMU_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/unlearned_models/RMU/bio_lr_1.00e-04_alpha_0.30_seed_42/final_model}"

# --- Analysis knobs ---
LAYERS="${LAYERS:-all}"
PROJECTIONS="${PROJECTIONS:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}"
DEVICE="${DEVICE:-auto}"
SPECTRA_LAYERS="${SPECTRA_LAYERS:-3,10,17,24}"
SAVE_SPECTRA_NPZ="${SAVE_SPECTRA_NPZ:-1}"

LAYER_TAG="$(printf '%s' "$LAYERS" | tr ',' '_' | tr '-' '_')"
DEFAULT_OUT="${REPO_ROOT}/outputs/wmdp/fingerprint/weight_fingerprint_layers_${LAYER_TAG}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUT}}"
mkdir -p "$OUTPUT_DIR"

CMD=(
  python -u scripts/fingerprint/weight_fingerprint.py
  --base-model-path "$BASE_MODEL_PATH"
  --candidate-model "snmf=${SNMF_PATH}"
  --candidate-model "maxent=${MAXENT_PATH}"
  --candidate-model "rmu=${RMU_PATH}"
  --output-dir "$OUTPUT_DIR"
  --layers "$LAYERS"
  --device "$DEVICE"
  --spectra-layers "$SPECTRA_LAYERS"
  --projections $PROJECTIONS
)
if [[ "$SAVE_SPECTRA_NPZ" == "0" || "$SAVE_SPECTRA_NPZ" == "false" ]]; then
  CMD+=(--no-save-spectra-npz)
fi

echo "================================================================"
echo " Weight-space unlearning fingerprint"
echo " Base:     $BASE_MODEL_PATH"
echo " SNMF:     $SNMF_PATH"
echo " MaxEnt:   $MAXENT_PATH"
echo " RMU:      $RMU_PATH"
echo " Layers:   $LAYERS"
echo " Projs:    $PROJECTIONS"
echo " Spectra:  $SPECTRA_LAYERS"
echo " Output:   $OUTPUT_DIR"
echo "================================================================"

"${CMD[@]}"

echo "================================================================"
echo " Done. Artefacts:"
echo "   $OUTPUT_DIR/metrics.csv"
echo "   $OUTPUT_DIR/singular_values.npz"
echo "   $OUTPUT_DIR/summary.json"
echo "   $OUTPUT_DIR/plots/"
echo "================================================================"
