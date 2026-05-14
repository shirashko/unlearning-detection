#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=snmf_unlearning_audit
#SBATCH --output=logs/unlearning_audit_%j.out
#SBATCH --error=logs/unlearning_audit_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

# Initial-check pipeline for the SNMF-based unlearning audit
# (Sections 3.2 + 3.3 of the proposal: "Intrinsic Auditing via SNMF").
#
# Pipeline:
#   1. Load Z = F (per-layer SNMF basis trained on M_base) from $SNMF_DIR/layer_*/snmf_factors.pt.
#   2. Run M_base    on $DATA_PATH, collect mlp_intermediate activations for $LAYERS.
#   3. Run M_unlearned on the SAME prompts, collect activations.
#   4. Project both onto Z via ridge least squares -> Y_base, Y_unlearned.
#   5. Compute per-feature delta = E[Y_base_max] - E[Y_unlearned_max] (overall + per group),
#      reconstruction residual ratios (weight-trace signal), summary plots.
#
# Defaults run the audit against the iter-1 SNMF basis
# (outputs/wmdp/results_data_part1_gemma2_2b) on the RMU bio-unlearned
# Gemma-2-2b checkpoint (bio_lr_1.00e-04_alpha_0.30_seed_42/final_model).
# To audit a different checkpoint (e.g. MaxEnt), override UNLEARNED_MODEL_PATH (see below).
#
# The output directory is derived automatically from
#   <SNMF_DIR basename>__<method>_<run_name>__<thr>_<layers>
# so each unlearned-model audit lands in its OWN folder and you can run several
# methods / hyperparam combinations without ever overwriting a previous result.
# Override the auto-tag via env UNLEARNED_TAG=... or override OUTPUT_DIR=...
#
# Examples:
#   env LAYERS=0-25 MAX_PER_GROUP=300 sbatch scripts/wmdp/run_unlearning_audit.sh
#   env UNLEARNED_MODEL_PATH=/path/to/MaxEnt/.../final_model sbatch scripts/wmdp/run_unlearning_audit.sh   # audit MaxEnt instead
#   env ROLE_THRESHOLD=0.18 sbatch scripts/wmdp/run_unlearning_audit.sh
#
# Run as a non-slurm shell (no GPU node) by invoking with bash directly:
#   bash scripts/wmdp/run_unlearning_audit.sh

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

REPO_ROOT="/home/morg/students/rashkovits/snmf"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
mkdir -p logs "$HF_HOME"

# --- Configurable inputs ---
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/gemma-2-2b}"
UNLEARNED_MODEL_PATH="${UNLEARNED_MODEL_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/unlearned_models/RMU/bio_lr_1.00e-04_alpha_0.30_seed_42/final_model}"
SNMF_DIR="${SNMF_DIR:-${REPO_ROOT}/outputs/wmdp/results_data_part1_gemma2_2b}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/bio_data_part1.json}"

# Use mid layers by default — that's where the supervised analysis showed the
# strongest forget-vs-retain separation in iter-1. Override via env LAYERS for full sweeps.
LAYERS="${LAYERS:-10-18}"
MAX_PER_GROUP="${MAX_PER_GROUP:-150}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-auto}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-4}"
ROLE_THRESHOLD="${ROLE_THRESHOLD:-0.22}"
ROLE_BASIS="${ROLE_BASIS:-pooled}"
SUPERVISED_JSON="${SUPERVISED_JSON:-feature_analysis_supervised_wmdp_bio.json}"
TOP_K_REPORT="${TOP_K_REPORT:-25}"
# How to rank bio_forget_lean latents by "how much was erased":
#   rel_delta_forget (default) -- fractional drop on forget prompts
#       delta_forget / (E_forget[Y_base] + eps); surfaces "surgical"
#       unlearning of niche features whose base activation was already small.
#   abs_rel_delta_forget       -- magnitude of fractional change either way.
#   delta_forget               -- raw signed decrease (E_forget[Y_base] - E_forget[Y_unl]).
#   abs_delta_forget           -- magnitude of raw decrease either way.
RANK_BY="${RANK_BY:-rel_delta_forget}"

# --- Logit-lens (output-side interpretation on M_base) ---
# Mirrors the controls in scripts/audit/run_general_unlearning_audit.sh.
VOCAB_LENS_TOP_K="${VOCAB_LENS_TOP_K:-15}"
SKIP_VOCAB_LENS="${SKIP_VOCAB_LENS:-0}"
LENS_CENTER_UNEMBED="${LENS_CENTER_UNEMBED:-1}"
LENS_MASK_SPECIAL_TOKENS="${LENS_MASK_SPECIAL_TOKENS:-1}"
VOCAB_LENS_AGGREGATE_TOP_K="${VOCAB_LENS_AGGREGATE_TOP_K:-20}"
LENS_DELTA_WEIGHTED="${LENS_DELTA_WEIGHTED:-0}"

SEED="${SEED:-42}"

THR_TAG="thr$(printf '%s' "$ROLE_THRESHOLD" | tr -d '.')"
LAYER_TAG="$(printf '%s' "$LAYERS" | tr ',' '_' | tr '-' '_')"

# --- Derive a stable, human-readable tag from the unlearned model path so each
#     unlearning method / hyperparam run gets its OWN output dir and never
#     overwrites a previous run.
#
# Layout in Localized-UNDO/.../unlearned_models is:
#   .../unlearned_models/<METHOD>/<RUN_NAME>/final_model
# We tag with "<method>_<run_name>" (lowercased, dot-stripped). If the path
# doesn't follow that convention we fall back to the basename of the parent dir.
# Override directly via env UNLEARNED_TAG=mytag if you want full control.
_unlearned_norm="${UNLEARNED_MODEL_PATH%/}"
_unlearned_norm="${_unlearned_norm%/final_model}"
_run_name="$(basename "$_unlearned_norm")"
_method_name="$(basename "$(dirname "$_unlearned_norm")")"
DEFAULT_UNLEARNED_TAG="$(printf '%s_%s' "$_method_name" "$_run_name" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_' '_' | tr -s '_' | sed 's/^_//;s/_$//')"
UNLEARNED_TAG="${UNLEARNED_TAG:-$DEFAULT_UNLEARNED_TAG}"

# SNMF basis tag: take the basename of $SNMF_DIR so e.g. results_data_part1_gemma2_2b
# becomes part of the output dir; this keeps multiple bases from colliding.
SNMF_TAG="$(basename "${SNMF_DIR%/}")"

DEFAULT_OUT="${REPO_ROOT}/outputs/wmdp/audit/${SNMF_TAG}__${UNLEARNED_TAG}__${THR_TAG}_layers_${LAYER_TAG}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUT}}"
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo " SNMF unlearning audit (initial check)"
echo " Base model:        $BASE_MODEL_PATH"
echo " Unlearned model:   $UNLEARNED_MODEL_PATH"
echo " SNMF dir:          $SNMF_DIR"
echo " Data:              $DATA_PATH    (max-per-group=$MAX_PER_GROUP)"
echo " Layers:            $LAYERS"
echo " Output:            $OUTPUT_DIR"
echo " Role basis/thr:    $ROLE_BASIS / $ROLE_THRESHOLD"
echo " Ridge lambda:      $RIDGE_LAMBDA"
echo " Supervised JSON:   $SUPERVISED_JSON"
echo " Rank by:           $RANK_BY"
echo " Vocab-lens top-k:  $VOCAB_LENS_TOP_K  (skip=$SKIP_VOCAB_LENS  center=$LENS_CENTER_UNEMBED  mask_specials=$LENS_MASK_SPECIAL_TOKENS)"
echo " Aggregate top-k:   $VOCAB_LENS_AGGREGATE_TOP_K  (delta_weighted=$LENS_DELTA_WEIGHTED)"
echo "================================================================"

CMD=(
  python -u experiments/audit/unlearning_audit.py
  --base-model-path "$BASE_MODEL_PATH"
  --unlearned-model-path "$UNLEARNED_MODEL_PATH"
  --snmf-dir "$SNMF_DIR"
  --data-path "$DATA_PATH"
  --output-dir "$OUTPUT_DIR"
  --layers "$LAYERS"
  --max-per-group "$MAX_PER_GROUP"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
  --ridge-lambda "$RIDGE_LAMBDA"
  --role-assignment-threshold "$ROLE_THRESHOLD"
  --role-label-basis "$ROLE_BASIS"
  --supervised-json-filename "$SUPERVISED_JSON"
  --top-k-report "$TOP_K_REPORT"
  --rank-by "$RANK_BY"
  --vocab-lens-top-k "$VOCAB_LENS_TOP_K"
  --vocab-lens-aggregate-top-k "$VOCAB_LENS_AGGREGATE_TOP_K"
  --seed "$SEED"
)
if [[ "$SKIP_VOCAB_LENS" == "1" || "$SKIP_VOCAB_LENS" == "true" ]]; then
  CMD+=(--skip-vocab-lens)
fi
if [[ "$LENS_CENTER_UNEMBED" == "0" || "$LENS_CENTER_UNEMBED" == "false" ]]; then
  CMD+=(--no-lens-center-unembed)
fi
if [[ "$LENS_MASK_SPECIAL_TOKENS" == "0" || "$LENS_MASK_SPECIAL_TOKENS" == "false" ]]; then
  CMD+=(--no-lens-mask-special-tokens)
fi
if [[ "$LENS_DELTA_WEIGHTED" == "1" || "$LENS_DELTA_WEIGHTED" == "true" ]]; then
  CMD+=(--lens-delta-weighted)
fi

"${CMD[@]}"

echo "================================================================"
echo " Audit done. Inspect:"
echo "   $OUTPUT_DIR/audit_summary.json"
echo "   $OUTPUT_DIR/audit_summary_per_layer.csv"
echo "   $OUTPUT_DIR/delta_by_role.png"
echo "   $OUTPUT_DIR/rel_delta_by_role.png"
echo "   $OUTPUT_DIR/delta_top_features.png"
echo "   $OUTPUT_DIR/rel_delta_top_features.png"
echo "================================================================"
