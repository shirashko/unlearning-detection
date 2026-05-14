#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=snmf_general_audit
#SBATCH --output=logs/general_unlearning_audit_%j.out
#SBATCH --error=logs/general_unlearning_audit_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

# Label-free SNMF unlearning audit + Gemini judge.
# Wraps experiments/audit/general_unlearning_audit.py.
#
# Pipeline:
#   1. Load Z = F per layer from $SNMF_DIR/layer_*/snmf_factors.pt
#      (basis trained on M_base).
#   2. Run M_base and M_candidate on the SAME unlabeled prompts in $DATA_PATH.
#   3. Project onto Z via ridge least squares -> Y_base, Y_candidate.
#   4. Pick the most-changed features. Default ranking is rel_delta =
#      (E[Y_base] - E[Y_candidate]) / (E[Y_base] + 1e-9), the fractional
#      drop in mean peak activation, which surfaces "surgical" unlearning
#      of niche features that were already small on M_base. Set
#      RANK_BY=delta to fall back to the raw signed decrease.
#      Pull each top feature's top-activating token windows, pack
#      everything into a single message and ask the judge LLM
#      (Gemini 2.5 Flash by default):
#        - confidence (0-100) that unlearning happened
#        - what concept it most plausibly targeted
#
# Defaults audit the iter-1 SNMF basis (outputs/.../results_data_part1_gemma2_2b)
# against the MaxEnt bio-unlearned Gemma-2-2b checkpoint, using the unlabeled
# pretrain general data we generate via scripts/audit/run_create_general_data.sh.
#
# The output directory is auto-derived from
#   <SNMF_DIR basename>__<method>_<run_name>__layers_<...>
# so each candidate-model audit lands in its OWN folder. Override OUTPUT_DIR
# or UNLEARNED_TAG if you want manual control.
#
# Examples:
#   env LAYERS=0-25 MAX_PROMPTS=600 sbatch scripts/audit/run_general_unlearning_audit.sh
#   env CANDIDATE_MODEL_PATH=/path/to/MaxEnt/.../final_model \
#       sbatch scripts/audit/run_general_unlearning_audit.sh
#   env DATA_PATH=data/general_data_part1.json \
#       JUDGE_MODEL=gemini-2.5-flash sbatch scripts/audit/run_general_unlearning_audit.sh
#   env SKIP_JUDGE=1 sbatch scripts/audit/run_general_unlearning_audit.sh
#   # tune / disable the rare-context-word ranking (uses wordfreq's Zipf
#   # frequency to surface topical vocabulary that recurs across each
#   # top feature's context windows):
#   env CONTEXT_RARE_ZIPF_CUTOFF=5.0 CONTEXT_RARE_TOP_N=20 \
#       sbatch scripts/audit/run_general_unlearning_audit.sh
#   env SKIP_CONTEXT_RARE_WORDS=1 sbatch scripts/audit/run_general_unlearning_audit.sh
#   # Full paths in judge_prompt (debug only; default is redacted):
#   env JUDGE_NO_ANONYMIZE_PATHS=1 sbatch scripts/audit/run_general_unlearning_audit.sh
#
# If ``pip install wordfreq`` fails with "No space left on device" while the
# login node's root filesystem (/) is full, point temp and pip cache at $HF_HOME:
#   mkdir -p "$HF_HOME/tmp" "$HF_HOME/pip-cache"
#   TMPDIR="$HF_HOME/tmp" PIP_CACHE_DIR="$HF_HOME/pip-cache" \\
#     pip install --no-cache-dir wordfreq
#
# Run on a non-slurm shell (e.g. local GPU) via:
#   bash scripts/audit/run_general_unlearning_audit.sh

set -euo pipefail

# --- Environment Setup ---
if [[ -f /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh ]]; then
  source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
  conda activate /home/morg/students/rashkovits/envs/snmf_env 2>/dev/null \
    || conda activate snmf_env
fi

export HF_HOME="${HF_HOME:-/home/morg/students/rashkovits/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
# Default TMPDIR under home (not /tmp): cluster login nodes often have a full
# root disk; pip/torch temp on NFS avoids Errno 28 during installs and long runs.
export TMPDIR="${TMPDIR:-$HF_HOME/tmp}"
mkdir -p "$HF_HOME" "$TMPDIR"

REPO_ROOT="/home/morg/students/rashkovits/snmf"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
mkdir -p logs "$HF_HOME"

# --- Configurable inputs ---
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/gemma-2-2b}"
CANDIDATE_MODEL_PATH="${CANDIDATE_MODEL_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/unlearned_models/RMU/bio_lr_1.00e-04_alpha_0.30_seed_42/final_model}"
SNMF_DIR="${SNMF_DIR:-${REPO_ROOT}/outputs/wmdp/audit/results_general_data_part1_300_rank}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/general_data_part1.json}"

LAYERS="${LAYERS:-10-18}"
MAX_PROMPTS="${MAX_PROMPTS:-400}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-auto}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-4}"
TOP_K_GLOBAL="${TOP_K_GLOBAL:-20}"
TOP_K_PER_LAYER="${TOP_K_PER_LAYER:-15}"
RANK_BY="${RANK_BY:-rel_delta}"
CONTEXTS_PER_FEATURE="${CONTEXTS_PER_FEATURE:-8}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-15}"
VOCAB_LENS_TOP_K="${VOCAB_LENS_TOP_K:-15}"
SKIP_VOCAB_LENS="${SKIP_VOCAB_LENS:-0}"
LENS_CENTER_UNEMBED="${LENS_CENTER_UNEMBED:-1}"
LENS_MASK_SPECIAL_TOKENS="${LENS_MASK_SPECIAL_TOKENS:-1}"
VOCAB_LENS_AGGREGATE_TOP_K="${VOCAB_LENS_AGGREGATE_TOP_K:-20}"
LENS_DELTA_WEIGHTED="${LENS_DELTA_WEIGHTED:-0}"
# Rare-context-word ranking (uses wordfreq's Zipf frequency to surface
# rare/topical vocabulary that recurs across a feature's top-activating
# contexts, not just the **emphasized** peak token). Set
# SKIP_CONTEXT_RARE_WORDS=1 to disable; CONTEXT_RARE_TOP_N=0 also disables.
CONTEXT_RARE_TOP_N="${CONTEXT_RARE_TOP_N:-15}"
CONTEXT_RARE_ZIPF_CUTOFF="${CONTEXT_RARE_ZIPF_CUTOFF:-5.5}"
CONTEXT_RARE_MIN_LEN="${CONTEXT_RARE_MIN_LEN:-3}"
SKIP_CONTEXT_RARE_WORDS="${SKIP_CONTEXT_RARE_WORDS:-0}"
SEED="${SEED:-42}"

# --- Judge LLM ---
JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-flash}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_MAX_OUTPUT_TOKENS="${JUDGE_MAX_OUTPUT_TOKENS:-1500}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-GOOGLE_API_KEY}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
# Judge prompt omits checkpoint paths by default (Python: judge_anonymize_paths=True).
# Set to 1 only for debugging (embeds full paths; can leak method/forget-set hints).
JUDGE_NO_ANONYMIZE_PATHS="${JUDGE_NO_ANONYMIZE_PATHS:-0}"

LAYER_TAG="$(printf '%s' "$LAYERS" | tr ',' '_' | tr '-' '_')"

# --- Auto-tag candidate model run for output dir ---
_cand_norm="${CANDIDATE_MODEL_PATH%/}"
_cand_norm="${_cand_norm%/final_model}"
_run_name="$(basename "$_cand_norm")"
_method_name="$(basename "$(dirname "$_cand_norm")")"
DEFAULT_CANDIDATE_TAG="$(printf '%s_%s' "$_method_name" "$_run_name" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_' '_' | tr -s '_' | sed 's/^_//;s/_$//')"
CANDIDATE_TAG="${CANDIDATE_TAG:-$DEFAULT_CANDIDATE_TAG}"

SNMF_TAG="$(basename "${SNMF_DIR%/}")"
DEFAULT_OUT="${REPO_ROOT}/outputs/wmdp/audit_general/${SNMF_TAG}__${CANDIDATE_TAG}__layers_${LAYER_TAG}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUT}}"
mkdir -p "$OUTPUT_DIR"

CMD=(
  python -u experiments/audit/general_unlearning_audit.py
  --base-model-path "$BASE_MODEL_PATH"
  --candidate-model-path "$CANDIDATE_MODEL_PATH"
  --snmf-dir "$SNMF_DIR"
  --data-path "$DATA_PATH"
  --output-dir "$OUTPUT_DIR"
  --layers "$LAYERS"
  --max-prompts "$MAX_PROMPTS"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
  --ridge-lambda "$RIDGE_LAMBDA"
  --top-k-global "$TOP_K_GLOBAL"
  --top-k-per-layer "$TOP_K_PER_LAYER"
  --rank-by "$RANK_BY"
  --contexts-per-feature "$CONTEXTS_PER_FEATURE"
  --context-window "$CONTEXT_WINDOW"
  --vocab-lens-top-k "$VOCAB_LENS_TOP_K"
  --vocab-lens-aggregate-top-k "$VOCAB_LENS_AGGREGATE_TOP_K"
  --context-rare-top-n "$CONTEXT_RARE_TOP_N"
  --context-rare-zipf-cutoff "$CONTEXT_RARE_ZIPF_CUTOFF"
  --context-rare-min-len "$CONTEXT_RARE_MIN_LEN"
  --judge-model "$JUDGE_MODEL"
  --judge-temperature "$JUDGE_TEMPERATURE"
  --judge-max-output-tokens "$JUDGE_MAX_OUTPUT_TOKENS"
  --judge-api-key-env "$JUDGE_API_KEY_ENV"
  --seed "$SEED"
)
if [[ "$SKIP_JUDGE" == "1" || "$SKIP_JUDGE" == "true" ]]; then
  CMD+=(--skip-judge)
fi
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
if [[ "$SKIP_CONTEXT_RARE_WORDS" == "1" || "$SKIP_CONTEXT_RARE_WORDS" == "true" ]]; then
  CMD+=(--skip-context-rare-words)
fi
if [[ "$JUDGE_NO_ANONYMIZE_PATHS" == "1" || "$JUDGE_NO_ANONYMIZE_PATHS" == "true" ]]; then
  CMD+=(--no-judge-anonymize-paths)
fi
echo "================================================================"
echo " SNMF general (label-free) unlearning audit + Gemini judge"
echo " Base model:        $BASE_MODEL_PATH"
echo " Candidate model:   $CANDIDATE_MODEL_PATH"
echo " SNMF dir:          $SNMF_DIR"
echo " Data:              $DATA_PATH    (max-prompts=$MAX_PROMPTS)"
echo " Layers:            $LAYERS"
echo " Output:            $OUTPUT_DIR"
echo " Top-K global:      $TOP_K_GLOBAL  (per-layer=$TOP_K_PER_LAYER, rank_by=$RANK_BY)"
echo " Contexts/feature:  $CONTEXTS_PER_FEATURE  (window=$CONTEXT_WINDOW)"
echo " Vocab-lens top-k:  $VOCAB_LENS_TOP_K  (skip=$SKIP_VOCAB_LENS  center=$LENS_CENTER_UNEMBED  mask_specials=$LENS_MASK_SPECIAL_TOKENS)"
echo " Aggregate top-k:   $VOCAB_LENS_AGGREGATE_TOP_K  (delta_weighted=$LENS_DELTA_WEIGHTED)"
echo " Rare-context words: top_n=$CONTEXT_RARE_TOP_N  zipf_cutoff=$CONTEXT_RARE_ZIPF_CUTOFF  min_len=$CONTEXT_RARE_MIN_LEN  (skip=$SKIP_CONTEXT_RARE_WORDS)"
echo " Judge model:       $JUDGE_MODEL  (skip=$SKIP_JUDGE  no_anonymize_paths=$JUDGE_NO_ANONYMIZE_PATHS)"
echo "================================================================"

"${CMD[@]}"

echo "================================================================"
echo " Audit done. Inspect:"
echo "   $OUTPUT_DIR/audit_summary.json"
echo "   $OUTPUT_DIR/audit_summary_per_layer.csv"
echo "   $OUTPUT_DIR/judge_prompt.txt"
echo "   $OUTPUT_DIR/judge_response.json"
echo "================================================================"
