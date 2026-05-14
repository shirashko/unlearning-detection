#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=snmf_forget_wmdp_bio
#SBATCH --output=logs/snmf_forget_pipe_%j.out
#SBATCH --error=logs/snmf_forget_pipe_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

# End-to-end: SNMF training -> supervised analysis -> forget ablation checkpoint (+ optional eval).
#
# Defaults: Config-1 WMDP-bio run for the base Gemma-2-2B checkpoint at
#   /home/morg/students/rashkovits/Localized-UNDO/models/wmdp/gemma-2-2b
# Fits SNMF (rank=100) on bio_data.json activations of that base model, then runs a
# pooled+bio_retain forget ablation (threshold=0.22, down_proj_only=1).
#
# Override anything via env before sbatch, e.g.:
#   RANK=200 sbatch scripts/wmdp/run_snmf_forget_pipeline.sh
#
# Skip steps (reuse existing artifacts):
#   SKIP_TRAIN=1    — skip train_snmf.py (expects layer_* under SNMF_OUTPUT_DIR)
#   SKIP_ANALYZE=1  — skip wmdp_bio_analyze_snmf_results.py
#   SKIP_PROBE=0    — run wmdp_bio_probe_snmf_results.py between analyze and ablation
#                     (default 1 = skip; set to 0 only if you use SELECTION_MODE=probe_topk or intersect)
#
# Latent-selection mode for the ablation step:
#   SELECTION_MODE=log_ratio  (default) — legacy role_labels_by_basis rule
#   SELECTION_MODE=probe_topk           — top-PROBE_TOP_K by L1-logistic probe weight
#   SELECTION_MODE=intersect            — log_ratio ∩ probe_topk

set -euo pipefail

# --- Environment Setup ---
source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate snmf_env

# --- Space & Cache Management ---
export HF_HOME="${HF_HOME:-/home/morg/students/rashkovits/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export TMPDIR="${TMPDIR:-$HF_HOME}"

# --- Project Setup ---
REPO_ROOT="/home/morg/students/rashkovits/snmf"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

mkdir -p logs "$HF_HOME"

# ========== Config-1 defaults (WMDP-bio, Gemma-2-2b, base model, rank=100) ==========
# Base model is the fresh Localized-UNDO Gemma-2-2b checkpoint; SNMF is fit on its activations.
MODEL_PATH="${MODEL_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/gemma-2-2b}"
DATA_PATH="${DATA_PATH:-/home/morg/students/rashkovits/snmf/data/bio_data.json}"

# SNMF factors + analysis JSONs for this rank=100 base-model run.
SNMF_OUTPUT_DIR="${SNMF_OUTPUT_DIR:-outputs/wmdp/results_bio_data_gemma2_2b_base_rank300_thr022_down_proj_only}"
# Forget-ablation metadata dir (separate from the SNMF layer_* tree).
ABLATION_OUTPUT_DIR="${ABLATION_OUTPUT_DIR:-outputs/wmdp/forget_ablation_bio_data_gemma2_2b_base_rank300_thr022_down_proj_only}"

# --- train_snmf.py (mirrors scripts/wmdp/train_snmf.sh) ---
LAYERS="${LAYERS:-0-25}"                     # Gemma-2-2b has 26 layers => indices 0..25
RANK="${RANK:-300}"
SNMF_MODE="${SNMF_MODE:-mlp_intermediate}"
SNMF_INIT="${SNMF_INIT:-svd}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
SPARSITY="${SPARSITY:-0.01}"
MAX_ITER="${MAX_ITER:-3000}"
TRAIN_SEED="${TRAIN_SEED:-42}"
# Note: train_snmf.py pins patience=1500 internally and normalize=False by default (no CLI override
# from this script), matching the Config-1 spec.

# --- wmdp_bio_analyze_snmf_results.py (mirrors scripts/wmdp/run_analyze_wmdp_bio.sh) ---
SUMMARY_FILE="${SUMMARY_FILE:-analysis_summary_wmdp_bio.json}"
ANALYZE_DEVICE="${ANALYZE_DEVICE:-cuda}"
ANALYZE_SEED="${ANALYZE_SEED:-42}"
# Threshold used when BAKING role_labels_by_basis into the supervised JSONs. Kept permissive
# because the ablation step below recomputes labels on the fly at $ROLE_ASSIGNMENT_THRESHOLD.
ANALYZE_ROLE_THRESHOLD="${ANALYZE_ROLE_THRESHOLD:-0.05}"
ANALYZE_CTX_TOP_N="${ANALYZE_CTX_TOP_N:-10}"
ANALYZE_CTX_WINDOW="${ANALYZE_CTX_WINDOW:-15}"

# --- create_forget_ablated_model (mirrors scripts/wmdp/run_create_forget_ablated_model.sh) ---
# Ablated checkpoint path for this base-model rank=100 run.
SAVE_PATH="${SAVE_PATH:-local_models/wmdp/iter1/pooled_and_bio_retain_thr022_down_proj_only_base_rank300}"
SAVE_PATH_RANDOM="${SAVE_PATH_RANDOM:-${SAVE_PATH}_random}"
SUPERVISED_JSON_FILENAME="${SUPERVISED_JSON_FILENAME:-feature_analysis_supervised_wmdp_bio.json}"
mkdir -p "$ABLATION_OUTPUT_DIR"

# Config-1 ablation knobs.
FORGET_ROLES="${FORGET_ROLES:-bio_forget_lean}"
ROLE_LABEL_BASES="${ROLE_LABEL_BASES:-pooled bio_retain}"
ROLE_BASIS_COMBINE="${ROLE_BASIS_COMBINE:-all}"
ROLE_ASSIGNMENT_THRESHOLD="${ROLE_ASSIGNMENT_THRESHOLD:-0.22}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-6}"
SPAN_PROJECTION_SCALE="${SPAN_PROJECTION_SCALE:-1.0}"
ABLATION_DEVICE="${ABLATION_DEVICE:-auto}"
DOWN_PROJ_ONLY="${DOWN_PROJ_ONLY:-1}"

# Random matched-count baseline controls.
RANDOM_BASELINE="${RANDOM_BASELINE:-0}"
RANDOM_SEED="${RANDOM_SEED:-1234}"

# Eval controls (run by default before/after ablation).
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_PRE_EVAL="${SKIP_PRE_EVAL:-1}"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
EVAL_MODE="${EVAL_MODE:-wmdp_bio}"
EVAL_LARGE="${EVAL_LARGE:-1}"
EVAL_NO_MMLU="${EVAL_NO_MMLU:-0}"
EVAL_WMDP_INCLUDE_PATH="${EVAL_WMDP_INCLUDE_PATH:-}"
EVAL_WMDP_TASK_NAME="${EVAL_WMDP_TASK_NAME:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_ANALYZE="${SKIP_ANALYZE:-0}"
SKIP_PROBE="${SKIP_PROBE:-1}"

# Probe step controls (only used when SKIP_PROBE=0 and/or SELECTION_MODE != log_ratio).
PROBE_WEIGHTS_FILENAME="${PROBE_WEIGHTS_FILENAME:-probe_weights_wmdp_bio.json}"
PROBE_SUMMARY_FILE="${PROBE_SUMMARY_FILE:-probe_summary_wmdp_bio.json}"
PROBE_C_GRID="${PROBE_C_GRID:-0.01,0.1,1.0,10.0}"
PROBE_CV_FOLDS="${PROBE_CV_FOLDS:-5}"
PROBE_TEST_SIZE="${PROBE_TEST_SIZE:-0.2}"
PROBE_SEED="${PROBE_SEED:-42}"
PROBE_MAX_ITER="${PROBE_MAX_ITER:-2000}"
PROBE_FEATURE_AGG="${PROBE_FEATURE_AGG:-prompt_max}"

# Latent-selection mode and probe knobs passed into create_forget_ablated_model.py.
SELECTION_MODE="${SELECTION_MODE:-log_ratio}"
PROBE_TOP_K="${PROBE_TOP_K:-5}"

# --- Parallelism Optimization ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "================================================================"
echo " SNMF → analyze → forget ablation pipeline (Config-1 iter-2)"
echo " Node: ${SLURMD_NODENAME:-local}"
echo " Base model (train + ablate):  $MODEL_PATH"
echo " Data path:                    $DATA_PATH"
echo " SNMF output dir:              $SNMF_OUTPUT_DIR"
echo " Ablation run artifacts:       $ABLATION_OUTPUT_DIR"
echo " Learned ablation save:        $SAVE_PATH"
echo " Random baseline save:         $SAVE_PATH_RANDOM  (if RANDOM_BASELINE=1)"
echo " SNMF:                         rank=$RANK layers=$LAYERS mode=$SNMF_MODE init=$SNMF_INIT max_iter=$MAX_ITER sparsity=$SPARSITY"
echo " Role bases / combine / thr:   $ROLE_LABEL_BASES / $ROLE_BASIS_COMBINE / $ROLE_ASSIGNMENT_THRESHOLD"
echo " Forget roles:                 $FORGET_ROLES"
echo " DOWN_PROJ_ONLY:               $DOWN_PROJ_ONLY    SPAN_PROJECTION_SCALE: $SPAN_PROJECTION_SCALE"
echo " Eval mode:                    $EVAL_MODE   (SKIP_EVAL=$SKIP_EVAL SKIP_PRE_EVAL=$SKIP_PRE_EVAL)"
echo " SKIP_TRAIN=$SKIP_TRAIN  SKIP_ANALYZE=$SKIP_ANALYZE  SKIP_PROBE=$SKIP_PROBE"
echo " SELECTION_MODE=$SELECTION_MODE  PROBE_TOP_K=$PROBE_TOP_K  PROBE_FEATURE_AGG=$PROBE_FEATURE_AGG"
echo "================================================================"

mkdir -p "$SNMF_OUTPUT_DIR"

# ========== 1) Train SNMF ==========
if [[ "$SKIP_TRAIN" == "1" ]]; then
  echo "[1/3] SKIP_TRAIN=1 — skipping train_snmf.py"
else
  echo "[1/3] Training SNMF → $SNMF_OUTPUT_DIR"
  python train_snmf.py \
    --model-path "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --output-dir "$SNMF_OUTPUT_DIR" \
    --layers "$LAYERS" \
    --rank "$RANK" \
    --mode "$SNMF_MODE" \
    --init "$SNMF_INIT" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --device "$TRAIN_DEVICE" \
    --sparsity "$SPARSITY" \
    --max-iter "$MAX_ITER" \
    --seed "$TRAIN_SEED"
  echo "[1/3] Training finished."
fi

# ========== 2) Analyze (supervised roles) ==========
if [[ "$SKIP_ANALYZE" == "1" ]]; then
  echo "[2/3] SKIP_ANALYZE=1 — skipping wmdp_bio_analyze_snmf_results.py"
else
  echo "[2/3] Analyzing SNMF results in $SNMF_OUTPUT_DIR"
  python wmdp_bio_analyze_snmf_results.py \
    --model-path "$MODEL_PATH" \
    --results-dir "$SNMF_OUTPUT_DIR" \
    --summary-filename "$SUMMARY_FILE" \
    --data-path "$DATA_PATH" \
    --role-assignment-threshold "$ANALYZE_ROLE_THRESHOLD" \
    --device "$ANALYZE_DEVICE" \
    --seed "$ANALYZE_SEED" \
    --activation-context-top-n "$ANALYZE_CTX_TOP_N" \
    --activation-context-window "$ANALYZE_CTX_WINDOW"
  echo "[2/3] Analysis finished ($SNMF_OUTPUT_DIR/$SUMMARY_FILE)."
fi

# ========== 2b) Probe (optional) ==========
# Runs only when SKIP_PROBE=0 OR when SELECTION_MODE requires probe weights. This is a lightweight
# per-layer L1-logistic regression fit on SNMF prompt-max activations and writes
# layer_*/$PROBE_WEIGHTS_FILENAME which the ablation step can then consume via --selection-mode.
NEED_PROBE=0
if [[ "$SKIP_PROBE" != "1" ]]; then
  NEED_PROBE=1
fi
if [[ "$SELECTION_MODE" != "log_ratio" && "$SKIP_PROBE" == "1" ]]; then
  echo "[2b/3] NOTE: SELECTION_MODE=$SELECTION_MODE requires probe weights but SKIP_PROBE=1;"
  echo "       assuming probe weights already exist under $SNMF_OUTPUT_DIR/layer_*/$PROBE_WEIGHTS_FILENAME."
fi
if [[ "$NEED_PROBE" == "1" ]]; then
  echo "[2b/3] Fitting L1-logistic probe per layer → $SNMF_OUTPUT_DIR/layer_*/$PROBE_WEIGHTS_FILENAME"
  python wmdp_bio_probe_snmf_results.py \
    --results-dir "$SNMF_OUTPUT_DIR" \
    --data-path "$DATA_PATH" \
    --summary-filename "$PROBE_SUMMARY_FILE" \
    --weights-filename "$PROBE_WEIGHTS_FILENAME" \
    --seed "$PROBE_SEED" \
    --test-size "$PROBE_TEST_SIZE" \
    --cv-folds "$PROBE_CV_FOLDS" \
    --c-grid "$PROBE_C_GRID" \
    --max-iter "$PROBE_MAX_ITER" \
    --feature-aggregation "$PROBE_FEATURE_AGG"
  echo "[2b/3] Probe fit finished ($SNMF_OUTPUT_DIR/$PROBE_SUMMARY_FILE)."
fi

# ========== 3) Forget ablation ==========
echo "[3/3] Forget ablation (read $SNMF_OUTPUT_DIR)"
EVAL_ARGS=()
if [[ "$SKIP_EVAL" == "1" ]]; then
  EVAL_ARGS+=(--skip-eval)
fi
if [[ "$SKIP_PRE_EVAL" == "1" ]]; then
  EVAL_ARGS+=(--skip-pre-eval)
fi
if [[ "$EVAL_LARGE" == "1" ]]; then
  EVAL_ARGS+=(--eval-large)
fi
if [[ "$EVAL_NO_MMLU" == "1" ]]; then
  EVAL_ARGS+=(--eval-no-mmlu)
fi
if [[ -n "$EVAL_WMDP_INCLUDE_PATH" ]]; then
  EVAL_ARGS+=(--eval-wmdp-include-path "$EVAL_WMDP_INCLUDE_PATH")
fi
if [[ -n "$EVAL_WMDP_TASK_NAME" ]]; then
  EVAL_ARGS+=(--eval-wmdp-task-name "$EVAL_WMDP_TASK_NAME")
fi
RANDOM_ARGS=()
if [[ "$RANDOM_BASELINE" == "1" ]]; then
  RANDOM_ARGS+=(--random-baseline)
  RANDOM_ARGS+=(--save-path-random "$SAVE_PATH_RANDOM")
  RANDOM_ARGS+=(--random-seed "$RANDOM_SEED")
fi
DOWN_ONLY_ARGS=()
if [[ "$DOWN_PROJ_ONLY" == "1" ]]; then
  DOWN_ONLY_ARGS+=(--down-proj-only)
fi

BASIS_ARGS=()
if [[ -n "$ROLE_LABEL_BASES" ]]; then
  # shellcheck disable=SC2206
  BASIS_ARGS=(--role-label-bases $ROLE_LABEL_BASES --role-basis-combine "$ROLE_BASIS_COMBINE")
  if [[ -n "$ROLE_ASSIGNMENT_THRESHOLD" ]]; then
    BASIS_ARGS+=(--role-assignment-threshold "$ROLE_ASSIGNMENT_THRESHOLD")
  fi
fi

python create_forget_ablated_model.py \
  --model-path "$MODEL_PATH" \
  --results-dir "$SNMF_OUTPUT_DIR" \
  --supervised-json-filename "$SUPERVISED_JSON_FILENAME" \
  --save-path "$SAVE_PATH" \
  --forget-roles $FORGET_ROLES \
  "${BASIS_ARGS[@]}" \
  --ridge-lambda "$RIDGE_LAMBDA" \
  --span-projection-scale "$SPAN_PROJECTION_SCALE" \
  --device "$ABLATION_DEVICE" \
  --eval-device "$EVAL_DEVICE" \
  --eval-mode "$EVAL_MODE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --selection-mode "$SELECTION_MODE" \
  --probe-weights-filename "$PROBE_WEIGHTS_FILENAME" \
  --probe-top-k "$PROBE_TOP_K" \
  "${RANDOM_ARGS[@]}" \
  "${DOWN_ONLY_ARGS[@]}" \
  "${EVAL_ARGS[@]}"

echo "[3/3] Done. Checkpoint: $SAVE_PATH"
echo "================================================================"
echo " Pipeline complete."
echo " SNMF dir:     $SNMF_OUTPUT_DIR"
echo " Ablation log: $ABLATION_OUTPUT_DIR"
echo " Eval JSON:    $SAVE_PATH/ablation_eval_comparison.json"
echo "================================================================"
