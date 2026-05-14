#!/bin/bash

# End-to-end iter-1 pipeline for WMDP-bio on Gemma-2-2b, "Config 1" variant.
#
# Background:
# iter-1 (the original run) trained SNMF on data/bio_data_part1.json and applied
# ablation with a POOLED retain basis (group_sums['pooled_retain']). The
# "bio_retain AND neutral" strict-AND recipe — standard from iter-2 onward — was
# never tried on the iter-1 basis. This wrapper fills that gap: same stock
# Gemma-2-2b base, same iter-1 SNMF basis, but the Config-1 ablation recipe.
#
#   base model                 = /home/morg/.../Localized-UNDO/models/wmdp/gemma-2-2b (stock)
#   SNMF basis (reused)        = outputs/wmdp/results_data_part1_gemma2_2b
#                                (rank=300, layers=0-25, mode=mlp_intermediate,
#                                 sparsity=0.01, init=svd, seed=42 — same recipe as iter-3/4)
#   SNMF data                  = data/bio_data_part1.json
#   role_label_bases           = bio_retain + neutral        (AND — strict specificity)
#   role_basis_combine         = all
#   role_assignment_threshold  = 0.22                        (user request)
#   both up_proj and down_proj (DOWN_PROJ_ONLY=0)
#   span_projection_scale      = 1.0                         (full projection)
#   RANDOM_BASELINE=1                                        (matched-count control)
#   EVAL_MODE=wmdp_bio                                       (stock wmdp_bio + MMLU)
#   SKIP_PRE_EVAL=0                                          (before / learned / random all measured)
#
# Since the iter-1 SNMF basis and its supervised analysis JSONs already exist
# (group_sums over bio_forget / neutral / bio_retain / pooled_retain are there,
# and create_forget_ablated_model.py recomputes role_labels_by_basis on the fly
# at ROLE_ASSIGNMENT_THRESHOLD), we reuse them: SKIP_TRAIN=1, SKIP_ANALYZE=1.
#
# Usage:
#   bash scripts/wmdp/run_iter1_config1_pipeline.sh
# Override threshold / data part / base / force retrain via env:
#   env THRESHOLD=0.18 bash scripts/wmdp/run_iter1_config1_pipeline.sh
#   env SKIP_TRAIN=0 SKIP_ANALYZE=0 bash scripts/wmdp/run_iter1_config1_pipeline.sh
#
# Logs:  logs/snmf_forget_pipe_<jobid>.{out,err}

set -euo pipefail

REPO_ROOT="/home/morg/students/rashkovits/snmf"
cd "$REPO_ROOT"

SBATCH_SCRIPT="scripts/wmdp/run_snmf_forget_pipeline.sh"

# --- Base model: stock Gemma-2-2b (same local copy iter-1 was originally built on) ----
ITER1_BASE="${ITER1_BASE:-/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/gemma-2-2b}"

# --- SNMF training data (part 1, same as the original iter-1 basis) -------------------
DATA_PART1="${REPO_ROOT}/data/bio_data_part1.json"

# --- Ablation threshold ---------------------------------------------------------------
THRESHOLD="${THRESHOLD:-0.22}"
THR_TAG="thr$(printf '%s' "$THRESHOLD" | tr -d '.')"    # 0.22 -> "thr022"

# --- iter-1 Config-1 artifact paths ---------------------------------------------------
# Keep iter-1's existing SNMF dir (results_data_part1_gemma2_2b) so we don't
# duplicate the basis. Only the ablation outputs / checkpoint are namespaced
# to the new Config-1 recipe.
ITER1_GROUP="iter1_data_part1_${THR_TAG}_both_up_down"
ITER1_CONFIG="bio_retain_and_neutral"
SNMF_OUTPUT_DIR="${SNMF_OUTPUT_DIR:-outputs/wmdp/results_data_part1_gemma2_2b}"
ABLATION_OUTPUT_DIR="${ABLATION_OUTPUT_DIR:-outputs/wmdp/forget_ablation_data_part1_gemma2_2b_${ITER1_GROUP}_${ITER1_CONFIG}}"
SAVE_PATH="${SAVE_PATH:-local_models/wmdp/${ITER1_GROUP}/${ITER1_CONFIG}}"
SAVE_PATH_RANDOM="${SAVE_PATH_RANDOM:-${SAVE_PATH}_random}"

# --- Skip train/analyze by default (reuse existing SNMF basis) ------------------------
# The iter-1 SNMF dir already has config + snmf_factors.pt per layer + supervised
# analysis JSONs with group_sums/group_counts over bio_forget / neutral / bio_retain
# / pooled_retain. That's enough for the ablation step to recompute role labels at
# THRESHOLD against the bio_retain + neutral strict-AND basis. Flip these to 0 if
# you want to rebuild the basis from scratch.
SKIP_TRAIN="${SKIP_TRAIN:-1}"
SKIP_ANALYZE="${SKIP_ANALYZE:-1}"

# --- SNMF training config (only used if SKIP_TRAIN=0; pinned to iter-1 basis) ---------
SNMF_RANK="300"
SNMF_LAYERS="0-25"
SNMF_MODE="mlp_intermediate"
SNMF_INIT="svd"
SNMF_SPARSITY="0.01"
SNMF_MAX_ITER="3000"
SNMF_TRAIN_BATCH_SIZE="8"
SNMF_TRAIN_SEED="42"

# --- Supervised-analysis config (only used if SKIP_ANALYZE=0) -------------------------
# Threshold used when baking role_labels_by_basis into the supervised JSONs. Kept at
# 0.05 for symmetry with iter-3/4; the ablation step recomputes labels on the fly at
# ROLE_ASSIGNMENT_THRESHOLD=$THRESHOLD.
ANALYZE_ROLE_THRESHOLD_PIN="0.05"
ANALYZE_SEED_PIN="42"

# --- Eval: stock wmdp_bio + MMLU (SKIP_PRE_EVAL=0 — before / learned / random) --------
EVAL_MODE_SEL="wmdp_bio"

echo "================================================================"
echo " iter-1 Config-1 WMDP-bio pipeline (Gemma-2-2b)"
echo " Base model:      $ITER1_BASE"
echo " SNMF data:       $DATA_PART1"
echo " SNMF dir:        $SNMF_OUTPUT_DIR   (reused; SKIP_TRAIN=$SKIP_TRAIN  SKIP_ANALYZE=$SKIP_ANALYZE)"
echo " Ablation dir:    $ABLATION_OUTPUT_DIR"
echo " Learned save:    $SAVE_PATH"
echo " Random save:     $SAVE_PATH_RANDOM"
echo " SNMF training:   rank=$SNMF_RANK layers=$SNMF_LAYERS mode=$SNMF_MODE init=$SNMF_INIT"
echo "                  sparsity=$SNMF_SPARSITY max_iter=$SNMF_MAX_ITER batch=$SNMF_TRAIN_BATCH_SIZE seed=$SNMF_TRAIN_SEED"
echo " Analyze:         role_threshold_baked=$ANALYZE_ROLE_THRESHOLD_PIN seed=$ANALYZE_SEED_PIN"
echo " Recipe:          role_label_bases='bio_retain neutral' combine=all thr=$THRESHOLD"
echo "                  DOWN_PROJ_ONLY=0  SPAN_PROJECTION_SCALE=1.0"
echo "                  RANDOM_BASELINE=1 (matched-count control)"
echo " Eval:            EVAL_MODE=$EVAL_MODE_SEL  (stock wmdp_bio + MMLU)"
echo "                  SKIP_PRE_EVAL=0 — before / learned / random all measured"
echo "================================================================"

MODEL_PATH="$ITER1_BASE" \
DATA_PATH="$DATA_PART1" \
SNMF_OUTPUT_DIR="$SNMF_OUTPUT_DIR" \
ABLATION_OUTPUT_DIR="$ABLATION_OUTPUT_DIR" \
SAVE_PATH="$SAVE_PATH" \
SAVE_PATH_RANDOM="$SAVE_PATH_RANDOM" \
SKIP_TRAIN="$SKIP_TRAIN" \
SKIP_ANALYZE="$SKIP_ANALYZE" \
RANK="$SNMF_RANK" \
LAYERS="$SNMF_LAYERS" \
SNMF_MODE="$SNMF_MODE" \
SNMF_INIT="$SNMF_INIT" \
SPARSITY="$SNMF_SPARSITY" \
MAX_ITER="$SNMF_MAX_ITER" \
TRAIN_BATCH_SIZE="$SNMF_TRAIN_BATCH_SIZE" \
TRAIN_SEED="$SNMF_TRAIN_SEED" \
ANALYZE_ROLE_THRESHOLD="$ANALYZE_ROLE_THRESHOLD_PIN" \
ANALYZE_SEED="$ANALYZE_SEED_PIN" \
ROLE_LABEL_BASES="bio_retain neutral" \
ROLE_BASIS_COMBINE="all" \
ROLE_ASSIGNMENT_THRESHOLD="$THRESHOLD" \
DOWN_PROJ_ONLY="0" \
SPAN_PROJECTION_SCALE="1.0" \
RANDOM_BASELINE="1" \
RANDOM_SEED="1234" \
SKIP_PRE_EVAL="0" \
EVAL_MODE="$EVAL_MODE_SEL" \
EVAL_LARGE="1" \
EVAL_NO_MMLU="0" \
  sbatch --job-name="iter1_cfg1_${THR_TAG}_${ITER1_CONFIG}" "$SBATCH_SCRIPT"

echo
echo "================================================================"
echo " Submitted iter-1 Config-1 job (threshold=$THRESHOLD). Monitor:"
echo "   squeue -u \$USER"
echo " Log:                              logs/snmf_forget_pipe_<jobid>.out"
echo " Learned eval JSON (when done):    $SAVE_PATH/ablation_eval_comparison.json"
echo "   will contain: before / after (learned) / random_baseline.after,"
echo "   each with wmdp_bio + mmlu."
echo "================================================================"
