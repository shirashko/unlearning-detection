#!/bin/bash

# End-to-end iter-3 pipeline for WMDP-bio on Gemma-2-2b.
#
# Builds on iter-2 top-up variant A (the only iter-2 top-up that kept MMLU intact):
#   base    = local_models/wmdp/iter2_topup_data_part2_thr018_both_up_down/bio_retain_and_neutral
#   iter-2 base after that top-up: wmdp_bio=0.4273  mmlu=0.4837
#     (iter-2 base alone was wmdp_bio=0.4800  mmlu=0.4950)
#
# iter-3 re-fits SNMF on that checkpoint's residual activations on a FRESH bio split
# (bio_data_part3.json — unused so far; iter-1 used bio_data.json, iter-2 used _part2),
# re-runs the supervised role analysis, then applies the variant-A ablation recipe:
#   role_label_bases = bio_retain + neutral   (AND — strict specificity)
#   role_basis_combine = all
#   role_assignment_threshold = 0.18          (same as winning iter-2 top-up)
#   both up_proj and down_proj                (DOWN_PROJ_ONLY=0)
#   span_projection_scale = 1.0               (full projection)
#
# RANDOM_BASELINE=1 this time (the iter-2 sweep declared it but the wrapper never
# actually exported it, so we have no matched-count control for that round — fix here).
#
# Usage:
#   bash scripts/wmdp/run_iter3_pipeline.sh
#
# The underlying script `run_snmf_forget_pipeline.sh` is the single sbatch job that
# does: (1) train_snmf.py, (2) wmdp_bio_analyze_snmf_results.py, (3)
# create_forget_ablated_model.py + eval. We just override its env defaults.
#
# Logs:  logs/snmf_forget_pipe_<jobid>.{out,err}

set -euo pipefail

REPO_ROOT="/home/morg/students/rashkovits/snmf"
cd "$REPO_ROOT"

SBATCH_SCRIPT="scripts/wmdp/run_snmf_forget_pipeline.sh"

# --- Base model to iterate on (iter-2 + variant-A top-up) -----------------------------
ITER3_BASE="/home/morg/students/rashkovits/snmf/local_models/wmdp/iter2_topup_data_part2_thr018_both_up_down/bio_retain_and_neutral"

# --- Fresh bio data split (unused in iter-1 / iter-2) ---------------------------------
DATA_PART3="${REPO_ROOT}/data/bio_data_part3.json"

# --- iter-3 artifact paths ------------------------------------------------------------
# Groups like iter1/, iter2/, iter2_topup_.../. iter-3 lives under its own group with
# one ckpt per selection config (here: bio_retain_and_neutral, replicating variant-A).
ITER3_GROUP="iter3_data_part3_thr018_both_up_down"
ITER3_CONFIG="bio_retain_and_neutral"
SNMF_OUTPUT_DIR="outputs/wmdp/results_data_part3_gemma2_2b_iter3_thr018_both_up_down"
ABLATION_OUTPUT_DIR="outputs/wmdp/forget_ablation_data_part3_gemma2_2b_${ITER3_GROUP}_${ITER3_CONFIG}"
SAVE_PATH="local_models/wmdp/${ITER3_GROUP}/${ITER3_CONFIG}"
SAVE_PATH_RANDOM="${SAVE_PATH}_random"

# --- Robust WMDP-bio eval (contamination-filtered) -----------------------------------
# Use the `wmdp_bio_robust` task group (6 sub-categories aggregated by size) instead of
# the stock lm-eval `wmdp_bio` task. YAMLs live in wmdp_bio_categorized_mcqa/.
EVAL_WMDP_INCLUDE_PATH="${REPO_ROOT}/wmdp_bio_categorized_mcqa"
EVAL_WMDP_TASK_NAME="wmdp_bio_robust"

# --- SNMF training config ------------------------------------------------------------
# Pinned to match the SNMF basis that produced $ITER3_BASE (i.e. the SNMF dir
# outputs/wmdp/results_data_part2_gemma2_2b_iter2_thr022_down_proj_only, see its
# config.json). Only `rank` actually differs from the pipeline defaults (300 vs 100);
# the rest are pinned for self-documentation so this script is immune to pipeline
# default drift.
SNMF_RANK="300"
SNMF_LAYERS="0-25"
SNMF_MODE="mlp_intermediate"
SNMF_INIT="svd"
SNMF_SPARSITY="0.01"
SNMF_MAX_ITER="3000"
SNMF_TRAIN_BATCH_SIZE="8"
SNMF_TRAIN_SEED="42"

# --- Supervised-analysis config ------------------------------------------------------
# Threshold used when baking role_labels_by_basis into the supervised JSONs. Kept at
# 0.05 to match the iter-2 data_part2 analysis summary; the ablation step below still
# re-computes labels on the fly at ROLE_ASSIGNMENT_THRESHOLD=0.18.
ANALYZE_ROLE_THRESHOLD_PIN="0.05"
ANALYZE_SEED_PIN="42"

echo "================================================================"
echo " iter-3 WMDP-bio pipeline (Gemma-2-2b)"
echo " Base model:      $ITER3_BASE"
echo " SNMF data:       $DATA_PART3"
echo " SNMF dir:        $SNMF_OUTPUT_DIR"
echo " Ablation dir:    $ABLATION_OUTPUT_DIR"
echo " Learned save:    $SAVE_PATH"
echo " Random save:     $SAVE_PATH_RANDOM"
echo " SNMF training:   rank=$SNMF_RANK layers=$SNMF_LAYERS mode=$SNMF_MODE init=$SNMF_INIT"
echo "                  sparsity=$SNMF_SPARSITY max_iter=$SNMF_MAX_ITER batch=$SNMF_TRAIN_BATCH_SIZE seed=$SNMF_TRAIN_SEED"
echo "                  (matches the iter-2 data_part2 SNMF basis behind \$ITER3_BASE)"
echo " Analyze:         role_threshold=$ANALYZE_ROLE_THRESHOLD_PIN seed=$ANALYZE_SEED_PIN"
echo " Recipe:          role_label_bases='bio_retain neutral' combine=all thr=0.18"
echo "                  DOWN_PROJ_ONLY=0  SPAN_PROJECTION_SCALE=1.0"
echo "                  RANDOM_BASELINE=1 (matched-count control)"
echo " Eval:            EVAL_MODE=wmdp_bio_categorized  task=$EVAL_WMDP_TASK_NAME (+ MMLU)"
echo "================================================================"

# Hand off to the full SNMF -> analyze -> ablate sbatch job, overriding its defaults.
MODEL_PATH="$ITER3_BASE" \
DATA_PATH="$DATA_PART3" \
SNMF_OUTPUT_DIR="$SNMF_OUTPUT_DIR" \
ABLATION_OUTPUT_DIR="$ABLATION_OUTPUT_DIR" \
SAVE_PATH="$SAVE_PATH" \
SAVE_PATH_RANDOM="$SAVE_PATH_RANDOM" \
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
ROLE_ASSIGNMENT_THRESHOLD="0.18" \
DOWN_PROJ_ONLY="0" \
SPAN_PROJECTION_SCALE="1.0" \
RANDOM_BASELINE="1" \
RANDOM_SEED="1234" \
SKIP_PRE_EVAL="1" \
EVAL_MODE="wmdp_bio_categorized" \
EVAL_WMDP_INCLUDE_PATH="$EVAL_WMDP_INCLUDE_PATH" \
EVAL_WMDP_TASK_NAME="$EVAL_WMDP_TASK_NAME" \
  sbatch --job-name="iter3_${ITER3_CONFIG}" "$SBATCH_SCRIPT"

echo
echo "================================================================"
echo " Submitted iter-3 job. Monitor:    squeue -u \$USER"
echo " Log:                              logs/snmf_forget_pipe_<jobid>.out"
echo " Learned eval JSON (when done):    $SAVE_PATH/ablation_eval_comparison.json"
echo " Random-matched eval (when done):  nested under random_baseline.after in same JSON"
echo "================================================================"
