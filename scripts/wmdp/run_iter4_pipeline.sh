#!/bin/bash

# End-to-end iter-4 pipeline for WMDP-bio on Gemma-2-2b.
#
# Builds on the canonical iter-3 checkpoint (job 276246):
#   base = local_models/wmdp/iter3_data_part3_thr018_both_up_down/bio_retain_and_neutral
#     iter-3 eval on wmdp_bio_robust: 0.3468 (vs random-matched 0.3594). MMLU was not
#     measured in iter-3 (EVAL_MODE=wmdp_bio_categorized silently drops MMLU).
#
# iter-4 re-fits SNMF on iter-3's residual activations on a FRESH bio split
# (bio_data_part4.json — unused in iter-1 / iter-2 / iter-3), re-runs the supervised
# role analysis, then applies the iter-3 variant-A ablation recipe with a STRICTER
# threshold of 0.30 (the iter-3b lesson: thr=0.18 admits too many weakly-forget-leaning
# latents). Full pipeline: train_snmf.py -> wmdp_bio_analyze_snmf_results.py ->
# create_forget_ablated_model.py + eval, with MMLU measured this time
# (EVAL_MODE=wmdp_bio instead of wmdp_bio_categorized).
#
#   role_label_bases = bio_retain + neutral   (AND — strict specificity)
#   role_basis_combine = all
#   role_assignment_threshold = 0.30          (iter-3 used 0.18; stricter this time)
#   both up_proj and down_proj                (DOWN_PROJ_ONLY=0)
#   span_projection_scale = 1.0               (full projection)
#   RANDOM_BASELINE=1                         (matched-count control)
#
# Usage:
#   bash scripts/wmdp/run_iter4_pipeline.sh
# Override base / threshold / data part (e.g. to iterate from iter-3b instead):
#   ITER4_BASE=/path/to/other/model  bash scripts/wmdp/run_iter4_pipeline.sh
#   THRESHOLD=0.35                   bash scripts/wmdp/run_iter4_pipeline.sh
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

# --- Base model to iterate on (canonical iter-3) --------------------------------------
ITER4_BASE="${ITER4_BASE:-/home/morg/students/rashkovits/snmf/local_models/wmdp/iter3_data_part3_thr018_both_up_down/bio_retain_and_neutral}"

# --- Fresh bio data split (unused in iter-1 / iter-2 / iter-3) ------------------------
DATA_PART4="${REPO_ROOT}/data/bio_data_part4.json"

# --- Ablation threshold (stricter than iter-3's 0.18) ---------------------------------
THRESHOLD="${THRESHOLD:-0.30}"
THR_TAG="thr$(printf '%s' "$THRESHOLD" | tr -d '.')"    # 0.30 -> "thr030"

# --- iter-4 artifact paths ------------------------------------------------------------
# Groups like iter1/, iter2/, iter3_.../. iter-4 lives under its own group with one ckpt
# per selection config (here: bio_retain_and_neutral, replicating the strict-AND recipe).
#
# Note on SNMF_OUTPUT_DIR: the SNMF basis is INDEPENDENT of the ablation threshold
# (threshold only recomputes role labels from stored log-ratios). To sweep threshold
# without retraining SNMF, pass SKIP_TRAIN=1 SKIP_ANALYZE=1 +
# SNMF_OUTPUT_DIR=<existing_dir>. The default below uses the "thr030" SNMF dir that
# the first iter-4 run produced, so threshold sweeps reuse it.
ITER4_GROUP="iter4_data_part4_${THR_TAG}_both_up_down"
ITER4_CONFIG="bio_retain_and_neutral"
SNMF_OUTPUT_DIR="${SNMF_OUTPUT_DIR:-outputs/wmdp/results_data_part4_gemma2_2b_iter4_thr030_both_up_down}"
ABLATION_OUTPUT_DIR="${ABLATION_OUTPUT_DIR:-outputs/wmdp/forget_ablation_data_part4_gemma2_2b_${ITER4_GROUP}_${ITER4_CONFIG}}"
SAVE_PATH="${SAVE_PATH:-local_models/wmdp/${ITER4_GROUP}/${ITER4_CONFIG}}"
SAVE_PATH_RANDOM="${SAVE_PATH_RANDOM:-${SAVE_PATH}_random}"

# --- SNMF training config (pinned to match iter-3) ------------------------------------
# Same as run_iter3_pipeline.sh — same architecture, same recipe, new data. Pinned for
# self-documentation so this script is immune to pipeline default drift.
SNMF_RANK="300"
SNMF_LAYERS="0-25"
SNMF_MODE="mlp_intermediate"
SNMF_INIT="svd"
SNMF_SPARSITY="0.01"
SNMF_MAX_ITER="3000"
SNMF_TRAIN_BATCH_SIZE="8"
SNMF_TRAIN_SEED="42"

# --- Supervised-analysis config (pinned to match iter-3) ------------------------------
# Threshold used when BAKING role_labels_by_basis into the supervised JSONs. Kept at
# 0.05 (permissive) — the ablation step below recomputes labels on the fly at
# ROLE_ASSIGNMENT_THRESHOLD=$THRESHOLD.
ANALYZE_ROLE_THRESHOLD_PIN="0.05"
ANALYZE_SEED_PIN="42"

# --- Eval: wmdp_bio (stock lm-eval) + MMLU --------------------------------------------
# iter-3 used wmdp_bio_categorized (contamination-filtered via wmdp_bio_robust YAMLs)
# but that mode silently drops MMLU (see evaluation/eveluate_model.py). iter-3b /
# iter-3c / iter-4 use EVAL_MODE=wmdp_bio so both wmdp_bio (1273 Qs) and MMLU are
# measured for before / learned / random.
EVAL_MODE_SEL="wmdp_bio"

echo "================================================================"
echo " iter-4 WMDP-bio pipeline (Gemma-2-2b)"
echo " Base model:      $ITER4_BASE"
echo " SNMF data:       $DATA_PART4"
echo " SNMF dir:        $SNMF_OUTPUT_DIR"
echo " Ablation dir:    $ABLATION_OUTPUT_DIR"
echo " Learned save:    $SAVE_PATH"
echo " Random save:     $SAVE_PATH_RANDOM"
echo " SNMF training:   rank=$SNMF_RANK layers=$SNMF_LAYERS mode=$SNMF_MODE init=$SNMF_INIT"
echo "                  sparsity=$SNMF_SPARSITY max_iter=$SNMF_MAX_ITER batch=$SNMF_TRAIN_BATCH_SIZE seed=$SNMF_TRAIN_SEED"
echo "                  (same as iter-3; only data_part4 + stricter threshold change)"
echo " Analyze:         role_threshold_baked=$ANALYZE_ROLE_THRESHOLD_PIN seed=$ANALYZE_SEED_PIN"
echo " Recipe:          role_label_bases='bio_retain neutral' combine=all thr=$THRESHOLD"
echo "                  DOWN_PROJ_ONLY=0  SPAN_PROJECTION_SCALE=1.0"
echo "                  RANDOM_BASELINE=1 (matched-count control)"
echo " Eval:            EVAL_MODE=$EVAL_MODE_SEL  (stock wmdp_bio + MMLU)"
echo "                  SKIP_PRE_EVAL=0 — before / learned / random all measured"
echo "================================================================"

# Hand off to the full SNMF -> analyze -> ablate sbatch job, overriding its defaults.
MODEL_PATH="$ITER4_BASE" \
DATA_PATH="$DATA_PART4" \
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
ROLE_ASSIGNMENT_THRESHOLD="$THRESHOLD" \
DOWN_PROJ_ONLY="0" \
SPAN_PROJECTION_SCALE="1.0" \
RANDOM_BASELINE="1" \
RANDOM_SEED="1234" \
SKIP_PRE_EVAL="0" \
EVAL_MODE="$EVAL_MODE_SEL" \
EVAL_LARGE="1" \
EVAL_NO_MMLU="0" \
  sbatch --job-name="iter4_${THR_TAG}_${ITER4_CONFIG}" "$SBATCH_SCRIPT"

echo
echo "================================================================"
echo " Submitted iter-4 job (threshold=$THRESHOLD). Monitor:"
echo "   squeue -u \$USER"
echo " Log:                              logs/snmf_forget_pipe_<jobid>.out"
echo " Learned eval JSON (when done):    $SAVE_PATH/ablation_eval_comparison.json"
echo "   will contain: before / after (learned) / random_baseline.after,"
echo "   each with wmdp_bio + mmlu."
echo "================================================================"
