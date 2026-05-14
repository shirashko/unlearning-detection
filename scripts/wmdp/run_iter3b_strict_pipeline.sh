#!/bin/bash

# iter-3b: re-ablation on top of the iter-3 SNMF basis with a STRICTER
# role_assignment_threshold — "heavy forgetters only".
#
# Context: iter-3 (job 276246) used threshold=0.18 on the AND of bio_retain
# and neutral bases, selected ~67 latents across 22 layers, and produced only
# a ~1.3pp gap on wmdp_bio_robust vs. the random-matched baseline (0.3468 vs
# 0.3594). MMLU was not measured in that run (EVAL_MODE=wmdp_bio_categorized
# silently drops MMLU). Working hypothesis: thr=0.18 admits too many
# weak_mixed / marginally-forget-leaning latents, diluting the signal.
#
# This wrapper re-uses the iter-3 SNMF layer_* factors and supervised JSONs
# (the underlying script has SKIP_TRAIN / SKIP_ANALYZE support and the
# threshold is applied at ablation time by create_forget_ablated_model.py's
# _recompute_role_labels_by_basis — so no re-analysis is needed) and does
# ONLY the ablate + full eval step.
#
# Usage (default threshold 0.30):
#   bash scripts/wmdp/run_iter3b_strict_pipeline.sh
# Sweep stricter / looser:
#   THRESHOLD=0.35 bash scripts/wmdp/run_iter3b_strict_pipeline.sh
#   THRESHOLD=0.40 bash scripts/wmdp/run_iter3b_strict_pipeline.sh
#
# Output: local_models/wmdp/iter3b_strict_data_part3_<tag>_both_up_down/...
#         ablation_eval_comparison.json will contain
#           before (base model)         — with wmdp_bio + mmlu
#           after  (learned ablation)   — with wmdp_bio + mmlu
#           random_baseline.after       — with wmdp_bio + mmlu
#
# Logs: logs/snmf_forget_pipe_<jobid>.{out,err}

set -euo pipefail

REPO_ROOT="/home/morg/students/rashkovits/snmf"
cd "$REPO_ROOT"

SBATCH_SCRIPT="scripts/wmdp/run_snmf_forget_pipeline.sh"

# --- Same base as iter-3 (iter-2 + variant-A top-up) ---
ITER3_BASE="/home/morg/students/rashkovits/snmf/local_models/wmdp/iter2_topup_data_part2_thr018_both_up_down/bio_retain_and_neutral"

# --- Reuse iter-3 SNMF artifacts from job 276246 (data_part3, rank=300) ---
SNMF_OUTPUT_DIR="outputs/wmdp/results_data_part3_gemma2_2b_iter3_thr018_both_up_down"
DATA_PART3="${REPO_ROOT}/data/bio_data_part3.json"   # unused at this step (SKIP_TRAIN/ANALYZE=1) but kept for logging self-consistency

# --- Stricter threshold: heavy forgetters only ---
THRESHOLD="${THRESHOLD:-0.30}"
# e.g. 0.30 -> "thr030", 0.35 -> "thr035", 0.4 -> "thr04"
THR_TAG="thr$(printf '%s' "$THRESHOLD" | tr -d '.')"

ITER3B_GROUP="iter3b_strict_data_part3_${THR_TAG}_both_up_down"
ITER3B_CONFIG="bio_retain_and_neutral"
ABLATION_OUTPUT_DIR="outputs/wmdp/forget_ablation_data_part3_gemma2_2b_${ITER3B_GROUP}_${ITER3B_CONFIG}"
SAVE_PATH="local_models/wmdp/${ITER3B_GROUP}/${ITER3B_CONFIG}"
SAVE_PATH_RANDOM="${SAVE_PATH}_random"

# --- Eval: stock lm-eval WMDP-bio + MMLU (apples-to-apples with iter-2) ---
# Note: EVAL_MODE=wmdp_bio_categorized (robust task) does NOT run MMLU —
# see evaluation/eveluate_model.py. Using EVAL_MODE=wmdp_bio here so both
# WMDP-bio (1273 Qs) and MMLU are measured for before / learned / random.
EVAL_MODE_SEL="wmdp_bio"

echo "================================================================"
echo " iter-3b STRICT re-ablation (reusing iter-3 SNMF basis)"
echo " Base model:      $ITER3_BASE"
echo " Reused SNMF dir: $SNMF_OUTPUT_DIR  (SKIP_TRAIN=1 SKIP_ANALYZE=1)"
echo " Threshold:       $THRESHOLD        (iter-3 used 0.18)"
echo " Ablation dir:    $ABLATION_OUTPUT_DIR"
echo " Learned save:    $SAVE_PATH"
echo " Random save:     $SAVE_PATH_RANDOM"
echo " Recipe:          role_label_bases='bio_retain neutral' combine=all"
echo "                  DOWN_PROJ_ONLY=0  SPAN_PROJECTION_SCALE=1.0"
echo "                  RANDOM_BASELINE=1 (matched-count control)"
echo " Eval:            EVAL_MODE=$EVAL_MODE_SEL  (stock WMDP-bio + MMLU)"
echo "                  SKIP_PRE_EVAL=0 — before/after/random all measured"
echo "================================================================"

# Hand off to the SNMF -> analyze -> ablate sbatch job. Training + analysis
# are skipped (we reuse job 276246's outputs); only create_forget_ablated_model.py
# + eval run.
MODEL_PATH="$ITER3_BASE" \
DATA_PATH="$DATA_PART3" \
SNMF_OUTPUT_DIR="$SNMF_OUTPUT_DIR" \
ABLATION_OUTPUT_DIR="$ABLATION_OUTPUT_DIR" \
SAVE_PATH="$SAVE_PATH" \
SAVE_PATH_RANDOM="$SAVE_PATH_RANDOM" \
SKIP_TRAIN="1" \
SKIP_ANALYZE="1" \
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
  sbatch --job-name="iter3b_${THR_TAG}_${ITER3B_CONFIG}" "$SBATCH_SCRIPT"

echo
echo "================================================================"
echo " Submitted iter-3b strict job (threshold=$THRESHOLD). Monitor:"
echo "   squeue -u \$USER"
echo " Log:                              logs/snmf_forget_pipe_<jobid>.out"
echo " Learned eval JSON (when done):    $SAVE_PATH/ablation_eval_comparison.json"
echo "   will contain: before / after (learned) / random_baseline.after,"
echo "   each with wmdp_bio and mmlu."
echo "================================================================"
