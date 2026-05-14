#!/bin/bash

# iter-3c: re-ablation on top of the iter-3 SNMF basis using a per-layer
# L1-logistic PROBE to pick forget directions.
#
# Why: iter-3b showed that the unary log-ratio + AND(bio_retain, neutral)
# selector @thr=0.30 produces only ~23 directions across 26 layers (many
# layers with zero directions) and under-removes bio knowledge on the
# already-twice-ablated iter-3 base. A multivariate L1-logistic probe is a
# complementary signal: it ranks latents JOINTLY by how much additional
# bio_forget-vs-retain information each one carries given the others, with
# calibrated magnitudes and principled sparsity — no hand-picked threshold.
#
# Smoke-test on iter-3's SNMF artifacts showed the two selectors disagree
# substantially: log_ratio@0.30 picks 23 total directions; probe_topk(K=5)
# picks 130; their intersection is only 10 (17/26 layers have empty
# intersection). 'intersect' is therefore too restrictive for a primary run.
#
# This wrapper's DEFAULT RUN is a clean A/B vs iter-3b:
#     MODE=probe_topk  PROBE_TOP_K=1    →  26 directions (≈ iter-3b's 23),
#     so any wmdp_bio / mmlu delta vs iter-3b is attributable to the
#     selection-rule change, not the direction count.
#
# Reuses iter-3's SNMF layer_* factors and supervised JSONs (SKIP_TRAIN=1
# SKIP_ANALYZE=1), fits the per-layer L1 probe via wmdp_bio_probe_snmf_results.py
# (SKIP_PROBE=0), then runs the ablation step with --selection-mode=${MODE}
# and full wmdp_bio + MMLU evaluation on base / learned / random-matched models.
#
# Usage:
#   bash scripts/wmdp/run_iter3c_probe_pipeline.sh                 # probe_topk K=1 (default)
#   MODE=probe_topk PROBE_TOP_K=5 bash scripts/wmdp/run_iter3c_probe_pipeline.sh
#   MODE=intersect  PROBE_TOP_K=10 THRESHOLD=0.30 bash scripts/wmdp/run_iter3c_probe_pipeline.sh
#
# Outputs:
#   local_models/wmdp/iter3c_probe_data_part3_<mode>_k<K>_<thrTag>_both_up_down/
#     bio_retain_and_neutral/ablation_eval_comparison.json
#       before             — iter-3 base model          (wmdp_bio + mmlu)
#       after              — learned probe ablation     (wmdp_bio + mmlu)
#       random_baseline    — matched-count random ctrl  (wmdp_bio + mmlu)
#
# Logs: logs/snmf_forget_pipe_<jobid>.{out,err}

set -euo pipefail

REPO_ROOT="/home/morg/students/rashkovits/snmf"
cd "$REPO_ROOT"

SBATCH_SCRIPT="scripts/wmdp/run_snmf_forget_pipeline.sh"

# --- Same base + SNMF artifacts as iter-3 / iter-3b ---
ITER3_BASE="/home/morg/students/rashkovits/snmf/local_models/wmdp/iter2_topup_data_part2_thr018_both_up_down/bio_retain_and_neutral"
SNMF_OUTPUT_DIR="outputs/wmdp/results_data_part3_gemma2_2b_iter3_thr018_both_up_down"
DATA_PART3="${REPO_ROOT}/data/bio_data_part3.json"

# --- Probe/selection knobs ---
MODE="${MODE:-probe_topk}"                # probe_topk | intersect
PROBE_TOP_K="${PROBE_TOP_K:-1}"
PROBE_FEATURE_AGG="${PROBE_FEATURE_AGG:-prompt_max}"
# Only consumed when MODE=intersect — must match iter-3b's rule so the intersection
# is meaningful (AND(bio_retain, neutral) @ threshold 0.30).
THRESHOLD="${THRESHOLD:-0.30}"
THR_TAG="thr$(printf '%s' "$THRESHOLD" | tr -d '.')"

ITER3C_GROUP="iter3c_probe_data_part3_${MODE}_k${PROBE_TOP_K}_${THR_TAG}_both_up_down"
ITER3C_CONFIG="bio_retain_and_neutral"
ABLATION_OUTPUT_DIR="outputs/wmdp/forget_ablation_data_part3_gemma2_2b_${ITER3C_GROUP}_${ITER3C_CONFIG}"
SAVE_PATH="local_models/wmdp/${ITER3C_GROUP}/${ITER3C_CONFIG}"
SAVE_PATH_RANDOM="${SAVE_PATH}_random"

# --- Eval: stock WMDP-bio + MMLU (apples-to-apples with iter-3b) ---
EVAL_MODE_SEL="wmdp_bio"

echo "================================================================"
echo " iter-3c PROBE re-ablation (reusing iter-3 SNMF basis)"
echo " Base model:       $ITER3_BASE"
echo " Reused SNMF dir:  $SNMF_OUTPUT_DIR  (SKIP_TRAIN=1 SKIP_ANALYZE=1)"
echo " Probe step:       SKIP_PROBE=0  feat_agg=$PROBE_FEATURE_AGG  (writes layer_*/probe_weights_wmdp_bio.json)"
echo " Selection mode:   $MODE   top_K per layer=$PROBE_TOP_K"
if [[ "$MODE" == "intersect" ]]; then
  echo " Intersected with: role_bases='bio_retain neutral' combine=all  threshold=$THRESHOLD"
fi
echo " Ablation dir:     $ABLATION_OUTPUT_DIR"
echo " Learned save:     $SAVE_PATH"
echo " Random save:      $SAVE_PATH_RANDOM"
echo " Recipe:           DOWN_PROJ_ONLY=0  SPAN_PROJECTION_SCALE=1.0  RANDOM_BASELINE=1"
echo " Eval:             EVAL_MODE=$EVAL_MODE_SEL  (wmdp_bio + mmlu)  SKIP_PRE_EVAL=0"
echo "================================================================"

# Hand off to the SNMF -> analyze -> probe -> ablate sbatch job.
# Training + analysis are skipped (reuse job 276246 outputs); the probe step
# runs, and create_forget_ablated_model.py consumes the per-layer probe weights
# via --selection-mode.
MODEL_PATH="$ITER3_BASE" \
DATA_PATH="$DATA_PART3" \
SNMF_OUTPUT_DIR="$SNMF_OUTPUT_DIR" \
ABLATION_OUTPUT_DIR="$ABLATION_OUTPUT_DIR" \
SAVE_PATH="$SAVE_PATH" \
SAVE_PATH_RANDOM="$SAVE_PATH_RANDOM" \
SKIP_TRAIN="1" \
SKIP_ANALYZE="1" \
SKIP_PROBE="0" \
PROBE_FEATURE_AGG="$PROBE_FEATURE_AGG" \
SELECTION_MODE="$MODE" \
PROBE_TOP_K="$PROBE_TOP_K" \
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
  sbatch --job-name="iter3c_${MODE}_k${PROBE_TOP_K}_${THR_TAG}_${ITER3C_CONFIG}" "$SBATCH_SCRIPT"

echo
echo "================================================================"
echo " Submitted iter-3c probe job (mode=$MODE, top_K=$PROBE_TOP_K, thr=$THRESHOLD). Monitor:"
echo "   squeue -u \$USER"
echo " Log:                              logs/snmf_forget_pipe_<jobid>.out"
echo " Probe weights (per layer):        $SNMF_OUTPUT_DIR/layer_*/probe_weights_wmdp_bio.json"
echo " Probe summary:                    $SNMF_OUTPUT_DIR/probe_summary_wmdp_bio.json"
echo " Learned eval JSON (when done):    $SAVE_PATH/ablation_eval_comparison.json"
echo "   (will contain: before / after (learned probe) / random_baseline.after,"
echo "    each with wmdp_bio + mmlu)"
echo "================================================================"
