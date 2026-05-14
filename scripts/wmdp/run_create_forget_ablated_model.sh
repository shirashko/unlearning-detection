#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=forget_ablate_wmdp_bio
#SBATCH --output=logs/forget_ablate_wmdp_bio_%j.out
#SBATCH --error=logs/forget_ablate_wmdp_bio_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

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

# ========== Defaults: WMDP-bio + Gemma-2-2b (iter-2 variant-A top-up, robust-eval re-run) ==========
# Base = iter-2 ablated checkpoint from snmf_forget_pipe job 272373 (SNMF re-fit on
# bio_data_part2 residual of the iter-1 model; stricter thr=0.22 / combine=all / down_proj_only).
# That run left wmdp_bio at 0.480 on stock lm-eval. The iter-2 top-up sweep (jobs
# 273212/213/214) tried three selection configs on top of it; variant-A (bio_retain AND
# neutral, thr=0.18, both up/down) was the only one that preserved MMLU — stock
# wmdp_bio dropped to 0.4273 with mmlu at 0.4837.
#
# This script's current defaults reproduce variant-A and re-evaluate on the robust
# WMDP-bio task group (wmdp_bio_robust under wmdp_bio_categorized_mcqa/) plus MMLU,
# and add the random-matched-count baseline that was missed in the original sweep.
# Ablation is deterministic, so the resulting weights match the existing variant-A
# ckpt; we write to a sibling path with `_robust_eval` suffix to preserve the original
# stock-wmdp_bio `ablation_eval_comparison.json` for historical comparison.
MODEL_PATH="${MODEL_PATH:-/home/morg/students/rashkovits/snmf/local_models/wmdp/iter2/data_part2_thr022_down_proj_only}"
# data_part2 iter-2 SNMF results dir (re-fit on iter-1 residual activations). data_part2
# showed the stronger iter-2 signal vs bio_data.json (−0.045 vs −0.029 Δwmdp_bio).
RESULTS_DIR="${RESULTS_DIR:-outputs/wmdp/results_data_part2_gemma2_2b_iter2_thr022_down_proj_only}"
SUPERVISED_JSON_FILENAME="${SUPERVISED_JSON_FILENAME:-feature_analysis_supervised_wmdp_bio.json}"

# Ablated checkpoint output paths. Tag reflects variant-A config + robust-eval re-run.
SAVE_PATH="${SAVE_PATH:-local_models/wmdp/iter2_topup_data_part2_thr018_both_up_down/bio_retain_and_neutral_robust_eval}"
SAVE_PATH_RANDOM="${SAVE_PATH_RANDOM:-${SAVE_PATH}_random}"

# Ablation controls — variant-A (strict specificity: forget-leaning both vs bio_retain AND vs neutral).
FORGET_ROLES="${FORGET_ROLES:-bio_forget_lean}"
# role_labels_by_basis from WMDP-bio JSON (space-separated): pooled | neutral | bio_retain.
# variant-A: AND across bio_retain and neutral — strongest specificity, only iter-2 top-up
# variant that preserved MMLU.
ROLE_LABEL_BASES="${ROLE_LABEL_BASES:-bio_retain neutral}"
ROLE_BASIS_COMBINE="${ROLE_BASIS_COMBINE:-all}"
# On-the-fly threshold override (min |log_forget_vs_retain|). Recomputes per-basis labels from
# the raw stats in each supervised JSON profile. Set to empty to fall back to the stored
# role_labels_by_basis (baked at analysis time). Only applies when ROLE_LABEL_BASES is set.
ROLE_ASSIGNMENT_THRESHOLD="${ROLE_ASSIGNMENT_THRESHOLD:-0.18}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-6}"
SPAN_PROJECTION_SCALE="${SPAN_PROJECTION_SCALE:-1.0}"
ABLATION_DEVICE="${ABLATION_DEVICE:-auto}"
DOWN_PROJ_ONLY="${DOWN_PROJ_ONLY:-0}"

# Random matched-count baseline controls. On by default — the iter-2 top-up sweep
# declared it in comments but the wrapper never exported it, so there's no matched-
# count control for variant-A yet. This re-run fixes that.
RANDOM_BASELINE="${RANDOM_BASELINE:-1}"
RANDOM_SEED="${RANDOM_SEED:-1234}"

# Eval controls (run by default after ablation). Robust WMDP-bio + MMLU.
#   EVAL_MODE=wmdp_bio_categorized routes through the YAML-based lm-eval task group.
#   EVAL_WMDP_INCLUDE_PATH / EVAL_WMDP_TASK_NAME point at wmdp_bio_robust (6 sub-
#   categories aggregated by size). MMLU stays on (EVAL_NO_MMLU=0).
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_PRE_EVAL="${SKIP_PRE_EVAL:-1}"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
EVAL_MODE="${EVAL_MODE:-wmdp_bio_categorized}"
EVAL_LARGE="${EVAL_LARGE:-1}"
EVAL_NO_MMLU="${EVAL_NO_MMLU:-0}"
EVAL_WMDP_INCLUDE_PATH="${EVAL_WMDP_INCLUDE_PATH:-/home/morg/students/rashkovits/snmf/wmdp_bio_categorized_mcqa}"
EVAL_WMDP_TASK_NAME="${EVAL_WMDP_TASK_NAME:-wmdp_bio_robust}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

# --- Parallelism Optimization ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "================================================================"
echo " Running create_forget_ablated_model.py (WMDP-bio)"
echo " Node: ${SLURMD_NODENAME:-local}"
echo " Model path:                   $MODEL_PATH"
echo " Results dir:                  $RESULTS_DIR"
echo " Supervised JSON filename:     $SUPERVISED_JSON_FILENAME"
echo " Save path (learned):          $SAVE_PATH"
echo " Save path (random baseline):  $SAVE_PATH_RANDOM  (if RANDOM_BASELINE=1)"
echo " Forget roles:                 $FORGET_ROLES"
if [[ -n "$ROLE_LABEL_BASES" ]]; then
  echo " Role label bases (WMDP-bio):  $ROLE_LABEL_BASES (combine=$ROLE_BASIS_COMBINE)"
  if [[ -n "$ROLE_ASSIGNMENT_THRESHOLD" ]]; then
    echo " Role assignment threshold:    $ROLE_ASSIGNMENT_THRESHOLD (recomputed from raw stats)"
  else
    echo " Role assignment threshold:    <unset> → use stored role_labels_by_basis"
  fi
else
  echo " Role label bases:             <unset> → use top-level role_label only"
fi
echo " Eval mode:                    $EVAL_MODE"
if [[ "$SKIP_EVAL" == "1" ]]; then
  echo " Eval:                         SKIPPED (SKIP_EVAL=1)"
elif [[ "$SKIP_PRE_EVAL" == "1" ]]; then
  echo " Pre-ablation baseline eval:   SKIPPED (SKIP_PRE_EVAL=1, default)"
  echo " Post-ablation eval:           enabled"
else
  echo " Pre-ablation baseline eval:   enabled"
  echo " Post-ablation eval:           enabled"
fi
if [[ "$EVAL_MODE" == "wmdp_bio" ]]; then
  echo " Eval tasks:                   lm-eval wmdp_bio (+ MMLU unless EVAL_NO_MMLU=1)"
elif [[ "$EVAL_MODE" == "wmdp_bio_categorized" ]]; then
  echo " Eval (categorized):           include=${EVAL_WMDP_INCLUDE_PATH:-<unset>} task=${EVAL_WMDP_TASK_NAME:-wmdp_bio_robust}"
fi
echo "================================================================"

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
if [[ "$EVAL_MODE" == "wmdp_bio_categorized" ]]; then
  if [[ -n "$EVAL_WMDP_INCLUDE_PATH" ]]; then
    EVAL_ARGS+=(--eval-wmdp-include-path "$EVAL_WMDP_INCLUDE_PATH")
  fi
  EVAL_ARGS+=(--eval-wmdp-task-name "${EVAL_WMDP_TASK_NAME:-wmdp_bio_robust}")
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
  --results-dir "$RESULTS_DIR" \
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
  "${RANDOM_ARGS[@]}" \
  "${DOWN_ONLY_ARGS[@]}" \
  "${EVAL_ARGS[@]}"

echo "================================================================"
echo " Done."
echo " Learned checkpoint: $SAVE_PATH"
echo " Eval JSON:          $SAVE_PATH/ablation_eval_comparison.json"
echo "================================================================"
