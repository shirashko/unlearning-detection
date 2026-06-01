#!/bin/bash
# ==============================================================================
# Slurm wrapper: score RMU rel_delta audit judge predictions (Gemini API only).
# ==============================================================================
#
# All three concepts (default):
#   sbatch scripts/evaluation/run_rmu_rel_delta_target_eval.sh
#
# Single concept:
#   CONCEPT=golf sbatch --job-name=target_eval_golf \
#     --output=logs/target_eval_golf_%j.out \
#     --error=logs/target_eval_golf_%j.err \
#     scripts/evaluation/run_rmu_rel_delta_target_eval.sh
#
# Local (no Slurm):
#   bash scripts/evaluation/run_rmu_rel_delta_target_eval.sh
#
# ==============================================================================

#SBATCH --job-name=target_eval_rmu_rel_delta
#SBATCH --output=logs/target_eval_rmu_rel_delta_%j.out
#SBATCH --error=logs/target_eval_rmu_rel_delta_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/morg/students/rashkovits/unlearning-detection}"
cd "$REPO_ROOT"

# shellcheck source=scripts/audit/audit_runner_env.sh
source "${REPO_ROOT}/scripts/audit/audit_runner_env.sh"

AUDIT_BASE="${AUDIT_BASE:-outputs/gemma_2_2b_it/rank350/data1/rmu/rel_delta}"
MAX_SAMPLES_PER_SET="${MAX_SAMPLES_PER_SET:-4}"
EVAL_MODEL="${EVAL_MODEL:-gemini-2.5-flash}"
SEED="${SEED:-42}"

run_one() {
  local concept="$1"
  local audit_dir="${REPO_ROOT}/${AUDIT_BASE}/${concept}"
  local labeled_data="${REPO_ROOT}/data/eval/${concept}_forget_retain.json"

  if [[ ! -d "$audit_dir" ]]; then
    echo "[eval] skip ${concept}: missing audit dir ${audit_dir}" >&2
    return 1
  fi
  if [[ ! -f "$labeled_data" ]]; then
    echo "[eval] skip ${concept}: missing labeled data ${labeled_data}" >&2
    return 1
  fi

  echo "[eval] concept=${concept}"
  echo "       audit_dir=${audit_dir}"
  echo "       labeled_data=${labeled_data}"

  python3 experiments/evaluation/run_target_evaluation.py \
    --audit-dir "$audit_dir" \
    --labeled-data "$labeled_data" \
    --max-samples-per-set "$MAX_SAMPLES_PER_SET" \
    --eval-model "$EVAL_MODEL" \
    --seed "$SEED"
}

if [[ -n "${CONCEPT:-}" ]]; then
  run_one "$CONCEPT"
else
  for concept in ancient_rome golf uranium; do
    run_one "$concept" || true
  done
fi
