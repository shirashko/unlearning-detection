#!/bin/bash
# ==============================================================================
# Re-run the Gemini judge for an existing audit output (CPU-only, no GPU).
# ==============================================================================
#
# With audit YAML (uses output_dir + judge section from config):
#   AUDIT_CONFIG="configs/audit/pisces/gemma2_2b_it/rel_delta/ancient_rome.yaml" \
#     sbatch scripts/audit/rerun_audit_judge.sh
#
# With explicit output directory:
#   OUTPUT_DIR="outputs/gemma_2_2b_it/rank350/data1/pisces/rel_delta/golf" \
#     sbatch scripts/audit/rerun_audit_judge.sh
#
# ==============================================================================

#SBATCH --job-name=audit_judge_rerun
#SBATCH --output=logs/audit_judge_rerun_%j.out
#SBATCH --error=logs/audit_judge_rerun_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/home/morg/students/rashkovits/unlearning-detection}"

# shellcheck source=scripts/audit/audit_runner_env.sh
source "${REPO_ROOT}/scripts/audit/audit_runner_env.sh"

DEFAULT_CONFIG="${REPO_ROOT}/configs/audit/pisces/gemma2_2b_it/rel_delta/ancient_rome.yaml"
CONFIG="${AUDIT_CONFIG:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

echo "[judge-rerun] REPO_ROOT : ${REPO_ROOT}"

cmd=(python -u experiments/audit/rerun_audit_judge.py)
if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${CONFIG}" ]]; then
  cmd+=(--config "${CONFIG}")
elif [[ -z "${OUTPUT_DIR}" ]]; then
  cmd+=(--config "${DEFAULT_CONFIG}")
fi
cmd+=("$@")

echo "[judge-rerun] CMD: ${cmd[*]}"
exec "${cmd[@]}"
