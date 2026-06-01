#!/bin/bash
# ==============================================================================
# Slurm wrapper: general SNMF unlearning audit
# ==============================================================================
#
# Default (Gemma-2-2B RMU rel_delta / ancient_rome):
#   sbatch scripts/audit/run_general_unlearning_audit.sh
#
# Override config:
#   AUDIT_CONFIG="configs/audit/gemma2_2b_it/pisces/rel_delta/golf.yaml" \
#     sbatch scripts/audit/run_general_unlearning_audit.sh
#
# Override config + extra Python flags:
#   AUDIT_CONFIG="configs/audit/gemma2_2b_it/rmu/rel_delta/golf.yaml" \
#     sbatch scripts/audit/run_general_unlearning_audit.sh --skip-judge
#
# Optional: custom Slurm log names when overriding:
#   AUDIT_CONFIG="..." sbatch --job-name=audit_pisces_golf \
#     --output=logs/audit_pisces_golf_%j.out \
#     --error=logs/audit_pisces_golf_%j.err \
#     scripts/audit/run_general_unlearning_audit.sh
#
# ==============================================================================

#SBATCH --job-name=audit_general
#SBATCH --output=logs/audit_general_%j.out
#SBATCH --error=logs/audit_general_%j.err
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

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${_SCRIPT_DIR}/audit_runner_env.sh"

DEFAULT_CONFIG="${REPO_ROOT}/configs/audit/gemma2_2b_it/rmu/rel_delta/ancient_rome.yaml"
CONFIG="${AUDIT_CONFIG:-$DEFAULT_CONFIG}"

echo "[audit] CONFIG: ${CONFIG}"
exec python -u experiments/audit/general_unlearning_audit.py --config "$CONFIG" "$@"
