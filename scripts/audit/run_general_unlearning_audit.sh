#!/bin/bash
# ==============================================================================
# Unified Slurm Wrapper for General Unlearning Audit Experiments
# ==============================================================================
#
# Default behavior (Gemma-2-2B-it RMU):
#   sbatch scripts/audit/run_general_unlearning_audit.sh
#
# Overridden behavior (custom config):
#   AUDIT_CONFIG="configs/audit/obsolete/gemma22b_wmdp_bio_rmu_general_audit.yaml" \
#   sbatch --job-name=audit_custom \
#          --output=logs/audit_custom_%j.out \
#          --error=logs/audit_custom_%j.err \
#          scripts/audit/run_general_unlearning_audit.sh
#
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Slurm Resource Allocation (Default: g22b-it RMU audit)
# ------------------------------------------------------------------------------
#SBATCH --job-name=audit_g22b_it_rmu
#SBATCH --output=logs/audit_g22b_it_rmu_%j.out
#SBATCH --error=logs/audit_g22b_it_rmu_%j.err
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

# ------------------------------------------------------------------------------
# 2. Environment Invariants & Bootstrapping
# ------------------------------------------------------------------------------
export REPO_ROOT="${REPO_ROOT:-/home/morg/students/rashkovits/unlearning-detection}"

# shellcheck source=scripts/audit/audit_runner_env.sh
source "${REPO_ROOT}/scripts/audit/audit_runner_env.sh"

# ------------------------------------------------------------------------------
# 3. Configuration Resolution
# ------------------------------------------------------------------------------
# Default fallback to Gemma-2-2B-it RMU audit config if AUDIT_CONFIG is missing.
DEFAULT_CONFIG="${REPO_ROOT}/configs/audit/rmu/gemma2_2b_it/rel_delta/ancient_rome.yaml"
CONFIG="${AUDIT_CONFIG:-$DEFAULT_CONFIG}"

echo "[audit] Infrastructure initialization complete."
echo "        -> REPO_ROOT : ${REPO_ROOT}"
echo "        -> CONFIG    : ${CONFIG}"

# ------------------------------------------------------------------------------
# 4. Payload Execution
# ------------------------------------------------------------------------------
exec python -u experiments/audit/general_unlearning_audit.py --config "$CONFIG" "$@"