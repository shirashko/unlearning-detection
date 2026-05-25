#!/bin/bash
# ==============================================================================
# Unified Slurm Wrapper for Baseline (Raw MLP Neuron) Unlearning Audit
# ==============================================================================
#
# Same layout as run_general_unlearning_audit.sh: shared AUDIT_CONFIG default,
# audit_runner_env.sh, and trailing "$@" passthrough. Optional ACTIVATIONS_CACHE
# forwards --activations-cache when set (offline tensor path).
#
# Default (Gemma-2-2B WMDP bio RMU): uses gemma22b_wmdp_bio_rmu_general_audit.yaml
# but forces a distinct baseline output_dir so SNMF general runs (same YAML) do not
# clobber artifacts. Override the dir with BASELINE_OUTPUT_DIR or pass --output-dir
# after the script args.
#
#   sbatch scripts/audit/run_baseline_unlearning_audit.sh
#
# With cached aligned activations:
#   ACTIVATIONS_CACHE="/path/to/aligned_activations.pt" \
#   sbatch scripts/audit/run_baseline_unlearning_audit.sh
#
# Gemma 0.3B arithmetic (alternate YAML only — set AUDIT_CONFIG):
#   AUDIT_CONFIG="${REPO_ROOT}/configs/audit/gemma03b_arithmetic_baseline_audit.yaml" \
#   sbatch --job-name=baseline_audit_g03b_arith \
#          --output=logs/baseline_audit_g03b_%j.out \
#          --error=logs/baseline_audit_g03b_%j.err \
#          scripts/audit/run_baseline_unlearning_audit.sh
#
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Slurm Resource Allocation (default: Gemma-2-2B WMDP bio RMU baseline)
# ------------------------------------------------------------------------------
#SBATCH --job-name=baseline_audit_g22b_wmdp
#SBATCH --output=logs/baseline_audit_g22b_wmdp_%j.out
#SBATCH --error=logs/baseline_audit_g22b_wmdp_%j.err
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
DEFAULT_CONFIG="${REPO_ROOT}/configs/audit/gemma22b_wmdp_bio_rmu_general_audit.yaml"
CONFIG="${AUDIT_CONFIG:-$DEFAULT_CONFIG}"

# When using the default g22b YAML, write under a baseline-only directory (YAML
# output_dir matches the SNMF general audit; CLI overrides it for this script).
DEFAULT_BASELINE_OUTPUT_DIR="${REPO_ROOT}/outputs/gemma22b_baseline_audit_rmu_layers_10_18"

echo "[baseline_audit] Infrastructure initialization complete."
echo "                -> REPO_ROOT          : ${REPO_ROOT}"
echo "                -> CONFIG             : ${CONFIG}"
if [[ -z "${AUDIT_CONFIG:-}" ]]; then
  echo "                -> output_dir (forced for baseline): ${BASELINE_OUTPUT_DIR:-$DEFAULT_BASELINE_OUTPUT_DIR}"
fi
if [[ -n "${ACTIVATIONS_CACHE:-}" ]]; then
  echo "                -> ACTIVATIONS_CACHE : ${ACTIVATIONS_CACHE}"
fi

# ------------------------------------------------------------------------------
# 4. Payload Execution
# ------------------------------------------------------------------------------
BASE_ARGS=( -u experiments/audit/baseline_unlearning_audit.py --config "$CONFIG" )
if [[ -z "${AUDIT_CONFIG:-}" ]]; then
  BASE_ARGS+=( --output-dir "${BASELINE_OUTPUT_DIR:-$DEFAULT_BASELINE_OUTPUT_DIR}" )
fi
if [[ -n "${ACTIVATIONS_CACHE:-}" ]]; then
  BASE_ARGS+=( --activations-cache "${ACTIVATIONS_CACHE}" )
fi

exec python "${BASE_ARGS[@]}" "$@"
