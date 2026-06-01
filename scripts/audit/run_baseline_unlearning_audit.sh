#!/bin/bash
# ==============================================================================
# Slurm wrapper: baseline (raw MLP neuron) unlearning audit
# ==============================================================================
#
# Default (Gemma-2-2B RMU rel_delta / ancient_rome; writes to *_baseline output):
#   sbatch scripts/audit/run_baseline_unlearning_audit.sh
#
# Override config:
#   AUDIT_CONFIG="configs/audit/gemma2_2b_it/pisces/rel_delta/golf.yaml" \
#     sbatch scripts/audit/run_baseline_unlearning_audit.sh
#
# With cached aligned activations:
#   ACTIVATIONS_CACHE="/path/to/aligned_activations.pt" \
#     sbatch scripts/audit/run_baseline_unlearning_audit.sh
#
# Override config + extra Python flags:
#   AUDIT_CONFIG="configs/audit/gemma2_2b_it/rmu/rel_delta/golf.yaml" \
#     sbatch scripts/audit/run_baseline_unlearning_audit.sh --skip-judge
#
# When using the default config, --output-dir is forced to a baseline-only path so
# SNMF general audits on the same YAML do not clobber artifacts. Override with
# BASELINE_OUTPUT_DIR or pass --output-dir after the script args.
#
# ==============================================================================

#SBATCH --job-name=baseline_audit_general
#SBATCH --output=logs/baseline_audit_general_%j.out
#SBATCH --error=logs/baseline_audit_general_%j.err
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

# shellcheck source=audit_runner_env.sh
source "${_SCRIPT_DIR}/audit_runner_env.sh"

DEFAULT_CONFIG="${REPO_ROOT}/configs/audit/gemma2_2b_it/rmu/rel_delta/ancient_rome.yaml"
CONFIG="${AUDIT_CONFIG:-$DEFAULT_CONFIG}"
DEFAULT_BASELINE_OUTPUT_DIR="${REPO_ROOT}/outputs/gemma_2_2b_it/rank350/data1/rmu/rel_delta/ancient_rome_baseline"

echo "[baseline_audit] CONFIG: ${CONFIG}"
if [[ -z "${AUDIT_CONFIG:-}" ]]; then
  echo "[baseline_audit] output_dir (default baseline): ${BASELINE_OUTPUT_DIR:-$DEFAULT_BASELINE_OUTPUT_DIR}"
fi
if [[ -n "${ACTIVATIONS_CACHE:-}" ]]; then
  echo "[baseline_audit] ACTIVATIONS_CACHE: ${ACTIVATIONS_CACHE}"
fi

cmd=(python -u experiments/audit/baseline_unlearning_audit.py --config "$CONFIG")
if [[ -z "${AUDIT_CONFIG:-}" ]]; then
  cmd+=(--output-dir "${BASELINE_OUTPUT_DIR:-$DEFAULT_BASELINE_OUTPUT_DIR}")
fi
if [[ -n "${ACTIVATIONS_CACHE:-}" ]]; then
  cmd+=(--activations-cache "${ACTIVATIONS_CACHE}")
fi
cmd+=("$@")

exec "${cmd[@]}"
