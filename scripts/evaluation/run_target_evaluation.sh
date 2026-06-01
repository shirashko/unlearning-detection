#!/bin/bash
# ==============================================================================
# Slurm wrapper: target evaluation for completed audit runs (Gemini API only).
# ==============================================================================
#
# Default (Gemma-2-2B RMU rel_delta — ancient_rome, golf, uranium):
#   sbatch scripts/evaluation/run_target_evaluation.sh
#
# Override audit config (method / rank_by / model / topic path):
#   AUDIT_CONFIG="configs/audit/gemma2_2b_it/pisces/rel_delta/golf.yaml" \
#     sbatch scripts/evaluation/run_target_evaluation.sh
#
# Single concept (uses sibling YAMLs in the same config directory):
#   CONCEPT=golf AUDIT_CONFIG="configs/audit/gemma2_2b_it/rmu/rel_delta/ancient_rome.yaml" \
#     sbatch scripts/evaluation/run_target_evaluation.sh
#
# Local (no Slurm):
#   bash scripts/evaluation/run_target_evaluation.sh
#
# Build labeled eval data first (from SNMF-Erasure):
#   bash scripts/evaluation/build_forget_retain_eval_data.sh
#
# ==============================================================================

#SBATCH --job-name=target_eval
#SBATCH --output=logs/target_eval_%j.out
#SBATCH --error=logs/target_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${_SCRIPT_DIR}/../.." && pwd)}}"
cd "$REPO_ROOT"

# shellcheck source=scripts/audit/audit_runner_env.sh
source "${REPO_ROOT}/scripts/audit/audit_runner_env.sh"

DEFAULT_CONFIG="${REPO_ROOT}/configs/audit/gemma2_2b_it/rmu/rel_delta/ancient_rome.yaml"
CONFIG="${AUDIT_CONFIG:-$DEFAULT_CONFIG}"
CONFIG_DIR="$(dirname "$CONFIG")"

MAX_SAMPLES_PER_SET="${MAX_SAMPLES_PER_SET:-5}"
EVAL_MODEL="${EVAL_MODEL:-gemini-2.5-flash}"
SEED="${SEED:-42}"
LABELED_DATA_DIR="${LABELED_DATA_DIR:-${REPO_ROOT}/data/eval}"
DEFAULT_CONCEPTS=(ancient_rome golf uranium)

resolve_audit_dir() {
  python3 - "$1" <<'PY'
import os
import sys
from pathlib import Path

import yaml

repo = Path(os.environ["REPO_ROOT"])
cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text()) or {}
out = os.path.expandvars(os.path.expanduser(str(cfg.get("output_dir", ""))))
if not out:
    raise SystemExit(f"output_dir missing in {cfg_path}")
out_path = Path(out)
if not out_path.is_absolute():
    out_path = (repo / out_path).resolve()
print(out_path)
PY
}

run_one() {
  local config_path="$1"
  shift
  local audit_dir concept labeled_data

  if [[ ! -f "$config_path" ]]; then
    echo "[eval] skip: missing audit config ${config_path}" >&2
    return 1
  fi

  audit_dir="$(resolve_audit_dir "$config_path")"
  concept="$(basename "$config_path" .yaml)"
  labeled_data="${LABELED_DATA_DIR}/${concept}_forget_retain.json"

  if [[ ! -d "$audit_dir" ]]; then
    echo "[eval] skip ${concept}: missing audit dir ${audit_dir}" >&2
    return 1
  fi
  if [[ ! -f "$labeled_data" ]]; then
    echo "[eval] skip ${concept}: missing labeled data ${labeled_data}" >&2
    return 1
  fi

  echo "[eval] config=${config_path}"
  echo "       concept=${concept}"
  echo "       audit_dir=${audit_dir}"
  echo "       labeled_data=${labeled_data}"

  python3 experiments/evaluation/run_target_evaluation.py \
    --audit-dir "$audit_dir" \
    --labeled-data "$labeled_data" \
    --max-samples-per-set "$MAX_SAMPLES_PER_SET" \
    --eval-model "$EVAL_MODEL" \
    --seed "$SEED" \
    "$@"
}

extra_args=("$@")
if [[ -n "${CONCEPT:-}" ]]; then
  run_one "${CONFIG_DIR}/${CONCEPT}.yaml" "${extra_args[@]}"
else
  for concept in "${DEFAULT_CONCEPTS[@]}"; do
    run_one "${CONFIG_DIR}/${concept}.yaml" "${extra_args[@]}" || true
  done
fi
