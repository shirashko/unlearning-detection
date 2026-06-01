#!/bin/bash
# ==============================================================================
# Build forget/retain JSON files for target evaluation
# ==============================================================================
#
# Default (ancient_rome, golf, uranium — 100 forget + 100 retain each):
#   bash scripts/evaluation/build_forget_retain_eval_data.sh
#
# Single concept:
#   CONCEPT=golf bash scripts/evaluation/build_forget_retain_eval_data.sh
#
# Override coupled sample size:
#   MAX_SAMPLES=300 bash scripts/evaluation/build_forget_retain_eval_data.sh
#
# All concepts in SNMF-Erasure samples file:
#   ALL_CONCEPTS=1 bash scripts/evaluation/build_forget_retain_eval_data.sh
#
# List available concept labels:
#   LIST_CONCEPTS=1 bash scripts/evaluation/build_forget_retain_eval_data.sh
#
# ==============================================================================

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/../.." && pwd)}"
cd "$REPO_ROOT"

SNMF_DATA_ROOT="${SNMF_DATA_ROOT:-${SNMF_ERASURE_DATA_ROOT:-/home/morg/students/rashkovits/SNMF-Erasure/data}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/eval}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
DEFAULT_CONCEPTS=(ancient_rome golf uranium)

cmd=(python3 experiments/evaluation/build_forget_retain_eval_data.py
  --snmf-data-root "$SNMF_DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --seed "$SEED"
  --max-samples "$MAX_SAMPLES"
)
if [[ -n "${LIST_CONCEPTS:-}" ]]; then
  cmd+=(--list-concepts)
fi
if [[ -n "${ALL_CONCEPTS:-}" ]]; then
  cmd+=(--all-concepts)
elif [[ -n "${CONCEPT:-}" ]]; then
  cmd+=(--concept "$CONCEPT")
else
  for concept in "${DEFAULT_CONCEPTS[@]}"; do
    cmd+=(--concept "$concept")
  done
fi
cmd+=("$@")

echo "[build-eval-data] CMD: ${cmd[*]}"
exec "${cmd[@]}"
