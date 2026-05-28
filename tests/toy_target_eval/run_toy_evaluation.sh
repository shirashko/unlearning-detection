#!/usr/bin/env bash
# Smoke-run target evaluation on committed toy fixtures (requires GOOGLE_API_KEY).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

FIXTURE_DIR="${ROOT}/tests/toy_target_eval"
OUT_DIR="${FIXTURE_DIR}/out"

python3 experiments/evaluation/run_target_evaluation.py \
  --audit-dir "${FIXTURE_DIR}/audit" \
  --labeled-data "${FIXTURE_DIR}/labeled.json" \
  --output-dir "${OUT_DIR}" \
  --max-samples-per-set "${MAX_SAMPLES_PER_SET:-2}" \
  --seed "${SEED:-0}" \
  --eval-model "${EVAL_MODEL:-gemini-2.5-flash}"

echo "Report: ${OUT_DIR}/target_evaluation_report.json"
