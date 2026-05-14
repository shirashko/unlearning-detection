#!/bin/bash
#SBATCH --job-name=audit_g03b_arith
#SBATCH --output=logs/general_unlearning_audit_g03b_%j.out
#SBATCH --error=logs/general_unlearning_audit_g03b_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

# Label-free general unlearning audit for Gemma-2-0.3B (arithmetic + English):
# same pipeline as scripts/audit/run_general_unlearning_audit.sh, with paths for this
# base / MaxEnt-unlearned pair and SNMF factors fit on data/general_data_part1.json.
#
# Prerequisite: train the basis first (same rank/layers as SNMF_DIR), e.g.:
#   sbatch scripts/audit/train_snmf_gemma03b_arithmetic_general_data.sh
#
# Run:
#   cd /home/morg/students/rashkovits/snmf
#   sbatch scripts/audit/run_general_unlearning_audit_gemma03b_arithmetic.sh
#
# Tunables (same as the WMDP wrapper), e.g.:
#   env MAX_PROMPTS=200 LAYERS=5-10 SKIP_JUDGE=1 \\
#     sbatch scripts/audit/run_general_unlearning_audit_gemma03b_arithmetic.sh
#
# Rare-context words: enabled by default; needs `wordfreq` in snmf_env unless
# SKIP_CONTEXT_RARE_WORDS=1.

set -euo pipefail

REPO_ROOT="/home/morg/students/rashkovits/snmf"

# Must match train_snmf_gemma03b_arithmetic_general_data.sh defaults unless you override.
RANK="${RANK:-300}"
export SNMF_DIR="${SNMF_DIR:-${REPO_ROOT}/outputs/non_wmdp/audit/snmf_gemma03b_arithmetic_eng_general_data_part1_rank${RANK}}"

# HF roots (directories containing config.json + weights).
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/non-wmdp/pretrained_models/gemma-2-0.3B_all_arithmetic+eng/final_model}"
export CANDIDATE_MODEL_PATH="${CANDIDATE_MODEL_PATH:-/home/morg/students/rashkovits/Localized-UNDO/models/non-wmdp/unlearned_models/MaxEnt/pretrained_models_gemma-2-0.3B_all_arithmetic+eng_final_model_lr_1.0e-04/final_model}"

export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/general_data_part1.json}"

# 0.3B has 14 transformer blocks (indices 0–13). Middle-ish band similar to 10–18 on 2B:
export LAYERS="${LAYERS:-5-10}"

# Keep audit artifacts under non-wmdp; override if you prefer.
SNMF_TAG="$(basename "${SNMF_DIR%/}")"
LAYER_TAG="$(printf '%s' "$LAYERS" | tr ',' '_' | tr '-' '_')"
_cand_norm="${CANDIDATE_MODEL_PATH%/}"
_cand_norm="${_cand_norm%/final_model}"
_run_name="$(basename "$_cand_norm")"
_method_name="$(basename "$(dirname "$_cand_norm")")"
DEFAULT_CANDIDATE_TAG="$(printf '%s_%s' "$_method_name" "$_run_name" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_' '_' | tr -s '_' | sed 's/^_//;s/_$//')"
export CANDIDATE_TAG="${CANDIDATE_TAG:-$DEFAULT_CANDIDATE_TAG}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/non_wmdp/audit_general/${SNMF_TAG}__${CANDIDATE_TAG}__layers_${LAYER_TAG}}"

exec bash "${REPO_ROOT}/scripts/audit/run_general_unlearning_audit.sh"
