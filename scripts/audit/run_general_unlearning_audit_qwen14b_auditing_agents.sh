#!/bin/bash
#SBATCH --job-name=audit_qwen14b_aa
#SBATCH --output=logs/general_unlearning_audit_qwen14b_aa_%j.out
#SBATCH --error=logs/general_unlearning_audit_qwen14b_aa_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#
# Label-free SNMF audit: Qwen3‑14B *base* vs auditing-agents PEFT head (merged).
#
#   • SNMF factors Z must be trained on M_base with the SAME general JSON the
#     audit uses (see train_snmf_qwen14b_general_hf.sh; default backbone qwen/qwen3-14b).
#   • BASE_MODEL_PATH defaults to qwen/qwen3-14b (matches LoRA adapter_config base).
#   • CANDIDATE_MODEL_PATH defaults to the high‑flattery redteam variant:
#       auditing-agents/qwen_14b_synth_docs_only_then_redteam_high_flattery
#     Collection: huggingface.co/collections/auditing-agents/qwen-collection-synth-docs-sft-adv-train
#
# Prerequisites:
#   - Run train_snmf_qwen14b_general_hf.sh first; set SNMF_DIR to its OUTPUT_DIR.
#   - Default CANDIDATE is a Hugging Face *LoRA-only* repo (adapter_config.json).
#     load_local_model merges adapter into the base named in adapter_config via `peft`.
#     Install: pip install peft (also listed in requirements.txt).
#   - The auditing-agents checkpoint lists base_model_name_or_path (see adapter_config.json on HF).
#     Fit SNMF on that same backbone if you want dimensions/layers to match the merged candidate.
#     Optional env PEFT_BASE_MODEL_OVERRIDE fixes a wrong base path in adapter_config when the
#     LoRA weights were trained against a different hub id (must match tensor shapes).
#   - export HF_TOKEN=... if required.
#
# Submit from repo root (logs/ under ./logs), e.g. cd /path/to/snmf && sbatch ....
# If Slurm stages the script under /var/spool, BASH_SOURCE no longer lies in the repo:
#   export SNMF_REPO_ROOT=/path/to/snmf
#
# Examples:
#   sbatch scripts/audit/run_general_unlearning_audit_qwen14b_auditing_agents.sh
#   env SNMF_DIR=/path/to/snmf_qwen... LAYERS=16-23 MAX_PROMPTS=200 \\
#       JUDGE_MAX_OUTPUT_TOKENS=8192 sbatch \\
#       scripts/audit/run_general_unlearning_audit_qwen14b_auditing_agents.sh
#
# To compare against a different auditing-agents head (same SNMF dir):
#   env CANDIDATE_MODEL_PATH=auditing-agents/other_qwen14b_ckpt \\
#       sbatch scripts/audit/run_general_unlearning_audit_qwen14b_auditing_agents.sh
#
set -euo pipefail

_resolve_repo_root() {
  if [[ -n "${SNMF_REPO_ROOT:-}" ]]; then
    local r
    r="$(cd "$SNMF_REPO_ROOT" && pwd)"
    if [[ -f "$r/train_snmf.py" ]]; then
      printf '%s' "$r"
      return 0
    fi
    echo "[run_general_unlearning_audit_qwen14b_auditing_agents.sh] SNMF_REPO_ROOT=$SNMF_REPO_ROOT but train_snmf.py not found there." >&2
    exit 1
  fi
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local via_script
  via_script="$(cd "$script_dir/../.." && pwd)"
  if [[ -f "$via_script/train_snmf.py" ]]; then
    printf '%s' "$via_script"
    return 0
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    local sub
    sub="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
    if [[ -f "$sub/train_snmf.py" ]]; then
      printf '%s' "$sub"
      return 0
    fi
  fi
  echo "[run_general_unlearning_audit_qwen14b_auditing_agents.sh] Could not find repo root (train_snmf.py)." >&2
  echo "  Fix: cd to the snmf checkout and sbatch from there, or export SNMF_REPO_ROOT=/path/to/snmf" >&2
  exit 1
}

REPO_ROOT="$(_resolve_repo_root)"

export BASE_MODEL_PATH="${BASE_MODEL_PATH:-qwen/qwen3-14b}"
export CANDIDATE_MODEL_PATH="${CANDIDATE_MODEL_PATH:-auditing-agents/qwen_14b_synth_docs_only_then_redteam_high_flattery}"

_slug_base="$(printf '%s' "$BASE_MODEL_PATH" | tr '/' '_' | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_' '_' | tr -s '_' | sed 's/^_//;s/_$//')"
_slug_cand="$(printf '%s' "$CANDIDATE_MODEL_PATH" | tr '/' '_' | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_' '_' | tr -s '_' | sed 's/^_//;s/_$//')"
_RANK_HINT="${SNMF_RANK:-256}"
export SNMF_DIR="${SNMF_DIR:-${REPO_ROOT}/outputs/hf_audits/snmf_${_slug_base}_general_p1_r${_RANK_HINT}}"

# Must match (or be a subset of) the layers you trained in SNMF_DIR (Qwen3: 0–39).
export LAYERS="${LAYERS:-16-23}"

export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/general_data_part1.json}"
export MAX_PROMPTS="${MAX_PROMPTS:-400}"
export BATCH_SIZE="${BATCH_SIZE:-1}"

SNMF_TAG="$(basename "${SNMF_DIR%/}")"
LAYER_TAG="$(printf '%s' "$LAYERS" | tr ',' '_' | tr '-' '_')"
DEFAULT_OUT="${REPO_ROOT}/outputs/hf_audits/audit_general/${SNMF_TAG}__${_slug_cand}__layers_${LAYER_TAG}"
export OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUT}"

# Higher default helps Gemini 2.5 avoid truncated judge JSON on long prompts.
export JUDGE_MAX_OUTPUT_TOKENS="${JUDGE_MAX_OUTPUT_TOKENS:-8192}"

exec bash "${REPO_ROOT}/scripts/audit/run_general_unlearning_audit.sh"
