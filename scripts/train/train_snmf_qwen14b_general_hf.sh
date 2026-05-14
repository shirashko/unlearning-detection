#!/bin/bash
#SBATCH --job-name=snmf_qwen14b_gen
#SBATCH --output=logs/train_snmf_qwen14b_general_hf_%j.out
#SBATCH --error=logs/train_snmf_qwen14b_general_hf_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#
# Fit SNMF (mlp_intermediate) on activations from Hugging Face Qwen‑14B-class
# checkpoints, using the same unlabeled prompts as the label-free audit
# (data/general_data_part1.json).
#
# Default MODEL_PATH is qwen/qwen3-14b — the backbone named in auditing-agents
# LoRA repos (adapter_config base_model_name_or_path), including the “high flattery”
# head (mlp intermediate_size=17408). This keeps SNMF Z aligned with merged candidates.
#
# Qwen3‑14B has num_hidden_layers=40 (indices 0–39). Defaults keep a middle slice
# analogous to the old Qwen2.5 job (lighter than all layers). For Qwen2.5‑14B
# (48 layers, d_mlp=13824) override explicitly, e.g. MODEL_PATH=Qwen/Qwen2.5-14B-Instruct
# LAYERS=20-27.
#
# Prerequisites:
#   - Hugging Face CLI / token if a repo is gated: export HF_TOKEN=...
#   - GPU with enough memory for ~14B in float32 forward (tight on 80 GB; try
#     BATCH_SIZE=1 and smaller MAX_PROMPTS for smoke tests).
#
# Examples:
#   sbatch scripts/audit/train_snmf_qwen14b_general_hf.sh
#   env MODEL_PATH=qwen/qwen3-14b RANK=256 LAYERS=16-23 \
#       MAX_PROMPTS=200 BATCH_SIZE=1 sbatch scripts/audit/train_snmf_qwen14b_general_hf.sh
#   # Legacy Qwen2.5 basis (does not match Qwen3 LoRA merges):
#   env MODEL_PATH=Qwen/Qwen2.5-14B-Instruct LAYERS=20-27 \
#       sbatch scripts/audit/train_snmf_qwen14b_general_hf.sh
#
# Submit from the snmf repo root (so #SBATCH output paths resolve under ./logs), e.g.:
#   cd /path/to/snmf && sbatch scripts/audit/train_snmf_qwen14b_general_hf.sh
# Or set SNMF_REPO_ROOT explicitly if your site copies batch scripts to Slurm spool (then
# BASH_SOURCE no longer lies under the repo and a naive ../.. would point at /var/spool/...).
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
    echo "[train_snmf_qwen14b_general_hf.sh] SNMF_REPO_ROOT=$SNMF_REPO_ROOT but train_snmf.py not found there." >&2
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
  echo "[train_snmf_qwen14b_general_hf.sh] Could not find repo root (train_snmf.py)." >&2
  echo "  Fix: cd to the snmf checkout and sbatch from there, or export SNMF_REPO_ROOT=/path/to/snmf" >&2
  exit 1
}

REPO_ROOT="$(_resolve_repo_root)"
# shellcheck disable=SC1090
if [[ -f /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh ]]; then
  source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
  conda activate /home/morg/students/rashkovits/envs/snmf_env 2>/dev/null \
    || conda activate snmf_env
fi

export HF_HOME="${HF_HOME:-/home/morg/students/rashkovits/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export TMPDIR="${TMPDIR:-$HF_HOME/tmp}"
mkdir -p "$HF_HOME" "$TMPDIR"

cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
mkdir -p logs

# Hub id or local HF directory (passed through to train_snmf.py → load_local_model).
MODEL_PATH="${MODEL_PATH:-qwen/qwen3-14b}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/general_data_part1.json}"
RANK="${RANK:-256}"
# Middle layers only (Qwen3‑14B: 40 layers → 0–39).
LAYERS="${LAYERS:-16-23}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SNMF_MODE="${SNMF_MODE:-mlp_intermediate}"
SNMF_INIT="${SNMF_INIT:-svd}"
DEVICE="${DEVICE:-cuda}"
SPARSITY="${SPARSITY:-0.01}"
MAX_ITER="${MAX_ITER:-3000}"
SEED="${SEED:-42}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
MAX_PROMPTS="${MAX_PROMPTS:-400}"

_model_slug="$(printf '%s' "$MODEL_PATH" | tr '/' '_' | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_' '_' | tr -s '_' | sed 's/^_//;s/_$//')"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/hf_audits/snmf_${_model_slug}_general_p1_r${RANK}}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"

mkdir -p "$OUTPUT_DIR"

if [[ "$REQUIRE_GPU" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[train_snmf_qwen14b_general_hf.sh] REQUIRE_GPU=1 but nvidia-smi unavailable."
    exit 1
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[train_snmf_qwen14b_general_hf.sh] REQUIRE_GPU=1 but no visible GPU."
    exit 1
  fi
fi

echo "----------------------------------------------------------------"
echo " SNMF (general audit basis) — Qwen 14B-class HF model (default: Qwen3 backbone for auditing-agents LoRA)"
echo " MODEL_PATH:    $MODEL_PATH"
echo " DATA_PATH:     $DATA_PATH"
echo " OUTPUT_DIR:    $OUTPUT_DIR"
echo " LAYERS:        $LAYERS"
echo " RANK:          $RANK"
echo " MAX_PROMPTS:   $MAX_PROMPTS  (cap in train_snmf via data subsample if added)"
echo " BATCH_SIZE:    $BATCH_SIZE"
echo "----------------------------------------------------------------"

# train_snmf.py loads the full JSON; use a temp JSON with at most MAX_PROMPTS strings
# when MAX_PROMPTS > 0 by delegating to Python one-liner subsample.
effective_data="$DATA_PATH"
if [[ "$MAX_PROMPTS" != "0" && -n "$MAX_PROMPTS" ]]; then
  effective_data="${OUTPUT_DIR}/.audit_prompts_cap_${MAX_PROMPTS}_seed_${SEED}.json"
  python3 - <<PY
import json, random
from pathlib import Path
src = Path("${DATA_PATH}")
out = Path("${effective_data}")
with src.open() as f:
    data = json.load(f)
if isinstance(data, list):
    prompts = [x for x in data if isinstance(x, str) and x.strip()]
elif isinstance(data, dict):
    prompts = []
    for v in data.values():
        if isinstance(v, list):
            prompts.extend(str(x) for x in v if isinstance(x, str) and x.strip())
else:
    raise SystemExit("Unsupported JSON schema in data file")
rng = random.Random(${SEED})
n = min(int(${MAX_PROMPTS}), len(prompts))
if n < len(prompts):
    prompts = rng.sample(prompts, n)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    json.dump(prompts, f, indent=0)
print(f"Wrote {len(prompts)} prompts to {out}")
PY
fi

python train_snmf.py \
  --model-path "$MODEL_PATH" \
  --data-path "$effective_data" \
  --output-dir "$OUTPUT_DIR" \
  --layers "$LAYERS" \
  --rank "$RANK" \
  --mode "$SNMF_MODE" \
  --init "$SNMF_INIT" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --sparsity "$SPARSITY" \
  --max-iter "$MAX_ITER" \
  --seed "$SEED"

echo "----------------------------------------------------------------"
echo " Done. Point SNMF_DIR / run_general_unlearning_audit*.sh at:"
echo "   $OUTPUT_DIR"
echo "----------------------------------------------------------------"
