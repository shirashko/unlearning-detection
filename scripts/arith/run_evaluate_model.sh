#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=evaluate_snmf
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

# --- Environment Setup ---
source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate snmf_env

# --- Space & Cache Management ---
export HF_HOME="/home/morg/students/rashkovits/hf_cache"
export TORCH_HOME="/home/morg/students/rashkovits/hf_cache/torch"
export TMPDIR="/home/morg/students/rashkovits/hf_cache"

# --- Project Setup ---
cd /home/morg/students/rashkovits/snmf
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p logs outputs/eval_results "$HF_HOME" cache

# --- Parallelism Optimization ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Config (override by exporting before sbatch/bash) ---
MODEL_PATH="${MODEL_PATH:-local_models/gemma-2-0.3B_reference_model}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
CACHE_DIR="${CACHE_DIR:-./cache}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-./cache}"
ENG_VALID_FILE="${ENG_VALID_FILE:-/home/morg/students/rashkovits/Localized-UNDO/datasets/pretrain/valid_eng.jsonl}"
RESULTS_JSON="${RESULTS_JSON:-outputs/eval_results/eval_${SLURM_JOB_ID:-local}_results.json}"

# --- Execute Evaluation ---
echo "--------------------------------------------------------"
echo "Starting SNMF Evaluation on Node: ${SLURMD_NODENAME:-local}"
echo "Model path: $MODEL_PATH"
echo "Validation file: $ENG_VALID_FILE"
echo "Results json: $RESULTS_JSON"
echo "--------------------------------------------------------"

python3 evaluation/eveluate_model.py \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --cache-dir "$CACHE_DIR" \
    --dataset-cache-dir "$DATASET_CACHE_DIR" \
    --eng-valid-file "$ENG_VALID_FILE" \
    --output-json "$RESULTS_JSON"

echo "--------------------------------------------------------"
echo "SNMF Evaluation Finished"
