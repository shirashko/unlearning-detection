#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=analyze_snmf
#SBATCH --output=logs/analyze_%j.out
#SBATCH --error=logs/analyze_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

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

mkdir -p logs $HF_HOME

# --- Analysis I/O ---
# Default matches scripts/wmdp/train_snmf.sh OUTPUT_DIR; set RESULTS_DIR=outputs/snmf_train_results to analyze an older tree.
RESULTS_DIR="${RESULTS_DIR:-outputs/snmf_train_results_pipeline}"
SUMMARY_FILE="${SUMMARY_FILE:-analysis_summary.json}"

# --- Parallelism Optimization ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Execute Analysis ---
echo "--------------------------------------------------------"
echo "Starting SNMF results analysis on Node: $SLURMD_NODENAME"
echo "Results directory: $RESULTS_DIR"
echo "Per-run summary (counts + role meanings): $RESULTS_DIR/$SUMMARY_FILE"
echo "--------------------------------------------------------"

python analyze_snmf_results.py \
    --model-path "local_models/gemma-2-0.3B_reference_model" \
    --results-dir "$RESULTS_DIR" \
    --summary-filename "$SUMMARY_FILE" \
    --role-assignment-threshold 0.05 \
    --device "auto" \
    --seed 42 \
    --activation-context-top-n 10 \
    --activation-context-window 15

echo "--------------------------------------------------------"
echo "SNMF analysis finished"
echo "Feature counts and role definitions: $RESULTS_DIR/$SUMMARY_FILE"