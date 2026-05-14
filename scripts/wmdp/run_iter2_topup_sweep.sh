#!/bin/bash

# Parallel sweep over role-basis combinations on top of the iter-2 ablated checkpoint,
# using the data_part2 iter-2 SNMF basis. All three configs share:
#   - Base model      = iter-2 ablated checkpoint (default in run_create_forget_ablated_model.sh)
#   - SNMF results    = outputs/wmdp/results_data_part2_gemma2_2b_iter2_thr022_down_proj_only
#   - Threshold       = 0.18  (looser than iter-2's 0.22; residual distribution is shifted)
#   - DOWN_PROJ_ONLY  = 0     (edit both up_proj and down_proj)
#   - SPAN_SCALE      = 1.0   (full projection; scale sweep is the follow-up round)
#   - RANDOM_BASELINE = 1     (matched-count control per variant)
#
# Variants differ only in ROLE_LABEL_BASES × ROLE_BASIS_COMBINE. Each lands in its own
# SAVE_PATH so we can diff ablation_eval_comparison.json across runs.
#
# Usage:  bash scripts/wmdp/run_iter2_topup_sweep.sh
# Logs:   logs/forget_ablate_wmdp_bio_<jobid>.{out,err}

set -euo pipefail

REPO_ROOT="/home/morg/students/rashkovits/snmf"
cd "$REPO_ROOT"

SBATCH_SCRIPT="scripts/wmdp/run_create_forget_ablated_model.sh"

# Pin the base checkpoint explicitly so a stale MODEL_PATH exported in the
# submitting shell cannot silently redirect these jobs at a non-existent path
# (root cause of the 275613 failure: $MODEL_PATH was inherited by sbatch and
# pointed at a directory that doesn't exist on disk).
MODEL_PATH="${REPO_ROOT}/local_models/wmdp/iter2/data_part2_thr022_down_proj_only"
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: iter-2 base checkpoint not found: $MODEL_PATH" >&2
  exit 1
fi
export MODEL_PATH
# All three variants land as siblings inside this group dir. The config-specific
# sub-directory (bio_retain_and_neutral / bio_retain_or_neutral / pooled_or_bio_retain)
# is the HF ckpt folder; its _random sibling is the matched-count control.
SAVE_GROUP="local_models/wmdp/iter2_topup_data_part2_thr018_both_up_down"

echo "================================================================"
echo " iter-2 top-up sweep — role basis × combine"
echo " Base (iter-2) + SNMF (data_part2) come from run_create_forget_ablated_model.sh defaults."
echo " thr=0.18, DOWN_PROJ_ONLY=0, SPAN_PROJECTION_SCALE=1.0, RANDOM_BASELINE=1"
echo " Save group:  $SAVE_GROUP/<variant>"
echo "================================================================"

# --- A: strict specificity — bio_retain AND neutral ---
#   Picks latents that are forget-leaning both within bio (vs bio_retain) AND vs general
#   text (vs neutral). Strongest specificity filter; expected smallest N, best bio/mmlu ratio.
TAG_A="bio_retain_and_neutral"
SAVE_A="${SAVE_GROUP}/${TAG_A}"
echo
echo "==> A [$TAG_A]  SAVE_PATH=$SAVE_A"
ROLE_LABEL_BASES="bio_retain neutral" \
ROLE_BASIS_COMBINE="all" \
SAVE_PATH="$SAVE_A" \
SAVE_PATH_RANDOM="${SAVE_A}_random" \
  sbatch --job-name="iter2_topup_${TAG_A}" "$SBATCH_SCRIPT"

# --- B: permissive OR — bio_retain OR neutral ---
#   Union selector; expected larger N; tests whether dropping the AND-intersection helps or
#   just adds collateral. Same bases as A so it's a controlled contrast on combine=all vs any.
TAG_B="bio_retain_or_neutral"
SAVE_B="${SAVE_GROUP}/${TAG_B}"
echo
echo "==> B [$TAG_B]  SAVE_PATH=$SAVE_B"
ROLE_LABEL_BASES="bio_retain neutral" \
ROLE_BASIS_COMBINE="any" \
SAVE_PATH="$SAVE_B" \
SAVE_PATH_RANDOM="${SAVE_B}_random" \
  sbatch --job-name="iter2_topup_${TAG_B}" "$SBATCH_SCRIPT"

# --- C: current family loosened — pooled OR bio_retain ---
#   Same bases as the existing iter-2 pipeline, but with combine=any and thr=0.18.
#   Baseline to compare A/B against: did swapping `pooled` out for `neutral` actually help?
TAG_C="pooled_or_bio_retain"
SAVE_C="${SAVE_GROUP}/${TAG_C}"
echo
echo "==> C [$TAG_C]  SAVE_PATH=$SAVE_C"
ROLE_LABEL_BASES="pooled bio_retain" \
ROLE_BASIS_COMBINE="any" \
SAVE_PATH="$SAVE_C" \
SAVE_PATH_RANDOM="${SAVE_C}_random" \
  sbatch --job-name="iter2_topup_${TAG_C}" "$SBATCH_SCRIPT"

echo
echo "================================================================"
echo " Submitted 3 sweep jobs. Monitor with:   squeue -u \$USER"
echo " Logs (per job):     logs/forget_ablate_wmdp_bio_<jobid>.{out,err}"
echo " Eval JSONs:         <SAVE_PATH>/ablation_eval_comparison.json"
echo " Summarize results:  bash scripts/wmdp/summarize_iter2_topup_sweep.sh"
echo "================================================================"
