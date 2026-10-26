#!/bin/bash
#SBATCH --job-name=B12-INTERACT
#SBATCH --partition=h100,a100,h200,l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --comment=accept_cost
#SBATCH --array=0-15%8
# B12: is the collapse the codec, the standard block, or the two together?
#
# The final run collapsed 8 of 42 cells to the uniform distribution (ln 4).
# Neither prior experiment tested the shipped pair: C2 ran the standard block
# with the hand-built dual stream, B11 ran the codec with the OLD block (its
# worktree has no block_output_projection flag at all). This crosses them at the
# four item sizes that actually collapsed.
#
#   arm A  dual  + standard block   <- the proposed fallback
#   arm B  typed + standard block   <- what the final run shipped; should collapse
set -euo pipefail
RERUN=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN
CD=/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking
source "${RERUN}/config/cluster.sh"
export DOMAIN3_REPO_ROOT_OVERRIDE="${CD}" PYTHONPATH="${CD}:${PYTHONPATH:-}"
rerun_activate_env

ARMS=(dual_std typed_std)
SEEDS=(42 43)
SIZES=(400 500 600 800)
i=${SLURM_ARRAY_TASK_ID}
ARM=${ARMS[$(( i / 8 ))]}
SEED=${SEEDS[$(( (i % 8) / 4 ))]}
SIZE=${SIZES[$(( i % 4 ))]}

if [[ "${ARM}" == "typed_std" ]]; then
    CODEC=(DOMAIN3_VALUE_CODEC=typed DOMAIN3_CODEC_INIT=unit)
else
    CODEC=(DOMAIN3_VALUE_CODEC=dual)
fi

R="${RERUN}/runs/codec-b12-s101"
mkdir -p "${R}/logs"
# 400 epochs: a collapsed run sits at ln(4) from early on, so the full budget is
# not needed to tell the two apart.
env DOMAIN3_DATA_ROOT="${RERUN}/runs/dataseed-s101/DATA/STAN/DOMAIN3-SIGMOID" \
    DOMAIN3_MAX_ITEM=500 DOMAIN3_SEED="${SEED}" DOMAIN3_EPOCHS=400 \
    DOMAIN3_DUAL_OUTER_RESIDUAL=none DOMAIN3_BLOCK_OUTPUT_PROJECTION=none \
    DOMAIN3_DATASET_LABEL=DOMAIN3-SIGMOID DOMAIN3_SPLIT_PREFIX=DOMAIN3-SIGMOID \
    DOMAIN3_EXP_PREFIX=domain3_sigmoid PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${CODEC[@]}" \
    bash "${CD}/scripts/DOMAIN3/run_marformer_group.sh" \
        --family-label Marformer \
        --output-root "${R}/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/${ARM}/Marformer/seed${SEED}" \
        --axis item --protocol transductive --sizes "${SIZE}" \
    > "${R}/logs/CELL-${ARM}-s${SEED}-K${SIZE}.log" 2>&1
