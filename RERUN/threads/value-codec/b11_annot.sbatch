#!/bin/bash
#SBATCH --job-name=B11-ANNOT
#SBATCH --partition=h100,a100,l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --comment=accept_cost
#SBATCH --array=0-11%6
# B11-annot: the same question on the annotator axis.
#
# The paper's design runs the annotator axis at J=10..50, and J=10 is a smaller
# data regime than any item cell, so if the codec has a small-data failure mode
# this is where it should show. Sizes 10/30/50 span the paper's range.
#
# B10 established parity at K=2000 and explicitly limited itself there. The
# open risk is B7's: with "scaled", 1 run in 12 blows up at K=300 (0.73 ->
# 1.06), orthogonal init fails at the same rate, and "unit" was never run below
# 2000 at all. The final run's item axis starts at K=100, so the small-K
# failure rate decides whether the codec can ship.
#
# Six model seeds x two sizes x two arms on the shipping draw (s101), paired on
# (seed, K). Results land in their own run tree so nothing in codec-b3 is
# overwritten; the data bundles are read from codec-b3-s101.
set -euo pipefail
RERUN=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN
CD=/weka/scratch/jhu/jeisner1/xwang397/Marformer-codec/imputer/ranking
source "${RERUN}/config/cluster.sh"
export DOMAIN3_REPO_ROOT_OVERRIDE="${CD}" PYTHONPATH="${CD}:${PYTHONPATH:-}"
rerun_activate_env

# 12 tasks = 2 arms x 6 seeds; each task runs both sizes.
ARMS=(codec_dual codec_unit)
SEEDS=(42 43 44 45 46 47)
ARM=${ARMS[$(( SLURM_ARRAY_TASK_ID / 6 ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 6 ))]}

if [[ "${ARM}" == "codec_unit" ]]; then
    CODEC_ENV=(DOMAIN3_VALUE_CODEC=typed DOMAIN3_CODEC_INIT=unit)
else
    CODEC_ENV=(DOMAIN3_VALUE_CODEC=dual)
fi

DATA="${RERUN}/runs/codec-b3-s101/DATA/STAN/DOMAIN3-SIGMOID"
R="${RERUN}/runs/codec-b11-s101"
mkdir -p "${R}/logs"

# max_item=500 is the shipped chunk. At K<=300 the chunk is never binding, so
# this matches the final configuration without changing what is measured.
env DOMAIN3_DATA_ROOT="${DATA}" \
    DOMAIN3_MAX_ITEM=500 DOMAIN3_SEED="${SEED}" \
    DOMAIN3_DATASET_LABEL=DOMAIN3-SIGMOID DOMAIN3_SPLIT_PREFIX=DOMAIN3-SIGMOID \
    DOMAIN3_EXP_PREFIX=domain3_sigmoid PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${CODEC_ENV[@]}" \
    bash "${CD}/scripts/DOMAIN3/run_marformer_group.sh" \
        --family-label Marformer \
        --output-root "${R}/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/${ARM}/Marformer/seed${SEED}" \
        --axis annot --protocol transductive --sizes 10,30,50 \
    > "${R}/logs/CELL-${ARM}-s${SEED}-J10_30_50.log" 2>&1
