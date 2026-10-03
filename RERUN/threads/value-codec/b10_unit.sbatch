#!/bin/bash
#SBATCH --job-name=B10-UNIT
#SBATCH --partition=h100,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --comment=accept_cost
#SBATCH --array=0-5
# B10: is the encoder scale required to be logit_high, or merely loud?
# "scaled" reaches parity at K=2000 but reads logit_high to set its std, so the
# hand-built constant is still in the loop. "unit" is N(0,1): no constant at all,
# output norm sqrt(d)=8.9 against 20. Six runs, two draws x three seeds, K=2000
# only -- that is where parity was established and where the small-data failures
# are absent, so it is the clean read.
set -euo pipefail
RERUN=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN
CD=/weka/scratch/jhu/jeisner1/xwang397/Marformer-codec/imputer/ranking
source "${RERUN}/config/cluster.sh"
export DOMAIN3_REPO_ROOT_OVERRIDE="${CD}" PYTHONPATH="${CD}:${PYTHONPATH:-}"
rerun_activate_env

DRAWS=(s101 s101 s101 s105 s105 s105)
SEEDS=(42 43 44 42 43 44)
DRAW=${DRAWS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
R="${RERUN}/runs/codec-b3-${DRAW}"
mkdir -p "${R}/logs"

env DOMAIN3_DATA_ROOT="${R}/DATA/STAN/DOMAIN3-SIGMOID" \
    DOMAIN3_MAX_ITEM=300 DOMAIN3_SEED="${SEED}" \
    DOMAIN3_DATASET_LABEL=DOMAIN3-SIGMOID DOMAIN3_SPLIT_PREFIX=DOMAIN3-SIGMOID \
    DOMAIN3_EXP_PREFIX=domain3_sigmoid PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    DOMAIN3_VALUE_CODEC=typed DOMAIN3_CODEC_INIT=unit \
    bash "${CD}/scripts/DOMAIN3/run_marformer_group.sh" \
        --family-label Marformer \
        --output-root "${R}/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/codec_unit/Marformer/seed${SEED}" \
        --axis item --protocol transductive --sizes 2000 \
    > "${R}/logs/CELL-codec_unit-s${SEED}-K2000.log" 2>&1
