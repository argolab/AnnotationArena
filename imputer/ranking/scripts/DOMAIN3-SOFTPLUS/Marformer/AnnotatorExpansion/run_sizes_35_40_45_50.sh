#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=D3OMAT1
#SBATCH --output=/home/psingh54/scratchjeisner1/psingh54/Marformer/imputer/ranking/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18GB
#SBATCH --gpus=1
#SBATCH --partition=a100
#SBATCH --exclude=c001
#SBATCH --mail-user="psingh54@jhu.edu"
#SBATCH --time=18:00:00

DOMAIN3_DEFAULT_CLUSTER_ROOT="/home/psingh54/scratchjeisner1/psingh54/Marformer/imputer/ranking"
DOMAIN3_DEFAULT_LOCAL_ROOT="/Users/prabhavsingh/Documents/JHU/JHUResearch/Marformer/imputer/ranking"

if [[ -n "${DOMAIN3_REPO_ROOT_OVERRIDE:-}" && -d "${DOMAIN3_REPO_ROOT_OVERRIDE}/scripts/DOMAIN3-SOFTPLUS" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_REPO_ROOT_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/scripts/DOMAIN3-SOFTPLUS" ]]; then
    DOMAIN3_REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -d "${DOMAIN3_DEFAULT_CLUSTER_ROOT}/scripts/DOMAIN3-SOFTPLUS" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_DEFAULT_CLUSTER_ROOT}"
elif [[ -d "${DOMAIN3_DEFAULT_LOCAL_ROOT}/scripts/DOMAIN3-SOFTPLUS" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_DEFAULT_LOCAL_ROOT}"
else
    DOMAIN3_REPO_ROOT="${PWD}"
fi

export DOMAIN3_REPO_ROOT_OVERRIDE="${DOMAIN3_REPO_ROOT}"

bash "${DOMAIN3_REPO_ROOT}/scripts/DOMAIN3-SOFTPLUS/run_marformer_group.sh" \
    --family-label "Marformer" \
    --output-root "RESULTS/MARFORMER/DOMAIN3-SOFTPLUS/ANNOT" \
    --axis "annot" \
    --protocol "transductive" \
    --sizes "35,40,45,50"
