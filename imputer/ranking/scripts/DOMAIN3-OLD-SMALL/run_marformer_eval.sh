#!/bin/bash

DOMAIN3_OLD_DEFAULT_CLUSTER_ROOT="/home/psingh54/scratchjeisner1/psingh54/Marformer/imputer/ranking"
DOMAIN3_OLD_DEFAULT_LOCAL_ROOT="/Users/prabhavsingh/Documents/JHU/JHUResearch/Marformer/imputer/ranking"

if [[ -n "${DOMAIN3_REPO_ROOT_OVERRIDE:-}" && -f "${DOMAIN3_REPO_ROOT_OVERRIDE}/scripts/DOMAIN3/run_marformer_eval.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_REPO_ROOT_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/DOMAIN3/run_marformer_eval.sh" ]]; then
    DOMAIN3_REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "${DOMAIN3_OLD_DEFAULT_CLUSTER_ROOT}/scripts/DOMAIN3/run_marformer_eval.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_OLD_DEFAULT_CLUSTER_ROOT}"
elif [[ -f "${DOMAIN3_OLD_DEFAULT_LOCAL_ROOT}/scripts/DOMAIN3/run_marformer_eval.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_OLD_DEFAULT_LOCAL_ROOT}"
else
    DOMAIN3_REPO_ROOT="$(pwd)"
fi

export DOMAIN3_REPO_ROOT_OVERRIDE="${DOMAIN3_REPO_ROOT}"
export DOMAIN3_DATASET_LABEL="DOMAIN3-OLD-SMALL"
export DOMAIN3_EXP_PREFIX="domain3_old_small"

bash "${DOMAIN3_REPO_ROOT}/scripts/DOMAIN3/run_marformer_eval.sh" "$@"
