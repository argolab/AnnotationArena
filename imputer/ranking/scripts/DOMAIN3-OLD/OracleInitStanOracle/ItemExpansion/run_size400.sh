#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=D3OOIT-OI-400
#SBATCH --output=/home/psingh54/scratchjeisner1/psingh54/Marformer/imputer/ranking/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=24GB
#SBATCH --partition=cpu
#SBATCH --time=48:00:00

DOMAIN3_DEFAULT_CLUSTER_ROOT="/home/psingh54/scratchjeisner1/psingh54/Marformer/imputer/ranking"
DOMAIN3_DEFAULT_LOCAL_ROOT="/Users/prabhavsingh/Documents/JHU/JHUResearch/Marformer/imputer/ranking"

if [[ -n "${DOMAIN3_REPO_ROOT_OVERRIDE:-}" && -f "${DOMAIN3_REPO_ROOT_OVERRIDE}/scripts/DOMAIN3/run_stan_group.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_REPO_ROOT_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/DOMAIN3/run_stan_group.sh" ]]; then
    DOMAIN3_REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "${DOMAIN3_DEFAULT_CLUSTER_ROOT}/scripts/DOMAIN3/run_stan_group.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_DEFAULT_CLUSTER_ROOT}"
elif [[ -f "${DOMAIN3_DEFAULT_LOCAL_ROOT}/scripts/DOMAIN3/run_stan_group.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_DEFAULT_LOCAL_ROOT}"
else
    DOMAIN3_REPO_ROOT="$(pwd)"
fi

export DOMAIN3_REPO_ROOT_OVERRIDE="${DOMAIN3_REPO_ROOT}"
# shellcheck disable=SC1091
source "${DOMAIN3_REPO_ROOT}/scripts/DOMAIN3/common.sh"

export DOMAIN3_DATASET_LABEL="DOMAIN3-OLD"
export DOMAIN3_DATA_ROOT="DATA/STAN/DOMAIN3-OLD"
export DOMAIN3_SPLIT_PREFIX="DOMAIN3-OLD"

domain3_setup_env

SIZE=400
STAN_FILE="STAN/stan_models/tensor_model.stan"
LOCAL_STAN_FILE="$(domain3_stage_stan_file "${STAN_FILE}")"
CSV_PATTERN="$(basename "${LOCAL_STAN_FILE%.stan}")-*.csv"

DATA_ROOT="DATA/STAN/DOMAIN3-OLD"
SPLIT_PREFIX="DOMAIN3-OLD"
SPLIT_NAME="${SPLIT_PREFIX}_Item_T_${SIZE}"
DATA_DIR="${DATA_ROOT}/ItemSplits/Transductive/${SPLIT_NAME}"
DATA_BUNDLE="${DATA_DIR}/data_bundle.json"

OUTPUT_ROOT="RESULTS/STAN/DOMAIN3-OLD/ORACLE_INIT_STAN/ITEM"
RUN_NAME="${SPLIT_NAME}_TENSOR_OI"

CHAINS=1
ITER_WARMUP=100
ITER_SAMPLING=200
ADAPT_DELTA=0.85
MAX_TREEDEPTH=13
SEED=42

domain3_enter_repo

domain3_print_block "DOMAIN3-OLD | OracleInitStanOracle | item | size=${SIZE}"
echo " DATA_DIR    : ${DATA_DIR}"
echo " OUTPUT_ROOT : ${OUTPUT_ROOT}"
echo " RUN_NAME    : ${RUN_NAME}"
echo " INIT        : ground_truth"
echo " CHAINS      : ${CHAINS}"
echo " ITER_WARMUP : ${ITER_WARMUP}"
echo " ITER_SAMPLING: ${ITER_SAMPLING}"

SCRIPT_START=$SECONDS

echo ""
echo "[1/2] Running MCMC with oracle init..."
FIT_START=$SECONDS
python STAN/stan_code/scripts/run_inference.py \
    --data-bundle        "${DATA_BUNDLE}" \
    --configs            "${DATA_DIR}/configs.json" \
    --output-dir         "${OUTPUT_ROOT}" \
    --run-name           "${RUN_NAME}" \
    --stan-type          "tensor" \
    --stan-file          "${LOCAL_STAN_FILE}" \
    --chains             "${CHAINS}" \
    --iter-warmup        "${ITER_WARMUP}" \
    --iter-sampling      "${ITER_SAMPLING}" \
    --adapt-delta        "${ADAPT_DELTA}" \
    --max-treedepth      "${MAX_TREEDEPTH}" \
    --seed               "${SEED}" \
    --init-strategy      "ground_truth" \
    --overwrite-existing-data
FIT_ELAPSED=$(( SECONDS - FIT_START ))
echo "  ↳ fit: $(( FIT_ELAPSED / 60 ))m $(( FIT_ELAPSED % 60 ))s"

echo ""
echo "[2/2] Evaluating predictions..."
EVAL_START=$SECONDS
python STAN/stan_code/scripts/evaluate_predictions.py \
    --data-bundle        "${DATA_BUNDLE}" \
    --mcmc-dir           "${OUTPUT_ROOT}/${RUN_NAME}" \
    --output-dir         "${OUTPUT_ROOT}" \
    --run-name           "${RUN_NAME}_eval" \
    --csv-pattern        "${CSV_PATTERN}" \
    --overwrite-existing-data \
    --verbose
EVAL_ELAPSED=$(( SECONDS - EVAL_START ))
echo "  ↳ eval: $(( EVAL_ELAPSED / 60 ))m $(( EVAL_ELAPSED % 60 ))s"

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
domain3_print_block "DOMAIN3-OLD | OracleInitStanOracle | size=${SIZE} complete"
echo " TOTAL_TIME  : $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " OUTPUT_ROOT : ${OUTPUT_ROOT}"
