#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=CPM_LLM_Rubric
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=12:00:00

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2

cd /home/psingh54/scratchjeisner1/psingh54/AnnotationArena/imputer/ranking

export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
set -e

SCRIPT_START=$SECONDS

SPLIT="LLMRubric_225_25_9_30"
DATA_DIR="DATA/LLM_RUBRIC/${SPLIT}"
DATA_BUNDLE="${DATA_DIR}/data_bundle.json"
OUTPUT_DIR="RESULTS/STAN/LLM_RUBRIC/CPM"
RUN_NAME="LLMRubric_225_25_9_30_NONTRANS"

CHAINS=1
ITER_WARMUP=300
ITER_SAMPLING=500
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42

echo ""
echo "============================================================"
echo " LLM_Rubric | CPM Tensor | ${SPLIT}"
echo "  RUN_NAME      : ${RUN_NAME}"
echo "  CHAINS        : ${CHAINS}"
echo "  ITER_WARMUP   : ${ITER_WARMUP}"
echo "  ITER_SAMPLING : ${ITER_SAMPLING}"
echo "============================================================"

echo "[1/2] Running MCMC (tensor)..."
python STAN/stan_code/scripts/run_inference.py \
    --data-bundle        "${DATA_BUNDLE}"            \
    --configs            "${DATA_DIR}/configs.json" \
    --output-dir         "${OUTPUT_DIR}"            \
    --run-name           "${RUN_NAME}"              \
    --stan-type          "tensor"                    \
    --chains             "${CHAINS}"                \
    --iter-warmup        "${ITER_WARMUP}"           \
    --iter-sampling      "${ITER_SAMPLING}"         \
    --adapt-delta        "${ADAPT_DELTA}"           \
    --max-treedepth      "${MAX_TREEDEPTH}"         \
    --seed               "${SEED}"                  \
    --no-transductive-use-test-observed \
    --overwrite-existing-data

echo "[2/2] Evaluating predictions..."
python STAN/stan_code/scripts/evaluate_predictions.py \
    --data-bundle        "${DATA_BUNDLE}"              \
    --mcmc-dir           "${OUTPUT_DIR}/${RUN_NAME}" \
    --output-dir         "${OUTPUT_DIR}"               \
    --run-name           "${RUN_NAME}_eval"           \
    --csv-pattern        "tensor_model-*.csv"          \
    --overwrite-existing-data                            \
    --verbose

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Results: ${OUTPUT_DIR}/${RUN_NAME}_eval"
echo "============================================================"
