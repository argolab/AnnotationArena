#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.


SCRIPT_START=$SECONDS

export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
set -e

export XDG_CACHE_HOME=/weka/scratch/jeisner1/xwang397/.cache
export TMPDIR=/weka/scratch/jeisner1/xwang397/tmp/${SLURM_JOB_ID:-manual}
mkdir -p "$TMPDIR"


SCRIPT_START=$SECONDS

SIZE=50
DATA_DIR="DATA/STAN/DOMAIN3-ITEM/Tensor_400_25_9_ItemTest_SharedThreshold/Tensor_400_25_9_ItemTest_SharedThreshold_${SIZE}"
DATA_BUNDLE="${DATA_DIR}/data_bundle.json"
OUTPUT_DIR="RESULTS/STAN/SPARSE"
RUN_NAME="Tensor_400_25_9_ItemTest_SharedThreshold_${SIZE}_NONTRANS"

CHAINS=1
ITER_WARMUP=300
ITER_SAMPLING=500
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42

STAN_FILE_SRC="STAN/stan_models/shared_threshold_tensor_model.stan"
STAN_STAGE_DIR="${XDG_CACHE_HOME}/stan_shared_threshold_${SLURM_JOB_ID:-$$}"
mkdir -p "$STAN_STAGE_DIR"
LOCAL_STAN_FILE="$STAN_STAGE_DIR/shared_threshold_tensor_model.stan"
cp "$STAN_FILE_SRC" "$LOCAL_STAN_FILE"


echo ""
echo "============================================================"
echo " Stan Tensor | Tensor_500_25_9_ItemTest_SharedThreshold_${SIZE}"
echo "  CHAINS       : ${CHAINS}"
echo "  ITER_WARMUP  : ${ITER_WARMUP}"
echo "  ITER_SAMPLING: ${ITER_SAMPLING}"
echo "============================================================"

echo "[1/2] Running MCMC (tensor)..."
python STAN/stan_code/scripts/run_inference.py \
    --data-bundle        "$DATA_BUNDLE"   \
    --configs            "${DATA_DIR}/configs.json" \
    --output-dir         "$OUTPUT_DIR"    \
    --run-name           "$RUN_NAME"      \
    --stan-type          "tensor"         \
    --stan-file          "$LOCAL_STAN_FILE" \
    --chains             "$CHAINS"        \
    --iter-warmup        "$ITER_WARMUP"   \
    --iter-sampling      "$ITER_SAMPLING" \
    --adapt-delta        "$ADAPT_DELTA"   \
    --max-treedepth      "$MAX_TREEDEPTH" \
    --seed               "$SEED"          \
    --no-transductive-use-test-observed \
    --overwrite-existing-data

echo "[2/2] Evaluating predictions..."
python STAN/stan_code/scripts/evaluate_predictions.py \
    --data-bundle        "$DATA_BUNDLE"              \
    --mcmc-dir           "${OUTPUT_DIR}/${RUN_NAME}" \
    --output-dir         "$OUTPUT_DIR"               \
    --run-name           "${RUN_NAME}_eval"           \
    --csv-pattern        "shared_threshold_tensor_model-*.csv"        \
    --overwrite-existing-data                        \
    --verbose

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Results: ${OUTPUT_DIR}/${RUN_NAME}_eval"
echo "============================================================"
