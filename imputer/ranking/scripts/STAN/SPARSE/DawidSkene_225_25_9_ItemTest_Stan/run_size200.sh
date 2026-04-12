#!/bin/bash

SCRIPT_START=$SECONDS

SIZE=200
DATA_DIR="DATA/STAN/SPARSE/DawidSkene_225_25_9_ItemTest/DawidSkene_225_25_9_ItemTest_${SIZE}"
DATA_BUNDLE="${DATA_DIR}/data_bundle.json"
OUTPUT_DIR="RESULTS/STAN/SPARSE"
RUN_NAME="DawidSkene_225_25_9_ItemTest_${SIZE}_DS"

CHAINS=4
ITER_WARMUP=300
ITER_SAMPLING=500
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42

echo ""
echo "============================================================"
echo " Stan Dawid-Skene | DawidSkene_225_25_9_ItemTest_${SIZE}"
echo "  CHAINS       : ${CHAINS}"
echo "  ITER_WARMUP  : ${ITER_WARMUP}"
echo "  ITER_SAMPLING: ${ITER_SAMPLING}"
echo "============================================================"

echo "[1/2] Running MCMC (dawid-skene)..."
python STAN/stan_code/scripts/run_inference.py \
    --data-bundle        "$DATA_BUNDLE"   \
    --configs            "${DATA_DIR}/configs.json" \
    --output-dir         "$OUTPUT_DIR"    \
    --run-name           "$RUN_NAME"      \
    --stan-type          "dawid-skene"    \
    --chains             "$CHAINS"        \
    --iter-warmup        "$ITER_WARMUP"   \
    --iter-sampling      "$ITER_SAMPLING" \
    --adapt-delta        "$ADAPT_DELTA"   \
    --max-treedepth      "$MAX_TREEDEPTH" \
    --seed               "$SEED"          \
    --overwrite-existing-data

echo "[2/2] Evaluating predictions..."
python STAN/stan_code/scripts/evaluate_predictions.py \
    --data-bundle        "$DATA_BUNDLE"              \
    --mcmc-dir           "${OUTPUT_DIR}/${RUN_NAME}" \
    --output-dir         "$OUTPUT_DIR"               \
    --run-name           "${RUN_NAME}_eval"           \
    --csv-pattern        "dawid_skene_model-*.csv"   \
    --overwrite-existing-data                        \
    --verbose

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Results: ${OUTPUT_DIR}/${RUN_NAME}_eval"
echo "============================================================"
