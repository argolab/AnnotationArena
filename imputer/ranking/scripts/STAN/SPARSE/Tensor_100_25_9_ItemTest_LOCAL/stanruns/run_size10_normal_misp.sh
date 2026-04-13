#!/bin/bash

SCRIPT_START=$SECONDS

SIZE=10
DATA_DIR="DATA/STAN/SPARSE/Tensor_100_25_9_ItemTest/Tensor_100_25_9_ItemTest_${SIZE}"
DATA_BUNDLE="${DATA_DIR}/data_bundle.json"
OUTPUT_DIR="RESULTS/STAN/SPARSE"
RUN_NAME="Tensor_100_25_9_ItemTest_${SIZE}_NORMAL_MISP"

CHAINS=1
ITER_WARMUP=300
ITER_SAMPLING=500
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42

echo ""
echo "============================================================"
echo " Stan Normal Misspecified | Tensor_100_25_9_ItemTest_${SIZE}"
echo "  CHAINS       : ${CHAINS}"
echo "  ITER_WARMUP  : ${ITER_WARMUP}"
echo "  ITER_SAMPLING: ${ITER_SAMPLING}"
echo "  (doubled priors: kappa=20.0, sigma_measurement=0.2, sigma_annotator=0.6)"
echo "============================================================"

echo "[1/2] Running MCMC (normal-noise-dot-product, misspecified)..."
python STAN/stan_code/scripts/run_inference.py \
    --data-bundle              "$DATA_BUNDLE"   \
    --configs                  "${DATA_DIR}/configs.json" \
    --output-dir               "$OUTPUT_DIR"    \
    --run-name                 "$RUN_NAME"      \
    --stan-type                "normal-noise-dot-product" \
    --chains                   "$CHAINS"        \
    --iter-warmup              "$ITER_WARMUP"   \
    --iter-sampling            "$ITER_SAMPLING" \
    --adapt-delta              "$ADAPT_DELTA"   \
    --max-treedepth            "$MAX_TREEDEPTH" \
    --seed                     "$SEED"          \
    --override-alpha-dirichlet 20.0             \
    --override-sigma-measurement 0.5            \
    --override-sigma-annotator 0.6              \
    --stan-arg                 d_annotator=16    \
    --stan-arg                 use_factored_annotator=0 \
    --overwrite-existing-data

echo "[2/2] Evaluating predictions..."
python STAN/stan_code/scripts/evaluate_predictions.py \
    --data-bundle        "$DATA_BUNDLE"              \
    --mcmc-dir           "${OUTPUT_DIR}/${RUN_NAME}" \
    --output-dir         "$OUTPUT_DIR"               \
    --run-name           "${RUN_NAME}_eval"           \
    --csv-pattern        "normal_noise_dot_product_model-*.csv" \
    --overwrite-existing-data                        \
    --verbose

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Results: ${OUTPUT_DIR}/${RUN_NAME}_eval"
echo "============================================================"
