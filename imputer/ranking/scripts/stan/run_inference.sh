#!/bin/bash
set -e
 
# ── Data ───────────────────────────────────────────────────────────────────────
DATA_DIR="DATA/STAN/Normal_250_20_9_AnnotatorTest/Normal_250_20_9_AnnotatorTest_3"          # path to generated data run dir
CONFIGS="$DATA_DIR/configs.json"                     # configs.json in same dir as bundle
DATA_BUNDLE="$DATA_DIR/data_bundle.json"
 
# ── MCMC hyperparameters ───────────────────────────────────────────────────────
CHAINS="${CHAINS:-1}"
ITER_WARMUP="${ITER_WARMUP:-1}"
ITER_SAMPLING="${ITER_SAMPLING:-1}"
ADAPT_DELTA="${ADAPT_DELTA:-0.85}"
MAX_TREEDEPTH="${MAX_TREEDEPTH:-12}"
SEED="${SEED:-42}"
 
# ── Model hyperparameters ──────────────────────────────────────────────────────
EMBEDDING_DIM="${EMBEDDING_DIM:-8}"
ALPHA_LLM="${ALPHA_LLM:-5.0}"
 
# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR="RESULTS/STAN"
RUN_NAME="Normal_250_20_9_AnnotatorTest_3"
 
# ==============================================================================
# factored-dot-product
# ==============================================================================
RUN_BASE="${RUN_NAME}_Normal"
 
echo "[1/2] Running MCMC (normal-noise-dot-product)..."
python STAN/stan_code/scripts/run_inference.py \
    --data-bundle   "$DATA_BUNDLE" \
    --configs       "$CONFIGS" \
    --output-dir    "$OUTPUT_DIR" \
    --run-name      "$RUN_BASE" \
    --stan-type     "normal-noise-dot-product" \
    --chains        "$CHAINS" \
    --iter-warmup   "$ITER_WARMUP" \
    --iter-sampling "$ITER_SAMPLING" \
    --adapt-delta   "$ADAPT_DELTA" \
    --max-treedepth "$MAX_TREEDEPTH" \
    --seed          "$SEED" \
    --override-D    "$EMBEDDING_DIM" \
    --alpha-llm     "$ALPHA_LLM" \
    --overwrite-existing-data
 
echo "[2/2] Evaluating predictions (factored-dot-product)..."
python STAN/stan_code/scripts/evaluate_predictions.py \
    --data-bundle "$DATA_BUNDLE" \
    --mcmc-dir    "$OUTPUT_DIR/$RUN_BASE" \
    --run-name    "${RUN_BASE}_eval" \
    --csv-pattern "normal_noise_dot_product_model-*.csv" \
    --overwrite-existing-data \
    --verbose

RUN_BASE="${RUN_NAME}_Factor"
CONFIGS="${CONFIGS//Normal/Factor}"

 
echo "[1/2] Running MCMC (factored-dot-product)..."
python STAN/stan_code/scripts/run_inference.py \
    --data-bundle   "$DATA_BUNDLE" \
    --configs       "$CONFIGS" \
    --output-dir    "$OUTPUT_DIR" \
    --run-name      "$RUN_BASE" \
    --stan-type     "factored-dot-product" \
    --chains        "$CHAINS" \
    --iter-warmup   "$ITER_WARMUP" \
    --iter-sampling "$ITER_SAMPLING" \
    --adapt-delta   "$ADAPT_DELTA" \
    --max-treedepth "$MAX_TREEDEPTH" \
    --seed          "$SEED" \
    --override-D    "$EMBEDDING_DIM" \
    --alpha-llm     "$ALPHA_LLM" \
    --overwrite-existing-data
 
echo "[2/2] Evaluating predictions (factored-dot-product)..."
python STAN/stan_code/scripts/evaluate_predictions.py \
    --data-bundle "$DATA_BUNDLE" \
    --mcmc-dir    "$OUTPUT_DIR/$RUN_BASE" \
    --run-name    "${RUN_BASE}_eval" \
    --csv-pattern "normal_noise_dot_product_model-*.csv" \
    --overwrite-existing-data \
    --verbose
