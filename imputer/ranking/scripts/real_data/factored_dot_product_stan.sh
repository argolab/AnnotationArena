

run_name="factored_dot_product_llm_rubric"




# ── Dataset (always llmrubric_dist) ───────────────────────────────────────────
DATA_DIR="OUTPUT/generated_data/${run_name}"

# ── Dirichlet hyperparameter ──────────────────────────────────────────────────
# Larger  → LLM observations are treated as more reliable (tighter Dirichlet)
# Smaller → LLM observations are treated as more noisy
ALPHA_LLM="${ALPHA_LLM:-5.0}"

# Include alpha in folder name so sweeps don't collide
RUN_BASE="${run_name}"

# ── MCMC hyperparameters ──────────────────────────────────────────────────────
CHAINS="${CHAINS:-1}"
ITER_WARMUP="${ITER_WARMUP:-1}"
ITER_SAMPLING="${ITER_SAMPLING:-1}"
ADAPT_DELTA="${ADAPT_DELTA:-0.85}"
MAX_TREEDEPTH="${MAX_TREEDEPTH:-12}"
SEED="${SEED:-42}"

# ── Stan model hyperparameters ────────────────────────────────────────────────
EMBEDDING_DIM="${EMBEDDING_DIM:-8}"

echo ""
echo "=================================================="
echo "Stan Dirichlet Real-Data Inference (LLMRubric)"
echo "  data_dir:       $DATA_DIR"
echo "  run_base:       $RUN_BASE"
echo "  alpha_llm:      $ALPHA_LLM"
echo "  chains:         $CHAINS  warmup=$ITER_WARMUP  sampling=$ITER_SAMPLING"
echo "  adapt_delta:    $ADAPT_DELTA  max_treedepth=$MAX_TREEDEPTH"
echo "  D=$EMBEDDING_DIM"
echo "=================================================="
echo ""

# ── Step 1: MCMC inference with Dirichlet model ───────────────────────────────
echo "[Step 1/2] Running Stan MCMC (Dirichlet model)..."
python stan/scripts/run_inference.py \
    --data-bundle    "$DATA_DIR/data_bundle.json" \
    --run-name       "$RUN_BASE" \
    --chains         "$CHAINS" \
    --iter-warmup    "$ITER_WARMUP" \
    --iter-sampling  "$ITER_SAMPLING" \
    --adapt-delta    "$ADAPT_DELTA" \
    --max-treedepth  "$MAX_TREEDEPTH" \
    --seed           "$SEED" \
    --override-D     "$EMBEDDING_DIM" \
    --alpha-llm      "$ALPHA_LLM" \
    --stan-type      "factored-dot-product" \
    --overwrite-existing-data

if [ $? -ne 0 ]; then
    echo "ERROR: Stan inference failed"
    exit 1
fi

# ── Step 2: Evaluate predictions ──────────────────────────────────────────────
echo ""
echo "[Step 2/2] Evaluating Stan predictions..."
python stan/scripts/evaluate_predictions.py \
    --data-bundle  "$DATA_DIR/data_bundle.json" \
    --mcmc-dir     "OUTPUT/domain_model/runs/$RUN_BASE" \
    --run-name     "${RUN_BASE}_eval" \
    --csv-pattern  "normal_noise_dot_product_model-*.csv" \
    --overwrite-existing-data \
    --verbose

if [ $? -ne 0 ]; then
    echo "ERROR: Stan evaluation failed"
    exit 1
fi

# ── Step 1: MCMC inference with Dirichlet model ───────────────────────────────
echo "[Step 1/2] Running Stan MCMC Train Only"
python stan/scripts/run_inference.py \
    --data-bundle    "$DATA_DIR/data_bundle.json" \
    --run-name       "${RUN_BASE}_train_only" \
    --chains         "$CHAINS" \
    --iter-warmup    "$ITER_WARMUP" \
    --iter-sampling  "$ITER_SAMPLING" \
    --adapt-delta    "$ADAPT_DELTA" \
    --max-treedepth  "$MAX_TREEDEPTH" \
    --seed           "$SEED" \
    --override-D     "$EMBEDDING_DIM" \
    --alpha-llm      "$ALPHA_LLM" \
    --stan-type      "factored-dot-product" \
    --overwrite-existing-data \
    --use-train-only

if [ $? -ne 0 ]; then
    echo "ERROR: Stan inference failed"
    exit 1
fi

# ── Step 2: Evaluate predictions ──────────────────────────────────────────────
echo ""
echo "[Step 2/2] Evaluating Stan predictions..."
python stan/scripts/evaluate_predictions.py \
    --data-bundle  "$DATA_DIR/data_bundle.json" \
    --mcmc-dir     "OUTPUT/domain_model/runs/${RUN_BASE}_train_only" \
    --run-name     "${RUN_BASE}_train_only_eval" \
    --csv-pattern  "normal_noise_dot_product_model_freeze-*.csv" \
    --overwrite-existing-data \
    --use-train-only \
    --verbose

if [ $? -ne 0 ]; then
    echo "ERROR: Stan evaluation failed"
    exit 1
fi