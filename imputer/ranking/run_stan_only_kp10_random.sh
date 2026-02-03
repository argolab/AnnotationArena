#!/bin/bash
# Stan-Only Experiment Runner
# Starts from Stan inference (skips data generation and Marformer training)
# Configure parameters below and run Stan inference + evaluation + visualization

# ============================================
# CONFIGURATION KNOBS - Adjust these values
# ============================================

# Run name - MUST match the run name used for data generation
RUN_NAME="marformer_kappa10_full_random_D4_simple_emb"  # Specify the existing run name

# Stan hyperparameters
STAN_4C_CHAINS=4
STAN_1C_CHAINS=1
STAN_1C_ITER_SAMPLING=800
STAN_1C_WARMUP=200
STAN_4C_ITER_SAMPLING=800
STAN_4C_WARMUP=200

# ============================================
# EXPERIMENT PIPELINE
# ============================================

if [ -z "$RUN_NAME" ]; then
    echo "ERROR: RUN_NAME must be specified"
    echo "Please set RUN_NAME to match the existing data generation run name"
    exit 1
fi

# Verify data bundle exists
DATA_BUNDLE="OUTPUT/generated_data/${RUN_NAME}/data_bundle.json"
if [ ! -f "$DATA_BUNDLE" ]; then
    echo "ERROR: Data bundle not found at: $DATA_BUNDLE"
    echo "Please ensure the data generation has been completed with run name: $RUN_NAME"
    exit 1
fi

echo "=============================================="
echo "STAN-ONLY EXPERIMENT RUNNER"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Run name: $RUN_NAME"
echo "  Data bundle: $DATA_BUNDLE"
echo ""
echo "Stan settings:"
echo "  4-chain: $STAN_4C_CHAINS chains, $STAN_4C_ITER_SAMPLING samples, $STAN_4C_WARMUP warmup"
echo "  1-chain: $STAN_1C_CHAINS chain, $STAN_1C_ITER_SAMPLING samples, $STAN_1C_WARMUP warmup"
echo "=============================================="
echo ""

# Step 1: Run Stan inference (4 chains)
echo "[Step 1/4] Running Stan inference (4 chains)..."
python stan/scripts/run_inference.py \
    --data-bundle $DATA_BUNDLE \
    --chains $STAN_4C_CHAINS \
    --iter-sampling $STAN_4C_ITER_SAMPLING \
    --iter-warmup $STAN_4C_WARMUP \
    --run-name ${RUN_NAME}_stan4c \
    --overwrite-existing-data </dev/null

if [ $? -ne 0 ]; then
    echo "ERROR: Stan inference (4 chains) failed"
    exit 1
fi

# Step 2: Run Stan inference (1 chain, long)
echo "[Step 2/4] Running Stan inference (1 chain, long)..."
python stan/scripts/run_inference.py \
    --data-bundle $DATA_BUNDLE \
    --chains $STAN_1C_CHAINS \
    --iter-sampling $STAN_1C_ITER_SAMPLING \
    --iter-warmup $STAN_1C_WARMUP \
    --run-name ${RUN_NAME}_stan1c \
    --overwrite-existing-data </dev/null

if [ $? -ne 0 ]; then
    echo "ERROR: Stan inference (1 chain) failed"
    exit 1
fi

# Step 3: Evaluate Stan predictions (4-chain version)
echo "[Step 3/4] Evaluating Stan predictions (4 chains)..."
python stan/scripts/evaluate_predictions.py \
    --data-bundle $DATA_BUNDLE \
    --mcmc-dir OUTPUT/domain_model/runs/${RUN_NAME}_stan4c \
    --run-name ${RUN_NAME}_stan4c_eval \
    --overwrite-existing-data </dev/null

if [ $? -ne 0 ]; then
    echo "ERROR: Stan evaluation (4 chains) failed"
    exit 1
fi

# Step 3b: Evaluate Stan predictions (1-chain version)
echo "[Step 3b/4] Evaluating Stan predictions (1 chain)..."
python stan/scripts/evaluate_predictions.py \
    --data-bundle $DATA_BUNDLE \
    --mcmc-dir OUTPUT/domain_model/runs/${RUN_NAME}_stan1c \
    --run-name ${RUN_NAME}_stan1c_eval \
    --overwrite-existing-data </dev/null

if [ $? -ne 0 ]; then
    echo "ERROR: Stan evaluation (1 chain) failed"
    exit 1
fi

# Step 4: Generate visualization plots
echo "[Step 4/4] Generating visualization plots..."
MARFORMER_DIR="OUTPUT/IMPUTER/${RUN_NAME}_marformer"
if [ ! -d "$MARFORMER_DIR" ]; then
    echo "WARNING: Marformer directory not found at: $MARFORMER_DIR"
    echo "  Skipping visualization (requires Marformer results)"
else
    python utils/visualize.py \
        --run-dir $MARFORMER_DIR \
        --stan-metrics OUTPUT/domain_model/eval/${RUN_NAME}_stan4c_eval/predictive_metrics.json </dev/null

    if [ $? -ne 0 ]; then
        echo "WARNING: Visualization failed (continuing anyway)"
    else
        echo "  - Plots saved to ${MARFORMER_DIR}/plots/"
    fi
fi

echo ""
echo "=============================================="
echo "STAN EXPERIMENT COMPLETE"
echo "=============================================="
echo "Run name: $RUN_NAME"
echo ""
echo "Results saved in:"
echo "  - Stan (4c): OUTPUT/domain_model/runs/${RUN_NAME}_stan4c"
echo "  - Stan (4c) Eval: OUTPUT/domain_model/eval/${RUN_NAME}_stan4c_eval"
echo "  - Stan (1c): OUTPUT/domain_model/runs/${RUN_NAME}_stan1c"
echo "  - Stan (1c) Eval: OUTPUT/domain_model/eval/${RUN_NAME}_stan1c_eval"
if [ -d "$MARFORMER_DIR" ]; then
    echo "  - Plots: ${MARFORMER_DIR}/plots/"
fi
echo "=============================================="

