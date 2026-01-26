#!/bin/bash
# Smoke test script: runs both imputer and stan on a minimal dataset
# This is a quick sanity check that both systems work end-to-end

set -e  # Exit on error

# Small dataset parameters for fast testing
K_TRAIN=5
K_TEST=3
I=3
J=9
D=16
C=5

# Imputer training parameters (minimal)
EPOCHS=5
LR=1e-3
MASKING_RATE=0.5

# Stan parameters (minimal)
STAN_CHAINS=1
STAN_WARMUP=20
STAN_SAMPLING=50

# Output directories
RUN_NAME="smoke_test_$(date +%Y%m%d_%H%M%S)"
DATA_ROOT="OUTPUT/generated_data"
DATA_DIR="${DATA_ROOT}/${RUN_NAME}"           # This is where data_bundle.json will live
IMPUTER_DIR="OUTPUT/IMPUTER/${RUN_NAME}_imputer"
STAN_DIR="OUTPUT/domain_model/${RUN_NAME}_stan"

echo "=========================================="
echo "Smoke Test: Imputer + Stan"
echo "=========================================="
echo "Run name: ${RUN_NAME}"
echo ""

# Step 1: Generate data (Stan)
echo "[Step 1/3] Generating data..."
python stan/scripts/generate_data.py \
    --output-dir "${DATA_ROOT}" \
    --run-name "${RUN_NAME}" \
    --K-train ${K_TRAIN} \
    --K-test ${K_TEST} \
    --I ${I} \
    --J ${J} \
    --D ${D} \
    --C ${C} \
    --seed 42 \
    --overwrite-existing-data

if [ $? -ne 0 ]; then
    echo "ERROR: Data generation failed"
    exit 1
fi
echo "✓ Data generated"

# Step 2: Run Imputer
echo ""
echo "[Step 2/3] Running Imputer (Lightning)..."
python imputer/run_imputer.py \
    --data-dir "${DATA_DIR}" \
    --run-name "${RUN_NAME}_imputer" \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --masking-rate ${MASKING_RATE} \
    --embedding-dim ${D} \
    --encoder-layers 2 \
    --attention-heads 4 \
    --dropout 0.1 \
    --max-rank-size 2 \
    --device cpu \
    --overwrite-existing-data

if [ $? -ne 0 ]; then
    echo "ERROR: Imputer training failed"
    exit 1
fi
echo "✓ Imputer training completed"

# Step 3: Run Stan inference
echo ""
echo "[Step 3/3] Running Stan inference..."
python stan/scripts/run_inference.py \
    --data-bundle "${DATA_DIR}/data_bundle.json" \
    --chains ${STAN_CHAINS} \
    --iter-warmup ${STAN_WARMUP} \
    --iter-sampling ${STAN_SAMPLING} \
    --run-name "${RUN_NAME}_stan" \
    --overwrite-existing-data

if [ $? -ne 0 ]; then
    echo "ERROR: Stan inference failed"
    exit 1
fi
echo "✓ Stan inference completed"

# Step 4: Evaluate Stan predictions
echo ""
echo "[Step 4/4] Evaluating Stan predictions..."
python stan/scripts/evaluate_predictions.py \
    --mcmc-dir "OUTPUT/domain_model/runs/${RUN_NAME}_stan" \
    --data-bundle "${DATA_DIR}/data_bundle.json" \
    --output-dir "OUTPUT/domain_model/eval" \
    --run-name "${RUN_NAME}_stan_eval" \
    --overwrite-existing-data

if [ $? -ne 0 ]; then
    echo "ERROR: Stan evaluation failed"
    exit 1
fi
echo "✓ Stan evaluation completed"

# Step 5: Generate visualization plots (Marformer + Stan baseline)
echo ""
echo "[Step 5/5] Generating visualization plots..."
python utils/visualize.py \
    --run-dir "OUTPUT/IMPUTER/${RUN_NAME}_imputer" \
    --stan-metrics "OUTPUT/domain_model/eval/${RUN_NAME}_stan_eval/predictive_metrics.json" </dev/null

if [ $? -ne 0 ]; then
    echo "WARNING: Visualization failed (continuing anyway)"
else
    echo "✓ Plots saved to OUTPUT/IMPUTER/${RUN_NAME}_imputer/plots/"
fi

echo ""
echo "=========================================="
echo "Smoke test completed successfully!"
echo "=========================================="
echo "Data: ${DATA_DIR}"
echo "Imputer: OUTPUT/IMPUTER/${RUN_NAME}_imputer"
echo "Stan: OUTPUT/domain_model/${RUN_NAME}_stan"
echo "Stan eval: OUTPUT/domain_model/eval/${RUN_NAME}_stan_eval"
echo "Plots: OUTPUT/IMPUTER/${RUN_NAME}_imputer/plots/"
echo ""
