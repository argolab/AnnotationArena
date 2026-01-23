#!/bin/bash
# Single Experiment Runner
# Configure parameters below and run a single experiment

# ============================================
# CONFIGURATION KNOBS - Adjust these values
# ============================================

# Data dimensions
BASE_I=3
BASE_J=12
BASE_C=5
BASE_K_TRAIN=10
BASE_K_TEST=10

# Data generation hyperparameters
BASE_D=4
BASE_SIGMA_ANNOTATOR=0.5
BASE_SIGMA_MEASUREMENT=0.1
BASE_KAPPA=1
BASE_PROTOCOL="tie_breaking"  # Options: "tie_breaking", "mcar", "extended_rankings"
BASE_PROTOCOL_CODE="smar"     # For naming: "smar", "mcar05", "smarext"
BASE_USE_CONCAT=0             # 0 = disabled, 1 = enabled for AtomCompositional embeddings

# Protocol-specific parameters (only used if protocol requires them)
MCAR_MISSING_RATE=0.5         # For MCAR protocol
EXTENDED_PAIRWISE_RATE=0.2    # For extended_rankings protocol

# Marformer hyperparameters
MARFORMER_EPOCHS=70
MARFORMER_LR=1e-4
MARFORMER_MASKING_RATE=0.15
MARFORMER_MASKED_LOSS_WEIGHT=15
MARFORMER_OBSERVED_LOSS_WEIGHT=1
MARFORMER_MASK_AUGMENTATIONS=5
MARFORMER_EARLY_STOPPING_PATIENCE=10
MARFORMER_DEVICE="cuda"
MARFORMER_EARLY_STOPPING=false  # true = enable, false = disable (disabled for experiment: fix early stopping metric)
MARFORMER_EARLY_STOPPING_METRIC="loss"  # "loss" or "accuracy"
MARFORMER_FULL_RANDOM=1         # 0 = disable randomness, 1 = enable randomness
MARFORMER_DROPOUT=0.1

# Stan hyperparameters
STAN_4C_CHAINS=4
STAN_1C_CHAINS=1
STAN_1C_ITER_SAMPLING=300
STAN_1C_WARMUP=100
STAN_4C_ITER_SAMPLING=300
STAN_4C_WARMUP=100

# Run name (auto-generated if empty)
RUN_NAME="marformer_kappa01_full_random"  # Leave empty for auto-generated name, or specify custom name

# ============================================
# EXPERIMENT PIPELINE
# ============================================

# Generate run name if not specified
if [ -z "$RUN_NAME" ]; then
    # Format values for folder naming (remove decimals for cleaner names)
    d_str=$BASE_D
    sa_str=$(echo $BASE_SIGMA_ANNOTATOR | sed 's/\.//g')
    sm_str=$(echo $BASE_SIGMA_MEASUREMENT | sed 's/\.//g')
    kp_str=$(echo $BASE_KAPPA | sed 's/\.//g')
    uc_str=$BASE_USE_CONCAT
    
    RUN_NAME="single_D${d_str}_sa${sa_str}_sm${sm_str}_kp${kp_str}_uc${uc_str}_${BASE_PROTOCOL_CODE}"
fi

echo "=============================================="
echo "SINGLE EXPERIMENT RUNNER"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Data dimensions: I=$BASE_I, J=$BASE_J, C=$BASE_C"
echo "  Items: K_train=$BASE_K_TRAIN, K_test=$BASE_K_TEST"
echo "  Hyperparameters: D=$BASE_D, σ_ann=$BASE_SIGMA_ANNOTATOR, σ_meas=$BASE_SIGMA_MEASUREMENT, κ=$BASE_KAPPA"
echo "  Protocol: $BASE_PROTOCOL ($BASE_PROTOCOL_CODE)"
echo "  use_concat_embedding: $BASE_USE_CONCAT"
echo ""
echo "Marformer settings:"
echo "  Epochs: $MARFORMER_EPOCHS, LR: $MARFORMER_LR"
echo "  Masking rate: $MARFORMER_MASKING_RATE"
echo "  Loss weights: masked=$MARFORMER_MASKED_LOSS_WEIGHT, observed=$MARFORMER_OBSERVED_LOSS_WEIGHT"
echo "  Early stopping: $MARFORMER_EARLY_STOPPING (metric=$MARFORMER_EARLY_STOPPING_METRIC, patience=$MARFORMER_EARLY_STOPPING_PATIENCE)"
echo "  Full random: $MARFORMER_FULL_RANDOM"
echo ""
echo "Stan settings:"
echo "  4-chain: $STAN_4C_CHAINS chains, $STAN_4C_ITER_SAMPLING samples, $STAN_4C_WARMUP warmup"
echo "  1-chain: $STAN_1C_CHAINS chain, $STAN_1C_ITER_SAMPLING samples, $STAN_1C_WARMUP warmup"
    echo ""
echo "Run name: $RUN_NAME"
echo "=============================================="
    echo ""

    # Determine protocol-specific arguments
PROTOCOL_ARGS=""
if [ "$BASE_PROTOCOL" == "extended_rankings" ]; then
    PROTOCOL_ARGS="--extended-pairwise-rate $EXTENDED_PAIRWISE_RATE"
elif [ "$BASE_PROTOCOL" == "mcar" ]; then
    PROTOCOL_ARGS="--mcar-missing-rate $MCAR_MISSING_RATE"
    fi

    # Toggle for concat-based AtomCompositional embeddings
CONCAT_FLAG=""
if [ "$BASE_USE_CONCAT" == "1" ]; then
    CONCAT_FLAG="--use-concat-embedding"
fi

# Early stopping flags
EARLY_STOPPING_FLAGS=""
if [ "$MARFORMER_EARLY_STOPPING" == "true" ]; then
    EARLY_STOPPING_FLAGS="--early-stopping --early-stopping-metric $MARFORMER_EARLY_STOPPING_METRIC --early-stopping-patience $MARFORMER_EARLY_STOPPING_PATIENCE"
fi

# Full random flag
FULL_RANDOM_FLAG=""
if [ "$MARFORMER_FULL_RANDOM" == "1" ]; then
    FULL_RANDOM_FLAG="--full_random"
    fi

    # Step 1: Generate data
    echo "[Step 1/6] Generating data..."
    python stan/scripts/generate_data.py \
        --K-train $BASE_K_TRAIN \
        --K-test $BASE_K_TEST \
        --I $BASE_I \
        --J $BASE_J \
    --D $BASE_D \
        --C $BASE_C \
    --observation-protocol $BASE_PROTOCOL \
    --sigma-annotator $BASE_SIGMA_ANNOTATOR \
    --sigma-measurement $BASE_SIGMA_MEASUREMENT \
    --kappa $BASE_KAPPA \
    --run-name $RUN_NAME \
        --overwrite-existing-data \
    $PROTOCOL_ARGS </dev/null

    if [ $? -ne 0 ]; then
    echo "ERROR: Data generation failed"
    exit 1
    fi

    # Step 2: Run Marformer
    echo "[Step 2/6] Running Marformer..."
MARFORMER_ARGS="--data-dir OUTPUT/generated_data/${RUN_NAME} \
    --run-name ${RUN_NAME}_marformer \
        --overwrite-existing-data \
    --embedding-dim $((BASE_D * 3)) \
        --epochs $MARFORMER_EPOCHS \
        --lr $MARFORMER_LR \
        --masking-rate $MARFORMER_MASKING_RATE \
        --masked-loss-weight $MARFORMER_MASKED_LOSS_WEIGHT \
        --observed-loss-weight $MARFORMER_OBSERVED_LOSS_WEIGHT \
        --mask-augmentations $MARFORMER_MASK_AUGMENTATIONS \
        --transductive_learning \
        --no-final-norm \
        --normalize-parameter \
        --device $MARFORMER_DEVICE \
        --save-model-every 5 \
        --dropout $MARFORMER_DROPOUT \
    $EARLY_STOPPING_FLAGS \
    $FULL_RANDOM_FLAG \
    $CONCAT_FLAG"

python imputer/run_imputer.py $MARFORMER_ARGS </dev/null

    if [ $? -ne 0 ]; then
    echo "ERROR: Marformer training failed"
    exit 1
    fi

    # Step 3: Run Stan inference (4 chains)
    echo "[Step 3/6] Running Stan inference (4 chains)..."
    python stan/scripts/run_inference.py \
    --data-bundle OUTPUT/generated_data/${RUN_NAME}/data_bundle.json \
        --chains $STAN_4C_CHAINS \
        --iter-sampling $STAN_4C_ITER_SAMPLING \
        --iter-warmup $STAN_4C_WARMUP \
    --run-name ${RUN_NAME}_stan4c \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
    echo "ERROR: Stan inference (4 chains) failed"
    exit 1
    fi

    # Step 4: Run Stan inference (1 chain, long)
    echo "[Step 4/6] Running Stan inference (1 chain, long)..."
    python stan/scripts/run_inference.py \
    --data-bundle OUTPUT/generated_data/${RUN_NAME}/data_bundle.json \
        --chains $STAN_1C_CHAINS \
        --iter-sampling $STAN_1C_ITER_SAMPLING \
        --iter-warmup $STAN_1C_WARMUP \
    --run-name ${RUN_NAME}_stan1c \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
    echo "ERROR: Stan inference (1 chain) failed"
    exit 1
    fi

    # Step 5: Evaluate Stan predictions (4-chain version)
    echo "[Step 5/6] Evaluating Stan predictions (4 chains)..."
    python stan/scripts/evaluate_predictions.py \
    --data-bundle OUTPUT/generated_data/${RUN_NAME}/data_bundle.json \
    --mcmc-dir OUTPUT/domain_model/runs/${RUN_NAME}_stan4c \
    --run-name ${RUN_NAME}_stan4c_eval \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
    echo "ERROR: Stan evaluation (4 chains) failed"
    exit 1
    fi

    # Step 5b: Evaluate Stan predictions (1-chain version)
    echo "[Step 5b/6] Evaluating Stan predictions (1 chain)..."
    python stan/scripts/evaluate_predictions.py \
    --data-bundle OUTPUT/generated_data/${RUN_NAME}/data_bundle.json \
    --mcmc-dir OUTPUT/domain_model/runs/${RUN_NAME}_stan1c \
    --run-name ${RUN_NAME}_stan1c_eval \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
    echo "ERROR: Stan evaluation (1 chain) failed"
    exit 1
    fi

    # Step 6: Generate visualization plots
    echo "[Step 6/6] Generating visualization plots..."
    python utils/visualize.py \
    --run-dir OUTPUT/IMPUTER/${RUN_NAME}_marformer \
    --stan-metrics OUTPUT/domain_model/eval/${RUN_NAME}_stan4c_eval/predictive_metrics.json </dev/null

if [ $? -ne 0 ]; then
    echo "WARNING: Visualization failed (continuing anyway)"
else
    echo "  - Plots saved to OUTPUT/IMPUTER/${RUN_NAME}_marformer/plots/"
fi

echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE"
echo "=============================================="
echo "Run name: $RUN_NAME"
echo ""
echo "Results saved in:"
echo "  - Data: OUTPUT/generated_data/${RUN_NAME}"
echo "  - Marformer: OUTPUT/IMPUTER/${RUN_NAME}_marformer"
echo "  - Stan (4c): OUTPUT/domain_model/runs/${RUN_NAME}_stan4c"
echo "  - Stan (4c) Eval: OUTPUT/domain_model/eval/${RUN_NAME}_stan4c_eval"
echo "  - Stan (1c): OUTPUT/domain_model/runs/${RUN_NAME}_stan1c"
echo "  - Stan (1c) Eval: OUTPUT/domain_model/eval/${RUN_NAME}_stan1c_eval"
echo "  - Plots: OUTPUT/IMPUTER/${RUN_NAME}_marformer/plots/"
echo "=============================================="
echo ""
echo "=============================================="
echo "COMMANDS SUMMARY (for easy copy/paste)"
echo "=============================================="
echo ""
echo "# [Step 1] Generate data:"
echo "python stan/scripts/generate_data.py \\"
echo "    --K-train $BASE_K_TRAIN \\"
echo "    --K-test $BASE_K_TEST \\"
echo "    --I $BASE_I \\"
echo "    --J $BASE_J \\"
echo "    --D $BASE_D \\"
echo "    --C $BASE_C \\"
echo "    --observation-protocol $BASE_PROTOCOL \\"
echo "    --sigma-annotator $BASE_SIGMA_ANNOTATOR \\"
echo "    --sigma-measurement $BASE_SIGMA_MEASUREMENT \\"
echo "    --kappa $BASE_KAPPA \\"
echo "    --run-name $RUN_NAME \\"
echo "    --overwrite-existing-data \\"
echo "    $PROTOCOL_ARGS"
echo ""
echo "# [Step 2] Train Marformer:"
echo "python imputer/run_imputer.py \\"
echo "    --data-dir OUTPUT/generated_data/${RUN_NAME} \\"
echo "    --run-name ${RUN_NAME}_marformer \\"
echo "    --overwrite-existing-data \\"
echo "    --embedding-dim $((BASE_D * 3)) \\"
echo "    --epochs $MARFORMER_EPOCHS \\"
echo "    --lr $MARFORMER_LR \\"
echo "    --masking-rate $MARFORMER_MASKING_RATE \\"
echo "    --masked-loss-weight $MARFORMER_MASKED_LOSS_WEIGHT \\"
echo "    --observed-loss-weight $MARFORMER_OBSERVED_LOSS_WEIGHT \\"
echo "    --mask-augmentations $MARFORMER_MASK_AUGMENTATIONS \\"
echo "    --transductive_learning \\"
echo "    --no-final-norm \\"
echo "    --normalize-parameter \\"
echo "    --device $MARFORMER_DEVICE \\"
echo "    --save-model-every 5 \\"
echo "    --dropout $MARFORMER_DROPOUT \\"
echo "    $EARLY_STOPPING_FLAGS \\"
echo "    $FULL_RANDOM_FLAG \\"
echo "    $CONCAT_FLAG"
echo ""
echo "# [Step 3] Stan inference (4 chains):"
echo "python stan/scripts/run_inference.py \\"
echo "    --data-bundle OUTPUT/generated_data/${RUN_NAME}/data_bundle.json \\"
echo "    --chains $STAN_4C_CHAINS \\"
echo "    --iter-sampling $STAN_4C_ITER_SAMPLING \\"
echo "    --iter-warmup $STAN_4C_WARMUP \\"
echo "    --run-name ${RUN_NAME}_stan4c \\"
echo "    --overwrite-existing-data"
echo ""
echo "# [Step 4] Stan inference (1 chain):"
echo "python stan/scripts/run_inference.py \\"
echo "    --data-bundle OUTPUT/generated_data/${RUN_NAME}/data_bundle.json \\"
echo "    --chains $STAN_1C_CHAINS \\"
echo "    --iter-sampling $STAN_1C_ITER_SAMPLING \\"
echo "    --iter-warmup $STAN_1C_WARMUP \\"
echo "    --run-name ${RUN_NAME}_stan1c \\"
echo "    --overwrite-existing-data"
echo ""
echo "# [Step 5] Evaluate Stan (4 chains):"
echo "python stan/scripts/evaluate_predictions.py \\"
echo "    --data-bundle OUTPUT/generated_data/${RUN_NAME}/data_bundle.json \\"
echo "    --mcmc-dir OUTPUT/domain_model/runs/${RUN_NAME}_stan4c \\"
echo "    --run-name ${RUN_NAME}_stan4c_eval \\"
echo "    --overwrite-existing-data"
echo ""
echo "# [Step 5b] Evaluate Stan (1 chain):"
echo "python stan/scripts/evaluate_predictions.py \\"
echo "    --data-bundle OUTPUT/generated_data/${RUN_NAME}/data_bundle.json \\"
echo "    --mcmc-dir OUTPUT/domain_model/runs/${RUN_NAME}_stan1c \\"
echo "    --run-name ${RUN_NAME}_stan1c_eval \\"
echo "    --overwrite-existing-data"
echo ""
echo "# [Step 6] Generate visualizations:"
echo "python utils/visualize.py \\"
echo "    --run-dir OUTPUT/IMPUTER/${RUN_NAME}_marformer \\"
echo "    --stan-metrics OUTPUT/domain_model/eval/${RUN_NAME}_stan4c_eval/predictive_metrics.json"
echo ""
echo "=============================================="