#!/bin/bash
# OFAT: Vary use_concat_embedding Experiments
# Based on center configuration from run_single_kp10_random_as_key_fresh_lightning.sh

# Fixed base parameters for all experiments
BASE_I=5
BASE_J=12
BASE_C=5
BASE_K_TRAIN=30
BASE_K_TEST=30

# Baseline hyperparameter values (center of OFAT space)
BASE_D=16
BASE_SIGMA_ANNOTATOR=0.5
BASE_SIGMA_MEASUREMENT=0.1
BASE_KAPPA=10
BASE_PROTOCOL="tie_breaking"  # SMAR
BASE_PROTOCOL_CODE="smar"

# Unique prefix for TensorBoard filtering (same across all OFAT scripts)
OFAT_PREFIX="ofat_nr_v2"

# Marformer hyperparameters (fixed across all experiments - center values)
MARFORMER_EPOCHS=300
MARFORMER_LR=5e-4
MARFORMER_MASKING_RATE=0.15
MARFORMER_MASKED_LOSS_WEIGHT=15
MARFORMER_OBSERVED_LOSS_WEIGHT=1
MARFORMER_MASK_AUGMENTATIONS=5
MARFORMER_DEVICE="cuda"
MARFORMER_DEVICES=1  # 1 = single GPU (recommended for stability); empty = let Lightning auto-detect

# Transformer architecture (center/medium size)
MARFORMER_EMBEDDING_DIM=72
MARFORMER_ENCODER_LAYERS=6
MARFORMER_ATTENTION_HEADS=8
MARFORMER_NUM_FFN_LAYERS=2
MARFORMER_WEIGHT_DECAY=0.01
MARFORMER_DROPOUT=0.1

# General batching
MARFORMER_BATCH_SIZE=1

# Gradient clipping and learning rate schedule
MARFORMER_GRADIENT_CLIP_VAL=0.0
MARFORMER_USE_COSINE_SCHEDULE=true
MARFORMER_WARMUP_STEPS=200

# Devices flag (only pass if MARFORMER_DEVICES is set and non-empty)
DEVICES_FLAG=""
if [ -n "$MARFORMER_DEVICES" ]; then
    DEVICES_FLAG="--devices $MARFORMER_DEVICES"
fi

# Stan hyperparameters
STAN_4C_CHAINS=4
STAN_1C_CHAINS=1
STAN_1C_ITER_SAMPLING=300
STAN_1C_WARMUP=100
STAN_4C_ITER_SAMPLING=300
STAN_4C_WARMUP=100

# Function to run complete experiment pipeline
run_experiment() {
    local use_concat_embedding=$1

    # Format values for folder naming
    local d_str=$BASE_D
    local sa_str=$(echo $BASE_SIGMA_ANNOTATOR | sed 's/\.//g')
    local sm_str=$(echo $BASE_SIGMA_MEASUREMENT | sed 's/\.//g')
    local kp_str=$(echo $BASE_KAPPA | sed 's/\.//g')
    local uc_str=$use_concat_embedding

    # Construct run name with unique prefix
    local run_name="${OFAT_PREFIX}_varyConcat_D${d_str}_sa${sa_str}_sm${sm_str}_kp${kp_str}_uc${uc_str}_${BASE_PROTOCOL_CODE}"

    echo ""
    echo "=========================================="
    echo "EXPERIMENT: $run_name"
    echo "  use_concat_embedding=$use_concat_embedding"
    echo "=========================================="
    echo ""

    # Determine protocol-specific arguments
    local protocol_args=""
    if [ "$BASE_PROTOCOL" == "extended_rankings" ]; then
        protocol_args="--extended-pairwise-rate 0.2"
    elif [ "$BASE_PROTOCOL" == "mcar" ]; then
        protocol_args="--mcar-missing-rate 0.5"
    fi

    # Toggle for concat-based AtomCompositional embeddings
    local concat_flag=""
    if [ "$use_concat_embedding" == "1" ]; then
        concat_flag="--use-concat-embedding"
    fi

    # Cosine schedule flags
    local cosine_schedule_flags=""
    if [ "$MARFORMER_USE_COSINE_SCHEDULE" == "true" ]; then
        cosine_schedule_flags="--use-cosine-schedule --warmup-steps $MARFORMER_WARMUP_STEPS"
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
        --run-name $run_name \
        --overwrite-existing-data \
        $protocol_args </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Data generation failed for $run_name"
        return 1
    fi

    # Step 2: Run Marformer with Lightning
    echo "[Step 2/6] Running Marformer with PyTorch Lightning..."
    # Use interactive mode if DEBUG is set, otherwise redirect stdin
    if [ -n "$DEBUG" ]; then
        echo "Running in DEBUG mode (interactive - breakpoints will work)"
        python imputer/run_imputer.py \
            --data-dir OUTPUT/generated_data/${run_name} \
            --run-name ${run_name} \
            --overwrite-existing-data \
            --embedding-dim $MARFORMER_EMBEDDING_DIM \
            --encoder-layers $MARFORMER_ENCODER_LAYERS \
            --attention-heads $MARFORMER_ATTENTION_HEADS \
            --num_ffn_layers $MARFORMER_NUM_FFN_LAYERS \
            --weight-decay $MARFORMER_WEIGHT_DECAY \
            --dropout $MARFORMER_DROPOUT \
            --epochs $MARFORMER_EPOCHS \
            --lr $MARFORMER_LR \
            --masking-rate $MARFORMER_MASKING_RATE \
            --masked-loss-weight $MARFORMER_MASKED_LOSS_WEIGHT \
            --observed-loss-weight $MARFORMER_OBSERVED_LOSS_WEIGHT \
            --mask-augmentations $MARFORMER_MASK_AUGMENTATIONS \
            --no-final-norm \
            --normalize-parameter \
            --device $MARFORMER_DEVICE \
            $DEVICES_FLAG \
            --save-model-every 5 \
            --batch-size $MARFORMER_BATCH_SIZE \
            --gradient-clip-val $MARFORMER_GRADIENT_CLIP_VAL \
            $concat_flag \
            $cosine_schedule_flags
    else
        python imputer/run_imputer.py \
            --data-dir OUTPUT/generated_data/${run_name} \
            --run-name ${run_name} \
            --overwrite-existing-data \
            --embedding-dim $MARFORMER_EMBEDDING_DIM \
            --encoder-layers $MARFORMER_ENCODER_LAYERS \
            --attention-heads $MARFORMER_ATTENTION_HEADS \
            --num_ffn_layers $MARFORMER_NUM_FFN_LAYERS \
            --weight-decay $MARFORMER_WEIGHT_DECAY \
            --dropout $MARFORMER_DROPOUT \
            --epochs $MARFORMER_EPOCHS \
            --lr $MARFORMER_LR \
            --masking-rate $MARFORMER_MASKING_RATE \
            --masked-loss-weight $MARFORMER_MASKED_LOSS_WEIGHT \
            --observed-loss-weight $MARFORMER_OBSERVED_LOSS_WEIGHT \
            --mask-augmentations $MARFORMER_MASK_AUGMENTATIONS \
            --no-final-norm \
            --normalize-parameter \
            --device $MARFORMER_DEVICE \
            $DEVICES_FLAG \
            --save-model-every 5 \
            --batch-size $MARFORMER_BATCH_SIZE \
            --gradient-clip-val $MARFORMER_GRADIENT_CLIP_VAL \
            $concat_flag \
            $cosine_schedule_flags </dev/null
    fi

    if [ $? -ne 0 ]; then
        echo "ERROR: Marformer training failed for $run_name"
        return 1
    fi

    # Step 3: Run Stan inference (4 chains)
    echo "[Step 3/6] Running Stan inference (4 chains)..."
    python stan/scripts/run_inference.py \
        --data-bundle OUTPUT/generated_data/${run_name}/data_bundle.json \
        --chains $STAN_4C_CHAINS \
        --iter-sampling $STAN_4C_ITER_SAMPLING \
        --iter-warmup $STAN_4C_WARMUP \
        --run-name ${run_name}_stan4c \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Stan inference (4 chains) failed for $run_name"
        return 1
    fi

    # Step 4: Run Stan inference (1 chain, long)
    echo "[Step 4/6] Running Stan inference (1 chain, long)..."
    python stan/scripts/run_inference.py \
        --data-bundle OUTPUT/generated_data/${run_name}/data_bundle.json \
        --chains $STAN_1C_CHAINS \
        --iter-sampling $STAN_1C_ITER_SAMPLING \
        --iter-warmup $STAN_1C_WARMUP \
        --run-name ${run_name}_stan1c \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Stan inference (1 chain) failed for $run_name"
        return 1
    fi

    # Step 5: Evaluate Stan predictions (4-chain version)
    echo "[Step 5/6] Evaluating Stan predictions (4 chains)..."
    python stan/scripts/evaluate_predictions.py \
        --data-bundle OUTPUT/generated_data/${run_name}/data_bundle.json \
        --mcmc-dir OUTPUT/domain_model/runs/${run_name}_stan4c \
        --run-name ${run_name}_stan4c_eval \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Stan evaluation (4 chains) failed for $run_name"
        return 1
    fi

    # Step 5b: Evaluate Stan predictions (1-chain version)
    echo "[Step 5b/6] Evaluating Stan predictions (1 chain)..."
    python stan/scripts/evaluate_predictions.py \
        --data-bundle OUTPUT/generated_data/${run_name}/data_bundle.json \
        --mcmc-dir OUTPUT/domain_model/runs/${run_name}_stan1c \
        --run-name ${run_name}_stan1c_eval \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Stan evaluation (1 chain) failed for $run_name"
        return 1
    fi

    # Step 6: Generate visualization plots
    echo "[Step 6/6] Generating visualization plots..."
    python utils/visualize.py \
        --run-dir OUTPUT/IMPUTER/${run_name} \
        --stan-metrics OUTPUT/domain_model/eval/${run_name}_stan4c_eval/predictive_metrics.json </dev/null

    if [ $? -ne 0 ]; then
        echo "WARNING: Visualization failed for $run_name (continuing anyway)"
    else
        echo "  - Plots saved to OUTPUT/IMPUTER/${run_name}/plots/"
    fi

    echo ""
    echo "✓ COMPLETED: $run_name"
    echo ""
}

# Main execution
echo "=============================================="
echo "OFAT: VARY use_concat_embedding EXPERIMENTS"
echo "=============================================="
echo ""
echo "Base parameters:"
echo "  I=$BASE_I, J=$BASE_J, C=$BASE_C"
echo "  K_train=$BASE_K_TRAIN, K_test=$BASE_K_TEST"
echo ""
echo "Center (baseline) hyperparameters:"
echo "  D=$BASE_D"
echo "  σ_annotator=$BASE_SIGMA_ANNOTATOR"
echo "  σ_measurement=$BASE_SIGMA_MEASUREMENT"
echo "  κ=$BASE_KAPPA"
echo "  Protocol=$BASE_PROTOCOL (SMAR)"
echo "  Masking rate=$MARFORMER_MASKING_RATE"
echo "  Transformer: embedding_dim=$MARFORMER_EMBEDDING_DIM, layers=$MARFORMER_ENCODER_LAYERS, heads=$MARFORMER_ATTENTION_HEADS"
echo ""
echo "Total experiments: 2"
echo "  Vary use_concat_embedding: 2 (0, 1)"
echo ""
echo "Each experiment runs:"
echo "  - 1 Marformer training (Lightning)"
echo "  - 1 Stan (4 chains)"
echo "  - 1 Stan (1 chain)"
echo "  - 1 Visualization"
echo "=============================================="
echo ""

# Track experiment count
TOTAL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

# ===== OFAT: Vary use_concat_embedding =====
echo "======================================"
echo "OFAT SET: Varying use_concat_embedding"
echo "======================================"
for UC in 0 1; do
    run_experiment $UC
    if [ $? -ne 0 ]; then
        ((FAILED_EXPERIMENTS++))
    fi
    ((TOTAL_EXPERIMENTS++))
done

# ===== Summary =====
echo ""
echo "=============================================="
echo "OFAT VARY use_concat_embedding EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo "Failed experiments: $FAILED_EXPERIMENTS"
echo "Successful experiments: $((TOTAL_EXPERIMENTS - FAILED_EXPERIMENTS))"
echo ""
echo "Results saved in:"
echo "  - Data: OUTPUT/generated_data/${OFAT_PREFIX}_varyConcat_*"
echo "  - Marformer: OUTPUT/IMPUTER/${OFAT_PREFIX}_varyConcat_*"
echo "  - Stan (4c): OUTPUT/domain_model/runs/${OFAT_PREFIX}_varyConcat_*_stan4c"
echo "  - Stan (4c) Eval: OUTPUT/domain_model/eval/${OFAT_PREFIX}_varyConcat_*_stan4c_eval"
echo "  - Stan (1c): OUTPUT/domain_model/runs/${OFAT_PREFIX}_varyConcat_*_stan1c"
echo "  - Stan (1c) Eval: OUTPUT/domain_model/eval/${OFAT_PREFIX}_varyConcat_*_stan1c_eval"
echo "=============================================="
