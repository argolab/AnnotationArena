#!/bin/bash
# Easy-data axis invariance: K-only mode (hold_I=1, hold_J=1, hold_K=0)
# Ratings depend only on items K, not on criteria or annotators.

set -e

# Fixed base parameters (mirrors OFAT center setup where possible)
# When hold_I=1 and hold_J=1 we use I=1, J=1 and MCAR (no 4-annotator constraint).
BASE_I=1
BASE_J=1
BASE_C=5
BASE_K_TRAIN=30
BASE_K_TEST=30

# Baseline hyperparameter values
BASE_D=8
BASE_SIGMA_ANNOTATOR=0.5
BASE_SIGMA_MEASUREMENT=0.1
BASE_KAPPA=10
BASE_PROTOCOL="mcar"  # MCAR when I=1,J=1 (hold I/J fix)
BASE_PROTOCOL_CODE="mcar"
BASE_USE_CONCAT=0  # Center value

# Unique prefix for TensorBoard / output filtering
EASY_PREFIX="easy_axis"
MODE_NAME="K_only"
HOLD_I=1
HOLD_J=1
HOLD_K=0

# Marformer hyperparameters (kept modest for sanity checks)
MARFORMER_EPOCHS=300
MARFORMER_LR=2e-4
MARFORMER_MASKING_RATE=0.15
MARFORMER_MASKED_LOSS_WEIGHT=15
MARFORMER_OBSERVED_LOSS_WEIGHT=1
MARFORMER_MASK_AUGMENTATIONS=5
MARFORMER_DEVICE="cuda"
MARFORMER_DEVICES=1

# Transformer architecture (medium size)
MARFORMER_EMBEDDING_DIM=72
MARFORMER_ENCODER_LAYERS=4
MARFORMER_ATTENTION_HEADS=4
MARFORMER_NUM_FFN_LAYERS=2
MARFORMER_WEIGHT_DECAY=0.01
MARFORMER_DROPOUT=0.1

# Batching and optimization
MARFORMER_BATCH_SIZE=1
MARFORMER_GRADIENT_CLIP_VAL=0.0
MARFORMER_USE_COSINE_SCHEDULE=true
MARFORMER_WARMUP_STEPS=100

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

# Format values for folder naming
d_str=$BASE_D
sa_str=$(echo $BASE_SIGMA_ANNOTATOR | sed 's/\.//g')
sm_str=$(echo $BASE_SIGMA_MEASUREMENT | sed 's/\.//g')
kp_str=$(echo $BASE_KAPPA | sed 's/\.//g')
uc_str=$BASE_USE_CONCAT

# Axis flags string for run name
hI_str=$HOLD_I
hJ_str=$HOLD_J
hK_str=$HOLD_K

# Construct run name
run_name="${EASY_PREFIX}_${MODE_NAME}_D${d_str}_sa${sa_str}_sm${sm_str}_kp${kp_str}_uc${uc_str}_hI${hI_str}_hJ${hJ_str}_hK${hK_str}_${BASE_PROTOCOL_CODE}"

echo ""
echo "=========================================="
echo "EASY AXIS EXPERIMENT: $run_name"
echo "  Mode: ${MODE_NAME}"
echo "  hold_I_constant=${HOLD_I}, hold_J_constant=${HOLD_J}, hold_K_constant=${HOLD_K}"
echo "=========================================="
echo ""

# Determine protocol-specific arguments
protocol_args=""
if [ "$BASE_PROTOCOL" == "extended_rankings" ]; then
    protocol_args="--extended-pairwise-rate 0.2"
elif [ "$BASE_PROTOCOL" == "mcar" ]; then
    protocol_args="--mcar-missing-rate 0.5"
fi

# Toggle for concat-based AtomCompositional embeddings
concat_flag=""
if [ "$BASE_USE_CONCAT" == "1" ]; then
    concat_flag="--use-concat-embedding"
fi

# Cosine schedule flags
cosine_schedule_flags=""
if [ "$MARFORMER_USE_COSINE_SCHEDULE" == "true" ]; then
    cosine_schedule_flags="--use-cosine-schedule --warmup-steps $MARFORMER_WARMUP_STEPS"
fi

# Step 1: Generate data (Stan)
echo "[Step 1/6] Generating data..."
hold_flags=""
[ "$HOLD_I" == "1" ] && hold_flags="$hold_flags --hold-I-constant"
[ "$HOLD_J" == "1" ] && hold_flags="$hold_flags --hold-J-constant"
[ "$HOLD_K" == "1" ] && hold_flags="$hold_flags --hold-K-constant"
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
    $hold_flags \
    $protocol_args </dev/null

if [ $? -ne 0 ]; then
    echo "ERROR: Data generation failed for $run_name"
    exit 1
fi

# Step 2: Run Marformer with Lightning
echo "[Step 2/6] Running Marformer with PyTorch Lightning..."
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
    exit 1
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
    exit 1
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
    exit 1
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
    exit 1
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
    exit 1
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
