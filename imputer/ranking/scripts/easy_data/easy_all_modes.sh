#!/bin/bash
# Easy-data ladder over axis invariance flags (I, J, K).
# Runs Stan data generation + Marformer imputer + Stan inference + evaluation
# for a series of increasingly difficult dependency patterns.
#
# Axis flags:
#   hold_I_constant = 1 -> no dependence on I (criteria)
#   hold_J_constant = 1 -> no dependence on J (annotators)
#   hold_K_constant = 1 -> no dependence on K (items)
#
# The seven non-empty dependence modes (plus the full baseline) are:
#   constant:  hold_I=1, hold_J=1, hold_K=1   -> no I/J/K dependence
#   I-only:    hold_I=0, hold_J=1, hold_K=1   -> depends only on I
#   J-only:    hold_I=1, hold_J=0, hold_K=1   -> depends only on J
#   K-only:    hold_I=1, hold_J=1, hold_K=0   -> depends only on K
#   IJ:        hold_I=0, hold_J=0, hold_K=1   -> depends on I,J
#   IK:        hold_I=0, hold_J=1, hold_K=0   -> depends on I,K
#   JK:        hold_I=1, hold_J=0, hold_K=0   -> depends on J,K
#   IJK:       hold_I=0, hold_J=0, hold_K=0   -> full dependence (current complex case)

set -e

# Fixed base parameters (mirrors OFAT center setup where possible)
BASE_I=5
BASE_J=12
BASE_C=5
BASE_K_TRAIN=30
BASE_K_TEST=30

# Baseline hyperparameter values
BASE_D=8
BASE_SIGMA_ANNOTATOR=0.5
BASE_SIGMA_MEASUREMENT=0.1
BASE_KAPPA=10
BASE_PROTOCOL="tie_breaking"  # SMAR
BASE_PROTOCOL_CODE="smar"
BASE_USE_CONCAT=0  # Center value

# Unique prefix for TensorBoard / output filtering
EASY_PREFIX="easy_axis"

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

# Helper: run full pipeline for one axis mode
#   $1: mode name (constant, I, J, K, IJ, IK, JK, IJK)
#   $2: hold_I_constant (0/1)
#   $3: hold_J_constant (0/1)
#   $4: hold_K_constant (0/1)
run_axis_mode() {
    local mode_name=$1
    local hold_I=$2
    local hold_J=$3
    local hold_K=$4

    # Invariance rule: hold axis constant → set that dimension to 1 (no replication). Use MCAR when I=1 or J=1.
    local I_val=$BASE_I
    local J_val=$BASE_J
    local protocol=$BASE_PROTOCOL
    local protocol_code=$BASE_PROTOCOL_CODE
    [ "$hold_I" == "1" ] && { I_val=1; protocol=mcar; protocol_code=mcar; }   # I invariant → I=1
    [ "$hold_J" == "1" ] && { J_val=1; protocol=mcar; protocol_code=mcar; }   # J invariant → J=1

    # Format values for folder naming
    local d_str=$BASE_D
    local sa_str=$(echo $BASE_SIGMA_ANNOTATOR | sed 's/\.//g')
    local sm_str=$(echo $BASE_SIGMA_MEASUREMENT | sed 's/\.//g')
    local kp_str=$(echo $BASE_KAPPA | sed 's/\.//g')
    local uc_str=$BASE_USE_CONCAT

    # Axis flags string for run name
    local hI_str=$hold_I
    local hJ_str=$hold_J
    local hK_str=$hold_K

    # Construct run name
    local run_name="${EASY_PREFIX}_${mode_name}_D${d_str}_sa${sa_str}_sm${sm_str}_kp${kp_str}_uc${uc_str}_hI${hI_str}_hJ${hJ_str}_hK${hK_str}_${protocol_code}"

    echo ""
    echo "=========================================="
    echo "EASY AXIS EXPERIMENT: $run_name"
    echo "  Mode: ${mode_name}"
    echo "  hold_I_constant=${hold_I}, hold_J_constant=${hold_J}, hold_K_constant=${hold_K}"
    echo "=========================================="
    echo ""

    # Determine protocol-specific arguments (use $protocol for hold_I/hold_J → mcar)
    local protocol_args=""
    if [ "$protocol" == "extended_rankings" ]; then
        protocol_args="--extended-pairwise-rate 0.2"
    elif [ "$protocol" == "mcar" ]; then
        protocol_args="--mcar-missing-rate 0.5"
    fi

    # Toggle for concat-based AtomCompositional embeddings
    local concat_flag=""
    if [ "$BASE_USE_CONCAT" == "1" ]; then
        concat_flag="--use-concat-embedding"
    fi

    # Cosine schedule flags
    local cosine_schedule_flags=""
    if [ "$MARFORMER_USE_COSINE_SCHEDULE" == "true" ]; then
        cosine_schedule_flags="--use-cosine-schedule --warmup-steps $MARFORMER_WARMUP_STEPS"
    fi

    # Step 1: Generate data (Stan)
    echo "[Step 1/6] Generating data..."
    python stan/scripts/generate_data.py \
        --K-train $BASE_K_TRAIN \
        --K-test $BASE_K_TEST \
        --I $I_val \
        --J $J_val \
        --D $BASE_D \
        --C $BASE_C \
        --observation-protocol $protocol \
        --sigma-annotator $BASE_SIGMA_ANNOTATOR \
        --sigma-measurement $BASE_SIGMA_MEASUREMENT \
        --alpha-dirichlet $BASE_KAPPA \
        --run-name $run_name \
        --overwrite-existing-data \
        $( [ "$hold_I" == "1" ] && echo "--stan-arg hold_I_constant=1" ) \
        $( [ "$hold_J" == "1" ] && echo "--stan-arg hold_J_constant=1" ) \
        $( [ "$hold_K" == "1" ] && echo "--stan-arg hold_K_constant=1" ) \
        $protocol_args </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Data generation failed for $run_name"
        return 1
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

echo "=============================================="
echo "EASY-DATA SMOKE TEST: ALL AXIS MODES"
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
echo "  Protocol=$BASE_PROTOCOL"
echo "  use_concat_embedding=$BASE_USE_CONCAT"
echo "  Masking rate=$MARFORMER_MASKING_RATE"
echo ""
echo "Axis modes (hold_I, hold_J, hold_K):"
echo "  constant: 1,1,1"
echo "  I-only:   0,1,1"
echo "  J-only:   1,0,1"
echo "  K-only:   1,1,0"
echo "  IJ:       0,0,1"
echo "  IK:       0,1,0"
echo "  JK:       1,0,0"
echo "  IJK:      0,0,0"
echo ""

TOTAL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

# constant (no I/J/K dependence)
run_axis_mode "constant" 1 1 1 || ((FAILED_EXPERIMENTS++))
((TOTAL_EXPERIMENTS++))

# Single-axis modes
run_axis_mode "I_only" 0 1 1 || ((FAILED_EXPERIMENTS++))
((TOTAL_EXPERIMENTS++))

run_axis_mode "J_only" 1 0 1 || ((FAILED_EXPERIMENTS++))
((TOTAL_EXPERIMENTS++))

run_axis_mode "K_only" 1 1 0 || ((FAILED_EXPERIMENTS++))
((TOTAL_EXPERIMENTS++))

# Two-axis modes
run_axis_mode "IJ" 0 0 1 || ((FAILED_EXPERIMENTS++))
((TOTAL_EXPERIMENTS++))

run_axis_mode "IK" 0 1 0 || ((FAILED_EXPERIMENTS++))
((TOTAL_EXPERIMENTS++))

run_axis_mode "JK" 1 0 0 || ((FAILED_EXPERIMENTS++))
((TOTAL_EXPERIMENTS++))

# Full IJK dependence (reference)
run_axis_mode "IJK" 0 0 0 || ((FAILED_EXPERIMENTS++))
((TOTAL_EXPERIMENTS++))

echo ""
echo "=============================================="
echo "EASY-DATA AXIS LADDER EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo "Failed experiments: $FAILED_EXPERIMENTS"
echo "Successful experiments: $((TOTAL_EXPERIMENTS - FAILED_EXPERIMENTS))"
echo ""
echo "Results saved in:"
echo "  - Data: OUTPUT/generated_data/${EASY_PREFIX}_*"
echo "  - Marformer: OUTPUT/IMPUTER/${EASY_PREFIX}_*"
echo "  - Stan (4c): OUTPUT/domain_model/runs/${EASY_PREFIX}_*_stan4c"
echo "  - Stan (4c) Eval: OUTPUT/domain_model/eval/${EASY_PREFIX}_*_stan4c_eval"
echo "  - Stan (1c): OUTPUT/domain_model/runs/${EASY_PREFIX}_*_stan1c"
echo "  - Stan (1c) Eval: OUTPUT/domain_model/eval/${EASY_PREFIX}_*_stan1c_eval"
echo "=============================================="

