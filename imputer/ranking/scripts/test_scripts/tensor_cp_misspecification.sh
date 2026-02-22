#!/bin/bash
# Tensor CP Misspecification Experiments
# Tests Stan robustness under different data generation assumptions.
# All use tensor_data_generation.stan with per-annotator thresholds, centered scores.
#
# 6 experiments:
#   1. D=8, J=32, log-space scores (breaks Stan's linear likelihood)
#   2. D=6, J=32, log-space scores
#   3. D=4, J=12, log-space scores
#   4. D=4, J=9,  log-space scores
#   5. D=6, J=32, logistic link (breaks Stan's probit assumption)
#   6. D=6, J=32, log-space scores, factor_decay=1.0 (equal components)
#
# Each: data gen + Marformer training + Stan 1C (commented out).
# No pairwise rankings. MCAR 0.5.

set -e

# ─── Data generation parameters ───
BASE_I=5
BASE_C=5
BASE_K_TRAIN=100
BASE_K_TEST=10

BASE_SIGMA_ANNOTATOR=0.5
BASE_SIGMA_MEASUREMENT=0.1
BASE_KAPPA=10

# ─── Training parameters (shared by both models) ───
MAX_ITEM=10
EPOCHS=350
LR=2e-4
MASKING_RATE=0.30
MASKED_LOSS_WEIGHT=15
OBSERVED_LOSS_WEIGHT=1
MASK_AUGMENTATIONS=5
DEVICE="cpu"
DEVICES=1

# Transformer architecture
EMBEDDING_DIM=72
ENCODER_LAYERS=4
ATTENTION_HEADS=4
NUM_FFN_LAYERS=2
D_FF=512
WEIGHT_DECAY=0.01
DROPOUT=0.1

# Batching and optimization
BATCH_SIZE=1
GRADIENT_CLIP_VAL=0.0
USE_COSINE_SCHEDULE=true
WARMUP_STEPS=5

# ─── Stan parameters ───
STAN_1C_CHAINS=1
STAN_1C_ITER_SAMPLING=300
STAN_1C_WARMUP=100

# ─── Experiment flags ───
DISABLE_RANKING=1
USE_CONCAT=0

EXP_PREFIX="tcp_misspec"

# ─── Derived flags ───
DEVICES_FLAG=""
if [ -n "$DEVICES" ]; then
    DEVICES_FLAG="--devices $DEVICES"
fi

ranking_args=""
if [ "$DISABLE_RANKING" == "1" ]; then
    ranking_args="--disable-pairwise-rankings"
fi

cosine_flags=""
if [ "$USE_COSINE_SCHEDULE" == "true" ]; then
    cosine_flags="--use-cosine-schedule --warmup-steps $WARMUP_STEPS"
fi

concat_flag=""
if [ "$USE_CONCAT" == "1" ]; then
    concat_flag="--use-concat-embedding"
fi

# ═══════════════════════════════════════════════════════════
# Helper: generate tensor CP data
#   $1 = run_name
#   $2 = D
#   $3 = J
#   $4 = factor_decay
#   $5 = extra_flags (e.g. "--use-log-scores" or "--use-logistic-link")
# ═══════════════════════════════════════════════════════════
generate_data() {
    local run_name=$1
    local D_val=$2
    local J_val=$3
    local decay=$4
    local extra_flags=$5

    echo "[data] Generating tensor CP data -> ${run_name}"
    echo "       D=${D_val}, J=${J_val}, factor_decay=${decay} ${extra_flags}"
    python stan/scripts/generate_data.py \
        --stan-file models/tensor_data_generation.stan \
        --K-train $BASE_K_TRAIN \
        --K-test $BASE_K_TEST \
        --I $BASE_I \
        --J $J_val \
        --D $D_val \
        --C $BASE_C \
        --d-annotator $D_val \
        --factor-decay $decay \
        --observation-protocol mcar \
        --mcar-missing-rate 0.5 \
        --sigma-annotator $BASE_SIGMA_ANNOTATOR \
        --sigma-measurement $BASE_SIGMA_MEASUREMENT \
        --kappa $BASE_KAPPA \
        --run-name "$run_name" \
        --overwrite-existing-data \
        $ranking_args \
        $extra_flags </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Data generation failed for $run_name"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════
# Helper: run Stan inference + evaluation
#   $1 = data_run_name
#   $2 = stan_run_name
#   $3 = chains
#   $4 = iter_sampling
#   $5 = iter_warmup
# ═══════════════════════════════════════════════════════════
run_stan() {
    local data_run_name=$1
    local stan_run_name=$2
    local chains=$3
    local iter_sampling=$4
    local iter_warmup=$5

    echo "[stan] Inference (${chains}c) -> ${stan_run_name}"
    python stan/scripts/run_inference.py \
        --data-bundle OUTPUT/generated_data/${data_run_name}/data_bundle.json \
        --chains $chains \
        --iter-sampling $iter_sampling \
        --iter-warmup $iter_warmup \
        --run-name ${stan_run_name} \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Stan inference failed for $stan_run_name"
        return 1
    fi

    echo "[stan] Evaluation -> ${stan_run_name}_eval"
    python stan/scripts/evaluate_predictions.py \
        --data-bundle OUTPUT/generated_data/${data_run_name}/data_bundle.json \
        --mcmc-dir OUTPUT/domain_model/runs/${stan_run_name} \
        --run-name ${stan_run_name}_eval \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Stan evaluation failed for ${stan_run_name}_eval"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════
# Helper: run Marformer
#   $1 = data_run_name (for --data-dir)
#   $2 = model_run_name (for --run-name)
# ═══════════════════════════════════════════════════════════
run_marformer() {
    local data_run_name=$1
    local model_run_name=$2

    echo "[train] Marformer -> ${model_run_name}"
    python imputer/run_imputer.py \
        --data-dir OUTPUT/generated_data/${data_run_name} \
        --run-name ${model_run_name} \
        --overwrite-existing-data \
        --embedding-dim $EMBEDDING_DIM \
        --encoder-layers $ENCODER_LAYERS \
        --attention-heads $ATTENTION_HEADS \
        --num_ffn_layers $NUM_FFN_LAYERS \
        --d-ff $D_FF \
        --weight-decay $WEIGHT_DECAY \
        --dropout $DROPOUT \
        --epochs $EPOCHS \
        --lr $LR \
        --masking-rate $MASKING_RATE \
        --masked-loss-weight $MASKED_LOSS_WEIGHT \
        --observed-loss-weight $OBSERVED_LOSS_WEIGHT \
        --mask-augmentations $MASK_AUGMENTATIONS \
        --no-final-norm \
        --normalize-parameter \
        --device $DEVICE \
        --max-item $MAX_ITEM \
        $DEVICES_FLAG \
        --save-model-every 5 \
        --batch-size $BATCH_SIZE \
        --gradient-clip-val $GRADIENT_CLIP_VAL \
        $concat_flag \
        $cosine_flags </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Marformer training failed for $model_run_name"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════
# Run one experiment
#   $1 = exp_name
#   $2 = D
#   $3 = J
#   $4 = factor_decay
#   $5 = extra_flags for data gen
# ═══════════════════════════════════════════════════════════
run_experiment() {
    local exp_name=$1
    local D_val=$2
    local J_val=$3
    local decay=$4
    local extra_flags=$5

    local base="${EXP_PREFIX}_${exp_name}"

    echo ""
    echo "=========================================="
    echo "EXPERIMENT: ${exp_name}"
    echo "  D=${D_val}, J=${J_val}, factor_decay=${decay}"
    echo "  extra: ${extra_flags}"
    echo "=========================================="

    # --- Generate data ---
    local data_name="${base}_data"
    generate_data "$data_name" "$D_val" "$J_val" "$decay" "$extra_flags"

    # --- Stan 1C (commented out) ---
    # run_stan "$data_name" "${base}_stan1c" $STAN_1C_CHAINS $STAN_1C_ITER_SAMPLING $STAN_1C_WARMUP

    # --- Marformer ---
    run_marformer "$data_name" "${base}_marformer"

    echo ""
    echo "  Experiment ${exp_name} complete"
}

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

echo "=============================================="
echo "TENSOR CP MISSPECIFICATION EXPERIMENTS"
echo "=============================================="
echo ""
echo "Parameters:"
echo "  I=${BASE_I}, C=${BASE_C}"
echo "  K_train=${BASE_K_TRAIN}, K_test=${BASE_K_TEST}, max_item=${MAX_ITEM}"
echo "  sigma_a=${BASE_SIGMA_ANNOTATOR}, sigma_m=${BASE_SIGMA_MEASUREMENT}"
echo "  kappa=${BASE_KAPPA}, protocol=mcar (missing_rate=0.5)"
echo "  epochs=${EPOCHS}, lr=${LR}, embedding_dim=${EMBEDDING_DIM}"
echo "  encoder_layers=${ENCODER_LAYERS}, heads=${ATTENTION_HEADS}, d_ff=${D_FF}"
echo ""
echo "Experiments:"
echo "  1. log_D8_J32:     D=8,  J=32, log-space, decay=0.9"
echo "  2. log_D6_J32:     D=6,  J=32, log-space, decay=0.9"
echo "  3. log_D4_J12:     D=4,  J=12, log-space, decay=0.9"
echo "  4. log_D4_J9:      D=4,  J=9,  log-space, decay=0.9"
echo "  5. logistic_D6_J32: D=6, J=32, logistic link, decay=0.9"
echo "  6. log_D6_J32_eq:  D=6,  J=32, log-space, decay=1.0 (equal components)"
echo ""

TOTAL_EXPS=0
FAILED_EXPS=0

# Exp 1: D=8, J=32, log-space
run_experiment "log_D8_J32" 8 32 0.9 "--use-log-scores" || ((FAILED_EXPS++))
((TOTAL_EXPS++))

# Exp 2: D=6, J=32, log-space
run_experiment "log_D6_J32" 6 32 0.9 "--use-log-scores" || ((FAILED_EXPS++))
((TOTAL_EXPS++))

# Exp 3: D=4, J=12, log-space
run_experiment "log_D4_J12" 4 12 0.9 "--use-log-scores" || ((FAILED_EXPS++))
((TOTAL_EXPS++))

# Exp 4: D=4, J=9, log-space
run_experiment "log_D4_J9" 4 9 0.9 "--use-log-scores" || ((FAILED_EXPS++))
((TOTAL_EXPS++))

# Exp 5: D=6, J=32, logistic link (no log-space)
run_experiment "logistic_D6_J32" 6 32 0.9 "--use-logistic-link" || ((FAILED_EXPS++))
((TOTAL_EXPS++))

# Exp 6: D=6, J=32, log-space, factor_decay=1.0
run_experiment "log_D6_J32_eq" 6 32 1.0 "--use-log-scores" || ((FAILED_EXPS++))
((TOTAL_EXPS++))

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Total experiments: $TOTAL_EXPS"
echo "Failed experiments: $FAILED_EXPS"
echo ""
echo "Results:"
echo "  Data:     OUTPUT/generated_data/${EXP_PREFIX}_*"
echo "  Stan:     OUTPUT/domain_model/runs/${EXP_PREFIX}_*"
echo "  Stan eval: OUTPUT/domain_model/eval/${EXP_PREFIX}_*"
echo "  Marformer: OUTPUT/IMPUTER/${EXP_PREFIX}_*"
echo "=============================================="
echo ""
