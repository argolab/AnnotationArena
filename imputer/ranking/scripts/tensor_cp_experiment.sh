#!/bin/bash
# Stan + Marformer on Tensor CP Data
# Data: Z_ijk = sum_d v_id * u_jd * e_kd * T_d
#   v, u, e ~ Exp(1), T_d = factor_decay^(d-1)
#   Per-annotator thresholds, centered scores, MCAR 0.5, no pairwise rankings.
#
# Runs:
#   1. Generate CP tensor data
#   2. Stan inference (4 chains) + evaluation
#   3. Stan inference (1 chain) + evaluation
#   4. Train Old Marformer on CP data

set -e

# ─── Data generation parameters ───
BASE_I=5
BASE_J=32
BASE_C=5
BASE_K_TRAIN=100
BASE_K_TEST=10

BASE_D=6            # CP rank
BASE_D_ANNOTATOR=6  # Same as D for CP model
BASE_FACTOR_DECAY=0.9
BASE_SIGMA_ANNOTATOR=0.5
BASE_SIGMA_MEASUREMENT=0.1
BASE_KAPPA=10

# ─── Training parameters (shared by both models) ───
MAX_ITEM=10
EPOCHS=300
LR=2e-4
MASKING_RATE=0.15
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
STAN_4C_CHAINS=4
STAN_1C_CHAINS=1
STAN_4C_ITER_SAMPLING=300
STAN_4C_WARMUP=100
STAN_1C_ITER_SAMPLING=300
STAN_1C_WARMUP=100

# ─── Experiment flags ───
DISABLE_RANKING=1
USE_CONCAT=0

EXP_PREFIX="tensor_cp_marformer_largeJ_K${BASE_K_TRAIN}"

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
#   $2 = I_val
#   $3... = hold flags (e.g. --hold-I-constant)
# ═══════════════════════════════════════════════════════════
generate_data() {
    local run_name=$1
    local I_val=$2
    shift 2
    local hold_flags="$@"

    echo "[data] Generating tensor CP data -> ${run_name}"
    python stan/scripts/generate_data.py \
        --stan-file models/tensor_data_generation.stan \
        --K-train $BASE_K_TRAIN \
        --K-test $BASE_K_TEST \
        --I $I_val \
        --J $BASE_J \
        --D $BASE_D \
        --C $BASE_C \
        --d-annotator $BASE_D_ANNOTATOR \
        --factor-decay $BASE_FACTOR_DECAY \
        --observation-protocol mcar \
        --mcar-missing-rate 0.5 \
        --sigma-annotator $BASE_SIGMA_ANNOTATOR \
        --sigma-measurement $BASE_SIGMA_MEASUREMENT \
        --kappa $BASE_KAPPA \
        --run-name "$run_name" \
        --overwrite-existing-data \
        $hold_flags \
        $ranking_args </dev/null

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
# Run one mode
#   $1 = mode_name
#   $2 = hold_I (0/1)
#   $3 = hold_J (0/1)
#   $4 = hold_K (0/1)
# ═══════════════════════════════════════════════════════════
run_mode() {
    local mode_name=$1
    local hold_I=$2
    local hold_J=$3
    local hold_K=$4

    # Adjust I when held constant (same logic as easy_all_modes.sh)
    local I_val=$BASE_I
    [ "$hold_I" == "1" ] && I_val=1

    # Build hold flags for data generation
    local hold_flags=""
    [ "$hold_I" == "1" ] && hold_flags="$hold_flags --hold-I-constant"
    [ "$hold_J" == "1" ] && hold_flags="$hold_flags --hold-J-constant"
    [ "$hold_K" == "1" ] && hold_flags="$hold_flags --hold-K-constant"

    local base="${EXP_PREFIX}_${mode_name}"

    echo ""
    echo "=========================================="
    echo "MODE: ${mode_name}"
    echo "  hold_I=${hold_I}, hold_J=${hold_J}, hold_K=${hold_K}"
    echo "  I=${I_val}, J=${BASE_J}, K_train=${BASE_K_TRAIN}, max_item=${MAX_ITEM}"
    echo "=========================================="

    # --- Generate tensor CP data ---
    echo ""
    echo "--- Tensor CP Data (Exp loadings, per-annotator thresholds, centered) ---"
    local data_name="${base}_data"
    generate_data "$data_name" "$I_val" $hold_flags

    # --- Stan inference (4 chains) ---
    # run_stan "$data_name" "${base}_stan4c" $STAN_4C_CHAINS $STAN_4C_ITER_SAMPLING $STAN_4C_WARMUP

    # --- Stan inference (1 chain) ---
    run_stan "$data_name" "${base}_stan1c" $STAN_1C_CHAINS $STAN_1C_ITER_SAMPLING $STAN_1C_WARMUP

    # --- Marformer ---
    run_marformer "$data_name" "${base}_marformer"

    echo ""
    echo "  Mode ${mode_name} complete"
}

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

echo "=============================================="
echo "STAN + MARFORMER ON TENSOR CP DATA"
echo "=============================================="
echo ""
echo "Data Model: Z_ijk = sum_d v_id * u_jd * e_kd * T_d"
echo "  v, u, e ~ Exp(1), centered scores"
echo "  T_d = ${BASE_FACTOR_DECAY}^(d-1)"
echo "  D (rank) = ${BASE_D}"
echo "  Per-annotator thresholds"
echo ""
echo "Parameters:"
echo "  I=${BASE_I}, J=${BASE_J}, C=${BASE_C}"
echo "  K_train=${BASE_K_TRAIN}, K_test=${BASE_K_TEST}, max_item=${MAX_ITEM}"
echo "  D=${BASE_D}, d_annotator=${BASE_D_ANNOTATOR}"
echo "  sigma_a=${BASE_SIGMA_ANNOTATOR}, sigma_m=${BASE_SIGMA_MEASUREMENT}"
echo "  kappa=${BASE_KAPPA}, factor_decay=${BASE_FACTOR_DECAY}"
echo "  protocol=mcar (missing_rate=0.5)"
echo ""
echo "Stan: ${STAN_4C_CHAINS}c/${STAN_4C_ITER_SAMPLING}s/${STAN_4C_WARMUP}w + ${STAN_1C_CHAINS}c/${STAN_1C_ITER_SAMPLING}s/${STAN_1C_WARMUP}w"
echo "Marformer: epochs=${EPOCHS}, lr=${LR}, dim=${EMBEDDING_DIM}, layers=${ENCODER_LAYERS}, heads=${ATTENTION_HEADS}"
echo ""

TOTAL_MODES=0
FAILED_MODES=0

# IJK: full dependence (hold_I=0, hold_J=0, hold_K=0)
run_mode "IJK" 0 0 0 || ((FAILED_MODES++))
((TOTAL_MODES++))

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Total modes: $TOTAL_MODES"
echo "Failed modes: $FAILED_MODES"
echo ""
echo "Results:"
echo "  Data:   OUTPUT/generated_data/${EXP_PREFIX}_*"
echo "  Stan:   OUTPUT/domain_model/runs/${EXP_PREFIX}_*"
echo "  Stan eval: OUTPUT/domain_model/eval/${EXP_PREFIX}_*"
echo "  Marformer: OUTPUT/IMPUTER/${EXP_PREFIX}_*"
echo "=============================================="
echo ""
