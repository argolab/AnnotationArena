#!/bin/bash
# Comparison: Old Data vs New Data x Marformer vs Unified Entity
# Modes: IJK and JK only. No Stan inference. No selective masking. MCAR 0.5.
# Uses item-chunking batching (max_item=5) for large K=100.
#
# For each mode, runs 4 experiments:
#   1. Old Data (spherical annotator) + Marformer
#   2. Old Data (spherical annotator) + Unified Entity
#   3. New Data (factored annotator)  + Marformer
#   4. New Data (factored annotator)  + Unified Entity
#
# Runs 1,2 share the same generated old data.
# Runs 3,4 share the same generated new data.

set -e

# ─── Data generation parameters ───
BASE_I=5
BASE_J=12
BASE_C=5
BASE_K_TRAIN=100
BASE_K_TEST=10

BASE_D=8
BASE_D_ANNOTATOR=2  # Annotator embedding dim for factored model (None→D=full rank)
BASE_SIGMA_ANNOTATOR=0.5
BASE_SIGMA_MEASUREMENT=0.1
BASE_KAPPA=10

# ─── Training parameters (shared by both models) ───
MAX_ITEM=5
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
WARMUP_STEPS=10

# ─── Experiment flags ───
DISABLE_RANKING=1
USE_CONCAT=0

EXP_PREFIX="small_D_comp_K${BASE_K_TRAIN}"

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
# Helper: generate data
#   $1 = run_name
#   $2 = data_type ("old" or "new")
#   $3 = I_val
#   $4... = hold flags (e.g. --hold-I-constant)
# ═══════════════════════════════════════════════════════════
generate_data() {
    local run_name=$1
    local data_type=$2
    local I_val=$3
    shift 3
    local hold_flags="$@"

    local data_model_flag=""
    if [ "$data_type" == "old" ]; then
        data_model_flag="--use-spherical-annotator"
    else
        data_model_flag="--d-annotator $BASE_D_ANNOTATOR"
    fi

    echo "[data] Generating ${data_type} data -> ${run_name}"
    python stan/scripts/generate_data.py \
        --K-train $BASE_K_TRAIN \
        --K-test $BASE_K_TEST \
        --I $I_val \
        --J $BASE_J \
        --D $BASE_D \
        --C $BASE_C \
        --observation-protocol mcar \
        --mcar-missing-rate 0.5 \
        --sigma-annotator $BASE_SIGMA_ANNOTATOR \
        --sigma-measurement $BASE_SIGMA_MEASUREMENT \
        --kappa $BASE_KAPPA \
        --run-name "$run_name" \
        --overwrite-existing-data \
        $hold_flags \
        $ranking_args \
        $data_model_flag </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Data generation failed for $run_name"
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
# Helper: run Unified Entity
#   $1 = data_run_name (for --data-dir)
#   $2 = model_run_name (for --run-name)
# ═══════════════════════════════════════════════════════════
run_unified_entity() {
    local data_run_name=$1
    local model_run_name=$2

    echo "[train] Unified Entity -> ${model_run_name}"
    python imputer/run_imputer.py \
        --data-dir OUTPUT/generated_data/${data_run_name} \
        --run-name ${model_run_name} \
        --overwrite-existing-data \
        --use-unified-entity \
        --use-prediction-head \
        --logit-high 20.0 \
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
        $cosine_flags </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Unified Entity training failed for $model_run_name"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════
# Run one mode (generates data twice: old + new, trains 4 models)
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

    # --- Old data (spherical annotator) ---
    echo ""
    echo "--- Old Data (spherical annotator) ---"
    local old_data="${base}_olddata"
    generate_data "$old_data" "old" "$I_val" $hold_flags

    run_marformer "$old_data" "${base}_old_marformer"
    # run_unified_entity "$old_data" "${base}_old_unified"

    # --- New data (factored annotator) ---
    echo ""
    echo "--- New Data (factored annotator) ---"
    local new_data="${base}_newdata"
    generate_data "$new_data" "new" "$I_val" $hold_flags

    run_marformer "$new_data" "${base}_new_marformer"
    # run_unified_entity "$new_data" "${base}_new_unified"

    echo ""
    echo "  Mode ${mode_name} complete"
}

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

echo "=============================================="
echo "COMPARISON: Marformer vs Unified Entity"
echo "           Old Data vs New Data"
echo "=============================================="
echo ""
echo "Parameters:"
echo "  I=${BASE_I}, J=${BASE_J}, C=${BASE_C}"
echo "  K_train=${BASE_K_TRAIN}, K_test=${BASE_K_TEST}, max_item=${MAX_ITEM}"
echo "  D=${BASE_D}, d_annotator=${BASE_D_ANNOTATOR}, sigma_a=${BASE_SIGMA_ANNOTATOR}, sigma_m=${BASE_SIGMA_MEASUREMENT}"
echo "  kappa=${BASE_KAPPA}, protocol=mcar (missing_rate=0.5)"
echo "  epochs=${EPOCHS}, lr=${LR}, embedding_dim=${EMBEDDING_DIM}"
echo "  encoder_layers=${ENCODER_LAYERS}, heads=${ATTENTION_HEADS}, d_ff=${D_FF}"
echo ""
echo "Modes: IJK, JK"
echo "Models: Marformer, Unified Entity"
echo "Data: Old (spherical), New (factored)"
echo "Total runs: 2 modes x 2 data x 2 models = 8 training runs"
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
echo "  Models: OUTPUT/IMPUTER/${EXP_PREFIX}_*"
echo "=============================================="
echo ""
