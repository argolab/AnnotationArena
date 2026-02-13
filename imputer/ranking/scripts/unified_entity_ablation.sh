#!/bin/bash
# Unified Entity ablation: normalize_param x architecture x BASE_D
# All old data (spherical annotator), IJK mode only.
#
# Variables:
#   normalize_param: {True, False}
#   Architecture:    {4 layers/6 heads, 6 layers/6 heads}
#   BASE_D:          {8, 32}
#
# = 8 training runs, 2 data generations (D=8 and D=32)
# All 4 training runs per D share the same generated data.

set -e

# ─── Data generation parameters ───
BASE_I=5
BASE_J=9
BASE_C=5
BASE_K_TRAIN=100
BASE_K_TEST=10

BASE_SIGMA_ANNOTATOR=0.5
BASE_SIGMA_MEASUREMENT=0.1
BASE_KAPPA=10

# ─── Training parameters ───
MAX_ITEM=5
EPOCHS=300
LR=2e-4
MASKING_RATE=0.15
MASKED_LOSS_WEIGHT=15
OBSERVED_LOSS_WEIGHT=1
MASK_AUGMENTATIONS=5
DEVICE="cpu"
DEVICES=1

# Transformer architecture (fixed across runs unless varied)
EMBEDDING_DIM=72
NUM_FFN_LAYERS=2
D_FF=512
WEIGHT_DECAY=0.01
DROPOUT=0.1

# Batching and optimization
BATCH_SIZE=1
GRADIENT_CLIP_VAL=0.0
USE_COSINE_SCHEDULE=true
WARMUP_STEPS=100

# ─── Experiment flags ───
DISABLE_RANKING=1

EXP_PREFIX="ue_ablation"

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

# ═══════════════════════════════════════════════════════════
# Helper: generate old data (spherical annotator)
#   $1 = run_name
#   $2 = D value
# ═══════════════════════════════════════════════════════════
generate_data() {
    local run_name=$1
    local D_val=$2

    echo "[data] Generating old data (D=${D_val}) -> ${run_name}"
    python stan/scripts/generate_data.py \
        --K-train $BASE_K_TRAIN \
        --K-test $BASE_K_TEST \
        --I $BASE_I \
        --J $BASE_J \
        --D $D_val \
        --C $BASE_C \
        --observation-protocol mcar \
        --mcar-missing-rate 0.5 \
        --sigma-annotator $BASE_SIGMA_ANNOTATOR \
        --sigma-measurement $BASE_SIGMA_MEASUREMENT \
        --kappa $BASE_KAPPA \
        --run-name "$run_name" \
        --overwrite-existing-data \
        --use-spherical-annotator \
        $ranking_args </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Data generation failed for $run_name"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════
# Helper: run Unified Entity
#   $1 = data_run_name
#   $2 = model_run_name
#   $3 = encoder_layers
#   $4 = attention_heads
#   $5 = normalize_param_flag ("--normalize-parameter" or "")
# ═══════════════════════════════════════════════════════════
run_unified_entity() {
    local data_run_name=$1
    local model_run_name=$2
    local enc_layers=$3
    local attn_heads=$4
    local norm_flag=$5

    echo "[train] Unified Entity -> ${model_run_name} (layers=${enc_layers}, heads=${attn_heads}, norm=${norm_flag:+yes}${norm_flag:-no})"
    python imputer/run_imputer.py \
        --data-dir OUTPUT/generated_data/${data_run_name} \
        --run-name ${model_run_name} \
        --overwrite-existing-data \
        --use-unified-entity \
        --use-prediction-head \
        --logit-high 20.0 \
        --embedding-dim $EMBEDDING_DIM \
        --encoder-layers $enc_layers \
        --attention-heads $attn_heads \
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
        $norm_flag \
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
# Run all 4 training configs on one dataset
#   $1 = data_run_name
#   $2 = name_prefix
# ═══════════════════════════════════════════════════════════
run_all_configs() {
    local data_run_name=$1
    local prefix=$2

    # Config 1: 4L/4H, normalize_param=True
    run_unified_entity "$data_run_name" "${prefix}_4L4H_normT" 4 4 "--normalize-parameter"

    # Config 2: 4L/4H, normalize_param=False
    run_unified_entity "$data_run_name" "${prefix}_4L4H_normF" 4 4 ""

    # Config 3: 6L/6H, normalize_param=True
    run_unified_entity "$data_run_name" "${prefix}_6L6H_normT" 6 6 "--normalize-parameter"

    # Config 4: 6L/6H, normalize_param=False
    run_unified_entity "$data_run_name" "${prefix}_6L6H_normF" 6 6 ""
}

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

echo "=============================================="
echo "UNIFIED ENTITY ABLATION"
echo "=============================================="
echo ""
echo "Variables:"
echo "  normalize_param: {True, False}"
echo "  Architecture:    {4L/4H, 6L/6H}"
echo "  BASE_D:          {8, 32}"
echo ""
echo "Fixed:"
echo "  I=${BASE_I}, J=${BASE_J}, C=${BASE_C}"
echo "  K_train=${BASE_K_TRAIN}, K_test=${BASE_K_TEST}, max_item=${MAX_ITEM}"
echo "  Data: old (spherical annotator), Mode: IJK"
echo "  embedding_dim=${EMBEDDING_DIM}, d_ff=${D_FF}"
echo "  epochs=${EPOCHS}, lr=${LR}"
echo ""
echo "Total: 2 data generations + 8 training runs"
echo ""

TOTAL=0
FAILED=0

# ── D=8 ──
echo "=========================================="
echo "BASE_D=8"
echo "=========================================="
DATA_D8="${EXP_PREFIX}_D8_data"
generate_data "$DATA_D8" 8 || { echo "FATAL: data gen failed"; exit 1; }

run_all_configs "$DATA_D8" "${EXP_PREFIX}_D8" || ((FAILED++))
((TOTAL+=4))

# ── D=32 ──
echo ""
echo "=========================================="
echo "BASE_D=32"
echo "=========================================="
DATA_D32="${EXP_PREFIX}_D32_data"
generate_data "$DATA_D32" 32 || { echo "FATAL: data gen failed"; exit 1; }

run_all_configs "$DATA_D32" "${EXP_PREFIX}_D32" || ((FAILED++))
((TOTAL+=4))

echo ""
echo "=============================================="
echo "ABLATION COMPLETE"
echo "=============================================="
echo "Total training runs: $TOTAL"
echo "Failed groups: $FAILED"
echo ""
echo "Results:"
echo "  Data:   OUTPUT/generated_data/${EXP_PREFIX}_D*_data"
echo "  Models: OUTPUT/IMPUTER/${EXP_PREFIX}_D*"
echo "=============================================="
echo ""
