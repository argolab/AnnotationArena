#!/bin/bash
# TCP Normal Loadings Experiment
# N(0,1) loadings + per-(i,j) thresholds + factor_decay=1.0
# D=6, J=32, linear scores (no log transform — same as old iclr model)
#
# Runs:
#   1. Data generation (single dataset, shared across all runs)
#   2. Stan 1C: D=3 (half), D=6 (correct), D=12 (double)
#   3. Marformer: embedding_dim=36, 72, 144
#
# No pairwise rankings. MCAR 0.5.

set -e

# ─── Data generation parameters ───
D=6
J=32
I=5
C=5
K_TRAIN=100
K_TEST=10
FACTOR_DECAY=1.0

SIGMA_ANNOTATOR=0.5
SIGMA_MEASUREMENT=0.1
KAPPA=10

# ─── Training parameters ───
MAX_ITEM=10
EPOCHS=300
LR=2e-4
MASKING_RATE=0.15
MASKED_LOSS_WEIGHT=15
OBSERVED_LOSS_WEIGHT=1
MASK_AUGMENTATIONS=5
DEVICE="cpu"
DEVICES=1

# Transformer architecture (defaults — embedding_dim varies per run)
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
STAN_CHAINS=1
STAN_ITER=300
STAN_WARMUP=100

# ─── Experiment flags ───
DISABLE_RANKING=1
USE_CONCAT=0

EXP_PREFIX="tcp_normal"
DATA_NAME="${EXP_PREFIX}_data"

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
# Generate data (once, shared by all runs)
# ═══════════════════════════════════════════════════════════
generate_data() {
    echo ""
    echo "=========================================="
    echo "[data] Generating TCP Normal data"
    echo "  D=${D}, J=${J}, factor_decay=${FACTOR_DECAY}"
    echo "  N(0,1) loadings, per-(i,j) thresholds, linear"
    echo "=========================================="

    python stan/scripts/generate_data.py \
        --stan-file models/tensor_data_generation.stan \
        --K-train $K_TRAIN \
        --K-test $K_TEST \
        --I $I \
        --J $J \
        --D $D \
        --C $C \
        --d-annotator $D \
        --factor-decay $FACTOR_DECAY \
        --observation-protocol mcar \
        --mcar-missing-rate 0.5 \
        --sigma-annotator $SIGMA_ANNOTATOR \
        --sigma-measurement $SIGMA_MEASUREMENT \
        --kappa $KAPPA \
        $ranking_args \
        --run-name "$DATA_NAME" \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Data generation failed"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════
# Run Stan 1C with optional D override
#   $1 = stan_run_name
#   $2 = override_D (empty string = use data D)
# ═══════════════════════════════════════════════════════════
run_stan() {
    local stan_name=$1
    local override_D=$2

    local override_flag=""
    if [ -n "$override_D" ]; then
        override_flag="--override-D $override_D"
        echo "[stan] ${stan_name} (D=${override_D})"
    else
        echo "[stan] ${stan_name} (D=${D})"
    fi

    python stan/scripts/run_inference.py \
        --data-bundle OUTPUT/generated_data/${DATA_NAME}/data_bundle.json \
        --chains $STAN_CHAINS \
        --iter-sampling $STAN_ITER \
        --iter-warmup $STAN_WARMUP \
        $override_flag \
        --run-name ${stan_name} \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Stan inference failed for $stan_name"
        return 1
    fi

    python stan/scripts/evaluate_predictions.py \
        --data-bundle OUTPUT/generated_data/${DATA_NAME}/data_bundle.json \
        --mcmc-dir OUTPUT/domain_model/runs/${stan_name} \
        --run-name ${stan_name}_eval \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Stan evaluation failed for ${stan_name}_eval"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════
# Run Marformer with specified embedding dimension
#   $1 = run_name
#   $2 = embedding_dim
# ═══════════════════════════════════════════════════════════
run_marformer() {
    local run_name=$1
    local embed_dim=$2

    echo "[train] Marformer -> ${run_name} (embed_dim=${embed_dim})"
    python imputer/run_imputer.py \
        --data-dir OUTPUT/generated_data/${DATA_NAME} \
        --run-name ${run_name} \
        --overwrite-existing-data \
        --embedding-dim $embed_dim \
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
        echo "ERROR: Marformer training failed for $run_name"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

echo "=============================================="
echo "TCP NORMAL LOADINGS EXPERIMENT"
echo "=============================================="
echo ""
echo "Data: D=${D}, J=${J}, I=${I}, C=${C}"
echo "  K_train=${K_TRAIN}, K_test=${K_TEST}"
echo "  factor_decay=${FACTOR_DECAY}, linear, MCAR 0.5"
echo "  N(0,1) loadings, per-(i,j) thresholds"
echo ""
echo "Stan 1C runs: D=3 (half), D=6 (correct), D=12 (double)"
echo "Marformer runs: embed_dim=36, 72, 144"
echo ""

TOTAL=0
FAILED=0

# --- Generate data ---
generate_data || ((FAILED++))
((TOTAL++))

# --- Marformer: embed_dim=72 ---
echo ""
echo "=========================================="
echo "MARFORMER: embed_dim=72"
echo "=========================================="
run_marformer "${EXP_PREFIX}_marformer_e72" 72 || ((FAILED++))
((TOTAL++))

# --- Stan 1C: half D ---
echo ""
echo "=========================================="
echo "STAN: Half D (D=3)"
echo "=========================================="
run_stan "${EXP_PREFIX}_stan1c_halfD" 3 || ((FAILED++))
((TOTAL++))

# --- Stan 1C: correct D ---
echo ""
echo "=========================================="
echo "STAN: Correct D (D=6)"
echo "=========================================="
run_stan "${EXP_PREFIX}_stan1c" "" || ((FAILED++))
((TOTAL++))

# --- Stan 1C: double D ---
echo ""
echo "=========================================="
echo "STAN: Double D (D=12)"
echo "=========================================="
run_stan "${EXP_PREFIX}_stan1c_doubleD" 12 || ((FAILED++))
((TOTAL++))

# --- Marformer: embed_dim=144 ---
echo ""
echo "=========================================="
echo "MARFORMER: embed_dim=144"
echo "=========================================="
run_marformer "${EXP_PREFIX}_marformer_e144" 144 || ((FAILED++))
((TOTAL++))

echo ""
echo "=============================================="
echo "ALL RUNS COMPLETE"
echo "=============================================="
echo "Total: $TOTAL, Failed: $FAILED"
echo ""
echo "Results:"
echo "  Data:      OUTPUT/generated_data/${DATA_NAME}"
echo "  Stan:      OUTPUT/domain_model/runs/${EXP_PREFIX}_stan1c*"
echo "  Stan eval: OUTPUT/domain_model/eval/${EXP_PREFIX}_stan1c*"
echo "  Marformer: OUTPUT/IMPUTER/${EXP_PREFIX}_marformer_*"
echo "=============================================="
