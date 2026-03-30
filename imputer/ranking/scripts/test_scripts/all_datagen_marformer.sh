#!/bin/bash

set -e

# ─── Data dimensions ───
I=5
J=32
D=6
C=5
K_TRAIN=100
K_TEST=10
SIGMA_ANNOTATOR=0.5
SIGMA_MEASUREMENT=0.1
KAPPA=10
FACTOR_DECAY=1.0     # for TCP only

# ─── Marformer architecture ───
EMBEDDING_DIM=72
ENCODER_LAYERS=4
ATTENTION_HEADS=4
NUM_FFN_LAYERS=2
D_FF=512
DROPOUT=0.1
WEIGHT_DECAY=0.01

# ─── Marformer training ───
EPOCHS=200
LR=2e-4
MASKING_RATE=0.15
MASKED_LOSS_WEIGHT=15
OBSERVED_LOSS_WEIGHT=1
MASK_AUGMENTATIONS=5
MAX_ITEM=10
BATCH_SIZE=1
GRADIENT_CLIP_VAL=0.0
DEVICE="cpu"
DEVICES=1

# Cosine schedule
USE_COSINE_SCHEDULE=true
WARMUP_STEPS=5

# ─── Fixed flags ───
# No pairwise rankings, no concat, MCAR
DISABLE_RANKING=1
USE_CONCAT=0

# ─── Derived flags ───
ranking_args="--disable-pairwise-rankings"

cosine_flags=""
if [ "$USE_COSINE_SCHEDULE" == "true" ]; then
    cosine_flags="--use-cosine-schedule --warmup-steps $WARMUP_STEPS"
fi

concat_flag=""
# USE_CONCAT=0, so concat_flag stays empty

devices_flag="--devices $DEVICES"

# ═══════════════════════════════════════════════════════════
# Helper: generate data
#   $1 = run_name (data dir name)
#   $2 = stan_file
#   $3 = extra flags for generate_data.py (e.g. --use-spherical-annotator)
#   $4 = "tcp" if TCP model (needs --factor-decay)
# ═══════════════════════════════════════════════════════════
generate_data() {
    local data_name=$1
    local stan_file=$2
    local extra_flags=$3
    local is_tcp=$4

    echo ""
    echo "=========================================="
    echo "[data] ${data_name}"
    echo "  stan: ${stan_file}"
    echo "  flags: ${extra_flags}"
    echo "=========================================="

    local tcp_flag=""
    if [ "$is_tcp" == "tcp" ]; then
        tcp_flag="--factor-decay $FACTOR_DECAY"
    fi

    python stan/scripts/generate_data.py \
        --stan-file ${stan_file} \
        --K-train $K_TRAIN \
        --K-test $K_TEST \
        --I $I \
        --J $J \
        --D $D \
        --C $C \
        --d-annotator $D \
        --observation-protocol mcar \
        --mcar-missing-rate 0.5 \
        --sigma-annotator $SIGMA_ANNOTATOR \
        --sigma-measurement $SIGMA_MEASUREMENT \
        --alpha-dirichlet $KAPPA \
        $tcp_flag \
        $ranking_args \
        $extra_flags \
        --run-name "${data_name}" \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Data generation failed for ${data_name}"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════
# Helper: run Marformer
#   $1 = run_name
#   $2 = data_name (the data dir to read from)
# ═══════════════════════════════════════════════════════════
run_marformer() {
    local run_name=$1
    local data_name=$2

    echo ""
    echo "=========================================="
    echo "[train] ${run_name}"
    echo "  data: OUTPUT/generated_data/${data_name}"
    echo "  embed_dim=${EMBEDDING_DIM}, epochs=${EPOCHS}"
    echo "=========================================="

    python imputer/run_imputer.py \
        --data-dir OUTPUT/generated_data/${data_name} \
        --run-name ${run_name} \
        --overwrite-existing-data \
        --embedding-dim $EMBEDDING_DIM \
        --encoder-layers $ENCODER_LAYERS \
        --attention-heads $ATTENTION_HEADS \
        --num_ffn_layers $NUM_FFN_LAYERS \
        --d-ff $D_FF \
        --dropout $DROPOUT \
        --weight-decay $WEIGHT_DECAY \
        --epochs $EPOCHS \
        --lr $LR \
        --masking-rate $MASKING_RATE \
        --masked-loss-weight $MASKED_LOSS_WEIGHT \
        --observed-loss-weight $OBSERVED_LOSS_WEIGHT \
        --mask-augmentations $MASK_AUGMENTATIONS \
        --max-item $MAX_ITEM \
        --batch-size $BATCH_SIZE \
        --gradient-clip-val $GRADIENT_CLIP_VAL \
        --no-final-norm \
        --normalize-parameter \
        --device $DEVICE \
        $devices_flag \
        --save-model-every 5 \
        $concat_flag \
        $cosine_flags </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Marformer failed for ${run_name}"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

echo "=============================================="
echo "ALL 6 DATA GENERATION METHODS + MARFORMER"
echo "=============================================="
echo "Dims: I=${I}, J=${J}, D=${D}, C=${C}"
echo "      K_train=${K_TRAIN}, K_test=${K_TEST}, MCAR 0.5"
echo "Marformer: embed_dim=${EMBEDDING_DIM}, epochs=${EPOCHS}"
echo ""

TOTAL=0
FAILED=0

# ──────────────────────────────────────────────
# 2. Spherical Exp
# ──────────────────────────────────────────────
DATA="spherical_exp_data"
RUN="spherical_exp_marformer_e72"
generate_data "$DATA" "models/iclr_exp_data_generation.stan" "--use-spherical-annotator" "" || ((FAILED++))
((TOTAL++))
run_marformer "$RUN" "$DATA" || ((FAILED++))
((TOTAL++))

# ──────────────────────────────────────────────
# 1. Spherical Gaussian
# ──────────────────────────────────────────────
DATA="spherical_gauss_data"
RUN="spherical_gauss_marformer_e72"
generate_data "$DATA" "models/iclr_data_generation.stan" "--use-spherical-annotator" "" || ((FAILED++))
((TOTAL++))
run_marformer "$RUN" "$DATA" || ((FAILED++))
((TOTAL++))

# ──────────────────────────────────────────────
# 3. M-matrix Gaussian
# ──────────────────────────────────────────────
DATA="mmatrix_gauss_data"
RUN="mmatrix_gauss_marformer_e72"
generate_data "$DATA" "models/iclr_data_generation.stan" "" "" || ((FAILED++))
((TOTAL++))
run_marformer "$RUN" "$DATA" || ((FAILED++))
((TOTAL++))

# ──────────────────────────────────────────────
# 4. M-matrix Exp
# ──────────────────────────────────────────────
DATA="mmatrix_exp_data"
RUN="mmatrix_exp_marformer_e72"
generate_data "$DATA" "models/iclr_exp_data_generation.stan" "" "" || ((FAILED++))
((TOTAL++))
run_marformer "$RUN" "$DATA" || ((FAILED++))
((TOTAL++))

# ──────────────────────────────────────────────
# 5. TCP Gaussian
# ──────────────────────────────────────────────
DATA="tcp_gauss_data"
RUN="tcp_gauss_marformer_e72"
generate_data "$DATA" "models/tensor_data_generation.stan" "--use-normal-loadings" "tcp" || ((FAILED++))
((TOTAL++))
run_marformer "$RUN" "$DATA" || ((FAILED++))
((TOTAL++))

# ──────────────────────────────────────────────
# 6. TCP Exp
# ──────────────────────────────────────────────
DATA="tcp_exp_data"
RUN="tcp_exp_marformer_e72"
generate_data "$DATA" "models/tensor_data_generation.stan" "" "tcp" || ((FAILED++))
((TOTAL++))
run_marformer "$RUN" "$DATA" || ((FAILED++))
((TOTAL++))

echo ""
echo "=============================================="
echo "ALL DONE"
echo "Total: $TOTAL, Failed: $FAILED"
echo ""
echo "Data dirs:"
echo "  OUTPUT/generated_data/spherical_gauss_data"
echo "  OUTPUT/generated_data/spherical_exp_data"
echo "  OUTPUT/generated_data/mmatrix_gauss_data"
echo "  OUTPUT/generated_data/mmatrix_exp_data"
echo "  OUTPUT/generated_data/tcp_gauss_data"
echo "  OUTPUT/generated_data/tcp_exp_data"
echo ""
echo "Marformer runs:"
echo "  OUTPUT/IMPUTER/spherical_gauss_marformer_e72"
echo "  OUTPUT/IMPUTER/spherical_exp_marformer_e72"
echo "  OUTPUT/IMPUTER/mmatrix_gauss_marformer_e72"
echo "  OUTPUT/IMPUTER/mmatrix_exp_marformer_e72"
echo "  OUTPUT/IMPUTER/tcp_gauss_marformer_e72"
echo "  OUTPUT/IMPUTER/tcp_exp_marformer_e72"
echo "=============================================="
