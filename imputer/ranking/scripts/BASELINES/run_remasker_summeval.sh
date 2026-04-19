#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

set -e

SCRIPT_START=$SECONDS

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT="DATA/SUMMEVAL"
OUTPUT_ROOT="RESULTS/BASELINES/REMASKER/SUMMEVAL"

# ── Splits ────────────────────────────────────────────────────────────────────
SPLITS=(
    "SummEval_1600_8_4_50"
    "SummEval_1600_8_4_100"
    "SummEval_1600_8_4_500"
    "SummEval_1600_8_4_750"
    "SummEval_1600_8_4_1000"
    "SummEval_1600_8_4_1280"
)

# ── Hyperparams ───────────────────────────────────────────────────────────────
EPOCHS=300
BATCH_SIZE=64
LR=1e-3
WEIGHT_DECAY=0.05
EMBED_DIM=32
DEPTH=4
NUM_HEADS=4
DECODER_DEPTH=2
MASK_RATIO=0.5
SEED=42
DEVICE="cpu"

echo ""
echo "============================================================"
echo " ReMasker | SummEval | All splits"
echo "  OUTPUT_ROOT : ${OUTPUT_ROOT}"
echo "  epochs      : ${EPOCHS}"
echo "============================================================"

for SPLIT in "${SPLITS[@]}"; do
    SPLIT_START=$SECONDS
    echo ""; echo "--- Split: ${SPLIT} ---"; echo ""

    python BASELINES/run_baselines.py \
        --method       remasker \
        --data-bundle  "${DATA_ROOT}/${SPLIT}/data_bundle.json" \
        --output-dir   "${OUTPUT_ROOT}/${SPLIT}" \
        --epochs       "$EPOCHS" \
        --batch-size   "$BATCH_SIZE" \
        --lr           "$LR" \
        --weight-decay "$WEIGHT_DECAY" \
        --embed-dim    "$EMBED_DIM" \
        --depth        "$DEPTH" \
        --num-heads    "$NUM_HEADS" \
        --decoder-depth "$DECODER_DEPTH" \
        --mask-ratio   "$MASK_RATIO" \
        --seed         "$SEED" \
        --device       "$DEVICE"

    SPLIT_ELAPSED=$(( SECONDS - SPLIT_START ))
    echo ""
    echo "  ↳ ${SPLIT} done in $(( SPLIT_ELAPSED / 60 ))m $(( SPLIT_ELAPSED % 60 ))s"
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " All SummEval splits done."
echo " Total time : $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Output     : ${OUTPUT_ROOT}"
echo "============================================================"
