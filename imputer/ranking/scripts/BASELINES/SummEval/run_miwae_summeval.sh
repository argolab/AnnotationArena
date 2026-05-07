#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

set -e

SCRIPT_START=$SECONDS

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT="DATA/SUMMEVAL"
OUTPUT_ROOT="RESULTS/BASELINES/MIWAE/SUMMEVAL"

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
EPOCHS=2000
BATCH_SIZE=64
LR=1e-4
LATENT_DIM=4
HIDDEN_DIM=128
K=20
L=50
SEED=42
DEVICE="cpu"

echo ""
echo "============================================================"
echo " MIWAE | SummEval | All splits"
echo "  OUTPUT_ROOT : ${OUTPUT_ROOT}"
echo "  epochs      : ${EPOCHS}"
echo "============================================================"

for SPLIT in "${SPLITS[@]}"; do
    SPLIT_START=$SECONDS
    echo ""; echo "--- Split: ${SPLIT} ---"; echo ""

    python BASELINES/run_baselines.py \
        --method      miwae \
        --data-bundle "${DATA_ROOT}/${SPLIT}/data_bundle.json" \
        --output-dir  "${OUTPUT_ROOT}/${SPLIT}" \
        --epochs      "$EPOCHS" \
        --batch-size  "$BATCH_SIZE" \
        --lr          "$LR" \
        --latent-dim  "$LATENT_DIM" \
        --hidden-dim  "$HIDDEN_DIM" \
        --K           "$K" \
        --L           "$L" \
        --seed        "$SEED" \
        --device      "$DEVICE"

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
