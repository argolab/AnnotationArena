#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

set -e

SCRIPT_START=$SECONDS

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT="DATA/LLM_RUBRIC"
OUTPUT_ROOT="RESULTS/BASELINES/MIWAE/LLMRUBRIC"

# ── Splits ────────────────────────────────────────────────────────────────────
SPLITS=(
    "LLMRubric_225_25_9_10"
    "LLMRubric_225_25_9_20"
    "LLMRubric_225_25_9_30"
    "LLMRubric_225_25_9_40"
    "LLMRubric_225_25_9_50"
    "LLMRubric_225_25_9_75"
    "LLMRubric_225_25_9_100"
    "LLMRubric_225_25_9_125"
    "LLMRubric_225_25_9_150"
    "LLMRubric_225_25_9_175"
)

# ── Hyperparams ───────────────────────────────────────────────────────────────
EPOCHS=3000
BATCH_SIZE=64
LR=1e-4
LATENT_DIM=16
HIDDEN_DIM=128
K=20
L=100
SEED=42
DEVICE="cpu"

echo ""
echo "============================================================"
echo " MIWAE | LLMRubric | All splits"
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
echo " All LLMRubric splits done."
echo " Total time : $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Output     : ${OUTPUT_ROOT}"
echo "============================================================"
