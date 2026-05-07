#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

set -e

SCRIPT_START=$SECONDS

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT="DATA/LLM_RUBRIC"
OUTPUT_ROOT="RESULTS/BASELINES/REMASKER/LLMRUBRIC"

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
echo " ReMasker | LLMRubric | All splits"
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
echo " All LLMRubric splits done."
echo " Total time : $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Output     : ${OUTPUT_ROOT}"
echo "============================================================"
