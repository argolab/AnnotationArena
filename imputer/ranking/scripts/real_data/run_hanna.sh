#!/bin/bash
# Run Marformer on the HANNA dataset (human-data-new + gpt-3.5-turbo-data-new).
#
# Two variants (controlled by BUNDLE):
#   hard  → OUTPUT/generated_data/hanna       (hard labels, one-hot CE)
#   dist  → OUTPUT/generated_data/hanna_dist  (soft labels, GPT distributions)
#
# Dataset:  K=1200, I=6 (Q1-Q6), J=19 (18 humans + 1 GPT), C=5
# LLM annotator: 0-indexed id=18  (1-indexed id=19 in bundle)
#
# Usage (from Marformer repo root):
#   bash scripts/real_data/run_hanna.sh              # hard labels
#   BUNDLE=dist bash scripts/real_data/run_hanna.sh  # soft labels

set -e

# ── Bundle variant ─────────────────────────────────────────────────────────────
BUNDLE="${BUNDLE:-hard}"

if [ "$BUNDLE" == "dist" ]; then
    DATA_DIR="OUTPUT/generated_data/hanna_dist"
    RUN_NAME="hanna_marformer_dist"
else
    DATA_DIR="OUTPUT/generated_data/hanna"
    RUN_NAME="hanna_marformer"
fi

echo ""
echo "=========================================="
echo "HANNA — Marformer"
echo "  bundle:   $BUNDLE"
echo "  data_dir: $DATA_DIR"
echo "  run_name: $RUN_NAME"
echo "=========================================="
echo ""

# ── Hyperparameters ────────────────────────────────────────────────────────────
EMBEDDING_DIM=72
ENCODER_LAYERS=4
ATTENTION_HEADS=4
NUM_FFN_LAYERS=1
D_FF=512
DROPOUT=0.1
WEIGHT_DECAY=0.01

EPOCHS=150
LR=2e-4
MASKING_RATE=0.15
MASKED_LOSS_WEIGHT=5
OBSERVED_LOSS_WEIGHT=1
MASK_AUGMENTATIONS=5
MAX_ITEM=10
BATCH_SIZE=1
GRADIENT_CLIP_VAL=0.0
DEVICE="cpu"
DEVICES=1

USE_COSINE_SCHEDULE=true
WARMUP_STEPS=5

# LLM annotator: 0-indexed id=18  (annotator 19 in bundle, 19-1=18)
LLM_ANNOTATOR_ID=18
HUMAN_OBSERVED_RATE=0.2

# ── Derived flags ──────────────────────────────────────────────────────────────
cosine_flags=""
if [ "$USE_COSINE_SCHEDULE" == "true" ]; then
    cosine_flags="--use-cosine-schedule --warmup-steps $WARMUP_STEPS"
fi

# ── Run ────────────────────────────────────────────────────────────────────────
echo "[Marformer] Starting training..."

python imputer/run_imputer.py \
    --data-dir $DATA_DIR \
    --run-name $RUN_NAME \
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
    --devices $DEVICES \
    --save-model-every 5 \
    --llm-annotator-id $LLM_ANNOTATOR_ID \
    --human-observed-rate $HUMAN_OBSERVED_RATE \
    $cosine_flags

echo ""
echo "Done: OUTPUT/IMPUTER/${RUN_NAME}"
echo ""
