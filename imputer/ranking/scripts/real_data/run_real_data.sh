#!/bin/bash
# Run Marformer on LLMRubric real data.
#
# Two variants (controlled by BUNDLE):
#   hard  → OUTPUT/generated_data/llm_rubric       (hard labels, one-hot CE)
#   dist  → OUTPUT/generated_data/llm_rubric_dist  (soft labels, LLM distributions)
#
# Usage (from Marformer repo root):
#   DEBUG=1 bash scripts/real_data/run_real_data.sh              # hard labels
#   DEBUG=1 BUNDLE=dist bash scripts/real_data/run_real_data.sh  # soft labels
#
# Stan + visualize steps are intentionally omitted for now.

set -e

# ── Bundle variant ────────────────────────────────────────────────────────────
BUNDLE="${BUNDLE:-hard}"

if [ "$BUNDLE" == "dist" ]; then
    DATA_DIR="OUTPUT/generated_data/llm_rubric_dist"
    RUN_NAME="TEST"
else
    DATA_DIR="OUTPUT/generated_data/llm_rubric"
    RUN_NAME="llm_rubric_marformer"
fi

echo ""
echo "=========================================="
echo "LLMRubric Real Data — Marformer"
echo "  bundle:   $BUNDLE"
echo "  data_dir: $DATA_DIR"
echo "  run_name: $RUN_NAME"
echo "=========================================="
echo ""

# ── Marformer hyperparameters (identical to all_datagen_marformer.sh) ─────────
EMBEDDING_DIM=72
W_INIT="identity" # Other options: xavier, random
ENCODER_LAYERS=4
ATTENTION_HEADS=4
NUM_FFN_LAYERS=1
D_FF=128
DROPOUT=0.1
EMBEDDING_DROPOUT=0.7
WEIGHT_DECAY=0.01

EPOCHS=180
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

USE_COSINE_SCHEDULE=true
WARMUP_STEPS=5

# ── Derived flags ─────────────────────────────────────────────────────────────
cosine_flags=""
if [ "$USE_COSINE_SCHEDULE" == "true" ]; then
    cosine_flags="--use-cosine-schedule --warmup-steps $WARMUP_STEPS"
fi

# ── Run ───────────────────────────────────────────────────────────────────────
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
    --llm-annotator-id 24 \
    --human-observed-rate 0.2 \
    --item-embedding-dropout $EMBEDDING_DROPOUT \
    --w-init $W_INIT \
    --loss-fn ce \
    --llm-input-dist \
    $cosine_flags

echo ""
echo "✓ Done: OUTPUT/IMPUTER/${RUN_NAME}"
echo ""
