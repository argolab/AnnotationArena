#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

cd /Users/prabhavsingh/Documents/JHU/JHUResearch/EntityMarformer/imputer/ranking
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

SCRIPT_START=$SECONDS

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT="DATA/STAN/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest_3"
OUTPUT_ROOT="RESULTS/MARFORMER_ANNOT_DROP/STAN"
RUN_NAME="Factor_250_20_9_AnnotatorTest_3_TRANSDUCTIVE_MASK03"

# ── Fixed hyperparams ─────────────────────────────────────────────────────────
SEED=42
TYPE_EMBEDDING_INIT="kaiming"
EMBEDDING_DIM=80
NUM_LAYERS=8
ATTENTION_HEADS=4
D_FF=128
NUM_FFN_LAYERS=1
DROPOUT=0.1
EPOCHS=300
LR=2e-4
LR_SCHEDULE="none"
LR_MIN=1e-5
WEIGHT_DECAY=0.01
MASKING_RATE=0.3
MASK_AUGMENTATIONS=5
MASKED_LOSS_WEIGHT=15.0
OBSERVED_LOSS_WEIGHT=1.0
DEVICE="cpu"
MAX_ITEM=10
ANNOTATOR_REG_WEIGHT=0.0

# ── Experiment-specific flags ─────────────────────────────────────────────────
ITEM_DROPOUT_RATE=0.0
ANNOTATOR_DROPOUT_RATE=0.0
ITEM_REG_WEIGHT=0.0
ATTRIBUTE_REG_WEIGHT=0.0
USE_PER_HEAD_REL=false
SCALE_SHARED_REL=true
USE_POINTER=true
USE_REL_VALUE=false
USE_ADDONE_ATTN=false
USE_DEVIATION_NORM=false
USE_GRAPH_MASK=false
LLM_INPUT_DIST=true
OVERWRITE_EXISTING=true

# ── Build CLI flags ───────────────────────────────────────────────────────────
PER_HEAD_FLAG="";      [ "$USE_PER_HEAD_REL"  = "false" ] && PER_HEAD_FLAG="--no-per-head-rel"
SCALE_FLAG="";         [ "$SCALE_SHARED_REL"  = "true"  ] && SCALE_FLAG="--scale-shared-rel"
POINTER_FLAG="";       [ "$USE_POINTER"        = "true"  ] && POINTER_FLAG="--use-pointer"
REL_VALUE_FLAG="";     [ "$USE_REL_VALUE"      = "true"  ] && REL_VALUE_FLAG="--use-rel-value"
ADDONE_FLAG="";        [ "$USE_ADDONE_ATTN"    = "true"  ] && ADDONE_FLAG="--use-addone-attn"
DEVNORM_FLAG="";       [ "$USE_DEVIATION_NORM" = "true"  ] && DEVNORM_FLAG="--use-deviation-norm"
GRAPHMASK_FLAG="";     [ "$USE_GRAPH_MASK"     = "true"  ] && GRAPHMASK_FLAG="--use-graph-mask"
LLM_DIST_FLAG="";      [ "$LLM_INPUT_DIST"     = "true"  ] && LLM_DIST_FLAG="--llm-input-dist"
OVERWRITE_FLAG="";     [ "$OVERWRITE_EXISTING" = "true"  ] && OVERWRITE_FLAG="--overwrite-existing-data"

echo ""
echo "============================================================"
echo " STAN | Marformer Transductive | Factor_250_20_9_AnnotatorTest_3"
echo "  MASKING_RATE     : ${MASKING_RATE}"
echo "  ANNOTATOR_DROPOUT: ${ANNOTATOR_DROPOUT_RATE}"
echo "  OUTPUT_ROOT      : ${OUTPUT_ROOT}"
echo "  RUN_NAME         : ${RUN_NAME}"
echo "============================================================"

python -u -m imputer.entity_mf.train \
    --data-dir               "${DATA_ROOT}"            \
    --run-name               "${RUN_NAME}"             \
    --output-root            "${OUTPUT_ROOT}"          \
    --seed                   "$SEED"                   \
    --embedding-dim          "$EMBEDDING_DIM"          \
    --num-layers             "$NUM_LAYERS"             \
    --attention-heads        "$ATTENTION_HEADS"        \
    --d-ff                   "$D_FF"                   \
    --num-ffn-layers         "$NUM_FFN_LAYERS"         \
    --dropout                "$DROPOUT"                \
    --item-dropout-rate      "$ITEM_DROPOUT_RATE"      \
    --annotator-dropout-rate "$ANNOTATOR_DROPOUT_RATE" \
    --epochs                 "$EPOCHS"                 \
    --lr                     "$LR"                     \
    --lr-schedule            "$LR_SCHEDULE"            \
    --lr-min                 "$LR_MIN"                 \
    --weight-decay           "$WEIGHT_DECAY"           \
    --masking-rate           "$MASKING_RATE"           \
    --mask-augmentations     "$MASK_AUGMENTATIONS"     \
    --masked-loss-weight     "$MASKED_LOSS_WEIGHT"     \
    --observed-loss-weight   "$OBSERVED_LOSS_WEIGHT"   \
    --device                 "$DEVICE"                 \
    --max-item               "$MAX_ITEM"               \
    --type-embedding-init    "$TYPE_EMBEDDING_INIT"    \
    --item-reg-weight        "$ITEM_REG_WEIGHT"        \
    --attribute-reg-weight   "$ATTRIBUTE_REG_WEIGHT"   \
    --annotator-reg-weight   "$ANNOTATOR_REG_WEIGHT"   \
    --transductive-learning                            \
    $PER_HEAD_FLAG                                     \
    $SCALE_FLAG                                        \
    $POINTER_FLAG                                      \
    $REL_VALUE_FLAG                                    \
    $ADDONE_FLAG                                       \
    $DEVNORM_FLAG                                      \
    $GRAPHMASK_FLAG                                    \
    $LLM_DIST_FLAG                                     \
    $OVERWRITE_FLAG

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Output: ${OUTPUT_ROOT}/${RUN_NAME}"
echo "============================================================"
