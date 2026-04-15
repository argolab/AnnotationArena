#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.
#
# STAN_sparse — Marformer with hard graph attention mask (--use-graph-mask).
# Same data layout as run_train.sh in this directory; distinct RUN_NAME for outputs.
#
# Example (cwd must be imputer/ranking):
#   cd /path/to/AA_new/imputer/ranking
#   PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
#     /home/xwang397/bin/sbatch_adapt \
#     scripts/STAN_sparse/MARFORMER/Factor_225_25_9_ItemTest/Factor_225_25_9_ItemTest_Size_175/run_train_hard_mask.sh

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_LEAF="$(basename "${_SCRIPT_DIR}")"
_PARENT="$(basename "$(dirname "${_SCRIPT_DIR}")")"
_RANKING_ROOT="$(cd "${_SCRIPT_DIR}/../../../../.." && pwd)"
cd "${_RANKING_ROOT}"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

SCRIPT_START=$SECONDS

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT="DATA/STAN_sparse/${_PARENT}/${_LEAF}"
OUTPUT_ROOT="RESULTS/MARFORMER/STAN_sparse"
RUN_NAME="${_LEAF}_hard_mask"

# ── Fixed hyperparams (aligned with SUMMEVAL MARFORMER) ─────────────────────────
SEED=42
TYPE_EMBEDDING_INIT="kaiming"
EMBEDDING_DIM=80
NUM_LAYERS=8
ATTENTION_HEADS=4
D_FF=128
NUM_FFN_LAYERS=1
DROPOUT=0.1
EPOCHS=200
LR=2e-4
LR_SCHEDULE="none"
LR_MIN=1e-5
WEIGHT_DECAY=0.01
MASKING_RATE=0.15
MASK_AUGMENTATIONS=5
MASKED_LOSS_WEIGHT=15.0
OBSERVED_LOSS_WEIGHT=1.0
DEVICE="cuda"
MAX_ITEM=10
ANNOTATOR_REG_WEIGHT=0.0

# ── Experiment-specific flags ─────────────────────────────────────────────────
ITEM_DROPOUT_RATE=0.7
ITEM_REG_WEIGHT=0.0
ATTRIBUTE_REG_WEIGHT=0.0
USE_PER_HEAD_REL=false
SCALE_SHARED_REL=true
USE_POINTER=true
USE_REL_VALUE=false
USE_ADDONE_ATTN=false
USE_DEVIATION_NORM=false
USE_GRAPH_MASK=true
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
echo " STAN_sparse | Marformer (hard mask) | ${_PARENT} / ${_LEAF}"
echo "  DATA_DIR     : ${DATA_ROOT}"
echo "  OUTPUT_ROOT  : ${OUTPUT_ROOT}"
echo "  RUN_NAME     : ${RUN_NAME}"
echo "  USE_GRAPH_MASK : true"
echo "============================================================"

python -u -m imputer.entity_mf.train \
    --data-dir             "${DATA_ROOT}"   \
    --run-name             "${RUN_NAME}"    \
    --output-root          "${OUTPUT_ROOT}" \
    --seed                 "$SEED"                   \
    --embedding-dim        "$EMBEDDING_DIM"          \
    --num-layers           "$NUM_LAYERS"             \
    --attention-heads      "$ATTENTION_HEADS"        \
    --d-ff                 "$D_FF"                   \
    --num-ffn-layers       "$NUM_FFN_LAYERS"         \
    --dropout              "$DROPOUT"                \
    --item-dropout-rate    "$ITEM_DROPOUT_RATE"      \
    --epochs               "$EPOCHS"                 \
    --lr                   "$LR"                     \
    --lr-schedule          "$LR_SCHEDULE"            \
    --lr-min               "$LR_MIN"                 \
    --weight-decay         "$WEIGHT_DECAY"           \
    --masking-rate         "$MASKING_RATE"           \
    --mask-augmentations   "$MASK_AUGMENTATIONS"     \
    --masked-loss-weight   "$MASKED_LOSS_WEIGHT"     \
    --observed-loss-weight "$OBSERVED_LOSS_WEIGHT"   \
    --device               "$DEVICE"                 \
    --max-item             "$MAX_ITEM"               \
    --type-embedding-init  "$TYPE_EMBEDDING_INIT"    \
    --item-reg-weight      "$ITEM_REG_WEIGHT"        \
    --attribute-reg-weight "$ATTRIBUTE_REG_WEIGHT"   \
    --annotator-reg-weight "$ANNOTATOR_REG_WEIGHT"   \
    $PER_HEAD_FLAG                                   \
    $SCALE_FLAG                                      \
    $POINTER_FLAG                                    \
    $REL_VALUE_FLAG                                  \
    $ADDONE_FLAG                                     \
    $DEVNORM_FLAG                                    \
    $GRAPHMASK_FLAG                                  \
    $LLM_DIST_FLAG                                   \
    $OVERWRITE_FLAG

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Output: ${OUTPUT_ROOT}/${RUN_NAME}"
echo "============================================================"
