#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=EMF3_ItemReg
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=18GB
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1
conda activate llm_rubric_env
cd /export/fs06/psingh54/EntityMarformer/imputer/ranking
export PYTHONPATH=.
set -e

# ── Ablation 3: ITEM L2 REG ───────────────────────────────────────────────────
# Base + L2 regularization on item deviation table only (1e-3).
# Tests whether penalising item deviations toward zero reduces overfitting.

BUNDLE="dist"
DATA_DIR="OUTPUT/generated_data/llm_rubric_dist"

USE_PER_HEAD_REL=false
USE_POINTER=true
USE_REL_VALUE=true
USE_ADDONE_ATTN=true
USE_LLM_INPUT_DIST=true

TYPE_EMBEDDING_INIT="kaiming"

EMBEDDING_DIM=80
NUM_LAYERS=8
ATTENTION_HEADS=4
D_FF=128
NUM_FFN_LAYERS=1
DROPOUT=0.1
ITEM_DROPOUT_RATE=0.7

EPOCHS=300
LR=2e-4
LR_SCHEDULE="none"
LR_MIN=1e-5
WEIGHT_DECAY=0.01
MASKING_RATE=0.15
MASK_AUGMENTATIONS=5
MASKED_LOSS_WEIGHT=15.0
OBSERVED_LOSS_WEIGHT=1.0
DEVICE="cuda"

LLM_ANNOTATOR_ID=24
HUMAN_OBSERVED_RATE=0.0
MAX_ITEM=10

ITEM_REG_WEIGHT=1e-3        # <-- ablated: 1e-3 (base = 0.0)
ATTRIBUTE_REG_WEIGHT=0.0

FLAGS=""
[ "$USE_PER_HEAD_REL"   = "true" ] && FLAGS="${FLAGS}perhead" || FLAGS="${FLAGS}shared"
[ "$USE_POINTER"        = "true" ] && FLAGS="${FLAGS}_ptr"
[ "$USE_REL_VALUE"      = "true" ] && FLAGS="${FLAGS}_relv"
[ "$USE_ADDONE_ATTN"    = "true" ] && FLAGS="${FLAGS}_addone"
[ "$USE_LLM_INPUT_DIST" = "true" ] && FLAGS="${FLAGS}_softinput"

RUN_NAME="ablation3_itemreg_${NUM_LAYERS}L${ATTENTION_HEADS}H_emb${EMBEDDING_DIM}_ffn${NUM_FFN_LAYERS}_${EPOCHS}ep"
RUN_NAME="${RUN_NAME}_itemdrop${ITEM_DROPOUT_RATE}_lr${LR}_sched${LR_SCHEDULE}"
RUN_NAME="${RUN_NAME}_ireg${ITEM_REG_WEIGHT}_areg${ATTRIBUTE_REG_WEIGHT}_init${TYPE_EMBEDDING_INIT}"
RUN_NAME="${RUN_NAME}_${FLAGS}_${BUNDLE}"

echo ""
echo "============================================================"
echo " Ablation 3: ITEM L2 REG = ${ITEM_REG_WEIGHT}"
echo "  run_name:     $RUN_NAME"
echo "  model:        ${NUM_LAYERS}L ${ATTENTION_HEADS}H emb=${EMBEDDING_DIM} ffn=${NUM_FFN_LAYERS}"
echo "  lr:           ${LR}  schedule=${LR_SCHEDULE}"
echo "  reg:          item=${ITEM_REG_WEIGHT}  attr=${ATTRIBUTE_REG_WEIGHT}"
echo "  init:         ${TYPE_EMBEDDING_INIT}  deviation_norm=false"
echo "  item_dropout: ${ITEM_DROPOUT_RATE}"
echo "============================================================"
echo ""

PER_HEAD_REL_FLAG=""
POINTER_FLAG=""
REL_VALUE_FLAG=""
ADDONE_ATTN_FLAG=""
LLM_INPUT_DIST_FLAG=""

[ "$USE_PER_HEAD_REL"   = "true"  ] && PER_HEAD_REL_FLAG="--use-per-head-rel"
[ "$USE_PER_HEAD_REL"   = "false" ] && PER_HEAD_REL_FLAG="--no-per-head-rel"
[ "$USE_POINTER"        = "true"  ] && POINTER_FLAG="--use-pointer"
[ "$USE_REL_VALUE"      = "true"  ] && REL_VALUE_FLAG="--use-rel-value"
[ "$USE_ADDONE_ATTN"    = "true"  ] && ADDONE_ATTN_FLAG="--use-addone-attn"
[ "$USE_LLM_INPUT_DIST" = "true"  ] && LLM_INPUT_DIST_FLAG="--llm-input-dist"

python -m imputer.entity_mf.train \
    --data-dir             "$DATA_DIR"               \
    --run-name             "$RUN_NAME"               \
    --output-root          "OUTPUT/ENTITY_MF"        \
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
    --llm-annotator-id     "$LLM_ANNOTATOR_ID"       \
    --human-observed-rate  "$HUMAN_OBSERVED_RATE"    \
    --max-item             "$MAX_ITEM"               \
    --type-embedding-init  "$TYPE_EMBEDDING_INIT"    \
    --item-reg-weight      "$ITEM_REG_WEIGHT"        \
    --attribute-reg-weight "$ATTRIBUTE_REG_WEIGHT"   \
    $PER_HEAD_REL_FLAG   \
    $POINTER_FLAG        \
    $REL_VALUE_FLAG      \
    $ADDONE_ATTN_FLAG    \
    $LLM_INPUT_DIST_FLAG \
    --overwrite-existing-data

echo ""
echo "Training complete. Output: OUTPUT/ENTITY_MF/$RUN_NAME"
