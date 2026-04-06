#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=EMF_STAN_HP_LR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18GB
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1
conda activate llm_rubric_env
cd /export/fs06/psingh54/MARFORMER/imputer/ranking
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

SCRIPT_START=$SECONDS

# ── Data ──────────────────────────────────────────────────────────────────────
SPLIT="Factor_250_20_9_AnnotatorTest_14"
DATA_ROOT="DATA/STAN/Factor_250_20_9_AnnotatorTest/${SPLIT}"
OUTPUT_ROOT="RESULTS/STAN_HP_SEARCH"

# ── Fixed hyperparams ─────────────────────────────────────────────────────────
SEED=42
TYPE_EMBEDDING_INIT="kaiming"
EMBEDDING_DIM=80
NUM_LAYERS=8
ATTENTION_HEADS=4
D_FF=128
NUM_FFN_LAYERS=1
DROPOUT=0.1
EPOCHS=150
LR_SCHEDULE="none"
LR_MIN=1e-5
WEIGHT_DECAY=0.01
MASKING_RATE=0.35
MASK_AUGMENTATIONS=5
MASKED_LOSS_WEIGHT=15.0
OBSERVED_LOSS_WEIGHT=1.0
DEVICE="cuda"
MAX_ITEM=10
ANNOTATOR_REG_WEIGHT=0.0
ITEM_REG_WEIGHT=0.0
ATTRIBUTE_REG_WEIGHT=0.0

# ── Fixed per this sweep: item dropout off, annotator dropout 0.5 ─────────────
ITEM_DROPOUT_RATE=0.0
ANNOTATOR_DROPOUT_RATE=0.5

# ── Model flags ───────────────────────────────────────────────────────────────
USE_PER_HEAD_REL=false
SCALE_SHARED_REL=true
USE_POINTER=true
USE_REL_VALUE=false
USE_ADDONE_ATTN=false
USE_DEVIATION_NORM=false
USE_GRAPH_MASK=false
LLM_INPUT_DIST=true
OVERWRITE_EXISTING=true

PER_HEAD_FLAG="";  [ "$USE_PER_HEAD_REL"  = "false" ] && PER_HEAD_FLAG="--no-per-head-rel"
SCALE_FLAG="";     [ "$SCALE_SHARED_REL"  = "true"  ] && SCALE_FLAG="--scale-shared-rel"
POINTER_FLAG="";   [ "$USE_POINTER"        = "true"  ] && POINTER_FLAG="--use-pointer"
REL_VALUE_FLAG=""; [ "$USE_REL_VALUE"      = "true"  ] && REL_VALUE_FLAG="--use-rel-value"
ADDONE_FLAG="";    [ "$USE_ADDONE_ATTN"    = "true"  ] && ADDONE_FLAG="--use-addone-attn"
DEVNORM_FLAG="";   [ "$USE_DEVIATION_NORM" = "true"  ] && DEVNORM_FLAG="--use-deviation-norm"
GRAPHMASK_FLAG=""; [ "$USE_GRAPH_MASK"     = "true"  ] && GRAPHMASK_FLAG="--use-graph-mask"
LLM_DIST_FLAG="";  [ "$LLM_INPUT_DIST"     = "true"  ] && LLM_DIST_FLAG="--llm-input-dist"
OVERWRITE_FLAG=""; [ "$OVERWRITE_EXISTING" = "true"  ] && OVERWRITE_FLAG="--overwrite-existing-data"

echo ""
echo "============================================================"
echo " STAN HP Search | LR Sweep | Marformer"
echo " Data    : ${DATA_ROOT}"
echo " Output  : ${OUTPUT_ROOT}"
echo " Epochs  : ${EPOCHS}"
echo " item_dropout=${ITEM_DROPOUT_RATE}  ann_dropout=${ANNOTATOR_DROPOUT_RATE}"
echo "============================================================"

# ── Run 1: LR = 2e-4 ─────────────────────────────────────────────────────────
LR=2e-4
RUN_NAME="${SPLIT}_lr2e-4_35masking"
RUN_START=$SECONDS

echo ""; echo "--- Run 1: LR=${LR}  run_name=${RUN_NAME} ---"; echo ""

python -u -m imputer.entity_mf.train \
    --data-dir               "${DATA_ROOT}"              \
    --run-name               "${RUN_NAME}"               \
    --output-root            "${OUTPUT_ROOT}"            \
    --seed                   "$SEED"                     \
    --embedding-dim          "$EMBEDDING_DIM"            \
    --num-layers             "$NUM_LAYERS"               \
    --attention-heads        "$ATTENTION_HEADS"          \
    --d-ff                   "$D_FF"                     \
    --num-ffn-layers         "$NUM_FFN_LAYERS"           \
    --dropout                "$DROPOUT"                  \
    --item-dropout-rate      "$ITEM_DROPOUT_RATE"        \
    --annotator-dropout-rate "$ANNOTATOR_DROPOUT_RATE"   \
    --epochs                 "$EPOCHS"                   \
    --lr                     "$LR"                       \
    --lr-schedule            "$LR_SCHEDULE"              \
    --lr-min                 "$LR_MIN"                   \
    --weight-decay           "$WEIGHT_DECAY"             \
    --masking-rate           "$MASKING_RATE"             \
    --mask-augmentations     "$MASK_AUGMENTATIONS"       \
    --masked-loss-weight     "$MASKED_LOSS_WEIGHT"       \
    --observed-loss-weight   "$OBSERVED_LOSS_WEIGHT"     \
    --device                 "$DEVICE"                   \
    --max-item               "$MAX_ITEM"                 \
    --type-embedding-init    "$TYPE_EMBEDDING_INIT"      \
    --item-reg-weight        "$ITEM_REG_WEIGHT"          \
    --attribute-reg-weight   "$ATTRIBUTE_REG_WEIGHT"     \
    --annotator-reg-weight   "$ANNOTATOR_REG_WEIGHT"     \
    $PER_HEAD_FLAG                                       \
    $SCALE_FLAG                                          \
    $POINTER_FLAG                                        \
    $REL_VALUE_FLAG                                      \
    $ADDONE_FLAG                                         \
    $DEVNORM_FLAG                                        \
    $GRAPHMASK_FLAG                                      \
    $LLM_DIST_FLAG                                       \
    $OVERWRITE_FLAG

echo "  ↳ done in $(( (SECONDS - RUN_START) / 60 ))m $(( (SECONDS - RUN_START) % 60 ))s"

# ── Run 2: LR = 2e-3 ─────────────────────────────────────────────────────────
LR=2e-3
RUN_NAME="${SPLIT}_lr2e-3"
RUN_START=$SECONDS

echo ""; echo "--- Run 2: LR=${LR}  run_name=${RUN_NAME} ---"; echo ""

python -u -m imputer.entity_mf.train \
    --data-dir               "${DATA_ROOT}"              \
    --run-name               "${RUN_NAME}"               \
    --output-root            "${OUTPUT_ROOT}"            \
    --seed                   "$SEED"                     \
    --embedding-dim          "$EMBEDDING_DIM"            \
    --num-layers             "$NUM_LAYERS"               \
    --attention-heads        "$ATTENTION_HEADS"          \
    --d-ff                   "$D_FF"                     \
    --num-ffn-layers         "$NUM_FFN_LAYERS"           \
    --dropout                "$DROPOUT"                  \
    --item-dropout-rate      "$ITEM_DROPOUT_RATE"        \
    --annotator-dropout-rate "$ANNOTATOR_DROPOUT_RATE"   \
    --epochs                 "$EPOCHS"                   \
    --lr                     "$LR"                       \
    --lr-schedule            "$LR_SCHEDULE"              \
    --lr-min                 "$LR_MIN"                   \
    --weight-decay           "$WEIGHT_DECAY"             \
    --masking-rate           "$MASKING_RATE"             \
    --mask-augmentations     "$MASK_AUGMENTATIONS"       \
    --masked-loss-weight     "$MASKED_LOSS_WEIGHT"       \
    --observed-loss-weight   "$OBSERVED_LOSS_WEIGHT"     \
    --device                 "$DEVICE"                   \
    --max-item               "$MAX_ITEM"                 \
    --type-embedding-init    "$TYPE_EMBEDDING_INIT"      \
    --item-reg-weight        "$ITEM_REG_WEIGHT"          \
    --attribute-reg-weight   "$ATTRIBUTE_REG_WEIGHT"     \
    --annotator-reg-weight   "$ANNOTATOR_REG_WEIGHT"     \
    $PER_HEAD_FLAG                                       \
    $SCALE_FLAG                                          \
    $POINTER_FLAG                                        \
    $REL_VALUE_FLAG                                      \
    $ADDONE_FLAG                                         \
    $DEVNORM_FLAG                                        \
    $GRAPHMASK_FLAG                                      \
    $LLM_DIST_FLAG                                       \
    $OVERWRITE_FLAG

echo "  ↳ done in $(( (SECONDS - RUN_START) / 60 ))m $(( (SECONDS - RUN_START) % 60 ))s"

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " LR sweep done. Total: $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Check val log loss in:"
echo "   ${OUTPUT_ROOT}/${SPLIT}_lr2e-4/training_history.json"
echo "   ${OUTPUT_ROOT}/${SPLIT}_lr2e-3/training_history.json"
echo " Then set BEST_LR in run_hp_dropout.sh before submitting."
echo "============================================================"
