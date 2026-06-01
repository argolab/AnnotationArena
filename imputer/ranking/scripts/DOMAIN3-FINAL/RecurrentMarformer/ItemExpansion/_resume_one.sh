#!/bin/bash
# Resume one Recurrent Marformer run to a higher epoch count.
# Set before sourcing: RUN_TAG, PRELUDE_DEPTH, NUM_CORE_LAYERS, NUM_RECURRENCE, CODA_DEPTH
#
# Optional: OUTPUT_ROOT, EPOCHS, DEVICE, RESUME_CHECKPOINT (default latest = highest epoch periodic/best)

: "${RUN_TAG:?RUN_TAG required}"
: "${PRELUDE_DEPTH:?PRELUDE_DEPTH required}"
: "${NUM_CORE_LAYERS:?NUM_CORE_LAYERS required}"
: "${NUM_RECURRENCE:?NUM_RECURRENCE required}"
: "${CODA_DEPTH:?CODA_DEPTH required}"

export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-DATA/DOMAIN3-OLD_Item_T_1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12}"
RUN_NAME="DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_${RUN_TAG}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-latest}"

SEED="${SEED:-42}"
TYPE_EMBEDDING_INIT="${TYPE_EMBEDDING_INIT:-kaiming}"
EMBEDDING_DIM="${EMBEDDING_DIM:-80}"
ATTENTION_HEADS="${ATTENTION_HEADS:-4}"
D_FF="${D_FF:-128}"
NUM_FFN_LAYERS="${NUM_FFN_LAYERS:-1}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-600}"
LR="${LR:-2e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-none}"
LR_MIN="${LR_MIN:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MASKING_RATE="${MASKING_RATE:-0.15}"
MASK_AUGMENTATIONS="${MASK_AUGMENTATIONS:-5}"
MASKED_LOSS_WEIGHT="${MASKED_LOSS_WEIGHT:-15.0}"
OBSERVED_LOSS_WEIGHT="${OBSERVED_LOSS_WEIGHT:-1.0}"
DEVICE="${DEVICE:-cuda}"
MAX_ITEM="${MAX_ITEM:-100}"
ANNOTATOR_REG_WEIGHT="${ANNOTATOR_REG_WEIGHT:-0.0}"
ITEM_DROPOUT_RATE="${ITEM_DROPOUT_RATE:-1.0}"
ANNOTATOR_DROPOUT_RATE="${ANNOTATOR_DROPOUT_RATE:-0.0}"
ITEM_REG_WEIGHT="${ITEM_REG_WEIGHT:-0.0}"
ATTRIBUTE_REG_WEIGHT="${ATTRIBUTE_REG_WEIGHT:-0.0}"
USE_PER_HEAD_REL="${USE_PER_HEAD_REL:-false}"
SCALE_SHARED_REL="${SCALE_SHARED_REL:-true}"
USE_POINTER="${USE_POINTER:-true}"
USE_REL_VALUE="${USE_REL_VALUE:-false}"
USE_ADDONE_ATTN="${USE_ADDONE_ATTN:-false}"
USE_DEVIATION_NORM="${USE_DEVIATION_NORM:-false}"
USE_GRAPH_MASK="${USE_GRAPH_MASK:-false}"
LLM_INPUT_DIST="${LLM_INPUT_DIST:-false}"

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: run dir not found: $RUN_DIR" >&2
    exit 1
fi
CKPT_DIR="${RUN_DIR}/checkpoints"
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: no checkpoints dir: $CKPT_DIR" >&2
    exit 1
fi
if [ "$RESUME_CHECKPOINT" = "latest" ] || [ "$RESUME_CHECKPOINT" = "last" ]; then
    shopt -s nullglob
    _periodic=( "${CKPT_DIR}"/periodic-epoch=*.ckpt )
    _best=( "${CKPT_DIR}"/best-*.ckpt )
    shopt -u nullglob
    if [ "${#_periodic[@]}" -eq 0 ] && [ "${#_best[@]}" -eq 0 ]; then
        echo "ERROR: no numbered checkpoints under $CKPT_DIR" >&2
        exit 1
    fi
elif [ ! -f "${CKPT_DIR}/${RESUME_CHECKPOINT}" ] && [ ! -f "${CKPT_DIR}/${RESUME_CHECKPOINT}.ckpt" ]; then
    echo "ERROR: checkpoint not found: ${RESUME_CHECKPOINT} in $CKPT_DIR" >&2
    exit 1
fi

PER_HEAD_FLAG="";      [ "$USE_PER_HEAD_REL"  = "false" ] && PER_HEAD_FLAG="--no-per-head-rel"
SCALE_FLAG="";         [ "$SCALE_SHARED_REL"  = "true"  ] && SCALE_FLAG="--scale-shared-rel"
POINTER_FLAG="";       [ "$USE_POINTER"        = "true"  ] && POINTER_FLAG="--use-pointer"
REL_VALUE_FLAG="";     [ "$USE_REL_VALUE"      = "true"  ] && REL_VALUE_FLAG="--use-rel-value"
ADDONE_FLAG="";        [ "$USE_ADDONE_ATTN"    = "true"  ] && ADDONE_FLAG="--use-addone-attn"
DEVNORM_FLAG="";       [ "$USE_DEVIATION_NORM" = "true"  ] && DEVNORM_FLAG="--use-deviation-norm"
GRAPHMASK_FLAG="";     [ "$USE_GRAPH_MASK"     = "true"  ] && GRAPHMASK_FLAG="--use-graph-mask"
LLM_DIST_FLAG="";      [ "$LLM_INPUT_DIST"     = "true"  ] && LLM_DIST_FLAG="--llm-input-dist"

EFF_DEPTH=$(( PRELUDE_DEPTH + NUM_CORE_LAYERS * NUM_RECURRENCE + CODA_DEPTH ))

echo ""
echo "============================================================"
echo " Resume RecurrentMarformer | ${RUN_TAG} -> ${EPOCHS} epochs"
echo " RUN_DIR     : ${RUN_DIR}"
echo " Checkpoint  : ${RESUME_CHECKPOINT}"
echo " Tuple       : prelude=${PRELUDE_DEPTH} core=${NUM_CORE_LAYERS} x${NUM_RECURRENCE} coda=${CODA_DEPTH}"
echo "============================================================"

python -u -m imputer.entity_mf.recurrent.train \
    --resume \
    --resume-checkpoint      "$RESUME_CHECKPOINT"      \
    --data-dir               "${DATA_DIR}"             \
    --run-name               "${RUN_NAME}"             \
    --output-root            "${OUTPUT_ROOT}"          \
    --seed                   "$SEED"                   \
    --embedding-dim          "$EMBEDDING_DIM"          \
    --prelude-depth          "$PRELUDE_DEPTH"          \
    --num-core-layers        "$NUM_CORE_LAYERS"        \
    --num-recurrence         "$NUM_RECURRENCE"         \
    --coda-depth             "$CODA_DEPTH"             \
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
    --transductive-learning                           \
    $PER_HEAD_FLAG                                     \
    $SCALE_FLAG                                        \
    $POINTER_FLAG                                      \
    $REL_VALUE_FLAG                                    \
    $ADDONE_FLAG                                       \
    $DEVNORM_FLAG                                      \
    $GRAPHMASK_FLAG                                    \
    $LLM_DIST_FLAG

echo " Resumed run finished: ${RUN_DIR}"
