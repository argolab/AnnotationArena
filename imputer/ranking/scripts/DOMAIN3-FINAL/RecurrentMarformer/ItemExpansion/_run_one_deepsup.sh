#!/bin/bash
# Train one Recurrent Marformer run with deep supervision (coda_depth=0 only).
#
# Required: RUN_TAG, PRELUDE_DEPTH, NUM_CORE_LAYERS, NUM_RECURRENCE
# Optional: DEEP_SUPERVISION_SCHEDULE (default exp_decay), DEEP_SUPERVISION_EXP_BASE (default 1.12)
# Plus overrides from _run_one.sh (DATA_DIR, OUTPUT_ROOT, EPOCHS, MAX_ITEM, ...)

: "${RUN_TAG:?RUN_TAG required}"
: "${PRELUDE_DEPTH:?PRELUDE_DEPTH required}"
: "${NUM_CORE_LAYERS:?NUM_CORE_LAYERS required}"
: "${NUM_RECURRENCE:?NUM_RECURRENCE required}"

CODA_DEPTH=0
if [ -n "${CODA_DEPTH_OVERRIDE:-}" ] && [ "${CODA_DEPTH_OVERRIDE}" != "0" ]; then
    echo "ERROR: deep supervision requires coda_depth=0 (got CODA_DEPTH_OVERRIDE=${CODA_DEPTH_OVERRIDE})" >&2
    exit 1
fi

export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-DATA/DOMAIN3-OLD_Item_T_1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-DEEPSUP-MAXITEM500}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-_DS_M500}"
if [ -n "${RUN_NAME:-}" ]; then
    :
else
    RUN_NAME="DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_${RUN_TAG}${RUN_NAME_SUFFIX}"
fi

DEEP_SUPERVISION_SCHEDULE="${DEEP_SUPERVISION_SCHEDULE:-exp_decay}"
DEEP_SUPERVISION_EXP_BASE="${DEEP_SUPERVISION_EXP_BASE:-1.12}"

SEED="${SEED:-42}"
TYPE_EMBEDDING_INIT="${TYPE_EMBEDDING_INIT:-kaiming}"
EMBEDDING_DIM="${EMBEDDING_DIM:-80}"
ATTENTION_HEADS="${ATTENTION_HEADS:-4}"
D_FF="${D_FF:-128}"
NUM_FFN_LAYERS="${NUM_FFN_LAYERS:-1}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-400}"
LR="${LR:-2e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-none}"
LR_MIN="${LR_MIN:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MASKING_RATE="${MASKING_RATE:-0.15}"
MASK_AUGMENTATIONS="${MASK_AUGMENTATIONS:-5}"
MASKED_LOSS_WEIGHT="${MASKED_LOSS_WEIGHT:-15.0}"
OBSERVED_LOSS_WEIGHT="${OBSERVED_LOSS_WEIGHT:-1.0}"
DEVICE="${DEVICE:-cuda}"
MAX_ITEM="${MAX_ITEM:-500}"
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
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-true}"

PER_HEAD_FLAG="";      [ "$USE_PER_HEAD_REL"  = "false" ] && PER_HEAD_FLAG="--no-per-head-rel"
SCALE_FLAG="";         [ "$SCALE_SHARED_REL"  = "true"  ] && SCALE_FLAG="--scale-shared-rel"
POINTER_FLAG="";       [ "$USE_POINTER"        = "true"  ] && POINTER_FLAG="--use-pointer"
REL_VALUE_FLAG="";     [ "$USE_REL_VALUE"      = "true"  ] && REL_VALUE_FLAG="--use-rel-value"
ADDONE_FLAG="";        [ "$USE_ADDONE_ATTN"    = "true"  ] && ADDONE_FLAG="--use-addone-attn"
DEVNORM_FLAG="";       [ "$USE_DEVIATION_NORM" = "true"  ] && DEVNORM_FLAG="--use-deviation-norm"
GRAPHMASK_FLAG="";     [ "$USE_GRAPH_MASK"     = "true"  ] && GRAPHMASK_FLAG="--use-graph-mask"
LLM_DIST_FLAG="";      [ "$LLM_INPUT_DIST"     = "true"  ] && LLM_DIST_FLAG="--llm-input-dist"
OVERWRITE_FLAG="";     [ "$OVERWRITE_EXISTING" = "true"  ] && OVERWRITE_FLAG="--overwrite-existing-data"

UNIQUE=$(( PRELUDE_DEPTH + NUM_CORE_LAYERS + CODA_DEPTH ))
EFF_DEPTH=$(( PRELUDE_DEPTH + NUM_CORE_LAYERS * NUM_RECURRENCE + CODA_DEPTH ))

echo ""
echo "============================================================"
echo " RecurrentMarformer DEEPSUP | ${RUN_TAG}"
echo " unique=${UNIQUE} eff_depth=${EFF_DEPTH} r=${NUM_RECURRENCE} coda=0"
echo " schedule=${DEEP_SUPERVISION_SCHEDULE} exp_base=${DEEP_SUPERVISION_EXP_BASE}"
echo " DATA_DIR=${DATA_DIR}  MAX_ITEM=${MAX_ITEM}"
echo " RUN_NAME=${RUN_NAME}"
echo " RUN_DIR=${OUTPUT_ROOT}/${RUN_NAME}"
echo "============================================================"

if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

python -u -m imputer.entity_mf.recurrent.train \
    --data-dir               "${DATA_DIR}"             \
    --run-name               "${RUN_NAME}"             \
    --output-root            "${OUTPUT_ROOT}"          \
    --seed                   "$SEED"                   \
    --embedding-dim          "$EMBEDDING_DIM"          \
    --prelude-depth          "$PRELUDE_DEPTH"          \
    --num-core-layers        "$NUM_CORE_LAYERS"        \
    --num-recurrence         "$NUM_RECURRENCE"         \
    --coda-depth             "$CODA_DEPTH"             \
    --deep-supervision                                 \
    --deep-supervision-schedule "$DEEP_SUPERVISION_SCHEDULE" \
    --deep-supervision-exp-base "$DEEP_SUPERVISION_EXP_BASE" \
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
    $LLM_DIST_FLAG                                     \
    $OVERWRITE_FLAG
status=$?
if [ "$status" -ne 0 ]; then
    echo "ERROR: training failed for ${OUTPUT_ROOT}/${RUN_NAME} (exit ${status})" >&2
    return "$status" 2>/dev/null || exit "$status"
fi

echo " Finished: ${OUTPUT_ROOT}/${RUN_NAME}"
