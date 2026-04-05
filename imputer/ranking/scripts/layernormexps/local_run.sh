DATA_DIR="OUTPUT/generated_data/llm_rubric_dist"
OUTPUT_ROOT="OUTPUT/ENTITY_MF/LAYERNORM_EXPS"
BUNDLE="dist"

# ── Fixed hyperparams ─────────────────────────────────────────────────────────
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
MASKING_RATE=0.15
MASK_AUGMENTATIONS=5
MASKED_LOSS_WEIGHT=15.0
OBSERVED_LOSS_WEIGHT=1.0
DEVICE="cuda"
LLM_ANNOTATOR_ID=24
HUMAN_OBSERVED_RATE=0.0
MAX_ITEM=0
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

# ── Build CLI flags ───────────────────────────────────────────────────────────
PER_HEAD_FLAG="";     [ "$USE_PER_HEAD_REL"   = "false" ] && PER_HEAD_FLAG="--no-per-head-rel"
SCALE_FLAG="";        [ "$SCALE_SHARED_REL"   = "true"  ] && SCALE_FLAG="--scale-shared-rel"
POINTER_FLAG="";      [ "$USE_POINTER"         = "true"  ] && POINTER_FLAG="--use-pointer"
REL_VALUE_FLAG="";    [ "$USE_REL_VALUE"       = "true"  ] && REL_VALUE_FLAG="--use-rel-value"
ADDONE_FLAG="";       [ "$USE_ADDONE_ATTN"     = "true"  ] && ADDONE_FLAG="--use-addone-attn"
DEVNORM_FLAG="";      [ "$USE_DEVIATION_NORM"  = "true"  ] && DEVNORM_FLAG="--use-deviation-norm"
GRAPHMASK_FLAG="";    [ "$USE_GRAPH_MASK"      = "true"  ] && GRAPHMASK_FLAG="--use-graph-mask"

# ── Run name base ─────────────────────────────────────────────────────────────
EXP_LABEL="pointer_graphmask_0"
RUN_BASE="lnexp_${EXP_LABEL}_${NUM_LAYERS}L${ATTENTION_HEADS}H_emb${EMBEDDING_DIM}_${EPOCHS}ep"
RUN_BASE="${RUN_BASE}_itemdrop${ITEM_DROPOUT_RATE}_ireg${ITEM_REG_WEIGHT}_areg${ATTRIBUTE_REG_WEIGHT}_${BUNDLE}"

echo ""
echo "============================================================"
echo " Exp: Base + Pointer + Graph Mask (3 runs)"
echo "  itemdrop=${ITEM_DROPOUT_RATE}  ireg=${ITEM_REG_WEIGHT}  areg=${ATTRIBUTE_REG_WEIGHT}"
echo "  flags: $PER_HEAD_FLAG $SCALE_FLAG $POINTER_FLAG $REL_VALUE_FLAG $ADDONE_FLAG $DEVNORM_FLAG $GRAPHMASK_FLAG"
echo "============================================================"

# ── Helper ────────────────────────────────────────────────────────────────────
_train() {
    local N=$1
    local SEED=$2
    echo ""; echo "--- Run ${N}/3: ${RUN_BASE}_run${N} (seed=${SEED}) ---"; echo ""
    python -m imputer.entity_mf.train \
        --data-dir             "$DATA_DIR"                  \
        --run-name             "${RUN_BASE}_run${N}"        \
        --output-root          "$OUTPUT_ROOT"               \
        --embedding-dim        "$EMBEDDING_DIM"             \
        --num-layers           "$NUM_LAYERS"                \
        --attention-heads      "$ATTENTION_HEADS"           \
        --d-ff                 "$D_FF"                      \
        --num-ffn-layers       "$NUM_FFN_LAYERS"            \
        --dropout              "$DROPOUT"                   \
        --item-dropout-rate    "$ITEM_DROPOUT_RATE"         \
        --epochs               "$EPOCHS"                    \
        --lr                   "$LR"                        \
        --lr-schedule          "$LR_SCHEDULE"               \
        --lr-min               "$LR_MIN"                    \
        --weight-decay         "$WEIGHT_DECAY"              \
        --masking-rate         "$MASKING_RATE"              \
        --mask-augmentations   "$MASK_AUGMENTATIONS"        \
        --masked-loss-weight   "$MASKED_LOSS_WEIGHT"        \
        --observed-loss-weight "$OBSERVED_LOSS_WEIGHT"      \
        --device               "$DEVICE"                    \
        --llm-annotator-id     "$LLM_ANNOTATOR_ID"          \
        --human-observed-rate  "$HUMAN_OBSERVED_RATE"       \
        --max-item             "$MAX_ITEM"                  \
        --type-embedding-init  "$TYPE_EMBEDDING_INIT"       \
        --item-reg-weight      "$ITEM_REG_WEIGHT"           \
        --attribute-reg-weight "$ATTRIBUTE_REG_WEIGHT"      \
        --annotator-reg-weight "$ANNOTATOR_REG_WEIGHT"      \
        $PER_HEAD_FLAG                                      \
        $SCALE_FLAG                                         \
        $POINTER_FLAG                                       \
        $REL_VALUE_FLAG                                     \
        $ADDONE_FLAG                                        \
        $DEVNORM_FLAG                                       \
        $GRAPHMASK_FLAG                                     \
        --llm-input-dist                                    \
        --seed                 "$SEED"                      \
        --overwrite-existing-data
}

_train 1 42