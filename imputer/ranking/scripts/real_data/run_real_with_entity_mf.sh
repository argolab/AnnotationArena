#!/bin/bash
# Run Stan baseline, original Marformer, and Entity Marformer on the same LLMRubric bundle.
#
# Usage (from repo root):
#   bash scripts/real_data/run_real_with_entity_mf.sh           # hard labels (default)
#   BUNDLE=dist bash scripts/real_data/run_real_with_entity_mf.sh  # soft distribution bundle
#
# Each section can be individually commented out if you only want to re-run one part.
# Stan and original Marformer sections are commented out by default — uncomment to enable.

set -e

# ── Bundle variant ───────────────────────────────────────────────────────────────
BUNDLE="${BUNDLE:-hard}"

if [ "$BUNDLE" == "dist" ]; then
    DATA_DIR="OUTPUT/generated_data/llm_rubric_dist"
else
    DATA_DIR="OUTPUT/generated_data/llm_rubric"
fi

# ── Architecture flags ───────────────────────────────────────────────────────────
# Attention design
USE_PER_HEAD_REL=true     # true = per-head relational bias (new); false = old shared-bias design
USE_POINTER=true          # K_aug obs-obs shared-identity pointer bias (like old Marformer)
USE_REL_VALUE=true        # Relation-specific value augmentation V_{ij} = V(x_j) + sum_r e_r * edge_mask[i,j,r]
USE_ADDONE_ATTN=true      # Add-one attention: attn = exp(s)/(1+sum exp(s)), sum<=1

# Soft distribution input (only meaningful when BUNDLE=dist)
USE_LLM_INPUT_DIST=true  # true = encode observed LLM ratings as log-probabilities; false = hard one-hot spike

# ── Model hyperparameters ────────────────────────────────────────────────────────
EMBEDDING_DIM=80          # Total model dim (feature_dim + param_dim). Must be divisible by ATTENTION_HEADS.
NUM_LAYERS=4              # Number of transformer encoder layers
ATTENTION_HEADS=4         # Number of attention heads. head_dim = EMBEDDING_DIM/ATTENTION_HEADS = 20
D_FF=128                  # Feed-forward hidden dimension
NUM_FFN_LAYERS=2          # Number of MLP layers inside each FFN block
DROPOUT=0.1               # Dropout rate (attention + FFN)
ITEM_DROPOUT_RATE=0.7     # Probability of dropping item deviation during training (1.0 = always drop; items are unseen at test time)

# ── Training hyperparameters ─────────────────────────────────────────────────────
EPOCHS=300
LR=2e-4
WEIGHT_DECAY=0.01
MASKING_RATE=0.15         # Fraction of observed vars masked per training step
MASK_AUGMENTATIONS=5      # Number of independent masking draws per epoch (training steps per epoch)
MASKED_LOSS_WEIGHT=15.0   # Weight on masked-token CE in training objective
OBSERVED_LOSS_WEIGHT=1.0  # Weight on observed-token CE in training objective
DEVICE="cpu"

# ── Data / evaluation settings ───────────────────────────────────────────────────
LLM_ANNOTATOR_ID=24       # 0-indexed annotator ID of the LLM (used for structured masking)
HUMAN_OBSERVED_RATE=0   # Fraction of human annotations kept as observed during training
MAX_ITEM=10               # Max items per forward pass (item chunking to limit graph size)

# ── Run name: encode key hyperparams for easy identification ─────────────────────
# Format: entity_mf_<layers>L<heads>H_emb<dim>_ff<dff>_<epochs>ep
#         _msk<masking_rate>_aug<augmentations>_wt<masked_wt>obs<obs_wt>
#         _itemdrop<item_dropout>_drop<dropout>
#         _<flags>_<bundle>
FLAGS=""
[ "$USE_PER_HEAD_REL" = "true"  ] && FLAGS="${FLAGS}perhead"  || FLAGS="${FLAGS}shared"
[ "$USE_POINTER"      = "true"  ] && FLAGS="${FLAGS}_ptr"
[ "$USE_REL_VALUE"    = "true"  ] && FLAGS="${FLAGS}_relv"
[ "$USE_ADDONE_ATTN"  = "true"  ] && FLAGS="${FLAGS}_addone"
[ "$USE_LLM_INPUT_DIST" = "true" ] && FLAGS="${FLAGS}_softinput"

RUN_NAME="entity_mf_${NUM_LAYERS}L${ATTENTION_HEADS}H_emb${EMBEDDING_DIM}_ff${D_FF}_${EPOCHS}ep"
RUN_NAME="${RUN_NAME}_msk${MASKING_RATE}_aug${MASK_AUGMENTATIONS}_wt${MASKED_LOSS_WEIGHT}obs${OBSERVED_LOSS_WEIGHT}"
RUN_NAME="${RUN_NAME}_itemdrop${ITEM_DROPOUT_RATE}_drop${DROPOUT}"
RUN_NAME="${RUN_NAME}_${FLAGS}_${BUNDLE}"

echo ""
echo "============================================================"
echo " Real-data run — Entity Marformer"
echo "  bundle:       $BUNDLE"
echo "  data_dir:     $DATA_DIR"
echo "  run_name:     $RUN_NAME"
echo "  model:        ${NUM_LAYERS}L ${ATTENTION_HEADS}H emb=${EMBEDDING_DIM} dff=${D_FF}"
echo "  training:     ${EPOCHS}ep lr=${LR} wd=${WEIGHT_DECAY}"
echo "  masking:      rate=${MASKING_RATE} aug=${MASK_AUGMENTATIONS}"
echo "  loss weights: masked=${MASKED_LOSS_WEIGHT} observed=${OBSERVED_LOSS_WEIGHT}"
echo "  dropout:      model=${DROPOUT} item=${ITEM_DROPOUT_RATE}"
echo "  flags:        per_head_rel=${USE_PER_HEAD_REL} pointer=${USE_POINTER} rel_value=${USE_REL_VALUE} addone=${USE_ADDONE_ATTN}"
echo "  llm_input_dist: ${USE_LLM_INPUT_DIST}"
echo "============================================================"
echo ""

# ── 1) Stan baseline ─────────────────────────────────────────────────────────────
echo "[1/3] Stan baseline..."
# DATASET="llmrubric" BUNDLE="$BUNDLE" bash scripts/real_data/run_stan_real.sh
echo "  (skipped — uncomment above line to run)"

# ── 2) Original Marformer ────────────────────────────────────────────────────────
echo ""
echo "[2/3] Original Marformer imputer..."
# BUNDLE="$BUNDLE" bash scripts/real_data/run_real_data.sh
echo "  (skipped — uncomment above line to run)"

# ── 3) Entity Marformer ──────────────────────────────────────────────────────────
echo ""
echo "[3/3] Entity Marformer (python -m imputer.entity_mf.train)..."

# Build boolean flags (only pass if true)
PER_HEAD_REL_FLAG=""
POINTER_FLAG=""
REL_VALUE_FLAG=""
ADDONE_ATTN_FLAG=""
LLM_INPUT_DIST_FLAG=""

[ "$USE_PER_HEAD_REL"    = "true" ] && PER_HEAD_REL_FLAG="--use-per-head-rel"
[ "$USE_PER_HEAD_REL"    = "false" ] && PER_HEAD_REL_FLAG="--no-per-head-rel"
[ "$USE_POINTER"         = "true" ] && POINTER_FLAG="--use-pointer"
[ "$USE_REL_VALUE"       = "true" ] && REL_VALUE_FLAG="--use-rel-value"
[ "$USE_ADDONE_ATTN"     = "true" ] && ADDONE_ATTN_FLAG="--use-addone-attn"
[ "$USE_LLM_INPUT_DIST"  = "true" ] && LLM_INPUT_DIST_FLAG="--llm-input-dist"

python -m imputer.entity_mf.train \
    --data-dir            "$DATA_DIR"                \
    --run-name            "$RUN_NAME"                \
    --output-root         "OUTPUT/ENTITY_MF"         \
    \
    --embedding-dim       "$EMBEDDING_DIM"           \
    --num-layers          "$NUM_LAYERS"              \
    --attention-heads     "$ATTENTION_HEADS"         \
    --d-ff                "$D_FF"                    \
    --num-ffn-layers      "$NUM_FFN_LAYERS"          \
    --dropout             "$DROPOUT"                 \
    --item-dropout-rate   "$ITEM_DROPOUT_RATE"       \
    \
    --epochs              "$EPOCHS"                  \
    --lr                  "$LR"                      \
    --weight-decay        "$WEIGHT_DECAY"            \
    --masking-rate        "$MASKING_RATE"            \
    --mask-augmentations  "$MASK_AUGMENTATIONS"      \
    --masked-loss-weight  "$MASKED_LOSS_WEIGHT"      \
    --observed-loss-weight "$OBSERVED_LOSS_WEIGHT"   \
    \
    --device              "$DEVICE"                  \
    --llm-annotator-id    "$LLM_ANNOTATOR_ID"        \
    --human-observed-rate "$HUMAN_OBSERVED_RATE"     \
    --max-item            "$MAX_ITEM"                \
    \
    $PER_HEAD_REL_FLAG   \
    $POINTER_FLAG        \
    $REL_VALUE_FLAG      \
    $ADDONE_ATTN_FLAG    \
    $LLM_INPUT_DIST_FLAG \
    --overwrite-existing-data

echo ""
echo "Entity Marformer training complete."
echo "Output written to: OUTPUT/ENTITY_MF/$RUN_NAME"
echo "TensorBoard logs:  OUTPUT/ENTITY_MF/$RUN_NAME/lightning_logs"
echo ""
