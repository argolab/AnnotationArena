#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=EMF_FA_Combo
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

# ── Exp 05: Exps 1 + 2 + 4 Together (Pointer + Rel Value + Add-One) ──────────
# Shared-bias + scale. Combines pointer, rel-value, and add-one attention.
# Per-head-rel (Exp 03) is intentionally excluded. Item dropout 0.7.

DATA_DIR="OUTPUT/generated_data/llm_rubric_dist"
OUTPUT_ROOT="OUTPUT/ENTITY_MF/FINAL_ABLATIONS"
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
MAX_ITEM=10
ANNOTATOR_REG_WEIGHT=0.0

# ── Experiment-specific flags ─────────────────────────────────────────────────
ITEM_DROPOUT_RATE=0.7
ITEM_REG_WEIGHT=0.0
ATTRIBUTE_REG_WEIGHT=0.0
USE_PER_HEAD_REL=false
SCALE_SHARED_REL=true
USE_POINTER=true
USE_REL_VALUE=true
USE_ADDONE_ATTN=true
USE_DEVIATION_NORM=false

# ── Build CLI flags ───────────────────────────────────────────────────────────
PER_HEAD_FLAG="";  [ "$USE_PER_HEAD_REL"   = "false" ] && PER_HEAD_FLAG="--no-per-head-rel"
SCALE_FLAG="";     [ "$SCALE_SHARED_REL"   = "true"  ] && SCALE_FLAG="--scale-shared-rel"
POINTER_FLAG="";   [ "$USE_POINTER"        = "true"  ] && POINTER_FLAG="--use-pointer"
REL_VALUE_FLAG=""; [ "$USE_REL_VALUE"      = "true"  ] && REL_VALUE_FLAG="--use-rel-value"
ADDONE_FLAG="";    [ "$USE_ADDONE_ATTN"    = "true"  ] && ADDONE_FLAG="--use-addone-attn"
DEVNORM_FLAG="";   [ "$USE_DEVIATION_NORM" = "true"  ] && DEVNORM_FLAG="--use-deviation-norm"

# ── Run name base ─────────────────────────────────────────────────────────────
EXP_LABEL="combo124"
RUN_BASE="final_${EXP_LABEL}_${NUM_LAYERS}L${ATTENTION_HEADS}H_emb${EMBEDDING_DIM}_${EPOCHS}ep"
RUN_BASE="${RUN_BASE}_itemdrop${ITEM_DROPOUT_RATE}_ireg${ITEM_REG_WEIGHT}_areg${ATTRIBUTE_REG_WEIGHT}_${BUNDLE}"

echo ""
echo "============================================================"
echo " Final Ablation Exp 05: Pointer + Rel Value + Add-One (3 runs)"
echo "  itemdrop=${ITEM_DROPOUT_RATE}  ireg=${ITEM_REG_WEIGHT}  areg=${ATTRIBUTE_REG_WEIGHT}"
echo "  flags: $PER_HEAD_FLAG $SCALE_FLAG $POINTER_FLAG $REL_VALUE_FLAG $ADDONE_FLAG $DEVNORM_FLAG"
echo "============================================================"

# ── Helper ────────────────────────────────────────────────────────────────────
_train() {
    local N=$1
    echo ""; echo "--- Run ${N}/3: ${RUN_BASE}_run${N} ---"; echo ""
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
        --llm-input-dist                                    \
        --overwrite-existing-data
}

_train 1
_train 2
_train 3

echo ""; echo "All 3 runs complete. Output: $OUTPUT_ROOT"
