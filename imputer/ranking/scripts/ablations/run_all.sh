#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=EMF_Abl_All
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

# ── Ablation: Base + pointer + add-one + rel-value (full best config) ─────────
# All three features enabled together. Should match the best run.

BUNDLE="dist"
DATA_DIR="OUTPUT/generated_data/llm_rubric_dist"

EMBEDDING_DIM=80
NUM_LAYERS=8
ATTENTION_HEADS=4
D_FF=128
NUM_FFN_LAYERS=1
DROPOUT=0.1
ITEM_DROPOUT_RATE=0.7
TYPE_EMBEDDING_INIT="kaiming"

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

ITEM_REG_WEIGHT=1e-3
ATTRIBUTE_REG_WEIGHT=1e-3
ANNOTATOR_REG_WEIGHT=0.0

RUN_NAME="ablation_all_${NUM_LAYERS}L${ATTENTION_HEADS}H_emb${EMBEDDING_DIM}_ffn${NUM_FFN_LAYERS}_${EPOCHS}ep"
RUN_NAME="${RUN_NAME}_itemdrop${ITEM_DROPOUT_RATE}_lr${LR}_sched${LR_SCHEDULE}"
RUN_NAME="${RUN_NAME}_ireg${ITEM_REG_WEIGHT}_areg${ATTRIBUTE_REG_WEIGHT}_init${TYPE_EMBEDDING_INIT}"
RUN_NAME="${RUN_NAME}_sharedbias_relscale_ptr_relv_addone_softinput_${BUNDLE}"

echo ""
echo "============================================================"
echo " Ablation: Base + pointer + add-one + rel-value (full)"
echo "  run_name: $RUN_NAME"
echo "============================================================"
echo ""

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
    --annotator-reg-weight "$ANNOTATOR_REG_WEIGHT"   \
    --no-per-head-rel                                \
    --scale-shared-rel                               \
    --use-pointer                                    \
    --use-addone-attn                                \
    --use-rel-value                                  \
    --llm-input-dist                                 \
    --overwrite-existing-data

echo ""
echo "Training complete. Output: OUTPUT/ENTITY_MF/$RUN_NAME"
