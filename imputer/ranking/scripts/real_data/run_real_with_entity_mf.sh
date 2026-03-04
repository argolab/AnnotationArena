#!/bin/bash
# Run Stan, original Marformer imputer, and Entity Marformer (entity_mf)
# on the same real-data bundle, then run the existing evaluation tools.
#
# This is a thin orchestrator around:
#   - scripts/real_data/run_stan_real.sh
#   - scripts/real_data/run_real_data.sh
#   - python -m imputer.entity_mf.train
#
# Usage (from repo root, same as other real_data scripts):
#   # LLMRubric, hard labels
#   bash scripts/real_data/run_real_with_entity_mf.sh
#   # LLMRubric, dist bundle
#   BUNDLE=dist bash scripts/real_data/run_real_with_entity_mf.sh
#
# Notes:
#   - Stan: run_stan_real.sh already runs evaluation and writes predictive_metrics.json.
#   - Imputer: run_real_data.sh trains the original Marformer.
#     You can then run utils/evaluate_checkpoint.py on the desired checkpoint.
#   - Entity Marformer: we run python -m imputer.entity_mf.train on the same data_dir.

set -e

# ── Bundle variant (llm_rubric hard/dist) ───────────────────────────────────────
BUNDLE="${BUNDLE:-hard}"

if [ "$BUNDLE" == "dist" ]; then
    DATA_DIR="OUTPUT/generated_data/llm_rubric_dist"
    IMPUTER_RUN_NAME="llm_rubric_marformer_dist_DFF_128_ITEM_DROP_0.5_MIXED_INIT_TRANSD"
else
    DATA_DIR="OUTPUT/generated_data/llm_rubric"
    IMPUTER_RUN_NAME="llm_rubric_marformer"
fi

echo ""
echo "============================================================"
echo "Real-data run (Stan + Marformer + EntityMarformer)"
echo "  bundle:       $BUNDLE"
echo "  data_dir:     $DATA_DIR"
echo "  imputer run:  $IMPUTER_RUN_NAME"
echo "============================================================"
echo ""

# ── 1) Stan baseline (inference + evaluation) ───────────────────────────────────
echo "[1/3] Running Stan baseline (scripts/real_data/run_stan_real.sh)..."
# DATASET="llmrubric" BUNDLE="$BUNDLE" bash scripts/real_data/run_stan_real.sh

# ── 2) Original Marformer imputer training ─────────────────────────────────────
echo ""
echo "[2/3] Running original Marformer imputer (scripts/real_data/run_real_data.sh)..."
# BUNDLE="$BUNDLE" bash scripts/real_data/run_real_data.sh

echo ""
echo "You can evaluate a specific Marformer checkpoint with:"
echo "  python utils/evaluate_checkpoint.py \\"
echo "      --model-path OUTPUT/IMPUTER/${IMPUTER_RUN_NAME}/model_epoch_XXXX.pt \\"
echo "      --data-dir ${DATA_DIR}"
echo ""

# ── 3) Entity Marformer training (entity_mf) ────────────────────────────────────
echo "[3/3] Running Entity Marformer (python -m imputer.entity_mf.train)..."

EPOCHS_ENTITY=180
LR_ENTITY=2e-4
WEIGHT_DECAY_ENTITY=0.01
MASKING_RATE_ENTITY=0.15
DEVICE_ENTITY="cuda"
LLM_ANNOTATOR_ID_ENTITY=24
HUMAN_OBSERVED_RATE_ENTITY=0.2
MAX_ITEM_ENTITY=10

python -m imputer.entity_mf.train \
    --data-dir "$DATA_DIR" \
    --epochs "$EPOCHS_ENTITY" \
    --lr "$LR_ENTITY" \
    --weight-decay "$WEIGHT_DECAY_ENTITY" \
    --masking-rate "$MASKING_RATE_ENTITY" \
    --device "$DEVICE_ENTITY" \
    --llm-annotator-id "$LLM_ANNOTATOR_ID_ENTITY" \
    --human-observed-rate "$HUMAN_OBSERVED_RATE_ENTITY" \
    --max-item "$MAX_ITEM_ENTITY"

echo ""
echo "Entity Marformer training finished."
echo "Lightning will have written logs/checkpoints under its default directory (e.g. lightning_logs/)." 
echo "You can inspect masked/observed/missing losses there to compare against Stan and the original Marformer."
echo ""

