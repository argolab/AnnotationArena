#!/bin/bash
# Sweep Entity Marformer over different max_item values.
# Usage (from repo root):
#   bash scripts/real_data/run_entity_mf_max_item_sweep.sh

set -e

# ---------- Shared real-data config ----------
BUNDLE="${BUNDLE:-hard}"  # or set BUNDLE=dist
if [ "$BUNDLE" == "dist" ]; then
    DATA_DIR="OUTPUT/generated_data/llm_rubric_dist"
else
    DATA_DIR="OUTPUT/generated_data/llm_rubric"
fi

EPOCHS_ENTITY=180
LR_ENTITY=2e-4
WEIGHT_DECAY_ENTITY=0.01
MASKING_RATE_ENTITY=0.15
DEVICE_ENTITY="cuda"
LLM_ANNOTATOR_ID_ENTITY=24
HUMAN_OBSERVED_RATE_ENTITY=0.2

# Values to sweep. Use the string "none" to mean --max-item omitted.
MAX_ITEM_VALUES=("10" "50" "none")

# Whether to use transductive learning. We sweep both settings.
TRANSDUCTIVE_VALUES=("true" "false")

echo "============================================================"
echo "Entity Marformer max_item sweep"
echo "  bundle:   $BUNDLE"
echo "  data_dir: $DATA_DIR"
echo "  max_item values: ${MAX_ITEM_VALUES[*]}"
echo "  transductive settings: ${TRANSDUCTIVE_VALUES[*]}"
echo "============================================================"
echo ""

for TRANSDUCTIVE in "${TRANSDUCTIVE_VALUES[@]}"; do
    for MAX_ITEM_ENTITY in "${MAX_ITEM_VALUES[@]}"; do
        echo ""
        echo ">>> Running Entity Marformer with max_item=${MAX_ITEM_ENTITY}, transductive=${TRANSDUCTIVE}"

        # Build max_item flag: omit entirely when set to "none"
        MAX_ITEM_FLAG=()
        if [ "$MAX_ITEM_ENTITY" != "none" ]; then
            MAX_ITEM_FLAG=(--max-item "$MAX_ITEM_ENTITY")
        fi

        # Build transductive flag: include only when true
        TRANSDUCTIVE_FLAG=()
        if [ "$TRANSDUCTIVE" == "true" ]; then
            TRANSDUCTIVE_FLAG=(--transductive-learning)
        fi

        python -m imputer.entity_mf.train \
            --data-dir "$DATA_DIR" \
            --epochs "$EPOCHS_ENTITY" \
            --lr "$LR_ENTITY" \
            --weight-decay "$WEIGHT_DECAY_ENTITY" \
            --masking-rate "$MASKING_RATE_ENTITY" \
            --device "$DEVICE_ENTITY" \
            --llm-annotator-id "$LLM_ANNOTATOR_ID_ENTITY" \
            --human-observed-rate "$HUMAN_OBSERVED_RATE_ENTITY" \
            "${TRANSDUCTIVE_FLAG[@]}" \
            "${MAX_ITEM_FLAG[@]}"

        echo "<<< Finished run with max_item=${MAX_ITEM_ENTITY}, transductive=${TRANSDUCTIVE}"
    done
done

echo ""
echo "All Entity Marformer max_item runs completed."
