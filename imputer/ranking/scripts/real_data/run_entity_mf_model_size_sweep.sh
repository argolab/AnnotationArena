#!/bin/bash
# Sweep Entity Marformer over model size:
#   - num_layers in {2, 4, 6, 8, 10}
#   - embedding_dim in {32, 72, 128, 256}
# Always uses max_item = 10.
#
# Usage (from repo root):
#   bash scripts/real_data/run_entity_mf_model_size_sweep.sh

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
MAX_ITEM_ENTITY=10

# Set to 1 to pass --overwrite-existing-data (reuse run dir when run-name exists).
OVERWRITE_EXISTING_DATA="${OVERWRITE_EXISTING_DATA:-0}"

# # Values to sweep.
# NUM_LAYERS_VALUES=("4" "6" "8" "10")
# EMBEDDING_DIM_VALUES=("32" "72" "128" "256")


# Values to sweep.
NUM_LAYERS_VALUES=( "10")
EMBEDDING_DIM_VALUES=("256")

# Whether to use transductive learning. We sweep both settings.
# TRANSDUCTIVE_VALUES=("true" "false")
TRANSDUCTIVE_VALUES=("false")

echo "============================================================"
echo "Entity Marformer model-size sweep"
echo "  bundle:            $BUNDLE"
echo "  data_dir:          $DATA_DIR"
echo "  num_layers values: ${NUM_LAYERS_VALUES[*]}"
echo "  embedding_dim:     ${EMBEDDING_DIM_VALUES[*]}"
echo "  max_item:          $MAX_ITEM_ENTITY"
echo "  transductive:      ${TRANSDUCTIVE_VALUES[*]}"
echo "============================================================"
echo ""

for TRANSDUCTIVE in "${TRANSDUCTIVE_VALUES[@]}"; do
    for EMBEDDING_DIM in "${EMBEDDING_DIM_VALUES[@]}"; do
        for NUM_LAYERS in "${NUM_LAYERS_VALUES[@]}"; do
            echo ""
            echo ">>> Running Entity Marformer with layers=${NUM_LAYERS}, D=${EMBEDDING_DIM}, max_item=${MAX_ITEM_ENTITY}, transductive=${TRANSDUCTIVE}"

            # Build transductive flag: include only when true
            TRANSDUCTIVE_FLAG=()
            if [ "$TRANSDUCTIVE" == "true" ]; then
                TRANSDUCTIVE_FLAG=(--transductive-learning)
            fi

            RUN_NAME="modelsize_L${NUM_LAYERS}_D${EMBEDDING_DIM}_T${TRANSDUCTIVE}"

            OVERWRITE_FLAG=()
            if [ "$OVERWRITE_EXISTING_DATA" = "1" ]; then
                OVERWRITE_FLAG=(--overwrite-existing-data)
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
                --max-item "$MAX_ITEM_ENTITY" \
                --embedding-dim "$EMBEDDING_DIM" \
                --num-layers "$NUM_LAYERS" \
                --run-name "$RUN_NAME" \
                "${OVERWRITE_FLAG[@]}" \
                "${TRANSDUCTIVE_FLAG[@]}"

            echo "<<< Finished run with layers=${NUM_LAYERS}, D=${EMBEDDING_DIM}, max_item=${MAX_ITEM_ENTITY}, transductive=${TRANSDUCTIVE}"
        done
    done
done

echo ""
echo "All Entity Marformer model-size runs completed."

