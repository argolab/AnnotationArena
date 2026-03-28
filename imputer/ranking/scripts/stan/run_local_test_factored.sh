#!/bin/bash

# Local test script — factored dot-product data.
# Runs 1 short run (3 epochs) to verify the pipeline end-to-end.
# No SLURM. Run from repo root: bash scripts/stan/run_local_test_factored.sh

set -e
export PYTHONPATH=.

DATA_DIR="OUTPUT/generated_data/K_train_200_K_test_25_I_9_J_25_factored_dot_product"
OUTPUT_ROOT="OUTPUT/ENTITY_MF/STAN_EXPS_LOCAL"

# Detect device
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
else
    DEVICE="cpu"
fi
echo "Using device: $DEVICE"

python -m imputer.entity_mf.train \
    --data-dir             "$DATA_DIR"                          \
    --run-name             "local_test_factored_pointer"        \
    --output-root          "$OUTPUT_ROOT"                       \
    --embedding-dim        80                                   \
    --num-layers           8                                    \
    --attention-heads      4                                    \
    --d-ff                 128                                  \
    --num-ffn-layers       1                                    \
    --dropout              0.1                                  \
    --item-dropout-rate    0.7                                  \
    --epochs               300                                    \
    --lr                   2e-4                                 \
    --lr-schedule          none                                 \
    --lr-min               1e-5                                 \
    --weight-decay         0.01                                 \
    --masking-rate         0.35                                 \
    --mask-augmentations   5                                    \
    --masked-loss-weight   15.0                                 \
    --observed-loss-weight 1.0                                  \
    --device               "$DEVICE"                           \
    --max-item             10                                   \
    --type-embedding-init  kaiming                              \
    --item-reg-weight      0.0                                  \
    --attribute-reg-weight 0.0                                  \
    --annotator-reg-weight 0.0                                  \
    --no-per-head-rel                                           \
    --scale-shared-rel                                          \
    --use-pointer                                               \
    --llm-input-dist                                            \
    --seed                 42                                   \
    --overwrite-existing-data

echo ""; echo "Local test complete. Output: $OUTPUT_ROOT"
