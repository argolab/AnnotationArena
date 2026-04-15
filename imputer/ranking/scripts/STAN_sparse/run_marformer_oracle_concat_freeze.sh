#!/bin/bash
set -euo pipefail

# Oracle concat+freeze diagnostic launcher.
# Uses oracle e_k + eff_pref (v_ij) from generated tensor bundles.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RANKING_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_RANKING_ROOT}"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1

# Point DATA_ROOT at a bundle whose folder name includes the cheating suffix from generate_data.py, e.g.:
#   ... --run-name Tensor_125_25_9_ItemTest_125 --run-name-suffix __cheating_oracle
# Regenerated “cheating” bundle (eff_pref): .../Tensor_125_25_9_ItemTest_125__cheating_oracle
DATA_ROOT="${DATA_ROOT:-DATA/STAN/SPARSE/Tensor_125_25_9_ItemTest_125__cheating_oracle}"
OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/MARFORMER/STAN_oracle_diag}"
RUN_NAME="${RUN_NAME:-Tensor_125_25_9_ItemTest_125_oracle_concat_freeze}"

SEED="${SEED:-42}"
EMBEDDING_DIM="${EMBEDDING_DIM:-80}"
NUM_LAYERS="${NUM_LAYERS:-8}"
ATTENTION_HEADS="${ATTENTION_HEADS:-4}"
D_FF="${D_FF:-128}"
NUM_FFN_LAYERS="${NUM_FFN_LAYERS:-1}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MASKING_RATE="${MASKING_RATE:-0.15}"
MASK_AUGMENTATIONS="${MASK_AUGMENTATIONS:-5}"
MASKED_LOSS_WEIGHT="${MASKED_LOSS_WEIGHT:-15.0}"
OBSERVED_LOSS_WEIGHT="${OBSERVED_LOSS_WEIGHT:-1.0}"
MAX_ITEM="${MAX_ITEM:-10}"
DEVICE="${DEVICE:-cuda}"

python -u -m imputer.entity_mf.train \
  --data-dir "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --seed "${SEED}" \
  --embedding-dim "${EMBEDDING_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --attention-heads "${ATTENTION_HEADS}" \
  --d-ff "${D_FF}" \
  --num-ffn-layers "${NUM_FFN_LAYERS}" \
  --dropout "${DROPOUT}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --masking-rate "${MASKING_RATE}" \
  --mask-augmentations "${MASK_AUGMENTATIONS}" \
  --masked-loss-weight "${MASKED_LOSS_WEIGHT}" \
  --observed-loss-weight "${OBSERVED_LOSS_WEIGHT}" \
  --max-item "${MAX_ITEM}" \
  --device "${DEVICE}" \
  --oracle-concat-freeze \
  --oracle-use-eff-pref
