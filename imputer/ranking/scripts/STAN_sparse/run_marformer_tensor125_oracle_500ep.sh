#!/bin/bash
#
# MARFORMER: tensor Tensor_125_25_9_ItemTest_125 with oracle concat+freeze (eff_pref + e_k), 500 epochs.
# All paths and hyperparameters are set below (edit this file to change them).
#
# Do NOT put #SBATCH lines in this file. sbatch_adapt prepends its own Slurm header (partition, GPUs,
# time, logs, modules, conda, chdir) and then appends *this entire script* as the job body. If this file
# starts with #SBATCH, the tool may only merge directives and stop before the training commands — dry
# run then ends right after "echo CWD" with no python -m imputer.entity_mf.train.
#
# Submit from imputer/ranking (sbatch_adapt is called with this script path only):
#   /home/xwang397/bin/sbatch_adapt scripts/STAN_sparse/run_marformer_tensor125_oracle_500ep.sh
# Or run interactively:
#   bash scripts/STAN_sparse/run_marformer_tensor125_oracle_500ep.sh
#
# Requires data_bundle.json with eff_pref (regenerate tensor data with current generate_data.py;
# e.g. --run-name Tensor_125_25_9_ItemTest_125 --run-name-suffix __cheating_oracle).

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RANKING_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_RANKING_ROOT}"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

SCRIPT_START=$SECONDS

# ── Data / output (edit if your regenerated folder name differs) ─────────────
DATA_ROOT="DATA/STAN/SPARSE/Tensor_125_25_9_ItemTest_125__cheating_oracle"
OUTPUT_ROOT="RESULTS/MARFORMER/STAN_oracle_diag"
RUN_NAME="Tensor_125_25_9_ItemTest_125_oracle_500ep"

# ── Training hyperparameters ─────────────────────────────────────────────────
SEED=42
TYPE_EMBEDDING_INIT="kaiming"
EMBEDDING_DIM=80
NUM_LAYERS=8
ATTENTION_HEADS=4
D_FF=128
NUM_FFN_LAYERS=1
DROPOUT=0.1
EPOCHS=500
LR="2e-4"
LR_SCHEDULE="none"
LR_MIN="1e-5"
WEIGHT_DECAY=0.01
MASKING_RATE=0.15
MASK_AUGMENTATIONS=5
MASKED_LOSS_WEIGHT=15.0
OBSERVED_LOSS_WEIGHT=1.0
DEVICE="cuda"
MAX_ITEM=10
ANNOTATOR_REG_WEIGHT=0.0

ITEM_DROPOUT_RATE=0.7
ITEM_REG_WEIGHT=0.0
ATTRIBUTE_REG_WEIGHT=0.0
USE_PER_HEAD_REL=false
SCALE_SHARED_REL=true
USE_POINTER=true
USE_REL_VALUE=false
USE_ADDONE_ATTN=false
USE_DEVIATION_NORM=false
USE_GRAPH_MASK=false
LLM_INPUT_DIST=true
# false = fail if RESULTS/.../RUN_NAME already exists (no --overwrite-existing-data).
# Set to true only if you intentionally re-run into the same output folder.
OVERWRITE_EXISTING=false

BUNDLE="${DATA_ROOT}/data_bundle.json"
if [[ ! -f "${BUNDLE}" ]]; then
  echo "ERROR: missing ${BUNDLE}"
  exit 1
fi
python - <<PY
import json
from pathlib import Path

p = Path("${BUNDLE}")
d = json.loads(p.read_text())
eff = d.get("eff_pref")
if eff is None:
    eg = d.get("extra_ground_truth")
    if isinstance(eg, dict):
        eff = eg.get("eff_pref")
if not eff:
    raise SystemExit(
        "data_bundle.json has no eff_pref. Regenerate tensor data with current generate_data.py "
        "(saves eff_pref). Example folder: Tensor_125_25_9_ItemTest_125__cheating_oracle"
    )
if "embeddings" not in d:
    raise SystemExit("data_bundle.json missing embeddings")
print("OK: eff_pref and embeddings present")
PY

PER_HEAD_FLAG="";      [ "$USE_PER_HEAD_REL"  = "false" ] && PER_HEAD_FLAG="--no-per-head-rel"
SCALE_FLAG="";         [ "$SCALE_SHARED_REL"  = "true"  ] && SCALE_FLAG="--scale-shared-rel"
POINTER_FLAG="";       [ "$USE_POINTER"        = "true"  ] && POINTER_FLAG="--use-pointer"
REL_VALUE_FLAG="";     [ "$USE_REL_VALUE"      = "true"  ] && REL_VALUE_FLAG="--use-rel-value"
ADDONE_FLAG="";        [ "$USE_ADDONE_ATTN"    = "true"  ] && ADDONE_FLAG="--use-addone-attn"
DEVNORM_FLAG="";       [ "$USE_DEVIATION_NORM" = "true"  ] && DEVNORM_FLAG="--use-deviation-norm"
GRAPHMASK_FLAG="";     [ "$USE_GRAPH_MASK"     = "true"  ] && GRAPHMASK_FLAG="--use-graph-mask"
LLM_DIST_FLAG="";      [ "$LLM_INPUT_DIST"     = "true"  ] && LLM_DIST_FLAG="--llm-input-dist"
OVERWRITE_FLAG="";     [ "$OVERWRITE_EXISTING" = "true"  ] && OVERWRITE_FLAG="--overwrite-existing-data"

echo ""
echo "============================================================"
echo " MARFORMER | Tensor_125 oracle concat+freeze | ${EPOCHS} epochs"
echo "  DATA_DIR     : ${DATA_ROOT}"
echo "  OUTPUT_ROOT  : ${OUTPUT_ROOT}"
echo "  RUN_NAME     : ${RUN_NAME}"
echo "============================================================"

python -u -m imputer.entity_mf.train \
    --data-dir             "${DATA_ROOT}"   \
    --run-name             "${RUN_NAME}"    \
    --output-root          "${OUTPUT_ROOT}" \
    --seed                 "$SEED"                   \
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
    --max-item             "$MAX_ITEM"               \
    --type-embedding-init  "$TYPE_EMBEDDING_INIT"    \
    --item-reg-weight      "$ITEM_REG_WEIGHT"        \
    --attribute-reg-weight "$ATTRIBUTE_REG_WEIGHT"   \
    --annotator-reg-weight "$ANNOTATOR_REG_WEIGHT"   \
    --oracle-concat-freeze \
    --oracle-use-eff-pref \
    $PER_HEAD_FLAG                                   \
    $SCALE_FLAG                                      \
    $POINTER_FLAG                                    \
    $REL_VALUE_FLAG                                  \
    $ADDONE_FLAG                                     \
    $DEVNORM_FLAG                                    \
    $GRAPHMASK_FLAG                                  \
    $LLM_DIST_FLAG                                   \
    $OVERWRITE_FLAG

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Output: ${OUTPUT_ROOT}/${RUN_NAME}"
echo "============================================================"
