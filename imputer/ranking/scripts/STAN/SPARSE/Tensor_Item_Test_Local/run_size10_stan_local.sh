#!/bin/bash
set -euo pipefail

SCRIPT_START=$SECONDS

# CmdStan compiles models with g++ (C++17). Compute nodes often omit /usr/bin/g++;
# load a GCC module in your job or shell first, e.g.:
#   module avail gcc
#   module load gcc/12.2.0
# If ~/.cmdstan/.../make/local hard-codes CXX=/usr/bin/g++, change it to CXX=g++
# or the full path from $(which g++).
#
# Link errors: undefined reference to __libc_single_threaded, or libtbb.so.2 wants
# GLIBC_2.32/2.34 — your $CMDSTAN tree was built on a newer OS than this node.
# Rebuild CmdStan *on this same node* (after module load gcc):
#   cd "${CMDSTAN:?set CMDSTAN to your cmdstan dir}"
#   make clean-all && make build -j8
# If __libc_single_threaded persists on EL8, rebuild CmdStan with an older GCC
# module (e.g. 9 or 10) that matches the cluster libc, not only gcc/12.
if ! command -v g++ >/dev/null 2>&1; then
    echo "ERROR: g++ not found in PATH. Load a compiler module, then re-run."
    echo "  Example: module load gcc"
    exit 1
fi
export CXX="$(command -v g++)"
if command -v gcc >/dev/null 2>&1; then
    export CC="$(command -v gcc)"
fi

SIZE=125
DATA_DIR="DATA/STAN/SPARSE/Tensor_125_25_9_ItemTest_125"
DATA_BUNDLE="${DATA_DIR}/data_bundle.json"
OUTPUT_DIR="RESULTS/STAN/SPARSE"
RUN_NAME="Tensor_125_25_9_ItemTest_125"

CHAINS=1
ITER_WARMUP=300
ITER_SAMPLING=500
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42

echo ""
echo "============================================================"
echo " Stan Dawid-Skene | Tensor_125_25_9_ItemTest_125"
echo "  CHAINS       : ${CHAINS}"
echo "  ITER_WARMUP  : ${ITER_WARMUP}"
echo "  ITER_SAMPLING: ${ITER_SAMPLING}"
echo "============================================================"

echo "[1/2] Running MCMC (tensor)..."
python STAN/stan_code/scripts/run_inference.py \
      --data-bundle        "$DATA_BUNDLE"   \
      --configs            "${DATA_DIR}/configs.json" \
      --output-dir         "$OUTPUT_DIR"    \
      --run-name           "$RUN_NAME"      \
      --stan-type          "tensor"         \
      --chains             "$CHAINS"        \
      --iter-warmup        "$ITER_WARMUP"   \
      --iter-sampling      "$ITER_SAMPLING" \
      --adapt-delta        "$ADAPT_DELTA"   \
      --max-treedepth      "$MAX_TREEDEPTH" \
      --seed               "$SEED"          \
      --overwrite-existing-data      \
      --show-stan-console

echo "[2/2] Evaluating predictions..."
python STAN/stan_code/scripts/evaluate_predictions.py \
      --data-bundle        "$DATA_BUNDLE"              \
      --mcmc-dir           "${OUTPUT_DIR}/${RUN_NAME}" \
      --output-dir         "$OUTPUT_DIR"               \
      --run-name           "${RUN_NAME}_eval"           \
      --csv-pattern        "tensor_model-*.csv"         \
      --overwrite-existing-data                         \
      --verbose

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Results: ${OUTPUT_DIR}/${RUN_NAME}_eval"
echo "============================================================"
