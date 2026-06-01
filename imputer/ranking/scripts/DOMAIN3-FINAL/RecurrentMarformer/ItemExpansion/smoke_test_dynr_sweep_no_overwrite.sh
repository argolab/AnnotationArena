#!/bin/bash
# Bash-only smoke test: sweep loop assigns a distinct RUN_DIR per RUN_TAG (no training).
#
#   cd imputer/ranking
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/smoke_test_dynr_sweep_no_overwrite.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " DYNR sweep path smoke test (no training)"
echo "============================================================"

# --- 1) Show the old bug pattern reuses one name ---
echo ""
echo "[1] Old bug: RUN_NAME=\"\${RUN_NAME:-...}\` keeps the first name"
_buggy=()
RUN_NAME=""
for tag in p6c2r3c4 p4c4r2c4 p0c4r3c8; do
    RUN_TAG="$tag"
    RUN_NAME="${RUN_NAME:-DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_${RUN_TAG}_DYNR}"
    _buggy+=("$RUN_NAME")
done
echo "    iter1: ${_buggy[0]}"
echo "    iter2: ${_buggy[1]}"
echo "    iter3: ${_buggy[2]}"
if [ "${_buggy[0]}" = "${_buggy[1]}" ] && [ "${_buggy[1]}" = "${_buggy[2]}" ]; then
    echo "    -> all identical (would overwrite)"
else
    echo "    -> unexpectedly different"
fi

# --- 2) Fixed pattern: 3 distinct names ---
echo ""
echo "[2] Fixed: unset RUN_NAME + RUN_NAME=\"..._\${RUN_TAG}_DYNR\""
_fixed=()
for tag in p6c2r3c4 p4c4r2c4 p0c4r3c8; do
    RUN_TAG="$tag"
    unset RUN_NAME
    RUN_NAME="DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_${RUN_TAG}_DYNR"
    _fixed+=("$RUN_NAME")
done
for i in 0 1 2; do
    echo "    iter$((i + 1)): ${_fixed[$i]}"
done
if [ "${_fixed[0]}" != "${_fixed[1]}" ] && [ "${_fixed[1]}" != "${_fixed[2]}" ]; then
    echo "    -> 3 distinct names OK"
else
    echo "FAIL: fixed pattern did not produce 3 distinct RUN_NAME values"
    exit 1
fi

# --- 3) Exact group-1 loop via _run_one_dynr.sh DRY_RUN=1 ---
echo ""
echo "[3] Group-1 loop (DRY_RUN=1, sources _run_one_dynr.sh)"

export DRY_RUN=1
export OUTPUT_ROOT="RESULTS/RECURRENT_MARFORMER/DYNR-SMOKE-PATHS-ONLY"
export EPOCHS=400

_paths=()
RECURRENCE_CONFIGS=(
  "p6c2r3c4   6  2  3  4   8"
  "p4c4r2c4   4  4  2  4   8"
  "p0c4r3c8   0  4  3  8  10"
)

for entry in "${RECURRENCE_CONFIGS[@]}"; do
    read -r RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE CODA_DEPTH RECURRENCE_MAX <<< "$entry"
    unset RUN_NAME
    export RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE CODA_DEPTH RECURRENCE_MAX
    out="$("${SCRIPT_DIR}/_run_one_dynr.sh" 2>&1 | awk -F'RUN_DIR=' '/RUN_DIR=/ {print $2}')"
    if [ -z "$out" ]; then
        echo "FAIL: could not parse RUN_DIR for RUN_TAG=${RUN_TAG}"
        exit 1
    fi
    if [[ "$out" != *"${RUN_TAG}_DYNR" ]]; then
        echo "FAIL: RUN_DIR does not contain RUN_TAG"
        echo "  RUN_TAG=${RUN_TAG}"
        echo "  RUN_DIR=${out}"
        exit 1
    fi
    _paths+=("$out")
    echo "    ${RUN_TAG} -> ${out}"
done

# All paths must be unique
if [ "${_paths[0]}" = "${_paths[1]}" ] || [ "${_paths[1]}" = "${_paths[2]}" ] || [ "${_paths[0]}" = "${_paths[2]}" ]; then
    echo "FAIL: duplicate RUN_DIR across iterations"
    exit 1
fi

echo ""
echo "============================================================"
echo " PASS: saving logic uses a different RUN_DIR per model."
echo " Full sweeps will write under:"
echo "   ${OUTPUT_ROOT}/DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_<tag>_DYNR/"
echo "============================================================"
