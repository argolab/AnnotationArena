#!/bin/bash
# Generate tensor bundles under DATA/STAN/SPARSE/Tensor_400_25_9_ItemTest/<LEAF><SUFFIX>/
# K_train=300, K_test=50, K_val=50 (400 items total), I=9, J=25, D=16 — aligned with
# generate_data_tensor_300_50_50_D16_mcar_cheating_oracle.sh (same Stan hyperparams).
#
# BUNDLE_SUFFIX distinguishes runs (default __cheating_oracle). For paths like
#   .../Tensor_400_25_9_ItemTest/Tensor_400_25_9_ItemTest_200
# with no suffix, run:  BUNDLE_SUFFIX= bash generate_tensor400_25_9_itemtest_splits.sh
#
# From imputer/ranking:
#   bash scripts/STAN_sparse/generate_tensor400_25_9_itemtest_splits.sh
#
# If Python dies with "Killed" after Stan finishes, the OS OOM-killer likely ran out of RAM while
# building data_bundle.json. Fix: run on a node with more memory (e.g. sbatch --mem=64G), or confirm
# with: dmesg -T | tail -20 | grep -i kill
#
# --omit-tensor-posterior-probs drops large optional arrays (MARFORMER does not use them).

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RANKING_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_RANKING_ROOT}"

export PYTHONPATH=.

# g++ only needed the first time CmdStan compiles the .stan model; if the exe already exists, warn only.
if ! command -v g++ >/dev/null 2>&1; then
  echo "WARNING: g++ not in PATH — Stan compile may fail on a fresh checkout. Try: module load gcc/12"
else
  export CXX="$(command -v g++)"
  export CC="$(command -v gcc 2>/dev/null || true)"
fi
export TBB_CXX_TYPE="${TBB_CXX_TYPE:-gcc}"

OUTPUT_FAMILY="DATA/STAN/SPARSE/Tensor_400_25_9_ItemTest"
# Append to each leaf folder name (empty string = no extra suffix; matches plain .../Tensor_400_25_9_ItemTest_200)
BUNDLE_SUFFIX="${BUNDLE_SUFFIX:-__cheating_oracle}"

# Add one entry per bundle to generate (same tensor size; e.g. different seeds → add loops with --seed).
LEAVES=(
  "Tensor_400_25_9_ItemTest_200"
)

for LEAF in "${LEAVES[@]}"; do
  echo ""
  echo "========== Generating: ${OUTPUT_FAMILY}/${LEAF}${BUNDLE_SUFFIX} =========="
  GEN_ARGS=(
    STAN/stan_code/scripts/generate_data.py
    --output-dir            "${OUTPUT_FAMILY}"
    --run-name              "${LEAF}"
    --stan-type             "tensor"
    --K-train               300
    --K-test                50
    --K-val                 50
    --I                     9
    --J                     25
    --C                     5
    --D                     16
    --kappa                 15.0
    --sigma-measurement     0.1
    --mcar-missing-rate     0.5
    --observation-protocol  mcar
    --seed                  42
    --stan-arg              T=3
    --stan-arg              sigma_u=1.0
    --stan-arg              sigma_v=1.0
    --stan-arg              sigma_uit=0.1
  )
  if [[ -n "${BUNDLE_SUFFIX}" ]]; then
    GEN_ARGS+=(--run-name-suffix "${BUNDLE_SUFFIX}")
  fi
  GEN_ARGS+=(--omit-tensor-posterior-probs)
  python -u "${GEN_ARGS[@]}"
  echo "Done: ${_RANKING_ROOT}/${OUTPUT_FAMILY}/${LEAF}${BUNDLE_SUFFIX}"
done

echo ""
echo "All requested splits finished."
