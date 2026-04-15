#!/bin/bash
# Generate tensor synthetic data (K_train=300, K_test=50, K_val=50, D=16, MCAR 0.5).
# Saves item embeddings (e_k) and oracle eff_pref (v_ij) in data_bundle.json when using current
# STAN/stan_code/pipeline/data_gen.py + --run-name-suffix __cheating_oracle.
#
# Run from imputer/ranking:
#   bash scripts/STAN_sparse/generate_data_tensor_300_50_50_D16_mcar_cheating_oracle.sh
#
# Requires: CmdStanPy, working g++, and (if needed) module load gcc; see scripts/STAN/SPARSE/Tensor_Item_Test_Local/run_size10_stan_local.sh

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RANKING_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_RANKING_ROOT}"

export PYTHONPATH=.

if ! command -v g++ >/dev/null 2>&1; then
  echo "ERROR: g++ not in PATH. Load a compiler module (e.g. module load gcc/12) and retry."
  exit 1
fi
export CXX="$(command -v g++)"
export CC="$(command -v gcc 2>/dev/null || true)"
export TBB_CXX_TYPE="${TBB_CXX_TYPE:-gcc}"

OUTPUT_DIR="DATA/STAN/SPARSE"
RUN_NAME="Tensor_300_50_50_9_25_D16"
RUN_SUFFIX="__cheating_oracle"

python -u STAN/stan_code/scripts/generate_data.py \
  --output-dir            "${OUTPUT_DIR}" \
  --run-name              "${RUN_NAME}" \
  --run-name-suffix       "${RUN_SUFFIX}" \
  --stan-type             "tensor" \
  --K-train               300 \
  --K-test                50 \
  --K-val                 50 \
  --I                     9 \
  --J                     25 \
  --C                     5 \
  --D                     16 \
  --kappa                 15.0 \
  --sigma-measurement     0.1 \
  --mcar-missing-rate     0.5 \
  --observation-protocol  mcar \
  --seed                  42 \
  --stan-arg              T=3 \
  --stan-arg              sigma_u=1.0 \
  --stan-arg              sigma_v=1.0 \
  --stan-arg              sigma_uit=0.1

# Do NOT pass --overwrite-existing-data by default: that flag rm -rf's the run folder if it
# already exists (see generate_data.py). If regeneration is intentional, add:
#   --overwrite-existing-data

echo ""
echo "Done. Bundle directory:"
echo "  ${_RANKING_ROOT}/${OUTPUT_DIR}/${RUN_NAME}${RUN_SUFFIX}"
