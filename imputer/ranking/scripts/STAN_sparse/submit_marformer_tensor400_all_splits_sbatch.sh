#!/bin/bash
# Submit one Slurm job per Tensor_400 leaf using the same MARFORMER config as
# run_marformer_tensor400_trans_noitemdev.sh (transductive, no item dev, 300 ep).
#
# sbatch_adapt is invoked with a generated wrapper script path only (no extra args).
# Wrappers are written under scripts/STAN_sparse/tensor400_jobs/ (gitignored).
#
# Edit LEAVES and BUNDLE_SUFFIX below, then from imputer/ranking:
#   bash scripts/STAN_sparse/submit_marformer_tensor400_all_splits_sbatch.sh
#
# Prerequisites: for each LEAF, bundle must exist at
#   DATA/STAN/SPARSE/Tensor_400_25_9_ItemTest/${LEAF}${BUNDLE_SUFFIX}/

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RANKING_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_RANKING_ROOT}"

JOBS_DIR="${_SCRIPT_DIR}/tensor400_jobs"
mkdir -p "${JOBS_DIR}"

# Must match generate_tensor400_25_9_itemtest_splits.sh LEAVES + BUNDLE_SUFFIX
LEAVES=(
  "Tensor_400_25_9_ItemTest_200"
  # "Tensor_400_25_9_ItemTest_150"
)
BUNDLE_SUFFIX="${BUNDLE_SUFFIX:-__cheating_oracle}"

RUNNER="${_SCRIPT_DIR}/run_marformer_tensor400_trans_noitemdev.sh"

for LEAF in "${LEAVES[@]}"; do
  SAFE_NAME="${LEAF//\//_}"
  WRAP="${JOBS_DIR}/wrap_${SAFE_NAME}.sh"
  cat > "${WRAP}" <<EOF
#!/bin/bash
set -euo pipefail
cd "${_RANKING_ROOT}"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
exec bash "${RUNNER}" "${LEAF}" "${BUNDLE_SUFFIX}"
EOF
  chmod +x "${WRAP}"
  echo "Submitting ${LEAF} -> ${WRAP}"
  PARTITION=a100 TIME=36:00:00 GPUS=1 CPUS_PER_TASK=16 /home/xwang397/bin/sbatch_adapt "${WRAP}"
done

echo "Submitted ${#LEAVES[@]} job(s)."
