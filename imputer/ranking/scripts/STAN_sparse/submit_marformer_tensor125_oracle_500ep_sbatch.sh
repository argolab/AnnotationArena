#!/bin/bash
# cd to imputer/ranking, then submit this job script only (no extra sbatch_adapt arguments).
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RANKING_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_RANKING_ROOT}"

PARTITION=a100 TIME=36:00:00 GPUS=1 CPUS_PER_TASK=16 /home/xwang397/bin/sbatch_adapt scripts/STAN_sparse/run_marformer_tensor125_oracle_500ep.sh
