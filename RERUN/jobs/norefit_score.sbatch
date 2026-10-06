#!/bin/bash
#SBATCH --job-name=NOREFIT
#SBATCH --partition=med
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --comment=accept_cost
# What would the corner say if we skipped the second stage?
#
# The shipped protocol carves 15% of the training partition as a validation
# block, selects a stopping epoch on it, then merges the block back and refits
# from scratch on 100% of the data. This scores the SEARCH run's own
# validation-best checkpoint instead -- the model you would report if you kept
# the carve and never refit. Same cells, same corner, so the difference is the
# value of the second stage.
#
# --exclude is cleared because it defaults to "-search", which would drop
# exactly the runs this is about.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
: "${SCORE_RUN:?}"
cd "$(dirname "${RERUN_DIR}")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
OUT="${RERUN_DIR}/runs/${SCORE_RUN}/corner_metrics"
mkdir -p "${OUT}"
python RERUN/analysis/norefit_corner.py \
    --run-id "${SCORE_RUN}" --device cpu \
    -o "${OUT}/corner_norefit.json"
python - "${OUT}/corner_norefit.json" <<'PY'
import json, sys
from pathlib import Path
rows = json.loads(Path(sys.argv[1]).read_text()).get("rows", [])
if not rows:
    sys.exit("FATAL: 0 rows -- the scorer exits 0 on an empty tree, so this is the guard.")
print(f"scored {len(rows)} search runs at their validation-best checkpoint")
PY
