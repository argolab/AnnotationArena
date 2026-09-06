#!/bin/bash
#SBATCH --job-name=TAIL-AVG
#SBATCH --partition=med
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --comment=accept_cost
# Score every periodic checkpoint of the resid-01 tree (3 arms x 3 seeds x 3
# sizes) on the corner, so we can ask whether averaging the last few
# checkpoints reduces the across-seed spread. If it does, every later suite
# needs fewer replicates for the same confidence, at zero GPU cost.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
cd "$(dirname "${RERUN_DIR}")"
mkdir -p "${RERUN_DIR}/side/out"
ARGS=()
while IFS= read -r d; do ARGS+=( --run-dir "$d" ); done < <(
  find "${RERUN_DIR}/runs/resid-01/RESULTS" -name "*_MARFORMER" -type d | sort )
python RERUN/fig6/corner_curve.py "${ARGS[@]}" --device cpu \
    -o "${RERUN_DIR}/side/out/tail_avg_resid01.json"
