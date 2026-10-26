#!/bin/bash
#SBATCH --job-name=OBSDEN-CURVE
#SBATCH --partition=med
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --comment=accept_cost
set -euo pipefail
# Corner curve for one representative cell per density (draw s101, seed 42).
# One array task per density; CURVE_RATES picks them.
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
cd "$(dirname "${RERUN_DIR}")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
read -r -a RATES <<< "${CURVE_RATES:-15 50 85}"
RATE="${RATES[${SLURM_ARRAY_TASK_ID:-0}]}"
DRAW="${CURVE_DRAW:-s101}"
RUN="${RERUN_DIR}/runs/obsden-${DRAW}-o${RATE}/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/valmcar-search/seed42/DOMAIN3-SIGMOID_Item_T_500_MARFORMER"
OUT="${RERUN_DIR}/threads/obs-density/out/corner_curve_${DRAW}_o${RATE}.json"
echo "cell obsden-${DRAW}-o${RATE}"
python RERUN/threads/obs-density/corner_curve.py \
    --run-dir "${RUN}" --device cpu --every "${CURVE_EVERY:-2}" -o "${OUT}"
