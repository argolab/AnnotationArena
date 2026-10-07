#!/bin/bash
#SBATCH --job-name=LENSFIG
#SBATCH --partition=med
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:40:00
#SBATCH --comment=accept_cost
# Redraw the two Logit Lens appendix figures from logit_lens.py's JSON.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
LENS_RUN="${LENS_RUN:-final-v2}"
LENS_CELL="${LENS_CELL:-DOMAIN3-SIGMOID_Item_T_500_MARFORMER}"
cd "$(dirname "${RERUN_DIR}")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
REP="${RERUN_DIR}/runs/${LENS_RUN}/reports/logit_lens"
python RERUN/analysis/plot_logit_lens.py \
    --json "${REP}/${LENS_CELL}.json" \
    --outdir "${RERUN_DIR}/runs/${LENS_RUN}/PLOTS/DOMAIN3-SYNTH/LENS"
echo done
