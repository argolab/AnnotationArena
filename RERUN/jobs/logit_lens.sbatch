#!/bin/bash
#SBATCH --job-name=LENS
#SBATCH --partition=med
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --comment=accept_cost
# Logit lens / tuned lens on a trained Marformer, layer by layer.
#
# The appendix's own script was never committed (D8), so RERUN/analysis/logit_lens.py
# is a reimplementation from the method description in main.tex. This runs it on
# the refit cell -- the model the paper actually reports, `last.ckpt` -- so the
# appendix numbers come from the same checkpoint as every other Domain 3 number.
#
# CPU on purpose: this is two forward passes and a handful of linear probes.
# corner_score.sbatch already scores the whole tree on cpu at K=4000, so the
# dense edge mask at K=500 is not the constraint here; see the
# never-compute-on-login-node rule for why this is a batch job at all.
#
#   LENS_RUN   = run id            (default final-v2)
#   LENS_CELL  = refit cell name   (default the item K=500 cell the appendix used)
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
LENS_RUN="${LENS_RUN:-final-v2}"
LENS_CELL="${LENS_CELL:-DOMAIN3-SIGMOID_Item_T_500_MARFORMER}"
cd "$(dirname "${RERUN_DIR}")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CELL_DIR="${RERUN_DIR}/runs/${LENS_RUN}/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/${LENS_CELL}"
OUT="${RERUN_DIR}/runs/${LENS_RUN}/reports/logit_lens"
mkdir -p "${OUT}"

python RERUN/analysis/logit_lens.py \
    --run "${CELL_DIR}" --ckpt last.ckpt --device cpu \
    --probe-epochs 300 --probe-lr 1e-2 --bootstrap 500 \
    -o "${OUT}/${LENS_CELL}.json"
echo done
