#!/bin/bash
#SBATCH --job-name=STAN-DIAG
#SBATCH --partition=med
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --comment=accept_cost
# MCMC convergence diagnostics for the Domain 3 Stan baselines.
#
# CPU-only and submitted as a job on purpose: it parses the CmdStan draw CSVs,
# which are large enough that doing it interactively stalls the shared login
# node. See the memory note "never compute on the login node".
#
# Walltime: reading + FFT-based ESS over a few hundred MB of CSV is a couple of
# minutes; 1h is ~20x margin, and med is uncontended so the generous limit costs
# nothing in scheduling.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env

ROOTS="${SD_ROOTS:-${RERUN_DIR}/runs/legacy/dataseed-s101/RESULTS/STAN/DOMAIN3-SIGMOID}"
mkdir -p "${RERUN_DIR}/side/out"
python -u "${RERUN_DIR}/side/stan_diagnostics.py" \
    --roots "${ROOTS}" \
    --pattern "${SD_PATTERN:-*_4000_*,*_1000_*}" \
    -o "${RERUN_DIR}/side/out/stan_diagnostics.json"
