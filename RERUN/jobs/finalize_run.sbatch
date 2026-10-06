#!/bin/bash
#SBATCH --job-name=FINALIZE
#SBATCH --partition=med
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --comment=accept_cost
#
# Everything that turns a scored run into the figures' inputs, in one job so it
# runs unattended after CORNER rather than waiting on a human to notice.
#
# Writes to runs/<id>/reports/. Deliberately does NOT fail on curve_qc's exit
# status: a flagged cell is a finding to read, not a reason to skip the tables.
#
#   FIN_RUN = run id
set -uo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
: "${FIN_RUN:?FIN_RUN is required}"
cd "${RERUN_DIR}"
R="runs/${FIN_RUN}/reports"
mkdir -p "${R}"

echo "=== curve QC (corner metric) ==="
python analysis/curve_qc.py --run-id "${FIN_RUN}" > "${R}/curve_qc.txt" 2>&1
echo "curve_qc exit=$? -> ${R}/curve_qc.txt"
grep -cE "DEGENERATE|FLAGGED" "${R}/curve_qc.txt" || true

echo "=== result tables ==="
python analysis/result_tables.py --run-id "${FIN_RUN}" > "${R}/tables.txt" 2>&1
echo "tables exit=$? -> ${R}/tables.txt"

echo "=== curve export for the artifact ==="
python analysis/curve_export.py --run-id "${FIN_RUN}" \
    --out "${R}/curves_${FIN_RUN}.json" > "${R}/curve_export.txt" 2>&1
echo "export exit=$? -> ${R}/curves_${FIN_RUN}.json"
tail -20 "${R}/curve_export.txt"

echo "=== LLM-Rubric Stan vs published ==="
python analysis/llmrubric_stan_compare.py --run-id final-s101 \
    > "${R}/stan_compare.txt" 2>&1
echo "stan_compare exit=$? -> ${R}/stan_compare.txt"
tail -8 "${R}/stan_compare.txt"

echo "ALL DONE"
