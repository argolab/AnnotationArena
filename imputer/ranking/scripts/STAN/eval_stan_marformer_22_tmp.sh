#!/usr/bin/env bash
#
# STAN MARFORMER — evaluate the 22 runs from stan_data_command_marformer.sh
# (ranked_eval: k=1,3,5,7 + last.ckpt → RANKED_RESULTS/by_val_missing_xent.json).
#
# From imputer/ranking, on a GPU node — four lighter jobs (one dataset family each):
#
#   bash scripts/STAN/eval_stan_marformer_22_tmp.sh 1
#   bash scripts/STAN/eval_stan_marformer_22_tmp.sh 2
#   bash scripts/STAN/eval_stan_marformer_22_tmp.sh 3
#   bash scripts/STAN/eval_stan_marformer_22_tmp.sh 4
#
# All families + summary tables:
#
#   bash scripts/STAN/eval_stan_marformer_22_tmp.sh
#
# Summary only (no ranked_eval; vertical text + PNGs under RESULTS/MARFORMER/STAN/reports/):
#
#   bash scripts/STAN/eval_stan_marformer_22_tmp.sh summary
#
# Regenerate tables/PNGs only:
#   python -m imputer.entity_mf.ranked_eval_report --mode stan [--no-png]
#
# Slurm example (group 1):
#   cd /path/to/AA_new/imputer/ranking && PARTITION=a100 GPUS=1 TIME=2:00:00 \
#     CPUS_PER_TASK=4 MEM_PER_CPU=8G /home/xwang397/bin/sbatch_adapt \
#     scripts/STAN/eval_stan_marformer_22_tmp.sh 1
#

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RANKING_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
cd "${_RANKING_ROOT}"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -euo pipefail

OUTPUT_ROOT="RESULTS/MARFORMER/STAN"

G1=(
  "Factor_250_20_9_AnnotatorTest_12"
  "Factor_250_20_9_AnnotatorTest_14"
  "Factor_250_20_9_AnnotatorTest_3"
  "Factor_250_20_9_AnnotatorTest_6"
  "Factor_250_20_9_AnnotatorTest_9"
)
G2=(
  "Factor_650_20_9_ItemTest_10"
  "Factor_650_20_9_ItemTest_100"
  "Factor_650_20_9_ItemTest_250"
  "Factor_650_20_9_ItemTest_50"
  "Factor_650_20_9_ItemTest_500"
  "Factor_650_20_9_ItemTest_600"
)
G3=(
  "Normal_250_20_9_AnnotatorTest_12"
  "Normal_250_20_9_AnnotatorTest_14"
  "Normal_250_20_9_AnnotatorTest_3"
  "Normal_250_20_9_AnnotatorTest_6"
  "Normal_250_20_9_AnnotatorTest_9"
)
G4=(
  "Normal_650_20_9_ItemTest_10"
  "Normal_650_20_9_ItemTest_100"
  "Normal_650_20_9_ItemTest_250"
  "Normal_650_20_9_ItemTest_50"
  "Normal_650_20_9_ItemTest_500"
  "Normal_650_20_9_ItemTest_600"
)

run_ranked_eval_for() {
  local run="$1"
  local rd="${OUTPUT_ROOT}/${run}"
  echo ""
  echo "--- ${run} ---"
  if [[ ! -d "${rd}" ]]; then
    echo "  SKIP: no run dir"
    return 0
  fi
  if [[ ! -f "${rd}/train_config.json" ]]; then
    echo "  SKIP: no train_config.json"
    return 0
  fi
  if ! compgen -G "${rd}/checkpoints/*.ckpt" > /dev/null; then
    echo "  SKIP: no checkpoints"
    return 0
  fi
  python -u -m imputer.entity_mf.ranked_eval \
    --run-dir "${rd}" \
    --ranks 1,3,5,7 \
    --device cuda
}

run_group() {
  local idx="$1"
  case "${idx}" in
    1) local -n _runs=G1 ;;
    2) local -n _runs=G2 ;;
    3) local -n _runs=G3 ;;
    4) local -n _runs=G4 ;;
    *) echo "Invalid group: ${idx}"; exit 1 ;;
  esac
  for run in "${_runs[@]}"; do
    run_ranked_eval_for "${run}"
  done
}

MODE="${1:-all}"

if [[ "${MODE}" == "summary" ]]; then
  :
elif [[ "${MODE}" == "all" ]]; then
  echo "============================================================"
  echo " STAN MARFORMER | ranked_eval | all 4 dataset families"
  echo "============================================================"
  run_group 1
  run_group 2
  run_group 3
  run_group 4
elif [[ "${MODE}" =~ ^[1-4]$ ]]; then
  echo "============================================================"
  echo " STAN MARFORMER | ranked_eval | group ${MODE} only"
  echo "============================================================"
  run_group "${MODE}"
else
  echo "Usage: $0 [ 1 | 2 | 3 | 4 | all | summary ]"
  exit 1
fi

echo ""
echo " STAN summary — vertical tables + lineplots under RESULTS/MARFORMER/STAN/reports/"
python -u -m imputer.entity_mf.ranked_eval_report --mode stan
