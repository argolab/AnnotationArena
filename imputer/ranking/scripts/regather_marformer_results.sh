#!/usr/bin/env bash
#
# One-shot: rerun ranked_eval for all MARFORMER SummEval (part B) + STAN runs, then
# regenerate vertical tables and lineplot PNGs (same as POST_TRAINING_REPORT_COMMANDS.md).
#
# Requires a GPU (CUDA): ranked_eval loads checkpoints and runs test/val loops.
#
# From anywhere:
#   bash /path/to/imputer/ranking/scripts/regather_marformer_results.sh
#
# Or after:
#   cd ~/AA_new/imputer/ranking && export PYTHONPATH=.
#
# Slurm (adjust partition/time/memory):
#   cd ~/AA_new/imputer/ranking
#   PARTITION=h100 GPUS=1 TIME=04:00:00 CPUS_PER_TASK=4 MEM_PER_CPU=8G \
#     /home/xwang397/bin/sbatch_adapt scripts/regather_marformer_results.sh
#
# Optional — skip a block if you only need the other (still exports PYTHONPATH in subshells):
#   SKIP_SUMMEVAL=1 bash scripts/regather_marformer_results.sh
#   SKIP_STAN=1     bash scripts/regather_marformer_results.sh
#

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RANKING_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "${_RANKING_ROOT}"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -euo pipefail

echo "================================================================"
echo " MARFORMER regather — $(date -Is)"
echo " Root: ${_RANKING_ROOT}"
echo "================================================================"

if [[ -z "${SKIP_SUMMEVAL:-}" ]]; then
  echo ""
  echo ">>> SummEval part B (750 / 1000 / 1280) + report + history summary"
  bash "${_SCRIPT_DIR}/SUMMEVAL/MARFORMER/TRAIN/eval_train_b_750_1000_1280_tmp.sh"
else
  echo ""
  echo ">>> SKIP SummEval (SKIP_SUMMEVAL set)"
fi

if [[ -z "${SKIP_STAN:-}" ]]; then
  echo ""
  echo ">>> STAN (22 runs) + reports"
  bash "${_SCRIPT_DIR}/STAN/eval_stan_marformer_22_tmp.sh"
else
  echo ""
  echo ">>> SKIP STAN (SKIP_STAN set)"
fi

echo ""
echo "================================================================"
echo " Done — $(date -Is)"
echo " SummEval: RESULTS/MARFORMER/SUMMEVAL/reports/"
echo " STAN:     RESULTS/MARFORMER/STAN/reports/"
echo "================================================================"
