#!/bin/bash
#SBATCH --job-name=ARM-EVAL-S1
#SBATCH --partition=med
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --comment=accept_cost
# Same as eval_arms.sbatch, but scores from the side/s1-featnorm worktree.
#
# Why a separate file: the S1 arms were trained with --feature-only-norm, so
# their LayerNorm parameters have width feature_dim (75), not model_dim (80).
# Rebuilding those checkpoints with the `patches` model.py fails with a
# size-mismatch on every blocks.*.norm_*.alpha (job 315450). The flag is
# recorded in train_config.json, so all the scorer needs is the branch's
# model.py on the import path -- which is what cd'ing into the worktree does.
# Nothing else differs; RERUN/runs and RERUN/side are symlinks back to main, so
# the same script scores the same checkpoints into the same out/ directory.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
S1_ROOT="${S1_ROOT:-/weka/scratch/jhu/jeisner1/xwang397/Marformer-s1}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
# cluster.sh exports PYTHONPATH=<main>/imputer/ranking, and that -- not the cwd
# -- is what "import model" resolves against. Repointing the cwd alone is not
# enough (job 315714 failed identically to 315450). Override both.
export RANKING_ROOT="${S1_ROOT}/imputer/ranking"
export DOMAIN3_REPO_ROOT_OVERRIDE="${RANKING_ROOT}"
export PYTHONPATH="${RANKING_ROOT}:${PYTHONPATH}"
cd "${S1_ROOT}"
mkdir -p "${RERUN_DIR}/side/out"
python "${RERUN_DIR}/side/s7_curves.py" \
    --run-dir "${RERUN_DIR}/runs/${AE_RUN}" \
    --eval-max-item "${AE_MI:-500}" \
    --device cpu \
    -o "${RERUN_DIR}/side/out/arms_${AE_RUN}.json"
