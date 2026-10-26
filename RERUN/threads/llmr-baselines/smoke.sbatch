#!/bin/bash
#SBATCH --job-name=LLMRBASE
#SBATCH --partition=med
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --comment=accept_cost
#SBATCH --output=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual/logs/llmrbase_%j.out
#SBATCH --error=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual/logs/llmrbase_%j.err
set -euo pipefail
RERUN_DIR=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
# ReMasker needs timm, which is not in the conda env and cannot be pip-installed
# from a compute node (no network). It is vendored instead: --no-deps into a
# side directory so nothing in the shared env moves, with torchvision pinned to
# 0.25.0 because that is the build that matches torch 2.10 -- the latest wheel
# loads but then dies with "operator torchvision::nms does not exist".
export PYTHONPATH="/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/vendor/pylibs:${PYTHONPATH:-}"
cd /weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking
export OMP_NUM_THREADS=16
SPLIT="${SPLIT:-LLMRubric_225_25_9_175}"
# One split each, to price the full 2x10 grid before committing to it.
set +e
for M in remasker miwae; do
  case "$M" in
    remasker) EP=300  ARGS="--embed-dim 32 --depth 4 --num-heads 4 --decoder-depth 2 --mask-ratio 0.5 --weight-decay 0.05" ;;
    miwae)    EP=3000 ARGS="--latent-dim 16 --hidden-dim 128 --K 20 --L 100" ;;
  esac
  echo "=== ${M} ${SPLIT} (${EP} epochs) ==="
  # /usr/bin/time is not installed on the med nodes; SECONDS is always there.
  T0=$SECONDS
  python BASELINES/run_baselines.py --method "$M" \
    --data-bundle "DATA/LLM_RUBRIC/${SPLIT}/data_bundle.json" \
    --output-dir  "RESULTS/BASELINES/$(echo $M | tr a-z A-Z)/LLMRUBRIC/${SPLIT}" \
    --epochs "$EP" --batch-size 64 --lr 1e-4 --seed 42 --device cpu $ARGS
  echo "  WALL ${M} ${SPLIT}: $(( SECONDS - T0 ))s"
done
