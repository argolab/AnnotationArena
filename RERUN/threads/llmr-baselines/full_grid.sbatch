#!/bin/bash
#SBATCH --job-name=LLMRBASE
#SBATCH --partition=med
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --comment=accept_cost
#SBATCH --output=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual/logs/llmrgrid_%j.out
#SBATCH --error=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual/logs/llmrgrid_%j.err
set -euo pipefail
# MIWAE and ReMasker, every LLM-Rubric split. They are absent from the current
# RESULTS tree and from every payload cache on disk (including the legacy
# replicate-01 one), so figures 10 and 11 draw them as "no data" while the prose
# still quotes their numbers. Nothing stored can recover the old predictions --
# only the published PNG, and reading numbers off a PNG is not a measurement.
#
# timm is vendored under RERUN/vendor/pylibs (--no-deps, torchvision pinned to
# 0.25.0 to match torch 2.10); the conda env is untouched.
export PYTHONPATH="/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/vendor/pylibs:${PYTHONPATH:-}"
RERUN_DIR=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
cd /weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking
export OMP_NUM_THREADS=16

SPLITS=(10 20 30 40 50 75 100 125 150 175)
for M in remasker miwae; do
  case "$M" in
    remasker) EP=300  ARGS="--embed-dim 32 --depth 4 --num-heads 4 --decoder-depth 2 --mask-ratio 0.5 --weight-decay 0.05" ;;
    miwae)    EP=3000 ARGS="--latent-dim 16 --hidden-dim 128 --K 20 --L 100" ;;
  esac
  UP=$(echo "$M" | tr a-z A-Z)
  for K in "${SPLITS[@]}"; do
    S="LLMRubric_225_25_9_${K}"
    T0=$SECONDS
    python BASELINES/run_baselines.py --method "$M" \
      --data-bundle "DATA/LLM_RUBRIC/${S}/data_bundle.json" \
      --output-dir  "RESULTS/BASELINES/${UP}/LLMRUBRIC/${S}" \
      --epochs "$EP" --batch-size 64 --lr 1e-4 --seed 42 --device cpu $ARGS
    echo "  WALL ${M} K=${K}: $(( SECONDS - T0 ))s"
  done
done
echo "ALLDONE"
