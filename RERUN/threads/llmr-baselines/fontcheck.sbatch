#!/bin/bash
#SBATCH --job-name=FONTCHK
#SBATCH --partition=med
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --comment=accept_cost
#SBATCH --output=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual/logs/fontchk_%j.out
#SBATCH --error=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/final-dual/logs/fontchk_%j.err
set -euo pipefail
RERUN_DIR=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
cd /weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking
# Redraw from the cached payloads only -- no model is evaluated, so this is a
# pure layout check of the font scale and needs no GPU.
python scripts/LLMRubric/plot_llmrubric_cpm_sigmoid.py --cache --device cpu
