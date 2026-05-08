#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=D3FMANN40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24GB
#SBATCH --gpus=1
#SBATCH --partition=a100
#SBATCH --time=96:00:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/../../run_marformer_group.sh" \
    --family-label "Marformer" \
    --output-root "RESULTS/MARFORMER/DOMAIN3-FINAL/ANNOT" \
    --axis "annot" \
    --protocol "transductive" \
    --sizes "40"
