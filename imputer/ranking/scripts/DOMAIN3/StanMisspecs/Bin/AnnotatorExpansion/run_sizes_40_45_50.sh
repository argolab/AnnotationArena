#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=D3BNAN3
#SBATCH --output=/home/psingh54/scratchjeisner1/psingh54/Marformer/imputer/ranking/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=24GB
#SBATCH --partition=cpu
#SBATCH --time=40:00:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/../../../run_stan_group.sh" \
    --family-label "StanMisspecs" \
    --model-label "Stan Bin" \
    --model-tag "BIN" \
    --stan-file "STAN/stan_models/tensor_shared_annotator_binning_raw_z.stan" \
    --output-root "RESULTS/STAN/DOMAIN3-FINAL/MISSPEC/ANNOT" \
    --axis "annot" \
    --sizes "40,45,50"
