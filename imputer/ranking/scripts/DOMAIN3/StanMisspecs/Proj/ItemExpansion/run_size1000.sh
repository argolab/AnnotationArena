#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=D3PJIT4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=24GB
#SBATCH --partition=cpu
#SBATCH --time=120:00:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/../../../run_stan_group.sh" \
    --family-label "StanMisspecs" \
    --model-label "Stan Proj" \
    --model-tag "PROJ" \
    --stan-file "STAN/stan_models/tensor_shared_annotator_projection.stan" \
    --output-root "RESULTS/STAN/DOMAIN3-FINAL/MISSPEC/ITEM" \
    --axis "item" \
    --sizes "1000"
