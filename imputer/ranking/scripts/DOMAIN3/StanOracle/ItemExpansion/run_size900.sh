#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=D3SOIT900
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=24GB
#SBATCH --partition=cpu
#SBATCH --time=120:00:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/../../run_stan_group.sh" \
    --family-label "StanOracle" \
    --model-label "Stan Oracle" \
    --model-tag "SHARED_THRESH_ORACLE" \
    --stan-file "STAN/stan_models/shared_threshold_tensor_model.stan" \
    --output-root "RESULTS/STAN/DOMAIN3-FINAL/ORACLE/ITEM" \
    --axis "item" \
    --sizes "900"
