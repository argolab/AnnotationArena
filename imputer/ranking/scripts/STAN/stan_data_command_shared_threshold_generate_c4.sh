#!/bin/bash
# Submit shared-threshold (C=4) synthetic data generation job.
# Run from imputer/ranking:
#   bash scripts/STAN/stan_data_command_shared_threshold_generate_c4.sh

PARTITION=cpu GPUS=0 TIME=24:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold/generate_data_c4.sh
