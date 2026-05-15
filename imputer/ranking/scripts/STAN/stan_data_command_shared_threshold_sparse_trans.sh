#!/bin/bash
# Submit shared-threshold synthetic transductive sparse runs.
# Run from imputer/ranking:
#   bash scripts/STAN/stan_data_command_shared_threshold_sparse_trans.sh

PARTITION=cpu GPUS=0 TIME=48:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold_Stan/run_size10.sh
PARTITION=cpu GPUS=0 TIME=48:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold_Stan/run_size50.sh
PARTITION=cpu GPUS=0 TIME=48:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold_Stan/run_size100.sh
PARTITION=cpu GPUS=0 TIME=48:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold_Stan/run_size200.sh
PARTITION=cpu GPUS=0 TIME=48:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold_Stan/run_size300.sh
