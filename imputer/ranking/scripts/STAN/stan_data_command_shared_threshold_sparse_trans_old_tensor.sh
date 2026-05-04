#!/bin/bash
# Submit shared-threshold synthetic transductive sparse runs
# using the old tensor_model.stan.
# Run from imputer/ranking:
#   bash scripts/STAN/stan_data_command_shared_threshold_sparse_trans_old_tensor.sh

PARTITION=cpu GPUS=0 TIME=36:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold_Stan_OldTensor/run_size10.sh
PARTITION=cpu GPUS=0 TIME=36:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold_Stan_OldTensor/run_size50.sh
PARTITION=cpu GPUS=0 TIME=36:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold_Stan_OldTensor/run_size100.sh
PARTITION=cpu GPUS=0 TIME=36:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold_Stan_OldTensor/run_size200.sh
PARTITION=cpu GPUS=0 TIME=36:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold_Stan_OldTensor/run_size300.sh
