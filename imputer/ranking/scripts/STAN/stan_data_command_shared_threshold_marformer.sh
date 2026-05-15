#!/bin/bash
# Submit shared-threshold synthetic MARFORMER training runs.
# Run from imputer/ranking:
#   bash scripts/STAN/stan_data_command_shared_threshold_marformer.sh

PARTITION=l40s GPUS=1 TIME=48:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold/Marformer/run_small_sizes.sh
PARTITION=l40s GPUS=1 TIME=48:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold/Marformer/run_size200.sh
PARTITION=l40s GPUS=1 TIME=48:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold/Marformer/run_size300.sh

PARTITION=l40s GPUS=1 TIME=48:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold/Marformer-NonTrans/run_small_sizes.sh
PARTITION=l40s GPUS=1 TIME=48:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold/Marformer-NonTrans/run_size200.sh
PARTITION=l40s GPUS=1 TIME=48:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/SPARSE/Tensor_400_25_9_ItemTest_SharedThreshold/Marformer-NonTrans/run_size300.sh
