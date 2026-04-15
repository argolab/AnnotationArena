#!/bin/bash
# Submit STAN_sparse MARFORMER training with hard graph mask (--use-graph-mask), 2 runs.
# Same layout as stan_sparse_marformer_commands.sh; outputs use RUN_NAME=*_hard_mask.
#
# Run from anywhere:
#   bash imputer/ranking/scripts/STAN_sparse/stan_sparse_marformer_hard_mask_commands.sh

# _RANKING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# cd "${_RANKING_ROOT}"

PARTITION=a100 GPUS=1 TIME=24:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN_sparse/MARFORMER/Factor_225_25_9_ItemTest/Factor_225_25_9_ItemTest_Size_175/run_train_hard_mask.sh
PARTITION=a100 GPUS=1 TIME=24:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN_sparse/MARFORMER/Normal_225_25_9_ItemTest/Normal_225_25_9_ItemTest_Size_175/run_train_hard_mask.sh
