#!/bin/bash
# Submit STAN_sparse MARFORMER training (2 runs).
#
# This file lives at imputer/ranking/scripts/STAN_sparse/ — we cd to imputer/ranking so
# sbatch_adapt sees the same cwd as scripts/STAN/stan_data_command_marformer.sh.
# (If you submit from the AA_new repo root without this, jobs get CWD=AA_new and
# "bash: scripts/STAN_sparse/.../run_train.sh: No such file or directory".)
#
# Run from anywhere:
#   bash imputer/ranking/scripts/STAN_sparse/stan_sparse_marformer_commands.sh
# Or after: cd imputer/ranking
#   bash scripts/STAN_sparse/stan_sparse_marformer_commands.sh
#
# Pasted one-liners must also be run from imputer/ranking (not AA_new).

_RANKING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${_RANKING_ROOT}"

PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN_sparse/MARFORMER/Factor_225_25_9_ItemTest/Factor_225_25_9_ItemTest_Size_175/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN_sparse/MARFORMER/Normal_225_25_9_ItemTest/Normal_225_25_9_ItemTest_Size_175/run_train.sh
