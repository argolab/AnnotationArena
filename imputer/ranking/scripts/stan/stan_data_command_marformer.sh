#!/bin/bash
# Submit all STAN MARFORMER training jobs (22 runs). Run from imputer/ranking:
#   bash scripts/STAN/stan_data_command_marformer.sh
# Or paste/submit lines individually.

PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest_12/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest_14/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest_3/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest_6/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest_9/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Factor_650_20_9_ItemTest/Factor_650_20_9_ItemTest_10/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Factor_650_20_9_ItemTest/Factor_650_20_9_ItemTest_100/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Factor_650_20_9_ItemTest/Factor_650_20_9_ItemTest_250/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Factor_650_20_9_ItemTest/Factor_650_20_9_ItemTest_50/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Factor_650_20_9_ItemTest/Factor_650_20_9_ItemTest_500/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Factor_650_20_9_ItemTest/Factor_650_20_9_ItemTest_600/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Normal_250_20_9_AnnotatorTest/Normal_250_20_9_AnnotatorTest_12/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Normal_250_20_9_AnnotatorTest/Normal_250_20_9_AnnotatorTest_14/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Normal_250_20_9_AnnotatorTest/Normal_250_20_9_AnnotatorTest_3/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Normal_250_20_9_AnnotatorTest/Normal_250_20_9_AnnotatorTest_6/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Normal_250_20_9_AnnotatorTest/Normal_250_20_9_AnnotatorTest_9/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Normal_650_20_9_ItemTest/Normal_650_20_9_ItemTest_10/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Normal_650_20_9_ItemTest/Normal_650_20_9_ItemTest_100/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Normal_650_20_9_ItemTest/Normal_650_20_9_ItemTest_250/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Normal_650_20_9_ItemTest/Normal_650_20_9_ItemTest_50/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Normal_650_20_9_ItemTest/Normal_650_20_9_ItemTest_500/run_train.sh
PARTITION=a100 GPUS=1 TIME=8:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER/Normal_650_20_9_ItemTest/Normal_650_20_9_ItemTest_600/run_train.sh
