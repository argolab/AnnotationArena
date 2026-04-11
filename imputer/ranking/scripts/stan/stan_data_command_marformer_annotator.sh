#!/bin/bash
# Submit STAN MARFORMER annotator-test jobs (10 runs): item dropout off, annotator dropout 0.7,
# masking rate 0.7 — scripts under MARFORMER_ANNOT_DROP/. Run from imputer/ranking:
#   bash scripts/STAN/stan_data_command_marformer_annotator.sh
# Or paste/submit lines individually.

PARTITION=a100 GPUS=1 TIME=24:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER_ANNOT_DROP/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest_12/run_train.sh
PARTITION=a100 GPUS=1 TIME=24:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER_ANNOT_DROP/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest_14/run_train.sh
PARTITION=a100 GPUS=1 TIME=24:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER_ANNOT_DROP/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest_3/run_train.sh
PARTITION=a100 GPUS=1 TIME=24:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER_ANNOT_DROP/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest_6/run_train.sh
PARTITION=a100 GPUS=1 TIME=24:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER_ANNOT_DROP/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest_9/run_train.sh
PARTITION=a100 GPUS=1 TIME=24:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER_ANNOT_DROP/Normal_250_20_9_AnnotatorTest/Normal_250_20_9_AnnotatorTest_12/run_train.sh
PARTITION=a100 GPUS=1 TIME=24:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER_ANNOT_DROP/Normal_250_20_9_AnnotatorTest/Normal_250_20_9_AnnotatorTest_14/run_train.sh
PARTITION=a100 GPUS=1 TIME=24:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER_ANNOT_DROP/Normal_250_20_9_AnnotatorTest/Normal_250_20_9_AnnotatorTest_3/run_train.sh
PARTITION=a100 GPUS=1 TIME=24:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER_ANNOT_DROP/Normal_250_20_9_AnnotatorTest/Normal_250_20_9_AnnotatorTest_6/run_train.sh
PARTITION=a100 GPUS=1 TIME=24:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G EXCLUDE_NODES=c012,c013 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/STAN/MARFORMER_ANNOT_DROP/Normal_250_20_9_AnnotatorTest/Normal_250_20_9_AnnotatorTest_9/run_train.sh
