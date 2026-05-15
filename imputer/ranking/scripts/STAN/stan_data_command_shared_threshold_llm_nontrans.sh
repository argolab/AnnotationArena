#!/bin/bash
# Submit shared-threshold LLM-Rubric non-transductive CPM runs.
# Run from imputer/ranking:
#   bash scripts/STAN/stan_data_command_shared_threshold_llm_nontrans.sh

PARTITION=cpu GPUS=0 TIME=24:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/LLM_RUBRIC/CPM_SHARED_THRESHOLD/NT/run_sizes_10_to_50.sh
PARTITION=cpu GPUS=0 TIME=24:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/LLM_RUBRIC/CPM_SHARED_THRESHOLD/NT/run_size75.sh
PARTITION=cpu GPUS=0 TIME=24:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/LLM_RUBRIC/CPM_SHARED_THRESHOLD/NT/run_size100.sh
PARTITION=cpu GPUS=0 TIME=24:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/LLM_RUBRIC/CPM_SHARED_THRESHOLD/NT/run_size125.sh
PARTITION=cpu GPUS=0 TIME=24:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/LLM_RUBRIC/CPM_SHARED_THRESHOLD/NT/run_size150.sh
PARTITION=cpu GPUS=0 TIME=24:00:00 CPUS_PER_TASK=16 \
  /home/xwang397/bin/sbatch_adapt \
  scripts/LLM_RUBRIC/CPM_SHARED_THRESHOLD/NT/run_size175.sh
