#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --comment=accept_cost
set -uo pipefail
source "${POOL_LIB}"
export REQUIRE_GPU=1
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
# Divide the node between the workers in this job. Without this each torch
# process opens one thread per core and they contend instead of computing.
export OMP_NUM_THREADS="$(( ${SLURM_CPUS_PER_TASK:-24} / POOL_PER_JOB ))"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
echo "job ${SLURM_JOB_ID} on $(hostname) [${SLURM_JOB_PARTITION}]: ${POOL_PER_JOB} workers, OMP_NUM_THREADS=${OMP_NUM_THREADS}"
for i in $(seq 1 "${POOL_PER_JOB}"); do
    POOL_GPU_BUDGET_GB="${POOL_GPU_BUDGET_GB:-}" \
    POOL_WORKER_LABEL="${SLURM_JOB_ID}.${i}" \
        bash "${POOL_WORKER}" "${i}" &
    sleep 5     # stagger, so they do not all race for the same task at t=0
done
wait
echo "job ${SLURM_JOB_ID} complete"
