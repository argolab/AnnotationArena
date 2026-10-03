source "/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/config/cluster.sh"
rerun_activate_env >/dev/null 2>&1
# Split the node between workers: unconstrained, each torch process opens one
# thread per core and they contend rather than compute (8.6 s/epoch with six of
# them, against 0.35 s/epoch for one).
export OMP_NUM_THREADS=$(( 24 / 1 )) MKL_NUM_THREADS=$(( 24 / 1 ))
export POOL_ROOT="/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/pool/queue" POOL_MAX_SECONDS=110690
# This allocation is held regardless, so its workers wait for work rather
# than releasing a node we keep paying for either way.
export POOL_WAIT_WHEN_EMPTY=1
for i in $(seq 1 1); do
    POOL_PER_JOB=1 POOL_GPU_BUDGET_GB="" POOL_WORKER_LABEL="local373263.${i}" \
        POOL_LIB_DIR="/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/pool/queue/rev_20260911_144320_2493795" bash "/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/pool/queue/rev_20260911_144320_2493795/worker.sh" "${i}" &
    sleep 5
done
wait
