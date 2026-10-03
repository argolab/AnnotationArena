#!/bin/bash
# A4: how much GPU memory and wall clock does each training context window cost?
#
# Before 500 or 600 can go into the final configuration we need to know it fits.
# Two reference points say the ceiling is real and close: max_item=1200 at K=4000
# needed ~76 GiB and died on an 80 GiB A100, and the 128x12 capacity arm needed
# 92.2 of 93.1 GiB. Peak memory is sampled from nvidia-smi during a 3-epoch run,
# which measures the whole process (allocator reserve included) rather than the
# allocator's own accounting -- the number that actually decides whether a job
# survives.
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env >/dev/null

DRAW="${DRAW:-dataseed-s101}"
OUT="${RERUN_DIR}/threads/context-window/out/a4_memory.json"
TMP="$(mktemp -d)"
export DOMAIN3_DATASET_LABEL=DOMAIN3-SIGMOID DOMAIN3_SPLIT_PREFIX=DOMAIN3-SIGMOID
export DOMAIN3_EXP_PREFIX=domain3_sigmoid DOMAIN3_EPOCHS=3 DOMAIN3_SEED=42
export DOMAIN3_DATA_ROOT="$(rerun_data_root "${DRAW}")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[" > "${OUT}"
first=true
for K in 2000 4000; do
  for CHUNK in 300 500 600 900; do
    export DOMAIN3_MAX_ITEM="${CHUNK}"
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "${TMP}/mem" 2>/dev/null &
    POLL=$!
    START=$(date +%s)
    bash "${RANKING_ROOT}/scripts/DOMAIN3/run_marformer_group.sh" \
        --family-label Marformer --output-root "${TMP}/run" \
        --axis item --protocol transductive --sizes "${K}" > "${TMP}/log" 2>&1 || true
    END=$(date +%s)
    kill "${POLL}" 2>/dev/null || true; wait "${POLL}" 2>/dev/null || true
    PEAK=$(sort -n "${TMP}/mem" | tail -1)
    SPE=$(sed -n 's/.*(\([0-9.]*\)s\/epoch.*/\1/p' "${TMP}/log" | tail -1)
    OOM=$(grep -ci "out of memory" "${TMP}/log" || true)
    ${first} || echo "," >> "${OUT}"; first=false
    printf ' {"K": %s, "max_item": %s, "peak_gpu_mib": %s, "sec_per_epoch": %s, "wall_s": %s, "oom": %s}' \
        "${K}" "${CHUNK}" "${PEAK:-null}" "${SPE:-null}" "$((END-START))" "${OOM}" >> "${OUT}"
    echo "K=${K} chunk=${CHUNK}  peak ${PEAK:-?} MiB  ${SPE:-?} s/epoch  oom=${OOM}"
    rm -rf "${TMP}/run"
  done
done
printf '\n]\n' >> "${OUT}"
rm -rf "${TMP}"
echo "wrote ${OUT}"
