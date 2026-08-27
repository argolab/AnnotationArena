#!/bin/bash
# Per-commit regression walk over the optimisation branch.
#
# For each commit, in a throwaway detached worktree:
#   1. run that commit's own test_equiv.py -- the contemporaneous numerical
#      check, so each step is validated against what it claimed to replace;
#   2. train item_T for a fixed short run and record the final CE, so a change
#      that passes the unit check but drifts the optimisation still shows up.
#
#   walk_commits.sh <jobid> [epochs] [seed]
set -uo pipefail

JOBID="$1"; EPOCHS="${2:-15}"; SEED="${3:-42}"
OPT=/weka/scratch/jhu/jeisner1/xwang397/Marformer-opt
WT=/weka/scratch/jhu/jeisner1/xwang397/Marformer-walk
DATA=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/replicate-01/DATA
SPLIT="${DATA}/STAN/DOMAIN3-SIGMOID/ItemSplits/Transductive/DOMAIN3-SIGMOID_Item_T_300"

COMMITS=$(cd "${OPT}" && git rev-list --reverse patches..marformer-optimization)

printf '%-10s %-52s %-10s %-10s\n' commit subject equiv final_ce
printf '%.0s-' {1..86}; echo

for C in ${COMMITS}; do
    SUBJ=$(cd "${OPT}" && git log -1 --format=%s "${C}" | cut -c1-50)
    rm -rf "${WT}"
    (cd "${OPT}" && git worktree remove --force "${WT}" >/dev/null 2>&1)
    (cd "${OPT}" && git worktree add --detach "${WT}" "${C}" >/dev/null 2>&1)
    mkdir -p "${WT}/RERUN/runs"
    ln -sfn /weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/replicate-01 \
            "${WT}/RERUN/runs/replicate-01"

    EQUIV=n/a
    if [[ -f "${WT}/RERUN/bench/test_equiv.py" ]]; then
        if srun --jobid="${JOBID}" --overlap bash -lc "
              eval \"\$(conda shell.bash hook)\"; conda activate torch
              cd ${WT}/imputer/ranking
              export PYTHONPATH=\$PWD:${WT}
              python -m RERUN.bench.test_equiv --data-dir ${SPLIT}" \
              >"${WT}/equiv.log" 2>&1; then
            EQUIV=PASS
        else
            EQUIV=FAIL
        fi
    fi

    # Two attempts: srun into a shared allocation occasionally fails to launch,
    # and a flaky launch must not be reported as a regression.
    CE=CRASH
    for attempt in 1 2; do
        srun --jobid="${JOBID}" --overlap bash -lc "
            eval \"\$(conda shell.bash hook)\"; conda activate torch
            cd ${WT}/imputer/ranking; export PYTHONPATH=\$PWD:${WT}
            rm -rf ${WT}/out
            python -u -m imputer.entity_mf.train --data-dir ${SPLIT} \
              --run-name run --output-root ${WT}/out --seed ${SEED} \
              --embedding-dim 80 --num-layers 8 --attention-heads 4 --d-ff 128 \
              --num-ffn-layers 1 --dropout 0.1 --item-dropout-rate 1.0 \
              --annotator-dropout-rate 0.0 --epochs ${EPOCHS} --lr 2e-4 \
              --lr-schedule none --lr-min 1e-5 --weight-decay 0.01 --masking-rate 0.15 \
              --mask-augmentations 5 --masked-loss-weight 15.0 --observed-loss-weight 1.0 \
              --device cuda --max-item 300 --type-embedding-init kaiming \
              --transductive-learning --no-per-head-rel --scale-shared-rel --use-pointer \
              --overwrite-existing-data" >"${WT}/train.log" 2>&1
        CE=$(srun --jobid="${JOBID}" --overlap bash -lc "
            eval \"\$(conda shell.bash hook)\"; conda activate torch
            python ${OPT}/RERUN/bench/_final_ce.py ${WT}/out/run/training_history.json" \
            2>/dev/null | tail -1)
        [[ "${CE}" != "CRASH" && -n "${CE}" ]] && break
        echo "  (attempt ${attempt} failed for ${C:0:9}; see ${WT}/train.log)" >&2
    done

    printf '%-10s %-52s %-10s %-10s\n' "${C:0:9}" "${SUBJ}" "${EQUIV}" "${CE:-CRASH}"
done

(cd "${OPT}" && git worktree remove --force "${WT}" >/dev/null 2>&1)
