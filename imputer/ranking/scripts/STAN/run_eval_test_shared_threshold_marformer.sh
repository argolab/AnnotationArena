#!/bin/bash

# Evaluate all SharedThreshold MARFORMER runs on test split across checkpoints,
# then write the best checkpoint summary to TEST_RESULTS/best.json per run.
#
# Run from imputer/ranking:
#   bash scripts/STAN/run_eval_test_shared_threshold_marformer.sh

set -euo pipefail

export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_START=$SECONDS

RESULTS_ROOT="RESULTS/MARFORMER/STAN/SPARSE"
RUN_GLOB="Tensor_400_25_9_ItemTest_SharedThreshold_*_MARFORMER"
DEVICE="${DEVICE:-cpu}"

echo ""
echo "============================================================"
echo " SharedThreshold MARFORMER — Test Evaluation (all checkpoints)"
echo " Results root : ${RESULTS_ROOT}"
echo " Device       : ${DEVICE}"
echo "============================================================"

shopt -s nullglob
RUN_DIRS=( "${RESULTS_ROOT}"/${RUN_GLOB} )
shopt -u nullglob

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "No run directories found under ${RESULTS_ROOT}/${RUN_GLOB}"
    exit 1
fi

for RUN_DIR in "${RUN_DIRS[@]}"; do
    if [ ! -d "$RUN_DIR" ]; then
        continue
    fi

    RUN_NAME="$(basename "$RUN_DIR")"
    TEST_RESULTS_DIR="${RUN_DIR}/TEST_RESULTS"

    echo ""
    echo "--- ${RUN_NAME} ---"

    rm -rf "${TEST_RESULTS_DIR}"

    python -u -m imputer.entity_mf.test \
        --run-dir "$RUN_DIR" \
        --checkpoint all \
        --device "$DEVICE"

    python -u -m imputer.entity_mf.test \
        --run-dir "$RUN_DIR" \
        --checkpoint last \
        --device "$DEVICE"

    python - "$RUN_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
test_dir = run_dir / "TEST_RESULTS"

candidates = []
for path in sorted(test_dir.glob("*.json")):
    if path.name == "best.json":
        continue
    with open(path) as f:
        data = json.load(f)
    missing = data.get("missing", {})
    log_loss = missing.get("log_loss")
    rmse = missing.get("rmse")
    n = missing.get("n", 0)
    if log_loss is None or n == 0:
        continue
    candidates.append((float(log_loss), float(rmse) if rmse is not None else math.inf, path.name, data))

if not candidates:
    raise SystemExit(f"No valid checkpoint results found in {test_dir}")

candidates.sort(key=lambda x: (x[0], x[1], x[2]))
best_log_loss, best_rmse, best_name, best_data = candidates[0]
best_data["selected_from"] = best_name
best_data["selection_metric"] = "missing.log_loss"

best_path = test_dir / "best.json"
with open(best_path, "w") as f:
    json.dump(best_data, f, indent=2)

print(f"  best checkpoint: {best_name}")
print(f"  best missing log_loss={best_log_loss:.6f} rmse={best_rmse:.6f}")
print(f"  saved -> {best_path}")
PY
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " All done. Total time: $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Best summaries saved under each run's TEST_RESULTS/best.json"
echo "============================================================"
