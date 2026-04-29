#!/bin/bash

# Local — evaluate all saved checkpoints for every DOMAIN3 sparse Marformer run
# and save the best test checkpoint summary to TEST_RESULTS/best.json.

set -euo pipefail

cd /Users/prabhavsingh/Documents/JHU/JHUResearch/EntityMarformer/imputer/ranking
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_START=$SECONDS

RESULTS_ROOT="RESULTS/MARFORMER/DOMAIN3/SPARSE"
RUN_GLOB="*_MARFORMER"

echo ""
echo "============================================================"
echo " DOMAIN3 SPARSE — Test Evaluation (all checkpoints)"
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
    TRAIN_CONFIG="${RUN_DIR}/train_config.json"

    echo ""
    echo "--- ${RUN_NAME} ---"

    python - "$TRAIN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
with open(cfg_path) as f:
    cfg = json.load(f)

data = cfg.get("data", {})
data_dir = data.get("data_dir")
if not data_dir:
    raise SystemExit(0)

current = Path(data_dir)
if current.exists():
    raise SystemExit(0)

old_prefix = Path("DATA/STAN/DOMAIN3-ITEM")
new_prefix = Path("DATA/STAN")
fallbacks = []

try:
    rel = current.relative_to(old_prefix)
    fallbacks.append(new_prefix / rel)
except ValueError:
    pass

old_sparse_prefix = Path("DATA/STAN/SPARSE")
old_sparse_root = Path("DATA/STAN/OLD-SPARSE")
try:
    rel = current.relative_to(old_sparse_prefix)
    fallbacks.append(old_sparse_root / rel)
except ValueError:
    pass

for candidate in fallbacks:
    if candidate.exists():
        cfg["data"]["data_dir"] = str(candidate)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  patched train_config.json data_dir -> {candidate}")
        raise SystemExit(0)

fallback_msg = ", ".join(str(p) for p in fallbacks) if fallbacks else "no fallback candidates"
raise SystemExit(
    f"Missing data_dir for {cfg_path.parent.name}: {current} does not exist, "
    f"and fallback(s) {fallback_msg} were not found."
)
PY

    rm -rf "${TEST_RESULTS_DIR}"

    python -u -m imputer.entity_mf.test \
        --run-dir    "$RUN_DIR" \
        --checkpoint all \
        --device     cpu

    python -u -m imputer.entity_mf.test \
        --run-dir    "$RUN_DIR" \
        --checkpoint last \
        --device     cpu

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
