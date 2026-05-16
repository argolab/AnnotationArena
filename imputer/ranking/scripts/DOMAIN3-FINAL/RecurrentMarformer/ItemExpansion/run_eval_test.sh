#!/bin/bash
# Evaluate Recurrent Marformer run on DOMAIN3-OLD_Item_T_1000.

set -euo pipefail
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_START=$SECONDS
RUN_DIR="RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD/DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p1c2r3c1"

echo "RecurrentMarformer — test eval: ${RUN_DIR}"

if [ ! -d "$RUN_DIR" ]; then
    echo "Run directory not found: ${RUN_DIR}"
    exit 1
fi

TEST_RESULTS_DIR="${RUN_DIR}/TEST_RESULTS"
rm -rf "${TEST_RESULTS_DIR}"

python -u -m imputer.entity_mf.recurrent.test \
    --run-dir "$RUN_DIR" \
    --checkpoint all \
    --device cuda

python -u -m imputer.entity_mf.recurrent.test \
    --run-dir "$RUN_DIR" \
    --checkpoint last \
    --device cuda

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
    raise SystemExit(f"No valid checkpoint results in {test_dir}")

candidates.sort(key=lambda x: (x[0], x[1], x[2]))
_, _, best_name, best_data = candidates[0]
best_data["selected_from"] = best_name
best_data["selection_metric"] = "missing.log_loss"
best_path = test_dir / "best.json"
with open(best_path, "w") as f:
    json.dump(best_data, f, indent=2)
print(f"best: {best_name} -> {best_path}")
PY

echo "Done in $(( (SECONDS - SCRIPT_START) / 60 ))m"
