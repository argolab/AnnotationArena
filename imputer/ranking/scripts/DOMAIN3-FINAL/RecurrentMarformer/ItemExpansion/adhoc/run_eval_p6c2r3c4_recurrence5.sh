#!/bin/bash
# Adhoc test eval: p6c2r3c4 trained with num_recurrence=3, evaluate with r=5.
# Eval chunk size: MAX_ITEM (default 300).
#
#   cd imputer/ranking && export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/adhoc/run_eval_p6c2r3c4_recurrence5.sh
#
# Writes to:
#   RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12/
#     DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p6c2r3c4/ADHOC_TEST_EVAL_num_recurrence_5/

set -euo pipefail
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../../.."

RUN_DIR="${RUN_DIR:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12/DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p6c2r3c4}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/ADHOC_TEST_EVAL_num_recurrence_5}"
NUM_RECURRENCE="${NUM_RECURRENCE:-5}"
MAX_ITEM="${MAX_ITEM:-300}"
DEVICE="${DEVICE:-cuda}"

echo "============================================================"
echo " Adhoc test eval | p6c2r3c4 | num_recurrence=${NUM_RECURRENCE}"
echo " MAX_ITEM: ${MAX_ITEM}"
echo " RUN_DIR : ${RUN_DIR}"
echo " OUT_DIR : ${OUT_DIR}"
echo "============================================================"

python -u -m imputer.entity_mf.recurrent.test \
    --run-dir "$RUN_DIR" \
    --out-dir "$OUT_DIR" \
    --num-recurrence "$NUM_RECURRENCE" \
    --max-item "$MAX_ITEM" \
    --checkpoint all \
    --device "$DEVICE"

python - "$OUT_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
candidates = []
for path in sorted(out_dir.glob("*.json")):
    if path.name == "best.json":
        continue
    data = json.loads(path.read_text())
    missing = data.get("missing", {})
    ll, n = missing.get("log_loss"), missing.get("n", 0)
    if ll is None or n == 0:
        continue
    rmse = missing.get("rmse")
    candidates.append((float(ll), float(rmse) if rmse is not None else math.inf, path.name, data))
if candidates:
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    _, _, name, best = candidates[0]
    best["selected_from"] = name
    (out_dir / "best.json").write_text(json.dumps(best, indent=2))
    print(f"best checkpoint: {name}  missing log_loss={candidates[0][0]:.4f}")
else:
    print("(no valid test metrics)")
PY

echo ""
echo "Done. JSONs under ${OUT_DIR}/"
