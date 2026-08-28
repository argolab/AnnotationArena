"""One-off cross-check: does split_missing_metrics reproduce the stored metric?

The pooled log-loss it computes must equal TEST_RESULTS/last.json when run at
the same chunk size test.py uses (500, hardcoded -- discrepancy D20). If it
does, the partition split derived from the same forward pass can be trusted.
"""
import json
from pathlib import Path

import torch

from RERUN.analysis.split_missing_metrics import evaluate

base = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN/runs/seeds-01"
            "/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM")
dev = torch.device("cuda")
for size in (500, 1000):
    rd = base / f"DOMAIN3-SIGMOID_Item_T_{size}_MARFORMER"
    stored = json.load(open(rd / "TEST_RESULTS/last.json"))["missing"]
    print(f"K={size}  stored last.json         log_loss={stored['log_loss']:.6f}  n={stored['n']}")
    for mi in (500, 300):
        rows, _ = evaluate(rd, dev, "last", mi)
        pooled = sum(r[1] for r in rows) / len(rows)
        tag = "max_item=500 (as test.py)" if mi == 500 else "max_item=300 (as trained)"
        print(f"K={size}  {tag:<26} log_loss={pooled:.6f}  n={len(rows)}")
    print()
