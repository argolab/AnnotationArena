#!/usr/bin/env python
"""Compact the E1 output into the payload the report page embeds."""
import json, sys
from pathlib import Path

RERUN = Path(__file__).resolve().parents[2]
OUT = RERUN / "threads/early-stopping/out"

def r(x, n=4):
    return None if x is None else round(float(x), n)

rows = json.loads((OUT / "e1_seed42.json").read_text())
payload = []
for row in rows:
    payload.append({
        "K": row["K"], "seed": row["seed"],
        "eStar": row["e_star"], "eOracle": row["e_oracle_full"],
        "atStar": r(row["test_at_e_star"]), "floor": r(row["test_floor_full"]),
        "regret": r(row["regret"]), "final": r(row["test_final_full"]),
        "noStop": r(row["cost_of_no_stopping"]),
        "obsStar": r(row["observed_at_e_star_full"], 8),
        "obsFinal": r(row["observed_final_full"], 8),
        # the dev curve is scored every epoch; every other point is plenty to draw
        "devEp": row["dev_epochs"][::2], "devLoss": [r(v) for v in row["dev_loss"][::2]],
        "fullEp": row["full_epochs"], "fullLoss": [r(v) for v in row["full_missing"]],
        "searchEp": row["search_epochs"], "searchLoss": [r(v) for v in row["search_missing"]],
    })
p = OUT / "e1_payload.json"
p.write_text(json.dumps(payload, separators=(",", ":")))
print("wrote", p, p.stat().st_size, "bytes")
n = len(payload)
print("mean regret", round(sum(x["regret"] for x in payload) / n, 4),
      "| worst", round(max(x["regret"] for x in payload), 4),
      "| mean cost of no stopping", round(sum(x["noStop"] for x in payload) / n, 4),
      "| worst", round(max(x["noStop"] for x in payload), 4))
print("observed loss range:", min(x["obsStar"] for x in payload), max(x["obsStar"] for x in payload))
