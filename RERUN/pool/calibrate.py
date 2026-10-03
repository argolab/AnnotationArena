#!/usr/bin/env python
"""How much longer do tasks actually take than their estimate?

The cost model behind each task's estimate was measured on an idle GPU, but pool
workers share a node on purpose, so real durations run long -- the first
completed task took 13 minutes against a 4-minute estimate. A worker that trusts
the raw estimate claims work it cannot finish and the walltime kills it
mid-training, wasting everything that task had done.

Prints a multiplier: the median of (actual / estimated) over completed tasks,
with headroom, floored at the caller's default. Falls back to the floor until
three tasks have finished, because two samples are not a median.

    calibrate.py <pool root> <floor>
"""
import json, pathlib, statistics, sys

root, floor = pathlib.Path(sys.argv[1]), float(sys.argv[2])
ratios = []
for d in (root / "state").glob("*.done"):
    try:
        parts = d.read_text().split()
        if not parts or not parts[-1].endswith("s"):
            continue          # a task marked done from its output marker, never timed
        actual = float(parts[-1][:-1])
        eta = json.loads((root / "tasks" / f"{d.stem}.json").read_text())["eta_s"]
        if eta > 0 and actual > 0:
            ratios.append(actual / eta)
    except Exception:
        continue
print(f"{max(floor, statistics.median(ratios) * 1.3):.2f}" if len(ratios) >= 3
      else f"{floor:.2f}")
