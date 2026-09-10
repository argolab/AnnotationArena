#!/usr/bin/env python
"""
Summarise the anneal sweep: does a terminal lr decay beat continuing flat?

Reads the per-epoch whole-graph missing-token cross-entropy that training writes
itself, so this needs no GPU and no re-scoring. That metric is NOT the reported
corner metric -- it averages every missing cell in the graph, ~13.6k of them at
K=4000, against the corner's ~316 -- but it is the same metric for both arms and
it has ~40x the sample size, so it is the better instrument for deciding whether
the decay does anything at all. The corner number is scored separately.

Every value is a 5-epoch tail mean. Adjacent-checkpoint sd on these curves is
0.008-0.018, the same order as the effects being measured, so a single final
epoch is not a reading (this is the winner's-curse lesson from W3d, applied to
the endpoint rather than to an argmin).
"""
from __future__ import annotations
import json, statistics
from pathlib import Path

RERUN = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN")
RUN = RERUN / "runs/anneal-s101"
RES = RUN / "RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM"
SIZES = [500, 1000, 2000, 4000]
TAIL = 5


def xents(history_path: Path):
    h = json.loads(history_path.read_text())
    return [e["combined_eval"]["metrics"]["missing"]["rating"]["xent"] for e in h]


def tail_mean(xs):
    return statistics.fmean(xs[-TAIL:])


def main() -> None:
    rows, epochs = [], None
    for k in SIZES:
        rn = f"DOMAIN3-SIGMOID_Item_T_{k}_MARFORMER"
        src = RERUN / ("runs/legacy/dataseed-s101/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/"
                       f"{rn}/training_history.json")
        arms = {a: RES / a / "Marformer/seed42" / rn / "training_history.json"
                for a in ("cont_const", "anneal_cos")}
        if not all(p.is_file() for p in [src, *arms.values()]):
            print(f"  K={k}: not complete yet, skipping")
            continue
        c, a = xents(arms["cont_const"]), xents(arms["anneal_cos"])
        epochs = len(a)
        rows.append({"size": k, "start": tail_mean(xents(src)),
                     "const": tail_mean(c), "cos": tail_mean(a),
                     "delta": tail_mean(a) - tail_mean(c)})
    out = {"epochs": epochs, "tail": TAIL, "metric": "whole-graph missing xent",
           "rows": rows}
    (RERUN / "side/out/anneal_summary.json").write_text(json.dumps(out, indent=1))
    print(f"{'K':>5} {'start':>8} {'+flat':>8} {'+cosine':>8} {'cos-flat':>9} {'cos-start':>10}")
    for r in rows:
        print(f"{r['size']:>5} {r['start']:8.4f} {r['const']:8.4f} {r['cos']:8.4f} "
              f"{r['delta']:+9.4f} {r['cos']-r['start']:+10.4f}")


if __name__ == "__main__":
    main()
