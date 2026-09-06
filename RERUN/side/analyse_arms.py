#!/usr/bin/env python
"""
Paired comparison of arm-sweep results.

Domain 3's unpaired noise makes any unpaired arm comparison worthless: the
generative draw alone moves the corner metric by sd 0.085, six times the size
of the effects these suites are looking for. So every number here is a PAIRED
difference against a named reference arm, over the (draw, model seed, K) cells
both arms share, and the reported spread is the spread of those differences.

    python analyse_arms.py --runs s2item-s101,s2item-s105 --ref d2_item1.0
"""
from __future__ import annotations
import argparse, json, statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Decision threshold, nats. 0.03 without tail-averaging; 0.015 is resolvable
# with tail=5 at K<=2000, where the paired sd drops to ~0.016.
THRESHOLD = 0.015


def load(runs: list[str], tail: int) -> dict:
    """
    (arm, family, draw, seed, K) -> corner log-loss, averaged over the last
    `tail` periodic checkpoints.

    Averaging the tail is a variance reduction, not a cosmetic choice. Measured
    on resid-01 (job 300293), the paired between-arm sd across model seeds:

        checkpoints averaged      K=100    K=300    K=500     mean
        1 (the last one)         0.0721   0.0336   0.0376   0.0478
        5                        0.0253   0.0160   0.0368   0.0260

    -- 1.8x tighter overall and 2.1x at K=300, worth roughly 4x the replicates
    for no GPU at all. It works because much of the seed-to-seed spread is SGD
    endpoint jitter rather than a real difference in where the arms converge.

    IMPORTANT LIMIT: this is only valid where the corner curve has flattened.
    The K=4000 transductive run is still descending at epoch 399, so averaging
    its tail would average over a trend and bias the number upward. Use tail=1
    at K=4000 until a longer run shows it has converged.
    """
    out: dict = {}
    for run in runs:
        p = HERE / "out" / f"arms_{run}.json"
        if not p.is_file():
            print(f"  [missing] {p.name}")
            continue
        rows = json.loads(p.read_text())
        # A run is finished only if it wrote training_history.json. `last.ckpt`
        # is rewritten by Lightning every epoch, so its existence means the run
        # STARTED, not that it finished -- using it as a completion marker
        # silently mixes half-trained runs into an arm comparison.
        rows = [r for r in rows
                if (Path(r["run"]) / "training_history.json").is_file()]
        draw = run.rsplit("-", 1)[-1]
        for r in rows:
            series = [x for x in r.get("series", []) if x.get("epoch", -1) >= 0]
            if not series:
                continue
            vals = [x["corner"] for x in series[-tail:]]
            parts = Path(r["run"]).parts
            # .../<AXIS>/<tag>/<family>/seed<N>/<run name>
            tag, fam = parts[-4], parts[-3]
            out[(tag, fam, draw, r["seed"], r["size"])] = sum(vals) / len(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run ids")
    ap.add_argument("--ref", required=True, help="reference arm tag")
    ap.add_argument("--tail", type=int, default=5,
                    help="average the corner loss over the last N periodic "
                         "checkpoints (1 = the final checkpoint only). Use 1 "
                         "where the curve has not flattened -- see load().")
    a = ap.parse_args()
    runs = [x for x in a.runs.split(",") if x]
    data = load(runs, a.tail)
    if not data:
        print("no rows"); return

    arms = sorted({k[0] for k in data})
    cells = sorted({(k[1], k[2], k[3], k[4]) for k in data})
    sizes = sorted({k[4] for k in data})
    print(f"arms={len(arms)}  cells={len(cells)}  sizes={sizes}\n")

    hdr = f"{'arm':<16}" + "".join(f"{('K=%d' % s):>22}" for s in sizes) + f"{'ALL':>22}"
    print(hdr); print("-" * len(hdr))
    for arm in arms:
        line = f"{arm:<16}"
        alld = []
        for s in sizes + [None]:
            ds = [data[(arm, f, d, sd, k)] - data[(a.ref, f, d, sd, k)]
                  for (f, d, sd, k) in cells
                  if (s is None or k == s)
                  and (arm, f, d, sd, k) in data and (a.ref, f, d, sd, k) in data]
            if s is None:
                ds = alld
            else:
                alld += ds
            if not ds:
                line += f"{'-':>22}"; continue
            m = st.mean(ds)
            sd = st.stdev(ds) if len(ds) > 1 else 0.0
            se = sd / len(ds) ** 0.5 if len(ds) > 1 else 0.0
            mark = "*" if abs(m) > THRESHOLD and abs(m) > 2 * se else " "
            line += f"{m:>+9.4f} +/-{se:>6.4f} (n={len(ds):>2}){mark}"[:22].rjust(22)
        print(line)
    print(f"\nreference arm: {a.ref} (its own row is 0 by construction)")
    print(f"* = paired mean exceeds the {THRESHOLD}-nat threshold AND 2 standard errors")
    print("\nabsolute corner log-loss, reference arm:")
    for s in sizes:
        vs = [data[(a.ref, f, d, sd, k)] for (f, d, sd, k) in cells
              if k == s and (a.ref, f, d, sd, k) in data]
        if vs:
            print(f"  K={s:<6} {st.mean(vs):.4f}  (n={len(vs)}, sd {st.stdev(vs) if len(vs)>1 else 0:.4f})")


if __name__ == "__main__":
    main()
