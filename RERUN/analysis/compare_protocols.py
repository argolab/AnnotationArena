#!/usr/bin/env python
"""
Compare checkpoint-selection protocols on the same splits.

Four ways to answer "which epoch do we report?", all on identical data and
seeds, so the differences are the protocol and nothing else:

  last          epoch 399, no selection. Honest but leaves overfitting in.
  best-on-test  argmin over test-scored checkpoints. What the paper's figures
                do; optimistic by construction (D14).
  val-block     stop where a held-out block of items/annotators says to, then
                refit on the whole training partition for that many epochs.
  val-mcar      same, with a random-cell holdout instead of a block.

The two validation columns are the *refit* models' last checkpoints, so nothing
in them was ever chosen on test.

    python compare_protocols.py --baseline RERUN/runs/seeds-01 \
        --arm val-block:RERUN/runs/val-block-01 \
        --arm val-mcar:RERUN/runs/val-mcar-01 [--axis item] [--family Marformer]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from pathlib import Path

RUN_RE = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")
SEED_RE = re.compile(r"^seed(\d+)$")
FAMILIES = (("MARFORMER", "Marformer"), ("MARFORMER-NT", "MarformerNT"))


def read(p: Path):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def scan(run_dir: Path, want_variant: str | None):
    """{(family, axis, size, seed): {last, best, best_epoch, epochs}}"""
    out = {}
    for fam_dir, family in FAMILIES:
        base = Path(run_dir) / "RESULTS" / fam_dir / "DOMAIN3-SIGMOID"
        if not base.is_dir():
            continue
        for d in sorted(base.rglob("*_MARFORMER")):
            m = RUN_RE.search(d.name)
            if not m:
                continue
            rel = str(d.relative_to(base))
            if want_variant:
                # the refit tree, not the search tree that sits beside it
                if f"{want_variant}/" not in rel or f"{want_variant}-search/" in rel:
                    continue
            elif "val" in rel:
                continue  # baseline scan: ignore any validation-protocol subtree
            sm = next((SEED_RE.match(p.name) for p in d.parents if SEED_RE.match(p.name)), None)
            seed = int(sm.group(1)) if sm else 42
            tr = d / "TEST_RESULTS"
            cands = []
            last = None
            for path in sorted(tr.glob("*.json")):
                if path.name in {"best.json", "launcher_metadata.json"}:
                    continue
                data = read(path)
                miss = (data or {}).get("missing", {})
                if miss.get("log_loss") is None or not miss.get("n"):
                    continue
                cands.append((float(miss["log_loss"]), float(miss.get("rmse") or math.inf),
                              path.name, miss))
                if path.name == "last.json":
                    last = miss
            if not cands:
                continue
            cands.sort(key=lambda r: (r[0], r[1], r[2]))
            cfg = read(d / "train_config.json") or {}
            out[(family, m.group(1).lower(), int(m.group(3)), seed)] = {
                "last": last, "best": cands[0][3], "best_source": cands[0][2],
                "epochs": cfg.get("training", {}).get("epochs"),
            }
    return out


def agg(values):
    v = [x for x in values if x is not None]
    if not v:
        return None, None
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--arm", action="append", default=[],
                    help="name:run-dir; the variant tag is derived as val<name-suffix>")
    ap.add_argument("--metric", default="log_loss")
    ap.add_argument("--axis", default="")
    ap.add_argument("--family", default="")
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()

    base = scan(Path(args.baseline), None)
    arms = {}
    for spec in args.arm:
        name, path = spec.split(":", 1)
        variant = "val" + name.split("-")[-1]
        arms[name] = scan(Path(path), variant)

    cells = sorted({k[:3] for k in base}, key=lambda t: (t[0], t[1], t[2]))
    rows = []
    for family, axis, size in cells:
        if args.family and family != args.family:
            continue
        if args.axis and axis != args.axis:
            continue
        seeds = sorted(s for (f, a, z, s) in base if (f, a, z) == (family, axis, size))
        row = {"family": family, "axis": axis, "size": size, "seeds": seeds}
        row["last"] = agg([base[(family, axis, size, s)]["last"][args.metric] for s in seeds])
        row["best"] = agg([base[(family, axis, size, s)]["best"][args.metric] for s in seeds])
        for name, table in arms.items():
            vals, eps = [], []
            for s in seeds:
                e = table.get((family, axis, size, s))
                if e and e["last"]:
                    vals.append(e["last"][args.metric])
                    eps.append(e["epochs"])
            row[name] = agg(vals)
            row[name + "_epochs"] = agg(eps)
        rows.append(row)

    names = ["last", "best"] + list(arms)
    head = f"{'family':<13}{'axis':<6}{'size':>6}" + "".join(f"{n:>18}" for n in names)
    print(f"test-missing {args.metric}, mean +/- sd over seeds\n")
    print(head)
    print("-" * len(head))
    for r in rows:
        line = f"{r['family']:<13}{r['axis']:<6}{r['size']:>6}"
        for n in names:
            m, sd = r.get(n, (None, None))
            line += "".rjust(18) if m is None else f"{m:>11.4f}+/-{sd:<5.3f}"
        print(line)

    for name in arms:
        deltas = [(r[name][0] - r["last"][0]) for r in rows
                  if r.get(name, (None,))[0] is not None and r["last"][0] is not None]
        vs_best = [(r[name][0] - r["best"][0]) for r in rows
                   if r.get(name, (None,))[0] is not None and r["best"][0] is not None]
        eps = [r[name + "_epochs"][0] for r in rows if r.get(name + "_epochs", (None,))[0]]
        if deltas:
            print(f"\n{name}: mean change vs last {st.mean(deltas):+.4f} nats "
                  f"({sum(1 for d in deltas if d < 0)}/{len(deltas)} cells better); "
                  f"vs best-on-test {st.mean(vs_best):+.4f}; "
                  f"mean refit budget {st.mean(eps):.0f} epochs")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"metric": args.metric, "rows": rows}, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
