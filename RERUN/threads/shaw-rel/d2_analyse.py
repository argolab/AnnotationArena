#!/usr/bin/env python
"""
D2: paired comparison of Shaw-style relational attention against the shipped
frozen-bit design.

Reads s7_curves.py output (which already scores every periodic checkpoint on
the TEST CORNER and records the epoch this run's own validation block would
select) and reports three reads of each arm, paired on (axis, seed, K) against
a reference arm:

  dev-sel   corner at the epoch the validation block selects. The primary read:
            it is the protocol the paper uses, and it is immune to the confound
            that an arm which overfits sooner looks worse at a fixed budget.
  last      corner at the end of the fixed budget. No selection at all.
  tail      corner averaged over the last few periodic checkpoints -- the usual
            thread read, a variance reduction on `last` (side/analyse_arms.py).
  best      corner at its own argmin. DIAGNOSTIC ONLY: this selects on test, so
            it is reported to show how much room the selector is leaving, never
            as a result.

Unpaired numbers would be worthless here -- the generative draw alone moves the
corner by sd 0.085 -- so every figure below is a difference within a cell.

    python d2_analyse.py --in out_d2.json --ref bits
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path

# side/analyse_arms.py's threshold: 0.03 on single checkpoints, 0.015 once the
# tail is averaged. Shaw's two changes are architectural, so anything that
# matters to the paper should clear 0.015.
THRESHOLD = 0.015


def read(path: Path, tail: int) -> dict:
    """(arm_tag, axis, seed, K) -> {read_name: corner}"""
    out: dict = {}
    for r in json.loads(path.read_text()):
        parts = Path(r["run"]).parts
        tag = parts[-4]                      # .../<AXIS>/<tag>/Marformer/seed<N>/<run>
        series = [x for x in r.get("series", []) if x["epoch"] >= 0]
        reads = {
            "dev-sel": r.get("corner_at_val_argmin"),
            "last": r.get("corner_at_last"),
            "best": r.get("corner_at_corner_argmin"),
            "tail": (sum(x["corner"] for x in series[-tail:]) / len(series[-tail:])
                     if series else None),
        }
        out[(tag, r["axis"], r["seed"], r["size"])] = {
            **reads,
            "_val_epoch": r.get("val_argmin_epoch"),
            "_best_epoch": r.get("corner_argmin_epoch"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--ref", default="bits")
    ap.add_argument("--tail", type=int, default=3)
    a = ap.parse_args()
    data = read(Path(a.inp), a.tail)
    if not data:
        print("no rows"); return

    arms = sorted({k[0] for k in data})
    cells = sorted({k[1:] for k in data})
    if a.ref not in arms:
        print(f"reference arm {a.ref!r} not present; have {arms}"); return

    print(f"arms={arms}\ncells={len(cells)}  reference={a.ref}\n")

    for read_name in ("dev-sel", "tail", "last", "best"):
        groups = [("item", [c for c in cells if c[0] == "item"]),
                  ("annot", [c for c in cells if c[0] == "annot"]),
                  ("ALL", cells)]
        label = read_name + ("   [DIAGNOSTIC ONLY: selects on test]"
                             if read_name == "best" else "")
        print(f"=== {label} ===")
        hdr = f"{'arm':<10}" + "".join(f"{g:>24}" for g, _ in groups)
        print(hdr); print("-" * len(hdr))
        for arm in arms:
            line = f"{arm:<10}"
            for _, cs in groups:
                ds = []
                for c in cs:
                    x, y = data.get((arm,) + c), data.get((a.ref,) + c)
                    if x and y and x[read_name] is not None and y[read_name] is not None:
                        ds.append(x[read_name] - y[read_name])
                if not ds:
                    line += f"{'-':>24}"; continue
                m = st.mean(ds)
                se = st.stdev(ds) / len(ds) ** 0.5 if len(ds) > 1 else 0.0
                mark = "*" if abs(m) > THRESHOLD and abs(m) > 2 * se else " "
                line += f"{m:>+9.4f} +/-{se:>6.4f} n={len(ds):<2}{mark}".rjust(24)
            print(line)
        print()

    print("* = paired mean beats the 0.015-nat threshold AND two standard errors."
          "\n  Positive = WORSE than the reference (these are cross-entropies).\n")

    print("=== per cell, dev-sel (absolute corner, and the epoch each arm stopped at) ===")
    hdr = f"{'cell':<14}" + "".join(f"{arm:>20}" for arm in arms)
    print(hdr); print("-" * len(hdr))
    for c in cells:
        line = f"{c[0]}/{c[2]} s{c[1]:<5}"[:14].ljust(14)
        for arm in arms:
            d = data.get((arm,) + c)
            line += (f"{d['dev-sel']:.4f} @{d['_val_epoch']}".rjust(20)
                     if d and d["dev-sel"] is not None else f"{'-':>20}")
        print(line)

    missing = [(arm,) + c for arm in arms for c in cells if (arm,) + c not in data]
    if missing:
        print(f"\n{len(missing)} cell(s) missing -- the paired ns above exclude them:")
        for m in missing[:20]:
            print(f"  {m}")


if __name__ == "__main__":
    main()
