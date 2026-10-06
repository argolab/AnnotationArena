#!/usr/bin/env python
"""Which single fixed epoch budget do the VALIDATION blocks pick, pooled over cells?

This is the test-blind version of the fixed-budget question. fixed_budget_vs_f13.py
shows 200 beating F13's per-cell selection on the test corner -- adopting it for
THAT reason would be selection on test (D14). But a single budget shared by every
cell is a one-parameter choice, and it can be made on the dev blocks alone, which
never touch the corner. If the dev-pooled argmin lands near 200, the budget is
legitimate and the corner agreement is a check rather than the criterion.

Per cell the dev curve is read at every logged epoch, so the grid is not limited to
the 25-epoch checkpoint stride: a retrain to a chosen budget can stop anywhere.

Two poolings, because a plain mean over cells is dominated by the badly scaled ones:
  mean dev loss(E)                       -- raw, for reference
  mean [dev loss(E) - min_E' dev loss(E')] -- per-cell dev regret, scale-centred

    python dev_pooled_budget.py
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs/final-dual"
FAM_DIR = {"Marformer": "MARFORMER", "MarformerNT": "MARFORMER-NT"}
AXIS_DIR = {"item": "ITEM", "annot": "ANNOT"}


def val_curve(run: Path) -> dict[int, float]:
    h = json.loads((run / "training_history.json").read_text())
    eps = h if isinstance(h, list) else h.get("epochs", h.get("history", []))
    out = {}
    for i, e in enumerate(eps):
        if not isinstance(e, dict):
            continue
        ep = int(e.get("epoch", i))
        v = None
        for k in ("val_missing_rating_ce", "val_ce", "val_rating_ce"):
            if isinstance(e.get(k), (int, float)):
                v = float(e[k]); break
        if v is None:
            m = ((e.get("val_eval") or {}).get("metrics") or {}).get("missing", {}).get("rating", {})
            for k in ("xent", "cross_entropy", "log_loss", "ce"):
                if isinstance(m.get(k), (int, float)):
                    v = float(m[k]); break
        if v is not None:
            out[ep] = v
    return out


def llmr_curves() -> tuple[list[dict[int, float]], list[str]]:
    """LLM-Rubric has a real validation split (val != test, verified epoch by
    epoch), so its budget can be pooled the same test-blind way."""
    root = (HERE.parent / "runs/final-s101/RESULTS/MARFORMER/LLM_RUBRIC")
    curves, labels = [], []
    for run in sorted(root.glob("LLMRubric_225_25_9_*")):
        h = json.loads((run / "training_history.json").read_text())
        eps = h if isinstance(h, list) else h.get("epochs", h.get("history", []))
        c = {}
        for i, e in enumerate(eps):
            if not isinstance(e, dict):
                continue
            m = (((e.get("val_eval") or {}).get("metrics") or {})
                 .get("missing", {}).get("rating", {}))
            if isinstance(m.get("xent"), (int, float)):
                c[int(e.get("epoch", i))] = float(m["xent"])
        if len(c) > 50:
            curves.append(c); labels.append(run.name)
    return curves, labels


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("synthetic", "llmrubric"), default="synthetic")
    a = ap.parse_args()
    if a.source == "llmrubric":
        curves, labels = llmr_curves()
        print(f"LLM-Rubric runs with a usable dev curve: {len(curves)}")
        grid = [e for e in range(25, 301, 25)]
        common = [e for e in grid if all(e in c for c in curves)]
        mins = [min(c.values()) for c in curves]
        rows = [(e, st.mean(c[e] for c in curves),
                 st.mean(c[e] - m for c, m in zip(curves, mins))) for e in common]
        best = min(rows, key=lambda r: r[2])
        print(f"\n{'epoch':>7}{'mean dev loss':>16}{'mean dev regret':>18}")
        for e, raw, reg in rows:
            print(f"{e:>7}{raw:>16.4f}{reg:>18.4f}"
                  + ("  <- argmin" if e == best[0] else ""))
        print(f"\ndev-pooled argmin: epoch {best[0]}  regret {best[2]:.4f}")
        print("\n=== per-run dev argmin (how stable is a single budget here?) ===")
        for c, l in zip(curves, labels):
            am = min(c, key=c.get)
            print(f"  {l:<28} dev argmin @{am:<4} ({c[am]:.4f})   "
                  f"dev@200 {c.get(200, float('nan')):.4f}")
        return 0
    curves, labels = [], []
    for fam, fd in FAM_DIR.items():
        for axis, ad in AXIS_DIR.items():
            root = RUNS / "RESULTS" / fd / "DOMAIN3-SIGMOID" / ad / "valblock-search"
            if not root.is_dir():
                continue
            for run in sorted(root.rglob("*_MARFORMER")):
                c = val_curve(run)
                if len(c) > 50:
                    curves.append(c); labels.append(f"{fam}/{axis}/{run.name}")
    print(f"cells with a usable dev curve: {len(curves)}")

    grid = [e for e in range(25, 1001, 25)]
    common = [e for e in grid if all(e in c for c in curves)]
    print(f"grid {common[0]}..{common[-1]} step 25, present in every cell\n")

    mins = [min(c.values()) for c in curves]
    rows = []
    for e in common:
        raw = st.mean(c[e] for c in curves)
        reg = st.mean(c[e] - m for c, m in zip(curves, mins))
        rows.append((e, raw, reg))

    best_raw = min(rows, key=lambda r: r[1])
    best_reg = min(rows, key=lambda r: r[2])
    print(f"{'epoch':>7}{'mean dev loss':>16}{'mean dev regret':>18}")
    for e, raw, reg in rows:
        if e > 600:
            continue
        mark = ""
        if e == best_raw[0]: mark += "  <- argmin raw"
        if e == best_reg[0]: mark += "  <- argmin regret"
        print(f"{e:>7}{raw:>16.4f}{reg:>18.4f}{mark}")

    print(f"\ndev-pooled argmin (raw)    : epoch {best_raw[0]}  loss {best_raw[1]:.4f}")
    print(f"dev-pooled argmin (regret) : epoch {best_reg[0]}  regret {best_reg[2]:.4f}")

    # How much worse is a flat 200 than the dev-pooled optimum, on dev?
    at = {e: (raw, reg) for e, raw, reg in rows}
    print("\n=== cost on DEV of rounding to a convenient budget ===")
    for e in (100, 150, 200, 250, 300, 400):
        if e in at:
            print(f"  {e:>4}: dev regret {at[e][1]:.4f}  "
                  f"(+{at[e][1]-best_reg[2]:.4f} vs the dev-pooled best)")

    # Per-cell: would a flat 200 be a disaster anywhere, judged on dev alone?
    print("\n=== per-cell dev regret at a flat 200 (dev-only view) ===")
    v = sorted(((c[200] - m, l) for c, m, l in zip(curves, mins, labels)), reverse=True)
    print(f"  mean {st.mean(x for x, _ in v):+.4f}  median {st.median([x for x, _ in v]):+.4f}")
    print("  worst 5:")
    for x, l in v[:5]:
        print(f"    {x:+.4f}  {l}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
