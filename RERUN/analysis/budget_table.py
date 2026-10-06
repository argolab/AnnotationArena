#!/usr/bin/env python
"""Join the dev (validation-block) loss onto the fixed-budget corner results.

fixed_budget_vs_f13.py scores corner log-loss and smECE at F13's selected epoch
and at fixed budgets {200,400,800}. It does NOT report the validation loss at
those same epochs, which is the number that decides whether the carve and the
corner agree. This reads it from each run's own training_history.json (the same
curve the protocol selects on, via side/s7_curves.val_curve) and prints the
three-column table.

READ fixed_budget_vs_f13.py's header before acting on this. Picking a budget
because it wins on the corner is selection on test.

    python budget_table.py                       # after the scoring job lands
    python budget_table.py --rows path/to.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import statistics as st

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs/final-dual"


def val_curve(run: Path) -> list[tuple[int, float]]:
    """(epoch, validation cross-entropy) from the run's own history.

    A copy of side/s7_curves.py::val_curve. Importing it there pulls in
    split_missing_metrics -> torch, and this script is pure JSON parsing that
    wants to stay runnable without a GPU allocation. Keep the two in step.
    """
    h = json.loads((run / "training_history.json").read_text())
    epochs = h if isinstance(h, list) else h.get("epochs", h.get("history", []))
    out = []
    for i, e in enumerate(epochs):
        if not isinstance(e, dict):
            continue
        ep = e.get("epoch", i)
        v = None
        for key in ("val_missing_rating_ce", "val_ce", "val_rating_ce"):
            if isinstance(e.get(key), (int, float)):
                v = float(e[key]); break
        if v is None:
            blob = e.get("val_eval") or {}
            m = (blob.get("metrics") or {}).get("missing", {}).get("rating", {})
            for key in ("xent", "cross_entropy", "log_loss", "ce"):
                if isinstance(m.get(key), (int, float)):
                    v = float(m[key]); break
        if v is not None:
            out.append((int(ep), v))
    return out

FAM_DIR = {"Marformer": "MARFORMER", "MarformerNT": "MARFORMER-NT"}
AXIS_DIR = {"item": "ITEM", "annot": "ANNOT"}


def find_run(family: str, axis: str, K: int) -> Path | None:
    root = RUNS / "RESULTS" / FAM_DIR[family] / "DOMAIN3-SIGMOID" / AXIS_DIR[axis] / "valblock-search"
    if not root.is_dir():
        return None
    tag = "NT" if family.endswith("NT") else "T"
    want = f"DOMAIN3-SIGMOID_{axis.capitalize()}_{tag}_{K}_MARFORMER"
    hits = sorted(root.rglob(want))
    return hits[0] if hits else None


def nearest(curve: list[tuple[int, float]], epoch: int):
    """The logged val point closest to `epoch`, as (epoch, loss)."""
    if not curve:
        return None, None
    return min(curve, key=lambda t: abs(t[0] - epoch))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(RUNS / "reports/fixed_budget_vs_f13.json"))
    ap.add_argument("-o", "--out", default=str(RUNS / "reports/budget_table.json"))
    a = ap.parse_args()

    rows = json.loads(Path(a.rows).read_text())
    # one history parse per run, not per row (2 MB each)
    cache: dict[tuple[str, str, int], list[tuple[int, float]]] = {}
    out = []
    for r in rows:
        key = (r["family"], r["axis"], r["K"])
        if key not in cache:
            run = find_run(*key)
            cache[key] = val_curve(run) if run else []
            if not cache[key]:
                print(f"  [warn] no val curve for {key}", file=sys.stderr)
        ve, vl = nearest(cache[key], r["actual_epoch"])
        out.append(dict(r, val_epoch=ve, val_loss=vl))

    Path(a.out).write_text(json.dumps(out, indent=2))

    idx = {(r["family"], r["axis"], r["K"], r["label"]): r for r in out}
    cells = sorted({(r["family"], r["axis"], r["K"]) for r in out})
    labels = ["F13-selected", "fixed-200", "fixed-400", "fixed-800"]

    hdr = (f"{'cell':<22}{'budget':<14}{'ep':>6}{'dev loss':>11}"
           f"{'corner ll':>11}{'corner smECE':>14}")
    print(hdr); print("=" * len(hdr))
    for c in cells:
        for lab in labels:
            r = idx.get(c + (lab,))
            if r is None:
                continue
            name = f"{c[0]}/{c[1]}/{c[2]}" if lab == labels[0] else ""
            ce = r.get("smece")
            vl = r.get("val_loss")
            vs = "--" if vl is None else "%.4f" % vl
            cs = "--" if ce is None else "%.4f" % ce
            print(f"{name:<22}{lab:<14}{r['actual_epoch']:>6}{vs:>11}"
                  f"{r['log_loss']:>11.4f}{cs:>14}")
        print("-" * len(hdr))

    print("\n=== means over the cells ===")
    print(f"{'budget':<14}{'dev loss':>11}{'corner ll':>11}{'corner smECE':>14}")
    for lab in labels:
        rs = [idx[c + (lab,)] for c in cells if c + (lab,) in idx]
        dv = [r["val_loss"] for r in rs if r.get("val_loss") is not None]
        ce = [r["smece"] for r in rs if r.get("smece") is not None]
        print(f"{lab:<14}"
              f"{(st.mean(dv) if dv else float('nan')):>11.4f}"
              f"{st.mean([r['log_loss'] for r in rs]):>11.4f}"
              f"{(st.mean(ce) if ce else float('nan')):>14.4f}")

    # F13's selected epoch IS the dev argmin by construction, so "dev prefers
    # F13-selected" is near-tautological and the agreement count below reads low
    # for that reason alone. The statistic that is not circular is the selector's
    # REGRET on the corner: how much corner loss is given up by stopping where the
    # dev curve bottoms instead of at the best of the four budgets. It is still a
    # diagnostic -- the best-of-four is an oracle -- but it measures the size of
    # the mis-selection rather than merely its direction.
    print("\n=== selector regret on the corner (dev-chosen epoch vs best of the four) ===")
    for field, name in (("log_loss", "log-loss regret"), ("smece", "smECE regret")):
        v = []
        for c in cells:
            sel = idx.get(c + ("F13-selected",))
            if sel is None or sel.get(field) is None:
                continue
            # item/T K=100 and annot/T J=10 have no validation block to carve, so
            # select_epochs.py gives them the axis median instead of a dev argmin.
            # Their "F13-selected" row is not a dev choice and cannot be charged
            # to the dev curve.
            if sel.get("val_loss") is None:
                continue
            alts = [idx[c + (l,)][field] for l in labels
                    if c + (l,) in idx and idx[c + (l,)].get(field) is not None]
            v.append(sel[field] - min(alts))
        if len(v) > 1:
            se = st.stdev(v) / len(v) ** 0.5
            print(f"{name:<18} mean {st.mean(v):+.4f} +/-{se:.4f}  "
                  f"median {st.median(v):+.4f}  max {max(v):+.4f}  (n={len(v)})")

    print("\n=== which budget the corner prefers, counted ===")
    pref = {}
    for c in cells:
        rs = [idx[c + (l,)] for l in labels if c + (l,) in idx]
        b = min(rs, key=lambda r: r["log_loss"])["label"]
        pref[b] = pref.get(b, 0) + 1
    for l in labels:
        print(f"  {l:<14}{pref.get(l, 0):>3}/{len(cells)}")

    # Does the dev curve point at the same budget the corner wants?
    print("\n=== per cell: which budget each metric prefers (F13-selected + the three fixed) ===")
    agree = 0; total = 0
    for c in cells:
        rs = [idx[c + (l,)] for l in labels if c + (l,) in idx]
        rs_dev = [r for r in rs if r.get("val_loss") is not None]
        if len(rs_dev) < 2:
            continue
        bd = min(rs_dev, key=lambda r: r["val_loss"])["label"]
        bc = min(rs, key=lambda r: r["log_loss"])["label"]
        total += 1; agree += (bd == bc)
        flag = "" if bd == bc else "   <-- disagree"
        print(f"{c[0]}/{c[1]}/{c[2]:<6} dev prefers {bd:<14} corner prefers {bc:<14}{flag}")
    if total:
        print(f"\nagree on {agree}/{total} cells ({100*agree/total:.0f}%)")

    # Rank correlation between dev loss and corner loss within each cell.
    print("\n=== within-cell Spearman(dev loss, corner ll) over the 4 budgets ===")
    rhos = []
    for c in cells:
        rs = [idx[c + (l,)] for l in labels if c + (l,) in idx]
        rs = [r for r in rs if r.get("val_loss") is not None]
        if len(rs) < 3:
            continue
        def rank(vals):
            order = sorted(range(len(vals)), key=lambda i: vals[i])
            rk = [0.0] * len(vals)
            for pos, i in enumerate(order):
                rk[i] = float(pos)
            return rk
        rd = rank([r["val_loss"] for r in rs]); rc = rank([r["log_loss"] for r in rs])
        n = len(rs)
        d2 = sum((rd[i] - rc[i]) ** 2 for i in range(n))
        rho = 1 - 6 * d2 / (n * (n * n - 1))
        rhos.append(rho)
        print(f"{c[0]}/{c[1]}/{c[2]:<6} rho={rho:+.2f}  (n={n})")
    if rhos:
        print(f"\nmean rho = {st.mean(rhos):+.3f} over {len(rhos)} cells "
              f"(+1 = dev ranks budgets exactly as the corner does, "
              f"-1 = exactly backwards)")
    print(f"\nwrote -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
