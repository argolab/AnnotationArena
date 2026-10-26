#!/usr/bin/env python
"""
Assemble the observation-density table: Marformer against the four Stan arms,
one column per observed rate.

Reads only numbers other people's trusted scorers already wrote:

  Marformer  runs/<cell>/corner_metrics/corner.json   (split_missing_metrics.py
             --corner-whole, i.e. the CORNER-ONLY log-loss). Deliberately NOT
             TEST_RESULTS/*.json: tree_eval's "missing" metric pools the corner
             with the training partition's own missing cells (D22), and Stan is
             scored on the corner alone, so mixing them is D23.
  Stan       runs/<cell>/RESULTS/STAN/.../<run>_<TAG>_eval/predictive_metrics.json
             rating_missing_log_likelihood is a per-rating mean log-lik, so
             log-loss is its negation.

The gate: within a cell, every Marformer row's `test.n` must equal every Stan
arm's `n_missing_ratings`. If they differ, the two families are being scored on
different populations and the table is meaningless -- that is exactly the D23
failure, and it is checked here rather than trusted.

    python collect.py [--prefix obsden] [-o out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

RERUN = Path(__file__).resolve().parents[2]
RUNS = RERUN / "runs"

DRAWS = ["s42", "s101", "s102", "s103", "s104", "s105"]
RATES = [15, 50, 85]
SIZE = 500
AXIS = "item"
FAMILY = "Marformer"
# tag -> (results subdir, display name). Oracle is the data-generating model:
# an upper bound, not a competitor.
# A 4-category uniform predictor scores ln(4) = 1.3863. A refit whose corner
# log-loss sits at that value has COLLAPSED -- it never left the uniform plateau,
# or the carve selected a stopping epoch before it did. That happened in 8 of 42
# cells of the first final run (with the typed codec; `dual` is what fixed it),
# and the sparsest cell here has only ~71 supervised tokens per step, so escape
# could be late. It must be detected, never averaged into a column.
UNIFORM_LL = math.log(4)
COLLAPSE_TOL = 0.01

STAN_ARMS = [
    ("TENSOR-SIG", "ORACLE", "Stan Oracle"),
    ("PROJ", "MISSPEC", "Stan Proj"),
    ("BIN", "MISSPEC", "Stan Bin"),
    ("BOTH", "MISSPEC", "Stan Both"),
]


def marformer_rows(cell_dir: Path) -> list[dict]:
    f = cell_dir / "corner_metrics" / "corner.json"
    if not f.exists():
        return []
    rows = json.loads(f.read_text()).get("rows", [])
    return [r for r in rows
            if r.get("family") == FAMILY and r.get("axis") == AXIS
            and int(r.get("size", -1)) == SIZE and r.get("test")]


def stan_metric(cell_dir: Path, tag: str, subdir: str) -> dict | None:
    f = (cell_dir / "RESULTS" / "STAN" / "DOMAIN3-SIGMOID" / subdir / AXIS.upper()
         / f"DOMAIN3-SIGMOID_Item_T_{SIZE}_{tag}_eval" / "predictive_metrics.json")
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    ll = d.get("rating_missing_log_likelihood")
    if ll is None:
        return None
    return {"log_loss": -float(ll),
            "n": int(d.get("n_missing_ratings", -1)),
            "calibration_error": d.get("rating_missing_calibration_error")}


def selected_epochs(cell_dir: Path, cap: int) -> dict:
    """
    The per-run stopping epochs the carve chose, plus whether any of them sat at
    the search cap. A selected epoch equal to the cap means the budget BOUND and
    the curve never turned -- F13's whole point was that a budget which binds
    costs real nats, so this has to be visible, not inferred.
    """
    f = cell_dir / "valmcar_epochs.json"
    if not f.exists():
        return {}
    table = json.loads(f.read_text()).get("epochs", {})
    picked = {k: int(v) for k, v in table.items() if f"|{AXIS}|{SIZE}|" in k}
    return {"epochs": picked,
            "at_cap": sorted(k for k, v in picked.items() if v >= cap)}


def refit_complete(cell_dir: Path, seed: int, want_epochs: int | None) -> str | None:
    """
    Returns a problem string if the refit did not actually finish.

    `last.ckpt` is not a completion marker -- a killed run leaves one behind and
    the corner scorer will happily score it, producing a plausible number from a
    truncated model. `training_history.json` is the marker (threads/PLAN.md), so
    its length is checked against the epoch budget the carve selected.
    """
    hist = (cell_dir / "RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM" / f"valmcar/seed{seed}"
            / f"DOMAIN3-SIGMOID_Item_T_{SIZE}_MARFORMER" / "training_history.json")
    if not hist.exists():
        return f"seed {seed}: no training_history.json -- refit never ran"
    n = len(json.loads(hist.read_text()))
    if want_epochs is not None and n < want_epochs:
        return (f"seed {seed}: refit TRUNCATED, training_history has {n} epochs "
                f"but the carve selected {want_epochs}")
    return None


def fmt(x, w=8, p=4):
    return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'--':>{w}}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="obsden")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--search-cap", type=int, default=2000,
                    help="the --search-epochs the search stage ran with; a selected "
                         "epoch at this value means the budget bound")
    args = ap.parse_args()

    cells: dict[tuple[str, int], dict] = {}
    problems: list[str] = []

    for draw in DRAWS:
        for rate in RATES:
            cell = f"{args.prefix}-{draw}-o{rate:02d}"
            d = RUNS / cell
            entry: dict = {"cell": cell, "draw": draw, "rate": rate,
                           "marformer": {}, "stan": {}, "n": None}
            if not d.exists():
                problems.append(f"{cell}: run dir missing")
                cells[(draw, rate)] = entry
                continue

            sel = selected_epochs(d, args.search_cap)
            entry["selected_epochs"] = sel.get("epochs", {})
            if sel.get("at_cap"):
                problems.append(
                    f"{cell}: epoch budget BOUND at the {args.search_cap} cap for "
                    f"{sel['at_cap']} -- the validation curve never turned, so the "
                    "refit is budget-limited, not carve-selected")

            ns: set[int] = set()
            for r in marformer_rows(d):
                entry["marformer"][int(r["seed"])] = {
                    "log_loss": r["test"]["log_loss"],
                    "n": r["test"]["n"],
                    "accuracy": r["test"].get("accuracy"),
                    "variant": r.get("variant"),
                    "max_item": r.get("max_item"),
                }
                ns.add(int(r["test"]["n"]))
                want = entry["selected_epochs"].get(
                    f"{FAMILY}|{AXIS}|{SIZE}|{int(r['seed'])}")
                bad = refit_complete(d, int(r["seed"]), want)
                if bad:
                    problems.append(f"{cell}: {bad}")
                if abs(r["test"]["log_loss"] - UNIFORM_LL) < COLLAPSE_TOL:
                    problems.append(
                        f"{cell}: seed {r['seed']} COLLAPSED TO UNIFORM "
                        f"(corner log-loss {r['test']['log_loss']:.4f} vs ln4="
                        f"{UNIFORM_LL:.4f}) -- the refit never left the plateau; "
                        "check the search curve's escape epoch against the "
                        "selected one, do not average this in")
            if not entry["marformer"]:
                problems.append(f"{cell}: no Marformer corner rows")

            for tag, subdir, name in STAN_ARMS:
                m = stan_metric(d, tag, subdir)
                if m is None:
                    problems.append(f"{cell}: {name} missing")
                    continue
                entry["stan"][name] = m
                ns.add(m["n"])

            if len(ns) > 1:
                problems.append(
                    f"{cell}: SCORED-POPULATION MISMATCH, n={sorted(ns)} "
                    "-- Marformer and Stan are not on the same cells (D23)")
            entry["n"] = sorted(ns)[0] if len(ns) == 1 else sorted(ns)
            cells[(draw, rate)] = entry

    # ---- per-density summary, draw-paired -----------------------------------
    print(f"\n{'='*94}")
    print("OBSERVATION DENSITY -- corner log-loss at K=500, item axis, transductive")
    print(f"{'='*94}")
    summary: dict[str, dict] = {}
    for rate in RATES:
        per_draw_mar, per_draw_stan = [], {n: [] for _, _, n in STAN_ARMS}
        n_scored = set()
        for draw in DRAWS:
            e = cells[(draw, rate)]
            seeds = [v["log_loss"] for v in e["marformer"].values()]
            if seeds:
                per_draw_mar.append(st.fmean(seeds))
            for _, _, name in STAN_ARMS:
                if name in e["stan"]:
                    per_draw_stan[name].append(e["stan"][name]["log_loss"])
            if isinstance(e["n"], int):
                n_scored.add(e["n"])

        row = {"n_draws_marformer": len(per_draw_mar),
               "n_scored_range": sorted(n_scored),
               "marformer": st.fmean(per_draw_mar) if per_draw_mar else None,
               "marformer_sd": st.stdev(per_draw_mar) if len(per_draw_mar) > 1 else None,
               "stan": {}}
        for _, _, name in STAN_ARMS:
            v = per_draw_stan[name]
            row["stan"][name] = {
                "log_loss": st.fmean(v) if v else None,
                "sd": st.stdev(v) if len(v) > 1 else None,
                "n_draws": len(v),
            }
        summary[f"o{rate:02d}"] = row

        print(f"\n--- {rate}% observed "
              f"(corner cells per draw: {sorted(n_scored) or 'n/a'}; "
              f"{len(per_draw_mar)}/{len(DRAWS)} draws) ---")
        print(f"  {'Marformer':<14}{fmt(row['marformer'])}"
              f"   sd(draws) {fmt(row['marformer_sd'], 6)}")
        for _, _, name in STAN_ARMS:
            s = row["stan"][name]
            gap = (row["marformer"] - s["log_loss"]
                   if row["marformer"] is not None and s["log_loss"] is not None else None)
            print(f"  {name:<14}{fmt(s['log_loss'])}"
                  f"   sd(draws) {fmt(s['sd'], 6)}"
                  f"   Marformer-minus {fmt(gap, 9)}"
                  f"{'  (Marformer better)' if isinstance(gap, float) and gap < 0 else ''}")

    if problems:
        print(f"\n{'='*94}\nINCOMPLETE / SUSPECT ({len(problems)})\n{'='*94}")
        for p in problems:
            print("  " + p)
    else:
        print("\nall 18 cells complete, scored populations consistent")

    out = args.out or str(Path(__file__).resolve().parent / "out" / "density_table.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"summary": summary,
         "cells": {f"{d}-o{r:02d}": cells[(d, r)] for d in DRAWS for r in RATES},
         "problems": problems}, indent=1))
    print(f"\nwrote {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
