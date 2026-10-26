#!/usr/bin/env python
"""Build the artifact's data payload: curves, selections, outcomes, baselines."""
from __future__ import annotations
import json, statistics as st
from pathlib import Path

RERUN = Path(__file__).resolve().parents[2]
RUNS = RERUN / "runs"
DRAWS = ["s42", "s101", "s102", "s103", "s104", "s105"]
RATES = [15, 50, 85]
SEEDS = [42, 43]
STEP = 10  # downsample 2000 epochs -> 200 points per curve

def curve(cell: str, seed: int):
    g = list((RUNS / cell / "RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM"
              / f"valmcar-search/seed{seed}").glob("*_MARFORMER/training_history.json"))
    if not g:
        return None
    h = json.loads(g[0].read_text())
    ep, tr, va, co = [], [], [], []
    for i, e in enumerate(h):
        if i % STEP and i != len(h) - 1:
            continue
        v = (e.get("val_eval") or {}).get("metrics", {}).get("missing", {}).get("rating", {})
        c = (e.get("combined_eval") or {}).get("metrics", {}).get("missing", {}).get("rating", {})
        if "xent" not in v:
            continue
        ep.append(e.get("epoch", i))
        tr.append(round(float(e["train_masked_ce"]), 4) if e.get("train_masked_ce") is not None else None)
        va.append(round(float(v["xent"]), 4))
        co.append(round(float(c["xent"]), 4) if "xent" in c else None)
    # full-resolution argmin, not the downsampled one
    full = [float(e["val_eval"]["metrics"]["missing"]["rating"]["xent"])
            for e in h if e.get("val_eval", {}).get("metrics")]
    return {"epoch": ep, "train_masked_ce": tr, "val_ce": va, "pooled_missing_ce": co,
            "val_n": int(v.get("n", 0)),
            "val_argmin_epoch": full.index(min(full)), "val_best": round(min(full), 4),
            "val_last": round(full[-1], 4), "epochs_run": len(h)}

out = {"rates": RATES, "draws": DRAWS, "seeds": SEEDS, "step": STEP, "cells": {}, "summary": {}}
table = json.loads((Path(__file__).parent / "out/density_table.json").read_text())

for d in DRAWS:
    for r in RATES:
        cell = f"obsden-{d}-o{r:02d}"
        tc = table["cells"].get(f"{d}-o{r:02d}", {})
        entry = {"draw": d, "rate": r, "corner_n": tc.get("n"),
                 "stan": {k: round(v["log_loss"], 4) for k, v in (tc.get("stan") or {}).items()},
                 "seeds": {}}
        for s in SEEDS:
            c = curve(cell, s)
            if c is None:
                continue
            mar = (tc.get("marformer") or {}).get(str(s), {})
            c["selected_epoch"] = (tc.get("selected_epochs") or {}).get(f"Marformer|item|500|{s}")
            c["corner_ll"] = round(mar["log_loss"], 4) if mar.get("log_loss") else None
            entry["seeds"][str(s)] = c
        out["cells"][f"{d}-o{r:02d}"] = entry

# per-density aggregates, paired by draw
for r in RATES:
    mars, stans, args_, degs, ns = [], {}, [], [], []
    for d in DRAWS:
        e = out["cells"][f"{d}-o{r:02d}"]
        lls = [v["corner_ll"] for v in e["seeds"].values() if v.get("corner_ll")]
        if lls:
            mars.append(st.fmean(lls))
        for k, v in e["stan"].items():
            stans.setdefault(k, []).append(v)
        for v in e["seeds"].values():
            args_.append(v["val_argmin_epoch"]); degs.append(v["val_last"] - v["val_best"])
        if isinstance(e["corner_n"], int):
            ns.append(e["corner_n"])
    row = {"marformer": round(st.fmean(mars), 4), "n_draws": len(mars),
           "corner_n_range": [min(ns), max(ns)] if ns else None,
           "argmin_mean": round(st.fmean(args_)), "argmin_min": min(args_), "argmin_max": max(args_),
           "degradation_mean": round(st.fmean(degs), 3), "stan": {}}
    for k, v in stans.items():
        diffs = [out["cells"][f"{d}-o{r:02d}"]["seeds"] for d in DRAWS]
        paired = []
        for d in DRAWS:
            e = out["cells"][f"{d}-o{r:02d}"]
            lls = [x["corner_ll"] for x in e["seeds"].values() if x.get("corner_ll")]
            if lls and k in e["stan"]:
                paired.append(st.fmean(lls) - e["stan"][k])
        row["stan"][k] = {"log_loss": round(st.fmean(v), 4),
                          "paired_diff": round(st.fmean(paired), 4),
                          "paired_sem": round(st.stdev(paired) / len(paired) ** 0.5, 4) if len(paired) > 1 else None,
                          "n": len(paired)}
    out["summary"][f"o{r:02d}"] = row

p = Path(__file__).parent / "out/payload.json"
p.write_text(json.dumps(out, separators=(",", ":")))
print(f"wrote {p} ({p.stat().st_size/1024:.0f} KB)")
for r in RATES:
    s = out["summary"][f"o{r:02d}"]
    print(f"  {r}%: marformer {s['marformer']} argmin {s['argmin_mean']} "
          f"[{s['argmin_min']},{s['argmin_max']}] degr {s['degradation_mean']} n={s['corner_n_range']}")
