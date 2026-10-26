#!/usr/bin/env python
"""
All-class reliability curves for every model at every density, pooled over the
six generative draws.

Follows the paper's own calibration convention exactly, by calling the same code
it does: `_all_class_calibration` (the full probability matrix flattened to
confidence/accuracy pairs, so 4 points per scored cell) then
`relplot.prepare_rel_diagram(..., report_CE=True, report_CE_std=True)` for the
smoothed curve and smECE. Nothing about the metric is re-derived here.

Two things this is careful about:

  * `plot_domain3_sigmoid.py` reaches its data through a module-level ROOT, which
    is why make_figures.sh symlinks the repo's DATA/RESULTS at a run. Doing that
    18 times would be serial and stateful, so instead each cell gets its own
    DatasetConfig via dataclasses.replace -- the repo symlinks are never touched.
  * That module has carried its own copies of D14/D22/D23. `METRIC_POPULATION`
    is set to "corner" explicitly rather than trusted, and --gate recomputes
    log-loss from the payloads and checks it against corner_metrics/corner.json,
    which the independent scorer wrote. If those disagree the payload population
    is wrong and the calibration curves would be too.

    python calibration.py -o out/calibration.json [--gate]
"""
from __future__ import annotations

import argparse, dataclasses, importlib.util, json, sys
from pathlib import Path

import numpy as np

RERUN = Path(__file__).resolve().parents[2]
RANK = RERUN.parent / "imputer/ranking"
RUNS = RERUN / "runs"
DRAWS = ["s42", "s101", "s102", "s103", "s104", "s105"]
RATES = [15, 50, 85]
SIZE = 500
SEED = 42
MODELS = [
    ("Marformer", None),
    ("Stan Oracle", "TENSOR-SIG"),
    ("Stan Proj", "PROJ"),
    ("Stan Bin", "BIN"),
    ("Stan Both", "BOTH"),
]


def load_module(name: str, path: Path):
    sys.path.insert(0, str(RANK))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def cell_cfg(pd3, cell: str):
    """ITEM_CFG retargeted at one cell's data and results."""
    root = RUNS / cell
    return dataclasses.replace(
        pd3.ITEM_CFG,
        display_sizes=[SIZE],
        data_root=root / "DATA/STAN/DOMAIN3-SIGMOID/ItemSplits/Transductive",
        marformer_root=root / "RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM",
        nt_data_root=root / "DATA/STAN/DOMAIN3-SIGMOID/ItemSplits/NonTransductive",
        marformer_nt_root=root / "RESULTS/MARFORMER-NT/DOMAIN3-SIGMOID/ITEM",
        stan_oracle_root=root / "RESULTS/STAN/DOMAIN3-SIGMOID/ORACLE/ITEM",
        misspec_root=root / "RESULTS/STAN/DOMAIN3-SIGMOID/MISSPEC/ITEM",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--gate", action="store_true", help="check log-loss against corner.json")
    a = ap.parse_args()

    pd3 = load_module("pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    pd3.METRIC_POPULATION = "corner"          # explicit, not inherited
    shared = pd3.shared
    import relplot as rp

    out = {"rates": RATES, "draws": DRAWS, "seed": SEED, "models": [m for m, _ in MODELS],
           "convention": "all-class calibration; confidence vs accuracy over the "
                         "flattened probability matrix, pooled over draws",
           "panels": {}, "gate": []}

    for rate in RATES:
        pooled = {name: {"p": [], "y": []} for name, _ in MODELS}
        for draw in DRAWS:
            cell = f"obsden-{draw}-o{rate:02d}"
            cfg = cell_cfg(pd3, cell)
            run = (cfg.marformer_root / f"valmcar/seed{SEED}"
                   / f"DOMAIN3-SIGMOID_Item_T_{SIZE}_MARFORMER")
            if run.exists() and (run / "checkpoints/last.ckpt").is_file():
                probs, labels = pd3._marformer_probs_all_missing(
                    run, "last.ckpt", pd3._marformer_eval_max_item(cfg, SIZE), a.device)
                probs, labels = np.asarray(probs, float), np.asarray(labels, np.int64)
                pooled["Marformer"]["p"].append(probs)
                pooled["Marformer"]["y"].append(labels)
                if a.gate:
                    nll = float(-np.mean(np.log(np.clip(
                        probs[np.arange(len(labels)), labels], 1e-300, None))))
                    cj = RUNS / cell / "corner_metrics/corner.json"
                    ref = None
                    if cj.exists():
                        rows = json.loads(cj.read_text())["rows"]
                        r = [x for x in rows if x["family"] == "Marformer" and x["seed"] == SEED]
                        ref = r[0]["test"]["log_loss"] if r else None
                    out["gate"].append({"cell": cell, "from_payload": round(nll, 6),
                                        "from_corner_json": ref,
                                        "delta": None if ref is None else round(nll - ref, 6),
                                        "n": int(len(labels))})
            oracle, _ = pd3._collect_oracle_payloads(cfg)
            for name, tag in MODELS:
                if tag is None:
                    continue
                pool = oracle if tag == "TENSOR-SIG" else pd3._collect_misspec_payloads(cfg, tag)[0]
                pay = pool.get(SIZE)
                if pay is None:
                    continue
                pooled[name]["p"].append(np.asarray(pay[0], float))
                pooled[name]["y"].append(np.asarray(pay[1], np.int64))

        for name, _ in MODELS:
            if not pooled[name]["p"]:
                continue
            P = np.vstack(pooled[name]["p"])
            Y = np.concatenate(pooled[name]["y"])
            conf, acc = shared._all_class_calibration(P, Y)
            d = rp.prepare_rel_diagram(conf, acc, num_bootstrap=a.bootstrap,
                                       report_CE=True, report_CE_std=True)
            # relplot interpolates the curve against diagram["mesh"]; there is
            # no "x" key. Fail loudly with the real key list if that changes.
            if "mesh" not in d or "mu" not in d:
                raise KeyError(f"unexpected prepare_rel_diagram keys: {sorted(d.keys())}")
            x = np.asarray(d["mesh"], float)
            mu = np.asarray(d["mu"], float)
            lo = np.asarray(d["lower"], float) if d.get("lower") is not None else None
            hi = np.asarray(d["upper"], float) if d.get("upper") is not None else None
            dens = np.asarray(d["density"], float) if d.get("density") is not None else None
            # thin to ~90 points: the artifact does not need the full mesh
            step = max(1, len(x) // 90)
            nll = float(-np.mean(np.log(np.clip(P[np.arange(len(Y)), Y], 1e-300, None))))
            out["panels"][f"{name}|o{rate:02d}"] = {
                "model": name, "rate": rate,
                "smece": float(d.get("ce", float("nan"))),
                "smece_ci": float(d.get("ce_ci_width", 0.0)),
                "log_loss": round(nll, 4),
                "n_cells": int(len(Y)),
                "x": [round(float(v), 4) for v in x[::step]],
                "mu": [round(float(v), 4) for v in mu[::step]],
                "lower": None if lo is None else [round(float(v), 4) for v in lo[::step]],
                "upper": None if hi is None else [round(float(v), 4) for v in hi[::step]],
                "density": None if dens is None else [round(float(v), 5) for v in dens[::step]],
                # point masses at exactly 0 and 1, which the paper's panels
                # draw separately because a KDE cannot represent an atom
                "pt0": [float(d.get("0pt_mu", float("nan"))), float(d.get("0pt_density", 0.0))],
                "pt1": [float(d.get("1pt_mu", float("nan"))), float(d.get("1pt_density", 0.0))],
            }
            print(f"  {rate:>3}%  {name:<13} smECE {d.get('ce', float('nan')):.4f} "
                  f"+-{d.get('ce_ci_width', 0):.4f}  logloss {nll:.4f}  n={len(Y)}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"\nwrote {a.out}")
    if a.gate:
        bad = [g for g in out["gate"] if g["delta"] is not None and abs(g["delta"]) > 2e-3]
        print(f"gate: {len(out['gate'])} cells checked, {len(bad)} mismatched")
        for g in bad:
            print("  MISMATCH", g)
        if bad:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
