#!/usr/bin/env python
"""The three result tables with every Marformer fit at a FIXED 200-epoch budget.

WHY 200 IS ALLOWED TO BE THE BUDGET. Picking it because it wins on the test corner
would be D14, the bug F13 exists to remove. It is not picked that way:
dev_pooled_budget.py pools the validation blocks over all 40 cells with a dev curve
and its argmin is epoch 200 on a dense 25-epoch grid from 25 to 1000, on both the
raw mean dev loss and the per-cell dev regret. A single budget shared by every cell
is one parameter and it is fitted on dev alone. The corner agreement is then a
check, not the criterion.

WHAT THE MARFORMER ROWS ARE. They are the 85%-data *search* runs read at epoch
~200, because those are the checkpoints that exist. A real fixed-budget protocol
needs no carve at all, so it would train on 100% of the training partition -- which
smece_factors.py measured as worth -0.022 (item/500) and -0.009 (annot/30), i.e.
these numbers are PESSIMISTIC by roughly 0.01-0.02 nats. That retrain has not been
run and is not assumed here.

Stan rows are unchanged by any of this: Stan has no epoch budget.

    python tables_fixed_budget.py --device cuda
"""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from pathlib import Path

import numpy as np

MAR = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer")
RANK = MAR / "imputer/ranking"
RERUN = MAR / "RERUN"
# Each dataset gets the budget ITS OWN pooled validation blocks pick
# (dev_pooled_budget.py): 200 for the synthetic sweep, 175 for LLM-Rubric.
# A single global constant is not supported by the dev curves -- see the
# per-run argmins, which spread from 120 to 260 on LLM-Rubric.
BUDGET = 200

ITEM_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000]
ANNOT_SIZES = [10, 15, 20, 25, 30, 35, 40, 45, 50]
LLMR_SIZES = [10, 20, 30, 40, 50, 75, 100, 125, 150, 175]
STAN_SERIES = [("Stan Oracle", "ORACLE", "TENSOR-SIG"),
               ("Stan Bin", "MISSPEC", "BIN"),
               ("Stan Proj", "MISSPEC", "PROJ"),
               ("Stan Both", "MISSPEC", "BOTH")]


def load(name: str, path: Path):
    sys.path.insert(0, str(RANK))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def smece(shared, probs, labels) -> float:
    conf, acc = shared._all_class_calibration(probs, labels)
    d = shared.rp.prepare_rel_diagram(conf, acc, num_bootstrap=500, report_CE=True)
    return float(d.get("ce", float("nan")))


def nll(probs, labels) -> float:
    p = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(p, 1e-12, None)).mean())


# ---------------------------------------------------------------- synthetic
def synthetic(device: str, budget: int = BUDGET) -> dict:
    """Marformer @200 from budget_table.json; Stan from the plot module payloads."""
    pd3 = load("pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    pd3.METRIC_POPULATION = "corner"
    shared = pd3.shared

    rows = json.loads((RERUN / "runs/final-dual/reports/budget_table.json").read_text())
    mar: dict[tuple[str, str, int], dict] = {}
    for r in rows:
        if r["label"] != f"fixed-{budget}":
            continue
        mar[(r["family"], r["axis"], r["K"])] = {
            "ll": r["log_loss"], "smece": r.get("smece"),
            "epoch": r["actual_epoch"], "dev": r.get("val_loss")}

    out: dict = {"budget": budget, "axes": {}}
    for axis, cfg, sizes in (("item", pd3.ITEM_CFG, ITEM_SIZES),
                             ("annot", pd3.ANNOT_CFG, ANNOT_SIZES)):
        series: dict[str, dict] = {}
        for fam, label in (("Marformer", "Marformer (T)"),
                           ("MarformerNT", "Marformer (NT)")):
            series[label] = {s: mar[(fam, axis, s)] for s in sizes
                             if (fam, axis, s) in mar}

        oracle, _ = pd3._collect_oracle_payloads(cfg)
        pools = {"Stan Oracle": oracle}
        for label, variant, tag in STAN_SERIES:
            if variant == "MISSPEC":
                p, _ = pd3._collect_misspec_payloads(cfg, tag)
                pools[label] = p
        for label, _v, _t in STAN_SERIES:
            pool = pools.get(label) or {}
            d = {}
            for s in sizes:
                pay = pool.get(s)
                if pay is None:
                    continue
                probs, labels = pay
                probs = np.asarray(probs, dtype=np.float64)
                labels = np.asarray(labels, dtype=np.int64)
                d[s] = {"ll": nll(probs, labels), "smece": smece(shared, probs, labels),
                        "epoch": None, "dev": None}
                print(f"  {axis}/{s:<5} {label:<14} ll={d[s]['ll']:.4f} "
                      f"smECE={d[s]['smece']:.4f}", flush=True)
            series[label] = d
        out["axes"][axis] = series
    return out


# ------------------------------------------------------------- llm-rubric
def llmrubric(device: str, budget: int = BUDGET) -> dict:
    import torch
    from imputer.entity_mf.data import variable_list_to_entity_graph
    from imputer.entity_mf.test import _load_checkpoint, _reconstruct
    lr = load("plr", RANK / "scripts/LLMRubric/plot_llmrubric_cpm_sigmoid.py")
    shared = lr.shared

    want = f"periodic-epoch={budget - 1:04d}.ckpt"
    series: dict[str, dict] = {"Marformer": {}, "Stan CPM": {}}

    for size in LLMR_SIZES:
        run = lr._marformer_run_dir(size)
        if run is None:
            print(f"  llmr/{size:<5} Marformer      no run dir"); continue
        ck = run / "checkpoints" / want
        if not ck.exists():
            print(f"  llmr/{size:<5} Marformer      {want} missing"); continue
        model, eval_vars, _cfg = _reconstruct(run)
        dev = torch.device(device)
        _load_checkpoint(model, ck, dev)
        model.to(dev).eval()
        nc = model.types["rating"].num_classes
        pr, lb = [], []
        # max_item=10 is the shipped LLM-Rubric eval chunk (D20/S8); changing it
        # here would be a second change on top of the budget.
        for chunk in lr._chunk_variables_by_item_limit(eval_vars, max_item=10):
            g = variable_list_to_entity_graph(chunk, model.types)
            params = model(g, device=dev)
            for i, v in enumerate(chunk):
                if not getattr(v, "is_missing", False):
                    continue
                pr.append(torch.softmax(params[0, i, 1:1 + nc], -1)
                          .detach().cpu().numpy().astype(np.float64))
                lb.append(int(v.rating_value))
        if not pr:
            print(f"  llmr/{size:<5} Marformer      no missing cells"); continue
        probs, labels = np.vstack(pr), np.asarray(lb, dtype=np.int64)
        series["Marformer"][size] = {"ll": nll(probs, labels),
                                     "smece": smece(shared, probs, labels),
                                     "epoch": budget - 1, "dev": None,
                                     "n": int(len(labels))}
        print(f"  llmr/{size:<5} Marformer      ll={series['Marformer'][size]['ll']:.4f} "
              f"smECE={series['Marformer'][size]['smece']:.4f} n={len(labels)}", flush=True)

        pay = lr._stan_probs_and_labels(size)
        if pay is not None:
            sp, sl = np.asarray(pay[0], dtype=np.float64), np.asarray(pay[1], dtype=np.int64)
            series["Stan CPM"][size] = {"ll": nll(sp, sl), "smece": smece(shared, sp, sl),
                                        "epoch": None, "dev": None, "n": int(len(sl))}
            print(f"  llmr/{size:<5} Stan CPM       ll={series['Stan CPM'][size]['ll']:.4f} "
                  f"smECE={series['Stan CPM'][size]['smece']:.4f}", flush=True)
    return {"budget": budget, "series": series}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-llmr", action="store_true")
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--budget", type=int, default=BUDGET,
                    help="synthetic budget (dev-pooled argmin: 200)")
    ap.add_argument("--llmr-budget", type=int, default=175,
                    help="LLM-Rubric budget (its own dev-pooled argmin: 175)")
    ap.add_argument("-o", "--out",
                    default=str(RERUN / "runs/final-dual/reports/tables_fixed200.json"))
    a = ap.parse_args()
    blob = {}
    if not a.skip_synth:
        print("=== synthetic ===", flush=True)
        blob["synthetic"] = synthetic(a.device, a.budget)
    if not a.skip_llmr:
        print("\n=== llm-rubric ===", flush=True)
        blob["llmrubric"] = llmrubric(a.device, a.llmr_budget)
    Path(a.out).write_text(json.dumps(blob, indent=2))
    print(f"\nwrote -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
