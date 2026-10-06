#!/usr/bin/env python
"""The three result tables under the SHIPPED F13 dev protocol, with smECE.

The companion to tables_fixed_budget.py, so the two protocols can be read side by
side. Here the Marformer rows are the published ones: carve 15% -> select the
stopping epoch on the validation block -> refit on 100% of the training partition
-> report `last`. corner_metrics/corner.json already holds their log-loss (the gate
in check_plot_corner.py ties it to the figures); this adds smECE on the SAME
checkpoints, which was never serialised for any size but item/500 and annot/30.

Stan rows are identical to the fixed-budget table by construction -- Stan has no
epoch budget -- so they are not recomputed here; render_f13.py reuses them.

LLM-Rubric's published row is its val-selected checkpoint (`last.ckpt`, which is
byte-identical to `best-epoch=*.ckpt`), scored at the shipped max_item=10 chunk.

    python tables_f13.py --device cuda
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

import numpy as np

MAR = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer")
RANK = MAR / "imputer/ranking"
RERUN = MAR / "RERUN"
RUNS = RERUN / "runs/final-dual"
LLMR_SIZES = [10, 20, 30, 40, 50, 75, 100, 125, 150, 175]


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
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-item", type=int, default=500)
    ap.add_argument("-o", "--out", default=str(RUNS / "reports/tables_f13.json"))
    a = ap.parse_args()

    pd3 = load("pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    pd3.METRIC_POPULATION = "corner"
    shared = pd3.shared

    corner = json.loads((RUNS / "corner_metrics/corner.json").read_text())
    FAM = {"Marformer": "MARFORMER", "MarformerNT": "MARFORMER-NT"}
    out: dict = {"protocol": "F13 (carve -> select on dev -> refit 100% -> last)",
                 "axes": {"item": {}, "annot": {}}}

    print("=== synthetic (42 published refit checkpoints) ===", flush=True)
    for r in corner["rows"]:
        fam, axis, size = r["family"], r["axis"], int(r["size"])
        run = (RUNS / "RESULTS" / FAM[fam] / "DOMAIN3-SIGMOID"
               / r["variant"] / f"seed{r['seed']}"
               / f"DOMAIN3-SIGMOID_{axis.capitalize()}_"
                 f"{'NT' if fam.endswith('NT') else 'T'}_{size}_MARFORMER")
        if not run.is_dir():
            print(f"  [miss] {fam}/{axis}/{size}: {run}"); continue
        probs, labels = pd3._marformer_probs_all_missing(
            run, "last.ckpt", a.max_item, a.device)
        ll, ce = nll(probs, labels), smece(shared, probs, labels)
        ref = r["test"]["log_loss"]
        # The published log-loss must come back out of this path, or the smECE
        # beside it belongs to a different checkpoint than the table's number.
        flag = "" if abs(ll - ref) < 2e-3 else f"  *** MISMATCH vs corner.json {ref:.4f}"
        label = "Marformer (T)" if fam == "Marformer" else "Marformer (NT)"
        out["axes"][axis].setdefault(label, {})[size] = {
            "ll": ref, "ll_recomputed": ll, "smece": ce, "n": int(len(labels))}
        print(f"  {axis}/{size:<5} {label:<16} ll={ll:.4f} smECE={ce:.4f}{flag}", flush=True)

    print("\n=== llm-rubric (published val-selected checkpoint) ===", flush=True)
    import torch
    from imputer.entity_mf.data import variable_list_to_entity_graph
    from imputer.entity_mf.test import _load_checkpoint, _reconstruct
    lr = load("plr", RANK / "scripts/LLMRubric/plot_llmrubric_cpm_sigmoid.py")
    series: dict[str, dict] = {"Marformer": {}, "Stan CPM": {}}
    for size in LLMR_SIZES:
        run = lr._marformer_run_dir(size)
        if run is None:
            continue
        rj = run / "TEST_RESULTS" / "last.json"
        ck_name = json.loads(rj.read_text()).get("checkpoint") if rj.is_file() else "last.ckpt"
        ck = run / "checkpoints" / (ck_name or "last.ckpt")
        if not ck.exists():
            ck = run / "checkpoints" / "last.ckpt"
        model, eval_vars, _ = _reconstruct(run)
        dev = torch.device(a.device)
        _load_checkpoint(model, ck, dev)
        model.to(dev).eval()
        nc = model.types["rating"].num_classes
        pr, lb = [], []
        for chunk in lr._chunk_variables_by_item_limit(eval_vars, max_item=10):
            g = variable_list_to_entity_graph(chunk, model.types)
            params = model(g, device=dev)
            for i, v in enumerate(chunk):
                if not getattr(v, "is_missing", False):
                    continue
                pr.append(torch.softmax(params[0, i, 1:1 + nc], -1)
                          .detach().cpu().numpy().astype(np.float64))
                lb.append(int(v.rating_value))
        probs, labels = np.vstack(pr), np.asarray(lb, dtype=np.int64)
        series["Marformer"][size] = {"ll": nll(probs, labels),
                                     "smece": smece(shared, probs, labels),
                                     "ckpt": ck.name, "n": int(len(labels))}
        print(f"  llmr/{size:<5} Marformer  ll={series['Marformer'][size]['ll']:.4f} "
              f"smECE={series['Marformer'][size]['smece']:.4f}  [{ck.name}]", flush=True)
    out["llmrubric"] = {"series": series}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
