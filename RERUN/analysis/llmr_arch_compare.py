#!/usr/bin/env python
"""LLM-Rubric: old block vs standard block, matched epoch, everything else equal.

The published K=10 log-loss moved 1.52 -> 1.95. The two runs' configs are
identical apart from the block (old: dual_outer_residual=all, no block projection,
no codec; new: none/none/dual) -- same delta_item=0.7, same max_item=10, same 300
epochs, and both select their reported checkpoint on the run's own validation set.
So matched-epoch scoring isolates the architecture.
"""
from __future__ import annotations
import argparse, importlib.util, json, re, sys
from pathlib import Path
import numpy as np

MAR = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer")
RANK = MAR / "imputer/ranking"; RERUN = MAR / "RERUN"
OLD = RERUN / "runs/replicate-01/RESULTS/MARFORMER/LLM_RUBRIC"
NEW = RERUN / "runs/final-s101/RESULTS/MARFORMER/LLM_RUBRIC"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10,30,150,175")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    sys.path.insert(0, str(RANK))
    from imputer.entity_mf.data import variable_list_to_entity_graph
    from imputer.entity_mf.test import _load_checkpoint, _reconstruct
    spec = importlib.util.spec_from_file_location(
        "plr", RANK / "scripts/LLMRubric/plot_llmrubric_cpm_sigmoid.py")
    lr = importlib.util.module_from_spec(spec); sys.modules["plr"] = lr
    spec.loader.exec_module(lr)
    shared = lr.shared

    def score(run: Path, ck: str):
        model, ev, _ = _reconstruct(run)
        dev = torch.device(a.device)
        _load_checkpoint(model, run / "checkpoints" / ck, dev)
        model.to(dev).eval()
        nc = model.types["rating"].num_classes
        pr, lb = [], []
        for chunk in lr._chunk_variables_by_item_limit(ev, max_item=10):
            g = variable_list_to_entity_graph(chunk, model.types)
            p = model(g, device=dev)
            for i, v in enumerate(chunk):
                if not getattr(v, "is_missing", False):
                    continue
                pr.append(torch.softmax(p[0, i, 1:1+nc], -1).detach().cpu().numpy().astype(np.float64))
                lb.append(int(v.rating_value))
        probs, labels = np.vstack(pr), np.asarray(lb, dtype=np.int64)
        ll = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())
        conf, acc = shared._all_class_calibration(probs, labels)
        ce = float(shared.rp.prepare_rel_diagram(conf, acc, num_bootstrap=500,
                                                 report_CE=True).get("ce", float("nan")))
        return ll, ce

    out = []
    for size in [int(x) for x in a.sizes.split(",")]:
        o, n = OLD / f"LLMRubric_225_25_9_{size}", NEW / f"LLMRubric_225_25_9_{size}"
        if not o.is_dir() or not n.is_dir():
            print(f"[miss] {size}"); continue
        def eps(p):
            return {int(m.group(1)) for f in (p/"checkpoints").glob("periodic-epoch=*.ckpt")
                    if (m := re.search(r"(\d+)\.ckpt$", f.name))}
        common = sorted(eps(o) & eps(n))
        print(f"\n--- K={size}  ({len(common)} shared checkpoints) ---", flush=True)
        rows = []
        for ep in common:
            ck = f"periodic-epoch={ep:04d}.ckpt"
            oll, oce = score(o, ck); nll, nce = score(n, ck)
            rows.append({"epoch": ep, "old_ll": oll, "new_ll": nll,
                         "old_smece": oce, "new_smece": nce})
            print(f"  ep={ep:<4} old ll {oll:.4f} / new ll {nll:.4f}   diff {nll-oll:+.4f}"
                  f"   | smECE {oce:.4f} / {nce:.4f}", flush=True)
        d = [r["new_ll"]-r["old_ll"] for r in rows]
        print(f"  == K={size}: architecture effect on log-loss "
              f"mean {np.mean(d):+.4f}, min {min(d):+.4f}, max {max(d):+.4f}")
        out.append({"size": size, "rows": rows, "mean_ll_diff": float(np.mean(d))})
    Path(RERUN / "runs/final-dual/reports/llmr_arch_compare.json").write_text(json.dumps(out, indent=2))
    print("\n=== architecture effect on LLM-Rubric log-loss (new block minus old) ===")
    for r in out:
        print(f"  K={r['size']:<5} {r['mean_ll_diff']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
