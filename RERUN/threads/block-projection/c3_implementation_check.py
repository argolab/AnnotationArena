#!/usr/bin/env python
"""C3: is the projection-removed model actually a healthy model?

The measured penalty for removing the block's output projection was large enough
to be suspicious, so before believing it, check the arm is not simply broken:

  1. every parameter that should receive gradient does, in BOTH arms -- nothing
     silently disconnected from the loss;
  2. no parameter is accidentally frozen;
  3. the removed modules are genuinely gone, and nothing else changed shape;
  4. the forward pass is finite and the two arms see identical inputs.
"""
from __future__ import annotations
import copy, json, os, sys
from pathlib import Path
import torch

RANKING_ROOT = Path(os.environ["RANKING_ROOT"]).resolve()
RERUN = RANKING_ROOT.parent.parent / "RERUN"
sys.path.insert(0, str(RERUN)); sys.path.insert(0, str(RANKING_ROOT))
from analysis.split_missing_metrics import chunk_by_item
from imputer.entity_mf.test import _reconstruct
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf import model as _m
from imputer.entity_mf.model import EntityMarformer
assert Path(_m.__file__).resolve().is_relative_to(RANKING_ROOT), _m.__file__

RUN = RERUN / "runs/replicate-01/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_300_MARFORMER"
OUT = Path(__file__).resolve().parent / "out/c3_implementation.json"


def build(cfg_src, types, n_rel, mode):
    cfg = copy.deepcopy(cfg_src)
    cfg.block_output_projection = mode
    # Removing the projection makes the block body the new stream value
    # (h' = h + attn + ff), which is the "replace" outer mode. Pairing it with
    # "all" instead doubles the stream every layer.
    cfg.dual_outer_residual = "none" if mode == "none" else "all"
    torch.manual_seed(0)
    return EntityMarformer(cfg, types, num_relationships=n_rel)


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref, eval_vars, _ = _reconstruct(RUN)
    n_rel = ref.blocks[0]["attn"].num_relationships
    chunk = chunk_by_item(eval_vars, 300)[0]
    graph = variable_list_to_entity_graph(chunk, ref.types)
    C = ref.types["rating"].num_classes

    report = {}
    for mode in ("full", "none"):
        m = build(ref.config, ref.types, n_rel, mode).to(dev).train()
        out = m(graph, device=dev)
        # a loss that touches every rating token, so anything feeding a
        # prediction must receive gradient
        idx = [i for i, t in enumerate(graph.tokens) if t.type_name == "rating"]
        logits = out[0, idx, 1:1 + C]
        loss = torch.nn.functional.cross_entropy(
            logits, torch.zeros(len(idx), dtype=torch.long, device=dev))
        m.zero_grad(); loss.backward()

        named = dict(m.named_parameters())
        frozen = [k for k, p in named.items() if not p.requires_grad]
        nograd = [k for k, p in named.items() if p.requires_grad and p.grad is None]
        zerograd = [k for k, p in named.items()
                    if p.grad is not None and float(p.grad.abs().max()) == 0.0]
        report[mode] = {
            "n_params": sum(p.numel() for p in m.parameters()),
            "n_tensors": len(named),
            "loss": round(float(loss), 6),
            "output_finite": bool(torch.isfinite(out).all()),
            "frozen": frozen,
            "no_gradient": nograd,
            "zero_gradient": zerograd,
            "grad_norm": round(float(sum(p.grad.pow(2).sum() for p in m.parameters()
                                         if p.grad is not None) ** 0.5), 4),
        }
        print(f"[{mode}] params {report[mode]['n_params']}  tensors {len(named)}  "
              f"loss {report[mode]['loss']}  grad-norm {report[mode]['grad_norm']}")
        for label, lst in (("frozen", frozen), ("no gradient", nograd),
                           ("zero gradient", zerograd)):
            print(f"    {label}: {len(lst)}" + (f"  {lst[:4]}" if lst else ""))

    # what actually differs between the two models
    a = set(dict(build(ref.config, ref.types, n_rel, "full").named_parameters()))
    b = set(dict(build(ref.config, ref.types, n_rel, "none").named_parameters()))
    only_full = sorted({k.rsplit(".", 1)[0].split(".")[-1] for k in a - b})
    report["only_in_full"] = only_full
    report["only_in_none"] = sorted(b - a)
    print(f"\nmodules only in 'full': {only_full}")
    print(f"modules only in 'none': {sorted(b - a) or 'none'}")

    ok = all(not report[m]["frozen"] and not report[m]["no_gradient"]
             and not report[m]["zero_gradient"] and report[m]["output_finite"]
             for m in ("full", "none"))
    report["healthy"] = ok
    print("\n" + ("OK: both arms are fully connected and trainable"
                  if ok else "PROBLEM: see lists above"))
    OUT.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
