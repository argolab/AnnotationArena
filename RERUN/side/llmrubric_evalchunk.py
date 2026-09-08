#!/usr/bin/env python
"""
D30: what does the paper's LLM-Rubric number cost by evaluating in chunks of 10?

The published figure is NOT produced by test.py. plot_llmrubric_cpm_sigmoid.py
reloads each checkpoint and recomputes predictions itself, chunking the
evaluation at max_item=10 on a 225-item bundle (its line 373). S4 showed on
Domain 3 that a too-small evaluation chunk is worth up to 0.11 nats and that the
optimum is interior, so this is worth measuring -- and unlike the training-chunk
question it needs no training at all, only a re-score of checkpoints that
already exist.

This calls the plot script's OWN recompute function so the metric is exactly the
published one, and varies only max_item. It writes nothing into the results
tree; the caller binds DATA/RESULTS and unbinds afterwards.

    python llmrubric_evalchunk.py --chunks 10,25,50,100,225 -o out.json
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from pathlib import Path
import numpy as np

RANKING = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
PLOT = RANKING / "scripts/LLMRubric/plot_llmrubric_cpm_sigmoid.py"


def load_plot_module():
    sys.path.insert(0, str(RANKING))
    spec = importlib.util.spec_from_file_location("llmr_plot", PLOT)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines dataclasses, and dataclasses
    # resolves a field's type by looking its owning module up in sys.modules.
    # Without this the import dies in dataclasses._is_type with
    # "'NoneType' object has no attribute '__dict__'".
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)
    return float(-np.mean(np.log(p)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="10,25,50,100,225")
    ap.add_argument("--sizes", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--marformer-root", default="",
                    help="Override SPEC.marformer_root, e.g. to score an S8 arm's "
                         "variant subtree instead of the shipped runs.")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    m = load_plot_module()
    if args.marformer_root:
        import dataclasses
        try:
            m.SPEC = dataclasses.replace(m.SPEC, marformer_root=Path(args.marformer_root))
        except Exception:
            m.SPEC.marformer_root = Path(args.marformer_root)
        print(f"marformer_root -> {m.SPEC.marformer_root}", flush=True)
    sizes = [int(x) for x in args.sizes.split(",") if x] or list(m.SPEC.sizes)
    chunks = [int(x) for x in args.chunks.split(",") if x]

    rows = []
    for size in sizes:
        for mi in chunks:
            try:
                payload = m._marformer_probs_and_labels(
                    size, device_name=args.device, force_best=False, max_item=mi)
            except Exception as exc:                      # noqa: BLE001
                print(f"size {size:>4} chunk {mi:>4}  ERROR {type(exc).__name__}: {exc}", flush=True)
                continue
            if payload is None:
                print(f"size {size:>4} chunk {mi:>4}  (no payload)", flush=True)
                continue
            probs, labels = payload[0], payload[1]
            ll = log_loss(np.asarray(probs), np.asarray(labels))
            rows.append({"size": size, "max_item": mi, "log_loss": ll, "n": int(len(labels))})
            print(f"size {size:>4} chunk {mi:>4}  log_loss {ll:.4f}  n={len(labels)}", flush=True)

    Path(args.out).write_text(json.dumps({"rows": rows}, indent=1))
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
