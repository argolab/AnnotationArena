#!/usr/bin/env python
"""
Reliability curves and smECE for the Domain 3 calibration figures.

The paper's two calibration panels (item K=500, annotator J=30) are the only
Domain 3 figures the summary metrics cannot reproduce: they need the per-example
predicted probabilities, not a mean. This reloads each checkpoint, recomputes
them, and runs the same relplot estimator the paper's plotting code uses
(`prepare_rel_diagram(..., num_bootstrap=500)`), so the numbers are comparable.

Each run is scored twice, at two evaluation chunk sizes:

  trained     max_item as the model was trained with (300 for Domain 3). This
              is the correct one.
  paper       max_item the plot script uses (500 on the item axis, unchunked on
              the annotator axis). Kept because it is what the published
              figures used, and the gap is discrepancy D20 showing up in
              calibration as well as in log-loss.

    python collect_calibration.py --run-dir RERUN/runs/seeds-01 -o cal.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.test import _find_checkpoint, _load_checkpoint, _reconstruct

RUN_RE = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")
SEED_RE = re.compile(r"^seed(\d+)$")


def chunk_by_item(variables, max_item):
    if max_item is None:
        return [variables]
    item_ids = sorted({i for v in variables for i in v.item_ids})
    chunks = []
    for i in range(0, len(item_ids), max_item):
        keep = set(item_ids[i:i + max_item])
        sel = [v for v in variables if all(j in keep for j in v.item_ids)]
        if sel:
            chunks.append(sel)
    return chunks or [variables]


@torch.no_grad()
def probs_and_labels(run_dir: Path, device: torch.device, max_item, which="last"):
    """Per-example predicted distributions over the test-missing cells."""
    model, eval_vars, _cfg = _reconstruct(run_dir)
    _load_checkpoint(model, _find_checkpoint(run_dir / "checkpoints", which), device)
    model.to(device).eval()
    C = model.types["rating"].num_classes
    rows, labels = [], []
    for chunk in chunk_by_item(eval_vars, max_item):
        graph = variable_list_to_entity_graph(chunk, model.types)
        params = model(graph, device=device)
        for idx, var in enumerate(chunk):
            if not getattr(var, "is_missing", False):
                continue
            p = torch.softmax(params[0, idx, 1:1 + C], dim=-1).double().cpu().numpy()
            rows.append(p)
            labels.append(int(var.rating_value))
    if not rows:
        raise RuntimeError(f"no missing-token probabilities for {run_dir}")
    return np.vstack(rows), np.asarray(labels, dtype=np.int64)


def reliability(probs: np.ndarray, labels: np.ndarray, bootstrap: int) -> dict:
    """All-class reliability curve + smECE, via the repo's own estimator."""
    import relplot as rp
    classes = np.arange(probs.shape[1], dtype=np.int64)
    conf = probs.flatten()
    acc = (labels[:, None] == classes[None, :]).astype(np.float32).flatten()
    diag = rp.prepare_rel_diagram(conf, acc, num_bootstrap=bootstrap,
                                  report_CE=True, report_CE_std=True)
    # relplot returns the smoothed curve under a few possible key names across
    # versions; take whichever pair of equal-length arrays is present.
    mesh = np.asarray(diag.get("mesh", [])).ravel()
    mu = np.asarray(diag.get("mu", [])).ravel()
    lo = np.asarray(diag.get("lower", [])).ravel()
    hi = np.asarray(diag.get("upper", [])).ravel()
    dens = np.asarray(diag.get("density", [])).ravel()
    step = max(1, mesh.size // 160) if mesh.size else 1

    def col(a):
        return [round(float(v), 5) for v in a[::step]] if a.size == mesh.size else []

    return {
        "smece": float(diag.get("ce", float("nan"))),
        "ci_width": float(diag.get("ce_ci_width", 0.0)),
        "n": int(conf.size),
        "mesh": col(mesh), "mu": col(mu),
        "lower": col(lo), "upper": col(hi), "density": col(dens),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--sizes", default="item:500,annot:30",
                    help="axis:size pairs to score; the paper's calibration panels")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--checkpoints", default="last,best",
                    help="which checkpoints to score; 'best' is the test-argmin the "
                         "paper's figures use, so the pair shows D14 in calibration too")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    wanted = {}
    for pair in args.sizes.split(","):
        axis, size = pair.split(":")
        wanted.setdefault(axis.strip(), set()).add(int(size))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(args.run_dir) / "RESULTS"
    out = []

    for family_dir, family in (("MARFORMER", "Marformer"), ("MARFORMER-NT", "MarformerNT")):
        base = root / family_dir / "DOMAIN3-SIGMOID"
        if not base.is_dir():
            continue
        for run_dir in sorted(base.rglob("*_MARFORMER")):
            m = RUN_RE.search(run_dir.name)
            if not m:
                continue
            axis = m.group(1).lower()
            size = int(m.group(3))
            if size not in wanted.get(axis, ()):
                continue
            if not (run_dir / "checkpoints" / "last.ckpt").is_file():
                continue
            sm = next((SEED_RE.match(p.name) for p in run_dir.parents if SEED_RE.match(p.name)), None)
            seed = int(sm.group(1)) if sm else 42
            with open(run_dir / "train_config.json") as fh:
                trained_max_item = json.load(fh)["training"].get("max_item")
            paper_max_item = 500 if axis == "item" else None

            entry = {"family": family, "axis": axis, "size": size, "seed": seed,
                     "path": str(run_dir.relative_to(Path(args.run_dir)))}
            for which in args.checkpoints.split(","):
                which = which.strip()
                for tag, mi in (("trained", trained_max_item), ("paper", paper_max_item)):
                    try:
                        probs, labels = probs_and_labels(run_dir, device, mi, which)
                    except (FileNotFoundError, RuntimeError) as exc:
                        print(f"  skip {run_dir.name} {which}/{tag}: {exc}")
                        continue
                    key = f"{which}_{tag}"
                    entry[key] = {"max_item": mi, "checkpoint": which,
                                  **reliability(probs, labels, args.bootstrap)}
                    print(f"{family:<12}{axis:<6}{size:>5} seed{seed}  {key:<16}"
                          f"max_item={str(mi):<5} smECE={entry[key]['smece']:.4f}"
                          f" +/-{entry[key]['ci_width']:.4f}  n={entry[key]['n']}")
            out.append(entry)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"calibration": out}, fh, separators=(",", ":"))
    print(f"\n{len(out)} runs -> {args.out}")


if __name__ == "__main__":
    main()
