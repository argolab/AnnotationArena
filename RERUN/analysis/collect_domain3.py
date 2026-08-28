#!/usr/bin/env python
"""
Collect a Domain 3 run tree into one JSON for the interactive report.

Walks RESULTS/{MARFORMER,MARFORMER-NT}/DOMAIN3-SIGMOID/{ITEM,ANNOT}[/seed<N>]/
and pulls, per trained model:

  * the hyperparameters it was actually trained with (train_config.json), so the
    report can show them on hover rather than asserting them in prose;
  * every scored checkpoint (TEST_RESULTS/*.json) as an epoch -> metric curve,
    which is what makes checkpoint selection visible instead of implicit;
  * the training-time curve (training_history.json);
  * wall clock (launcher_metadata.json).

The seed layout follows submit_all.sh --seeds: the first seed is canonical and
sits directly under <AXIS>/, later seeds under <AXIS>/seed<N>/.

    python collect_domain3.py --run-dir RERUN/runs/seeds-01 -o out.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

RUN_RE = re.compile(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$")
SEED_RE = re.compile(r"^seed(\d+)$")
CKPT_RE = re.compile(r"periodic-epoch=(\d+)\.json$")

# Hyperparameters worth surfacing on hover. Everything else in train_config is
# either a path, a duplicate, or a bookkeeping field.
MODEL_KEYS = ["embedding_dim", "num_layers", "attention_heads", "d_ff",
              "num_ffn_layers", "dropout", "use_per_head_rel", "use_pointer",
              "scale_shared_rel", "type_embedding_init", "logit_high",
              "dual_outer_residual", "inner_residual", "global_param_dim"]
TRAIN_KEYS = ["epochs", "lr", "weight_decay", "masking_rate", "mask_augmentations",
              "masked_loss_weight", "observed_loss_weight", "max_item",
              "item_dropout_rate", "annotator_dropout_rate", "annotator_reg_weight",
              "transductive_learning", "lr_schedule", "device",
              "batch_augmentations", "tf32", "compile_mode"]


def read_json(path: Path) -> Any:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def collect_checkpoints(run_dir: Path) -> dict:
    """Every scored checkpoint, as {kind: metrics} plus an epoch-indexed curve."""
    out: dict[str, Any] = {"curve": [], "last": None, "best": None, "best_epoch": None}
    tr = run_dir / "TEST_RESULTS"
    if not tr.is_dir():
        return out
    for path in sorted(tr.glob("*.json")):
        data = read_json(path)
        if not isinstance(data, dict) or "missing" not in data:
            continue
        m = data["missing"]
        entry = {k: m.get(k) for k in ("log_loss", "accuracy", "rmse", "n",
                                       "spearman_r", "kendall_tau", "pearson_r")}
        if path.name == "last.json":
            out["last"] = entry
        elif path.name == "best.json":
            out["best"] = entry
            out["best_epoch"] = _epoch_of(data.get("checkpoint"))
        else:
            match = CKPT_RE.search(path.name)
            if match:
                out["curve"].append({"epoch": int(match.group(1)), **entry})
    out["curve"].sort(key=lambda e: e["epoch"])
    return out


def _epoch_of(name: str | None) -> int | None:
    if not name:
        return None
    m = re.search(r"epoch=(\d+)", name)
    return int(m.group(1)) if m else None


def collect_history(run_dir: Path, stride: int) -> list:
    """Training-time curve: train CE and the per-epoch evaluation."""
    hist = read_json(run_dir / "training_history.json")
    if not isinstance(hist, list):
        return []
    rows = []
    for i, e in enumerate(hist):
        if stride > 1 and i % stride and i != len(hist) - 1:
            continue
        row = {"epoch": e.get("epoch"), "train_ce": e.get("train_masked_ce")}
        for key in ("combined_eval", "val_eval", "train_eval"):
            m = e.get(key, {}).get("metrics", {}).get("missing", {}).get("rating", {})
            if "xent" in m:
                row["eval_ce"] = m["xent"]
                row["eval_acc"] = m.get("acc")
                break
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--history-stride", type=int, default=2,
                    help="keep every Nth training epoch (the curves are 400 long)")
    ap.add_argument("--label", default="", help="tag shown in the report header")
    args = ap.parse_args()

    root = Path(args.run_dir)
    results = root / "RESULTS"
    runs = []

    for family_dir, protocol in (("MARFORMER", "transductive"),
                                 ("MARFORMER-NT", "nontransductive")):
        base = results / family_dir / "DOMAIN3-SIGMOID"
        if not base.is_dir():
            continue
        for run_dir in sorted(base.rglob("*_MARFORMER")):
            m = RUN_RE.search(run_dir.name)
            if not m:
                continue
            axis = m.group(1).lower()
            size = int(m.group(3))
            # <AXIS>/<run>  (canonical) or <AXIS>/seed<N>/<run>
            parent = run_dir.parent.name
            sm = SEED_RE.match(parent)
            cfg = read_json(run_dir / "train_config.json") or {}
            seed = int(sm.group(1)) if sm else int(cfg.get("run", {}).get("seed", 42))
            meta = read_json(run_dir / "launcher_metadata.json") or {}
            model_cfg = cfg.get("model", {})
            train_cfg = cfg.get("training", {})
            runs.append({
                "family": "Marformer" if family_dir == "MARFORMER" else "MarformerNT",
                "protocol": protocol,
                "axis": axis,
                "size": size,
                "seed": seed,
                "canonical": sm is None,
                "path": str(run_dir.relative_to(root)),
                "hyper": {**{k: model_cfg.get(k) for k in MODEL_KEYS if k in model_cfg},
                          **{k: train_cfg.get(k) for k in TRAIN_KEYS if k in train_cfg},
                          "seed": seed},
                "runtime_min": (float(meta["elapsed_seconds"]) / 60.0
                                if meta.get("elapsed_seconds") is not None else None),
                "test": collect_checkpoints(run_dir),
                "history": collect_history(run_dir, args.history_stride),
            })

    payload = {
        "run_id": root.name,
        "label": args.label or root.name,
        "n_runs": len(runs),
        "runs": runs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    size_mb = out.stat().st_size / 2**20
    complete = sum(1 for r in runs if r["test"]["last"])
    print(f"{len(runs)} runs ({complete} with test results) -> {out}  {size_mb:.1f} MiB")
    by = {}
    for r in runs:
        by.setdefault((r["family"], r["axis"]), set()).add(r["seed"])
    for k, v in sorted(by.items()):
        print(f"  {k[0]:<13} {k[1]:<6} seeds {sorted(v)}")


if __name__ == "__main__":
    main()
