#!/usr/bin/env python
"""Is the overfitting overconfidence, or is the prediction itself getting worse?

Jason's question on the overfitting thread (2026-10-06): the dev curve turns up
while training CE keeps falling, and he asked whether that is the
heteroskedastic-regression failure -- the location stays fine and the model just
gets too sure of it -- or whether the location genuinely degrades. His phrasing:
"break down these losses in terms of mean and variance ... (not mean & variance
exactly, since these are Likert variables, but mean and entropy might be ok)".

For every periodic checkpoint of a run this scores three splits and reports, per
split, a LOCATION family and a SPREAD family, so the two can move independently:

  location   mae  = mean |E_p[y] - y|          Likert "mean" error
             rmse = sqrt(mean (E_p[y] - y)^2)
             acc  = argmax accuracy
  spread     ent  = mean H(p), nats            Likert "entropy"
             pvar = mean Var_p[y]              the model's own claimed variance
             var_ratio = mean pvar / mean (E_p[y]-y)^2
                    < 1  => the model claims less spread than it actually has,
                            i.e. OVERCONFIDENT in exactly Jason's sense
  split      T*   = scalar temperature minimising NLL on this split
                    (> 1 => logits too sharp => overconfident)
             nll_T = NLL after dividing logits by T*
             d_oc  = nll - nll_T >= 0, the part of the loss a single global
                     confidence knob could remove.  nll_T is what is left: the
                     part that is genuinely a worse prediction.

d_oc vs nll_T is the headline. If the dev-loss rise is all d_oc, the model is
overconfident and nothing more; if nll_T rises too, the predictions themselves
decayed and no amount of temperature scaling will save them.

T* is fitted ON the split it is reported for, so nll_T is an optimistic bound on
what recalibration could buy -- that is deliberate. It is a decomposition of a
curve, not a reportable metric, and the optimism makes the "it is NOT only
overconfidence" conclusion conservative.

    python mean_entropy_curves.py --run RUNDIR [RUNDIR ...] --device cuda -o out.json
"""
from __future__ import annotations

import argparse
import json
import random
import math
import re
import sys
from pathlib import Path

import torch

REPO = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
sys.path.insert(0, str(REPO))

from imputer.data import DataConverter                                    # noqa: E402
from imputer.entity_mf.data import variable_list_to_entity_graph           # noqa: E402
from imputer.entity_mf.test import _load_checkpoint                        # noqa: E402
from imputer.entity_mf.masking import MCARMasking, MCARConfig               # noqa: E402
from imputer.entity_mf.validation import _entity_key                       # noqa: E402

EPOCH_RE = re.compile(r"periodic-epoch=(\d+)\.ckpt$")


# ── rebuilding the run, with the validation block put back ───────────────────
def reconstruct(run_dir: Path):
    """model + the three partitions, with the training-time val carve reapplied.

    `test._reconstruct` loads the bundle straight off disk, where the val block
    does not exist -- the carve happens in memory inside train.py. Replaying it
    here is what makes the dev split below the SAME cells the dev curve in
    training_history.json was computed on; without it "dev" would silently be a
    slice of train.

    The replay reads the entity ids train_config.json recorded rather than
    re-running `carve_validation`'s RNG, so it cannot drift if the sampling
    order ever changes, and the recorded cell counts are asserted against it.
    """
    from imputer.entity_mf.test import _reconstruct as _r  # builds model + types
    cfg = json.loads((run_dir / "train_config.json").read_text())
    model, _, train_cfg = _r(run_dir)

    sizes = cfg["resolved_sizes"]
    conv = DataConverter(num_attributes=sizes["num_attributes"],
                         num_annotators=sizes["num_annotators"],
                         num_items=sizes["num_items"],
                         num_likert_classes=sizes["num_likert_classes"],
                         max_rank_size=sizes.get("max_rank_size", 2))
    data_dir = Path(cfg["data"]["data_dir"])
    bundle = conv.load_bundle_data(data_dir / "data_bundle.json")

    vs = cfg.get("validation") or {}
    protocol = vs.get("protocol", "none")
    if protocol == "block":
        key = _entity_key(vs["axis"])
        chosen = set(vs["val_entities"])
        moved = {"observed": 0, "missing": 0}
        for name, rows in (("observed", bundle.observed_ratings),
                           ("missing", bundle.missing_ratings)):
            for r in rows:
                if r["instance"] == "train" and r[key] in chosen:
                    r["instance"] = "val"
                    moved[name] += 1
        for name in moved:
            want = vs[f"val_{name}"]
            if moved[name] != want:
                raise SystemExit(
                    f"{run_dir}: replayed val carve moved {moved[name]} {name} "
                    f"cells, train_config.json recorded {want}")
    elif protocol != "none":
        raise SystemExit(f"{run_dir}: val protocol {protocol!r} not handled here")

    types = model.types
    part = {}
    for p in ("train", "val", "test"):
        for s in ("observed", "missing"):
            part[(p, s)] = conv.create_variables_from_bundle(
                bundle, partition=p, status=s)
    return model, types, part, train_cfg, protocol, vs


def split_graphs(part, mask_rate: float = 0.15, mask_seed: int = 0):
    """(name, graph variables, predicate selecting the cells to score).

    train_masked  THE TRAINING OBJECTIVE'S OWN POPULATION. The loss is computed
                  on observed training cells that MCAR masking has hidden for
                  this step -- not on the training partition's missing cells,
                  which carry no label into the loss at all. So "what does the
                  model look like in-sample" means masked observed cells, and
                  anything else answers a different question. The same
                  MCARMasking the trainer uses is replayed here under a fixed
                  seed, at the trained masking_rate, and the masked cells
                  (status 1) are the ones scored.
    dev           exactly train.py's `val_eval_vars` under the block protocol --
                  val_all, which brings its own observed context (train.py ~180).
    corner        the paper's test metric: the whole test block in one graph
                  with training context, never split across chunks (D22/D23).

    Each split is scored on its own graph, the way training and evaluation each
    build their own; the three are not slices of one forward pass.
    """
    out = []
    tr_o, tr_m = part[("train", "observed")], part[("train", "missing")]
    va_o, va_m = part[("val", "observed")], part[("val", "missing")]
    te_o, te_m = part[("test", "observed")], part[("test", "missing")]

    if tr_o:
        random.seed(mask_seed)                      # MCARMasking draws from `random`
        masked = MCARMasking(MCARConfig(masking_rate=mask_rate)).mask(tr_o)
        hidden = {id(v) for v in masked if getattr(v, "status", None) == 1}
        if hidden:
            out.append(("train_masked", masked + tr_m, hidden))
    if tr_m:
        out.append(("train_missing", tr_o + tr_m, set(map(id, tr_m))))
    if va_m and va_o:
        out.append(("dev", va_o + va_m, set(map(id, va_m))))
    if te_m:
        out.append(("corner", tr_o + te_o + te_m, set(map(id, te_m))))
    return out


# ── metrics ──────────────────────────────────────────────────────────────────
def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Scalar T minimising NLL of softmax(logits / T). Grid then bisect on log T."""
    def nll(t: float) -> float:
        return float(torch.nn.functional.cross_entropy(logits / t, y))
    lo, hi = -2.5, 2.5                       # log T in [exp(-2.5), exp(2.5)]
    grid = [lo + (hi - lo) * i / 40 for i in range(41)]
    best = min(grid, key=lambda g: nll(math.exp(g)))
    a, b = best - (hi - lo) / 40, best + (hi - lo) / 40
    for _ in range(40):                      # golden section
        m1, m2 = a + 0.382 * (b - a), a + 0.618 * (b - a)
        if nll(math.exp(m1)) < nll(math.exp(m2)):
            b = m2
        else:
            a = m1
    return math.exp((a + b) / 2)


def stats(logits: torch.Tensor, y: torch.Tensor) -> dict:
    """Location, spread, and -- the part that actually answers the question --
    the same quantities split by whether the argmax was RIGHT or WRONG.

    Mean entropy alone is ambiguous: a sharp distribution is good when it is on
    the true class and catastrophic when it is not. `conf_wrong` (the mean
    probability the model assigned to its own choice on the cells it got wrong)
    is the direct reading of "confidently wrong", and `conf - acc` is the signed
    confidence gap: positive means the model's stated probability of being right
    exceeds how often it is right.
    """
    C = logits.shape[-1]
    k = torch.arange(C, dtype=logits.dtype, device=logits.device)
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    mu = (p * k).sum(-1)
    pvar = (p * (k - mu[:, None]) ** 2).sum(-1)
    err2 = (mu - y.to(logits.dtype)) ** 2
    nll_i = -logp.gather(1, y[:, None]).squeeze(1)      # per cell
    ent_i = -(p * logp).sum(-1)
    conf_i, pred = p.max(-1)                            # the model's own bet
    right = pred == y
    wrong = ~right
    n, nr, nw = len(y), int(right.sum()), int(wrong.sum())
    m = lambda t, msk: float(t[msk].mean()) if int(msk.sum()) else float("nan")

    T = fit_temperature(logits, y)
    nll = float(nll_i.mean())
    return {
        "n": n,
        "nll": nll,
        "acc": nr / n,
        # --- location ---
        "mae": float((mu - y.to(logits.dtype)).abs().mean()),
        "rmse": float(err2.mean().sqrt()),
        # --- spread ---
        "ent": float(ent_i.mean()),
        "pvar": float(pvar.mean()),
        "var_ratio": float(pvar.mean() / err2.mean().clamp_min(1e-12)),
        # --- confidence vs correctness: the direct decomposition ---
        "conf": float(conf_i.mean()),                   # stated P(my answer is right)
        "conf_gap": float(conf_i.mean()) - nr / n,      # > 0 => overconfident
        "conf_right": m(conf_i, right),
        "conf_wrong": m(conf_i, wrong),                 # "confidently wrong"
        "ent_right": m(ent_i, right),
        "ent_wrong": m(ent_i, wrong),
        "nll_right": m(nll_i, right),
        "nll_wrong": m(nll_i, wrong),
        # how much of the total loss the wrong cells are responsible for
        "loss_share_wrong": (float(nll_i[wrong].sum()) / float(nll_i.sum())
                             if nw and float(nll_i.sum()) > 0 else 0.0),
        # --- the temperature split ---
        "T_star": T,
        "nll_T": float(torch.nn.functional.cross_entropy(logits / T, y)),
        "d_oc": nll - float(torch.nn.functional.cross_entropy(logits / T, y)),
    }


@torch.no_grad()
def score(model, types, variables, score_ids, max_item, device) -> dict | None:
    """Forward the graph (chunked like training) and pool the scored cells."""
    from analysis.split_missing_metrics import chunk_by_item
    C = model.types["rating"].num_classes
    L, Y = [], []
    keep = {i for v in variables if id(v) in score_ids for i in v.item_ids}
    for chunk in chunk_by_item(variables, max_item, keep_whole=keep or None):
        graph = variable_list_to_entity_graph(chunk, types)
        params = model(graph, device=device)
        for idx, var in enumerate(chunk):
            if id(var) not in score_ids or var.rating_value is None:
                continue
            L.append(params[0, idx, 1:1 + C])
            Y.append(int(var.rating_value))
    if not L:
        return None
    return stats(torch.stack(L).float(), torch.tensor(Y, device=L[0].device))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=1,
                    help="score every Nth periodic checkpoint (they are 25 epochs apart)")
    ap.add_argument("--max-epoch", type=int, default=None)
    ap.add_argument("--splits", default="train_masked,train_missing,dev,corner")
    ap.add_argument("--mask-rate", type=float, default=None,
                    help="default: the rate this run trained at")
    ap.add_argument("--mask-seed", type=int, default=0)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    want = set(a.splits.split(","))
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = []

    for rd in a.run:
        run = Path(rd)
        print(f"\n=== {run}", flush=True)
        model, types, part, train_cfg, protocol, vs = reconstruct(run)
        model.to(device).eval()
        max_item = train_cfg.get("max_item")
        rate = a.mask_rate if a.mask_rate is not None else train_cfg.get("masking_rate", 0.15)
        graphs = [g for g in split_graphs(part, rate, a.mask_seed) if g[0] in want]
        print(f"  protocol={protocol} max_item={max_item} mask_rate={rate} "
              + " ".join(f"{n}:{len(ids)}" for n, _, ids in graphs), flush=True)

        ckpts = []
        for p in (run / "checkpoints").glob("periodic-epoch=*.ckpt"):
            m = EPOCH_RE.search(p.name)
            if m:
                ckpts.append((int(m.group(1)), p))
        ckpts.sort()
        if a.max_epoch is not None:
            ckpts = [c for c in ckpts if c[0] <= a.max_epoch]
        ckpts = ckpts[::a.stride]

        for ep, path in ckpts:
            _load_checkpoint(model, path, device)
            model.to(device).eval()
            row = {"run": str(run), "epoch": ep}
            for name, variables, ids in graphs:
                s = score(model, types, variables, ids, max_item, device)
                if s:
                    row[name] = s
            out.append(row)
            d = row.get("dev") or row.get("corner") or {}
            t = row.get("train_masked") or row.get("train_missing") or {}
            print(f"  ep{ep:<5} train nll={t.get('nll', float('nan')):.4f} "
                  f"ent={t.get('ent', float('nan')):.3f} | "
                  f"dev nll={d.get('nll', float('nan')):.4f} "
                  f"ent={d.get('ent', float('nan')):.3f} "
                  f"mae={d.get('mae', float('nan')):.3f} "
                  f"vr={d.get('var_ratio', float('nan')):.2f} "
                  f"T*={d.get('T_star', float('nan')):.2f} "
                  f"d_oc={d.get('d_oc', float('nan')):+.4f}", flush=True)

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))

    print(f"\nwrote {len(out)} checkpoint rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
