#!/usr/bin/env python
"""B2: is the aligned, frozen typed codec the same function as the dual stream?

The meeting's proposal was: initialise the encoder as a projection into the last
six dimensions of R^d, decode by reading the last five, freeze it, and expect
bit-exactness with the shipped dual-stream architecture.

This checks that claim three ways:

  1. one forward pass on a real graph, dual vs aligned-frozen typed;
  2. the same under the batched-augmentation path;
  3. five optimizer steps, asserting the shared parameters stay identical -- so
     the equivalence survives training, not just inference.

It also runs the same comparison against a plain identity encoder, which is what
the proposal literally says, to show why the scale factor is needed: the dual
stream writes logit_high (20.0) in the observed class's slot, not 1.0.
"""
from __future__ import annotations
import copy, json, os, sys
from pathlib import Path
import torch

# See threads/block-projection/c1_check.py: RERUN/threads is a shared symlink, so
# the import root must come from RANKING_ROOT, not from __file__.
RANKING_ROOT = Path(os.environ["RANKING_ROOT"]).resolve()
RERUN = RANKING_ROOT.parent.parent / "RERUN"
sys.path.insert(0, str(RERUN))
sys.path.insert(0, str(RANKING_ROOT))
from analysis.split_missing_metrics import chunk_by_item
from imputer.entity_mf.test import _reconstruct
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf import model as _model_mod
from imputer.entity_mf.model import EntityMarformer
from imputer.entity_mf.fastgraph import graph_tensors

assert Path(_model_mod.__file__).resolve().is_relative_to(RANKING_ROOT), (
    f"imported model.py from {_model_mod.__file__}, not from {RANKING_ROOT}")

RUN = RERUN / "runs/replicate-01/RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/DOMAIN3-SIGMOID_Item_T_300_MARFORMER"
OUT = Path(__file__).resolve().parent / "out/b2_equivalence.json"
TOL = 1e-4          # float32 over 8 blocks, against outputs of scale ~20


def build_typed(dual, types, n_rel, *, init="aligned", freeze=True, scale=True,
                reserve=True):
    """A linear-codec twin of `dual` sharing every parameter they have in common.

    reserve=False is the configuration the tail-pin hooks used to rescue: the
    aligned codec without its claim on the value range, so the centroids are
    model_dim wide and gradient can reach into the slots the codec writes.
    """
    cfg = copy.deepcopy(dual.config)
    cfg.value_codec = (f"linear:init={init}" + (",freeze" if freeze else "")
                       + ("" if reserve else ",reserve=false"))
    torch.manual_seed(0)
    m = EntityMarformer(cfg, types, num_relationships=n_rel)
    sd = dual.state_dict()
    # Centroids widen from feature_dim to model_dim; the codec owns the tail, so
    # the shipped centroid is copied into the head and the tail left at zero.
    own = m.state_dict()
    for k, v in sd.items():
        if k not in own:
            continue
        if own[k].shape == v.shape:
            own[k] = v.clone()
        elif own[k].dim() == v.dim() and own[k].shape[-1] > v.shape[-1]:
            w = own[k].clone(); w[..., : v.shape[-1]] = v; own[k] = w
    m.load_state_dict(own, strict=True)
    if not scale:   # the literal proposal: an identity encoder, no logit_high
        for codec in m.codecs.values():
            w = codec.encoder.proj.weight.data
            w[w != 0] = 1.0
    return m


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dual, eval_vars, _ = _reconstruct(RUN)
    dual.to(dev).eval()
    n_rel = dual.blocks[0]["attn"].num_relationships
    chunk = chunk_by_item(eval_vars, 300)[0]
    graph = variable_list_to_entity_graph(chunk, dual.types)
    C = dual.types["rating"].num_classes
    res = {}

    # Only tokens that CARRY a value are comparable. The dual stream's parameter
    # slice happens to hold something for entity tokens too (whatever the block
    # stack left there), and the loss and the evaluation path both ignore it; the
    # typed decoder correctly emits nothing for a token with no value. Comparing
    # those positions would be comparing a number nobody reads.
    gt0 = graph_tensors(graph, dual.types, dual.param_dim, dev)
    value_tokens = gt0.is_variable.nonzero(as_tuple=False).squeeze(-1)

    def logits(out):
        return out[:, value_tokens, 1 : 1 + C]

    # 1. single forward -------------------------------------------------------
    typed = build_typed(dual, dual.types, n_rel).to(dev).eval()
    with torch.no_grad():
        a, b = dual(graph, device=dev), typed(graph, device=dev)
    res["forward_max_abs_diff"] = float((logits(a) - logits(b)).abs().max())

    # 1b. the literal proposal, without the logit_high scale -------------------
    unscaled = build_typed(dual, dual.types, n_rel, scale=False).to(dev).eval()
    with torch.no_grad():
        c = unscaled(graph, device=dev)
    res["forward_max_abs_diff_identity_encoder"] = float((logits(a) - logits(c)).abs().max())

    # 1c. the same, with the value scale set to 1.0 ---------------------------
    # If the shipped architecture is rewritten to write 1.0 instead of 20.0 into
    # an observed value's slot, then the aligned encoder IS the plain identity
    # injection -- the literal form proposed at the meeting, with no scale factor
    # to remember. A fresh graph is needed because the precomputed value rows are
    # cached per graph.
    for t in dual.types.values():
        if hasattr(t, "logit_high"):
            t.logit_high = 1.0
    graph1 = variable_list_to_entity_graph(chunk, dual.types)
    typed1 = build_typed(dual, dual.types, n_rel).to(dev).eval()
    gt1 = graph_tensors(graph1, dual.types, dual.param_dim, dev)
    vt1 = gt1.is_variable.nonzero(as_tuple=False).squeeze(-1)
    with torch.no_grad():
        a1, b1 = dual(graph1, device=dev), typed1(graph1, device=dev)
    res["forward_max_abs_diff_scale1"] = float(
        (a1[:, vt1, 1 : 1 + C] - b1[:, vt1, 1 : 1 + C]).abs().max())
    enc1 = typed1.codecs["rating"].encoder.proj.weight.data
    res["scale1_encoder_is_plain_injection"] = bool(
        ((enc1 == 0) | (enc1 == 1)).all())
    for t in dual.types.values():
        if hasattr(t, "logit_high"):
            t.logit_high = 20.0

    # 2. batched-augmentation path -------------------------------------------
    gt = gt0
    status = torch.where(gt.is_variable, torch.full_like(gt.type_ids, 2),
                         torch.zeros_like(gt.type_ids))
    status = status.unsqueeze(0).repeat(3, 1)
    status[1, gt.is_variable.nonzero(as_tuple=False).squeeze(-1)[:5]] = 1   # mask a few
    with torch.no_grad():
        ab = dual.forward_batched(graph, status, device=dev)
        bb = typed.forward_batched(graph, status, device=dev)
    res["batched_max_abs_diff"] = float((logits(ab) - logits(bb)).abs().max())

    # 3. five optimizer steps -------------------------------------------------
    # Training uses dropout, and the two models consume the RNG at different
    # points, so each step is re-seeded identically for both -- otherwise this
    # measures the dropout masks rather than the architecture.
    target = torch.zeros(value_tokens.shape[0], dtype=torch.long, device=dev)

    def train_drift(pin_tail: bool) -> list[float]:
        d2 = copy.deepcopy(dual).to(dev).train()
        t2 = build_typed(dual, dual.types, n_rel, reserve=pin_tail).to(dev).train()
        od = torch.optim.AdamW([p for p in d2.parameters() if p.requires_grad], lr=2e-4)
        ot = torch.optim.AdamW([p for p in t2.parameters() if p.requires_grad], lr=2e-4)
        diffs = []
        for step in range(5):
            for m, opt in ((d2, od), (t2, ot)):
                torch.manual_seed(1000 + step)
                out = m(graph, device=dev)
                loss = torch.nn.functional.cross_entropy(logits(out)[0], target)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                sd_d, sd_t = d2.state_dict(), t2.state_dict()
                diffs.append(max(float((sd_t[k] - v).abs().max())
                                 for k, v in sd_d.items()
                                 if k in sd_t and sd_t[k].shape == v.shape))
        return diffs

    step_diffs = train_drift(pin_tail=True)
    res["shared_parameter_drift_over_5_steps"] = step_diffs
    # What the reservation is worth: without it the codec's centroids and
    # deviations are model_dim wide, so gradient reaches into the value slice,
    # which the dual stream's narrower centroids cannot do. (This used to be a
    # gradient hook pinning the tail; the reservation makes it structural.)
    res["drift_if_embedding_tail_not_pinned"] = train_drift(pin_tail=False)

    print(json.dumps(res, indent=1))
    ok = (res["forward_max_abs_diff"] < TOL
          and res["forward_max_abs_diff_scale1"] < TOL
          and res["scale1_encoder_is_plain_injection"]
          and res["batched_max_abs_diff"] < TOL
          and max(step_diffs) < TOL)
    res["equivalent"] = ok
    OUT.write_text(json.dumps(res, indent=1))
    print("\nEQUIVALENT" if ok else "\nNOT EQUIVALENT")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
