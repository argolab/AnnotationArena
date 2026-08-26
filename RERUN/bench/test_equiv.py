"""Equivalence check for the tensorised graph path.

Every optimisation in this branch replaces a per-token Python loop with a
gather. This asserts the replacements return the same numbers as the loops they
replaced, on a real Domain 3 split:

  1. EntityMarformer._build_initial_streams   vs  ..._ref  (the original loop)
  2. eval._rating_breakdown_fast              vs  RatingVariableType.compute_loss_breakdown
  3. evaluate_entity_marformer_split          vs  a reference loop written here

Dropout is disabled (model.eval()) so the comparison is exact up to float
associativity; entity dropout is a distribution-level equivalence, not a
value-level one, and is checked separately by the training curve.

  python -m RERUN.bench.test_equiv --data-dir <split> [--device cuda]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from imputer.entity_mf.train import load_bundle_and_converter
from imputer.entity_mf.config import EntityMarformerConfig
from imputer.entity_mf.data import variable_list_to_entity_graph
from imputer.entity_mf.eval import (
    _aggregate_loss_from_breakdowns,
    evaluate_entity_marformer_split,
)
from imputer.entity_mf.model import EntityMarformer
from imputer.entity_mf.types import build_default_domain3_types

FAILURES: list[str] = []


def check(name: str, got, want, tol: float = 1e-5) -> None:
    if isinstance(got, torch.Tensor):
        diff = (got - want).abs().max().item()
        ok = diff <= tol
        detail = f"max|diff|={diff:.3e}"
    else:
        diff = abs(float(got) - float(want))
        ok = diff <= tol
        detail = f"got={got!r} want={want!r} diff={diff:.3e}"
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:<44} {detail}")
    if not ok:
        FAILURES.append(name)


def reference_eval_loop(params, graph, variables, num_classes, device):
    """The evaluate_entity_marformer_split loop as it was before vectorisation."""
    arange = torch.arange(num_classes, device=device, dtype=torch.float32)
    m_preds, m_true, o_preds, o_true = [], [], [], []
    correct = total = 0
    for idx, _var in enumerate(variables):
        tok = graph.tokens[idx]
        if tok.type_name != "rating" or tok.status not in (0, 2):
            continue
        raw = tok.raw_data or {}
        rv = raw.get("rating_value", None)
        if rv is None:
            continue
        logits = params[0, idx, 1 : 1 + num_classes]
        expected = float((torch.softmax(logits, dim=-1) * arange).sum().item())
        if tok.status == 0:
            if int(torch.argmax(logits).item()) == int(rv):
                correct += 1
            total += 1
            m_preds.append(expected)
            m_true.append(int(rv))
        else:
            o_preds.append(expected)
            o_true.append(int(rv))
    return m_preds, m_true, o_preds, o_true, (correct / total if total else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-vars", type=int, default=6000,
                    help="cap the variable list; the reference paths are slow")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(0)

    bundle, converter, sizes = load_bundle_and_converter(Path(args.data_dir))

    types = build_default_domain3_types(
        num_attributes=sizes["num_attributes"],
        num_annotators=sizes["num_annotators"],
        num_items=sizes["num_items"],
        num_likert_classes=sizes["num_likert_classes"],
        max_rank_size=sizes.get("max_rank_size", converter.max_rank_size),
        logit_high=20.0,
        item_dropout_rate=1.0,
        annotator_dropout_rate=0.0,
    )

    obs = converter.create_variables_from_bundle(bundle, partition="train", status="observed")
    mis = converter.create_variables_from_bundle(bundle, partition="train", status="missing")
    variables = (obs + mis)[: args.max_vars]
    # Mark a slice as masked so all three statuses appear in the graph.
    for v in variables[:200]:
        if v.status == 2:
            v.status = 1
    graph = variable_list_to_entity_graph(variables, types)

    config = EntityMarformerConfig()
    config.embedding_dim = 80
    config.num_layers = 8
    config.attention_heads = 4
    config.d_ff = 128
    config.num_ffn_layers = 1
    config.dropout = 0.1
    config.use_per_head_rel = False
    config.scale_shared_rel = True
    config.use_pointer = True
    model = EntityMarformer(config=config, types=types,
                            num_relationships=graph.num_relationships).to(device)
    # Non-zero deviations, otherwise the gather path is trivially right.
    with torch.no_grad():
        for tbl in model.deviation_tables.values():
            tbl.normal_(0.0, 0.1)
    model.eval()

    print(f"device={device}  L={graph.num_tokens}  vars={len(variables)}")

    print("\n1. initial streams")
    with torch.no_grad():
        f_new, p_new, m_new = model._build_initial_streams(graph, device)
        f_ref, p_ref, m_ref = model._build_initial_streams_ref(graph, device)
    check("features", f_new, f_ref)
    check("params", p_new, p_ref)
    # The vectorised path returns None (nothing is padded) where the reference
    # returned an all-true mask; that is the same mask, minus a masked_fill over
    # the score tensor per layer.
    check("attn_mask all-valid", float(m_new is None and bool(m_ref.all())), 1.0, tol=0)

    print("\n2. rating loss breakdown")
    with torch.no_grad():
        params = model(graph, device=device)
        info = _aggregate_loss_from_breakdowns(params, graph, types,
                                               model.global_param_dim, device)
        type_mask = torch.tensor([t.type_name == "rating" for t in graph.tokens],
                                 device=device, dtype=torch.bool).unsqueeze(0)
        ref = types["rating"].compute_loss_breakdown(
            predicted_params=params, tokens=graph.tokens,
            type_mask=type_mask, global_param_dim=model.global_param_dim)
    check("loss_masked", info["loss_masked"], ref.loss_masked)
    check("loss_observed", info["loss_observed"], ref.loss_observed)
    check("loss_missing", info["loss_missing"], ref.loss_missing)
    check("n_masked", info["n_masked"], ref.n_masked, tol=0)
    check("n_observed", info["n_observed"], ref.n_observed, tol=0)
    check("n_missing", info["n_missing"], ref.n_missing, tol=0)
    check("trainable_masked_loss", info["trainable_masked_loss"].item(),
          ref.trainable_loss.item())

    print("\n3. evaluate_entity_marformer_split")
    with torch.no_grad():
        res = evaluate_entity_marformer_split(
            model=model, split="train", variables=variables, types=types,
            global_param_dim=model.global_param_dim, device=device, max_item=None)
        graph2 = variable_list_to_entity_graph(variables, types)
        params2 = model(graph2, device=device)
        m_p, m_t, o_p, o_t, acc = reference_eval_loop(
            params2, graph2, variables, types["rating"].num_classes, device)
    check("n_missing_preds", len(res.missing_preds), len(m_p), tol=0)
    check("n_observed_preds", len(res.observed_preds), len(o_p), tol=0)
    check("missing_preds", torch.tensor(res.missing_preds), torch.tensor(m_p), tol=1e-4)
    check("missing_true", torch.tensor(res.missing_true, dtype=torch.float),
          torch.tensor(m_t, dtype=torch.float), tol=0)
    check("observed_preds", torch.tensor(res.observed_preds), torch.tensor(o_p), tol=1e-4)
    check("acc", res.metrics["missing"]["rating"]["acc"], acc)

    print("\n4. batched-augmentation path (B rows must equal B separate passes)")
    from imputer.entity_mf.eval import compute_trainable_loss
    from imputer.entity_mf.fastgraph import graph_tensors

    # sample_status's contract is the training chunk layout: the first
    # n_maskable tokens are exactly the maskable observed variables. Rebuild the
    # graph that way rather than reusing the one section 2 mutated.
    for v in variables:
        if v.status == 1:
            v.status = 2
    maskable = [v for v in variables if v.status == 2]
    rest = [v for v in variables if v.status != 2]
    graph4 = variable_list_to_entity_graph(maskable + rest, types)
    gt = graph_tensors(graph4, types, model.global_param_dim, device)
    graph = graph4
    B = 3
    torch.manual_seed(7)
    status = gt.sample_status(n_maskable=len(maskable), masking_rate=0.15, batch=B)
    with torch.no_grad():
        batched = model.forward_batched(graph, status, device=device)
        # Same statuses, one row at a time through the unbatched path.
        rows = []
        for b in range(B):
            for i, tok in enumerate(graph.tokens):
                tok.status = int(status[b, i].item())
                if tok.raw_data is not None:
                    tok.raw_data["is_masked"] = tok.status == 1
            rows.append(model(graph, device=device)[0])
        stacked = torch.stack(rows, dim=0)
    check("forward_batched params", batched, stacked)
    check("per-row mask counts", int((status == 1).sum(dim=1).unique().numel()), 1, tol=0)

    with torch.no_grad():
        loss_b, mce_b, oce_b, nm_b, no_b = compute_trainable_loss(
            batched, graph, types, model.global_param_dim, device,
            masked_loss_weight=15.0, observed_loss_weight=1.0, status=status)
        mces, nms = [], []
        for b in range(B):
            for i, tok in enumerate(graph.tokens):
                tok.status = int(status[b, i].item())
                if tok.raw_data is not None:
                    tok.raw_data["is_masked"] = tok.status == 1
            gt.refresh(graph)
            _, mce, _, nm, _ = compute_trainable_loss(
                stacked[b : b + 1], graph, types, model.global_param_dim, device,
                masked_loss_weight=15.0, observed_loss_weight=1.0)
            mces.append(mce)
            nms.append(nm)
    check("batched masked CE == mean of rows", mce_b, sum(mces) / len(mces), tol=1e-5)
    check("batched n_masked == sum of rows", nm_b, sum(nms), tol=0)

    print()
    if FAILURES:
        raise SystemExit(f"FAILED: {', '.join(FAILURES)}")
    print("all equivalence checks passed")


if __name__ == "__main__":
    main()
