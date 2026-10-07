#!/usr/bin/env python
"""Logit-lens / tuned-lens analysis of a trained Marformer, layer by layer.

The appendix "Logit Lens Analysis" in main.tex quotes exact numbers (80d-Tuned
Lens log-loss 1.386 -> 0.566 across layer 0->1, Param Lens smECE 0.163 at layer
1 falling to 0.027 at the output) but the script that produced them was never
committed -- D8 in RERUN/DISCREPANCIES.md.  This is a reimplementation from the
appendix's own method description, run against final-v2 so the numbers belong to
the architecture the paper now reports (standard block: dual_outer_residual=none
with block_output_projection=none).

Why the lens is meaningful here at all.  With value_codec=dual the residual
stream is [features ; params] and the model's output logits are read straight out
of `params[1 : 1+C]` -- the same slots at every depth, never re-projected.  So
`params` is literally a distribution at every layer, and reading it at layer l is
the parameter-stream analogue of a logit lens.  Under the standard block the
per-layer update is h' = h + attn + ff, so the stream is additive and the layer-0
encoding survives to the top; see RERUN/reports/RESIDUAL_ABLATION.md.

Three readouts at each layer l in {0..L}, exactly the appendix's three:

  Param Lens        softmax(psi_l[1 : 1+C]).  No learned transformation at all.
  Tuned Param Lens  a learned linear map psi_l (P dims) -> C logits.
  Tuned 80d-Lens    the same, on the full hidden state h_l = [f_l ; psi_l].

Both tuned probes are fitted to mimic the FINAL model's predictions by
minimising KL(q_final || p_probe), on training-missing cells -- cells the
training objective never put a label on (the masked-autoencoding loss scores
masked *observed* cells), so they are held out from the model and carry no test
information.  The probes are then evaluated against true labels on the test
corner, which is the paper's own D22/D23-correct population.

Two honest deviations from the appendix, both recorded in the output JSON:

  * Input standardisation.  Each probe input dimension is centred and scaled by
    statistics from the probe-fitting split.  A linear probe composed with an
    affine reparametrisation is still exactly a linear probe -- the function
    class is identical -- but without it 300 epochs of AdamW does not converge on
    a stream whose param slots carry +-20 logits while its feature slots are
    O(1).  `probe_fit_kl` is reported per probe so convergence is checkable.
  * Learning rate.  The appendix records "300 epochs with AdamW" and no step
    size; this uses full-batch AdamW at --probe-lr (default 1e-2).

    python logit_lens.py --run RUNDIR --device cuda -o out.json
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RERUN = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN")
RANK = Path("/weka/scratch/jhu/jeisner1/xwang397/Marformer/imputer/ranking")
sys.path.insert(0, str(RERUN)); sys.path.insert(0, str(RANK))

from imputer.entity_mf.data import variable_list_to_entity_graph         # noqa: E402
from imputer.entity_mf.test import _load_checkpoint                      # noqa: E402


def load_shared():
    """The paper's own calibration path, so smECE here is the paper's statistic."""
    spec = importlib.util.spec_from_file_location(
        "pd3", RANK / "scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py")
    m = importlib.util.module_from_spec(spec); sys.modules["pd3"] = m
    spec.loader.exec_module(m)
    return m.shared, m.shared.rp


# ── layerwise capture ────────────────────────────────────────────────────────
@torch.no_grad()
def layer_streams(model, graph, device):
    """Replay _dual_blocks, keeping (features, params) at every depth 0..L.

    This duplicates the block arithmetic rather than hooking it, because the
    residual adds happen in _dual_blocks itself and not inside any submodule, so
    there is nothing to hang a hook on.  Duplicated arithmetic can drift from the
    real forward pass, so the caller asserts the final layer against model(graph)
    bit-for-bit; if the architecture ever changes under this script, that assert
    is what catches it instead of a plausible-looking wrong curve.
    """
    if model.value_codec != "dual" or model.use_param_output_head:
        raise SystemExit(
            f"this lens reads the dual param stream; got value_codec="
            f"{model.value_codec!r} use_param_output_head={model.use_param_output_head}")

    features, params, attn_mask = model._build_initial_streams(graph, device=device)
    edge_flat, edge_rel = graph.build_edge_index(device)
    edge_mask = model._dense_edge_mask(graph, device)
    K_aug = model._get_k_aug(graph, device)

    streams = [(features, params)]
    combined = torch.cat([features, params], dim=-1)
    nb = len(model.blocks)
    for layer_idx, block in enumerate(model.blocks):
        normed_attn = block["norm_1"](combined)
        attn_out = block["attn"](normed_attn, edge_flat=edge_flat, edge_rel=edge_rel,
                                 attn_mask=attn_mask, K_aug=K_aug, edge_mask=edge_mask)
        combined = combined + attn_out if model.inner_residual == "both" else attn_out

        normed_ff = block["norm_2"](combined)
        z_ff = block["ff"](normed_ff)
        combined = combined + z_ff if model.inner_residual in ("both", "ff_only") else z_ff

        if model.block_output_projection == "full":
            feat_update = block["dropout_2"](block["proj_out"](combined))
            param_update = block["dropout_2"](block["W_param"](combined))
        else:
            upd = block["dropout_2"](combined)
            feat_update = upd[..., : model.feature_dim]
            param_update = upd[..., model.feature_dim:]

        mode = model._outer_residual_mode(layer_idx, nb)
        if mode == "add_h":
            features = features + feat_update
            params = params + param_update
        elif mode == "replace":
            features, params = feat_update, param_update
        else:                                            # add_c
            features = combined[..., : model.feature_dim] + feat_update
            params = combined[..., model.feature_dim:] + param_update
        combined = torch.cat([features, params], dim=-1)
        streams.append((features, params))
    return streams


@torch.no_grad()
def collect(model, eval_vars, max_item, device):
    """Per-layer hidden states for every labelled missing cell, tagged by partition.

    The population and the chunking are split_missing_metrics.evaluate's
    --corner-whole path verbatim: the FULL variable list (so every cell sees the
    context the paper's scorer gave it) chunked by item with every test item kept
    whole.  Anything else silently scores a different population -- that is D22
    and D23, and it is exactly how the old appendix came to report 414 cells.
    One pass serves both jobs: the `train` cells fit the probes, the `test` cells
    are the corner the paper reports.

    Returns psi [L+1, n, P], h [L+1, n, D], y [n], tag [n].
    """
    from analysis.split_missing_metrics import chunk_by_item
    C = model.types["rating"].num_classes
    psi_by_layer, h_by_layer, Y, TAG = None, None, [], []
    keep = {i for v in eval_vars
            if getattr(v, "instance", None) == "test" for i in v.item_ids}
    for chunk in chunk_by_item(eval_vars, max_item, keep_whole=keep or None):
        graph = variable_list_to_entity_graph(chunk, model.types)
        streams = layer_streams(model, graph, device)
        # gate the replay against the shipped forward pass
        ref = model(graph, device=device)
        got = streams[-1][1]
        if not torch.equal(ref, got):
            d = float((ref - got).abs().max())
            raise SystemExit(f"layer replay diverged from model(graph): max|d|={d:.3e}")

        rows = [idx for idx, var in enumerate(chunk)
                if getattr(var, "is_missing", False) and var.rating_value is not None]
        if not rows:
            continue
        sel = torch.tensor(rows, device=streams[0][1].device)
        if psi_by_layer is None:
            psi_by_layer = [[] for _ in streams]
            h_by_layer = [[] for _ in streams]
        for li, (f, p) in enumerate(streams):
            psi_by_layer[li].append(p[0].index_select(0, sel).float().cpu())
            h_by_layer[li].append(torch.cat([f[0], p[0]], -1).index_select(0, sel).float().cpu())
        Y += [int(chunk[i].rating_value) for i in rows]
        TAG += [getattr(chunk[i], "instance", "?") for i in rows]

    if psi_by_layer is None:
        raise SystemExit("no scored cells")
    psi = torch.stack([torch.cat(x) for x in psi_by_layer])
    h = torch.stack([torch.cat(x) for x in h_by_layer])
    return psi, h, torch.tensor(Y), TAG, C


# ── probes ───────────────────────────────────────────────────────────────────
def fit_probe(X_fit, q_fit, C, epochs, lr, device, seed=0):
    """Linear map X -> C logits, fitted to mimic q_fit by KL(q_fit || p_probe).

    Standardisation is folded in (see the module docstring): it leaves the probe
    exactly linear in the hidden state, and is what makes a fixed 300-epoch
    budget enough on a stream whose coordinates differ by an order of magnitude.
    """
    torch.manual_seed(seed)
    mu = X_fit.mean(0, keepdim=True)
    sd = X_fit.std(0, keepdim=True).clamp_min(1e-6)
    Z = ((X_fit - mu) / sd).to(device)
    q = q_fit.to(device)
    probe = torch.nn.Linear(Z.shape[1], C).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.kl_div(torch.log_softmax(probe(Z), -1), q,
                        reduction="batchmean")      # KL(q || p)
        loss.backward()
        opt.step()
    return (probe, mu.to(device), sd.to(device)), float(loss.detach())


def apply_probe(fitted, X, device):
    probe, mu, sd = fitted
    with torch.no_grad():
        return probe(((X.to(device) - mu) / sd)).cpu()


# ── metrics ──────────────────────────────────────────────────────────────────
def eval_logits(logits, y, shared, rp, nboot):
    """log-loss, accuracy, and the paper's all-class smECE for one readout."""
    logits = logits.float()
    nll = float(F.cross_entropy(logits, y))
    probs = torch.softmax(logits, -1).numpy()
    yn = y.numpy()
    # MBR under L2: the Bayes act is the posterior mean, so the risk the second
    # panel of the appendix figure plots is mean (E_p[y] - y)^2.
    k = np.arange(probs.shape[1], dtype=float)
    mu = (probs * k).sum(1)
    conf, acc = shared._all_class_calibration(probs, yn)
    d = rp.prepare_rel_diagram(conf, acc, num_bootstrap=nboot,
                               report_CE=True, report_CE_std=True)
    return dict(
        log_loss=nll,
        mbr_l2=float(((mu - yn) ** 2).mean()),
        mae=float(np.abs(mu - yn).mean()),
        accuracy=float((probs.argmax(1) == yn).mean()),
        mean_conf=float(probs.max(1).mean()),
        entropy=float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum(1).mean()),
        smece=float(d["ce"]), smece_ci=float(d.get("ce_ci_width", 0.0)),
        probs=[[round(float(v), 6) for v in r] for r in probs],
        rel=dict(mesh=_thin(d["mesh"]), mu=_thin(d["mu"]),
                 lower=_thin(d["lower"]), upper=_thin(d["upper"]),
                 density=_thin(d["density"])),
    )


def _thin(a, n=120):
    a = np.asarray(a, dtype=float)
    if len(a) <= n:
        return [round(float(v), 5) for v in a]
    idx = np.linspace(0, len(a) - 1, n).round().astype(int)
    return [round(float(v), 5) for v in a[idx]]


def gate_against_corner(run: Path, final: dict, n: int):
    """The final layer IS the paper's reported model, so it must reproduce the
    paper's reported number. If it does not, the lens is reading a different
    population and every layer above it is wrong in the same way -- which is the
    failure the old appendix shipped. Fail loudly rather than plot it.
    """
    import re as _re
    m = _re.search(r"DOMAIN3-SIGMOID_(Item|Annot)_(T|NT)_(\d+)_MARFORMER$", run.name)
    corner = next((d / "corner_metrics" / "corner.json" for d in run.resolve().parents
                   if (d / "corner_metrics" / "corner.json").is_file()), None)
    if not (m and corner):
        print(f"  [gate] skipped: no corner.json beside {run.name}", flush=True)
        return
    axis, size = m.group(1).lower(), int(m.group(3))
    want = [r for r in json.loads(corner.read_text())["rows"]
            if r["axis"] == axis and r["size"] == size
            and r["family"] == ("Marformer" if m.group(2) == "T" else "Marformer NT")]
    if not want:
        print(f"  [gate] skipped: no {axis}/{size} row in {corner}", flush=True)
        return
    t = want[0]["test"]
    dn, dl = n - t["n"], abs(final["log_loss"] - t["log_loss"])
    print(f"  [gate] corner.json n={t['n']} log_loss={t['log_loss']:.4f}  "
          f"lens n={n} log_loss={final['log_loss']:.4f}", flush=True)
    if dn or dl > 1e-6:
        raise SystemExit(
            f"FATAL: the lens's final layer does not reproduce the reported corner "
            f"(dn={dn}, dlog_loss={dl:.2e}). The lens is on a different population.")
    print("  [gate] final layer reproduces the reported corner exactly", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a refit cell directory")
    ap.add_argument("--ckpt", default="last.ckpt",
                    help="which checkpoint in <run>/checkpoints (default: the reported one)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probe-epochs", type=int, default=300)
    ap.add_argument("--probe-lr", type=float, default=1e-2)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    shared, rp = load_shared()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    run = Path(a.run)

    from imputer.entity_mf.test import _reconstruct as _r
    model, eval_vars, train_cfg = _r(run)
    ck = run / "checkpoints" / a.ckpt
    _load_checkpoint(model, ck, device)
    model.to(device).eval()
    max_item = train_cfg.get("max_item")

    print(f"=== {run.name}  ckpt={a.ckpt}  L={len(model.blocks)}  "
          f"D={model.feature_dim + model.param_dim}  P={model.param_dim}", flush=True)
    print("  collecting layer streams (one --corner-whole pass)...", flush=True)
    psi, h, y, tag, C = collect(model, eval_vars, max_item, device)
    nL = psi.shape[0]

    fit = torch.tensor([t == "train" for t in tag])
    ev = torch.tensor([t == "test" for t in tag])
    if not int(fit.sum()):
        raise SystemExit("no training-missing cells to fit probes on")
    psi_f, h_f = psi[:, fit], h[:, fit]
    psi_e, h_e, y_e = psi[:, ev], h[:, ev], y[ev]
    print(f"  probe-fit n={int(fit.sum())}  corner n={int(ev.sum())}  "
          f"layers=0..{nL - 1}  C={C}", flush=True)

    # the final model's own prediction is the probes' regression target
    q_fit = torch.softmax(psi_f[-1][:, 1:1 + C], -1)
    final = eval_logits(psi_e[-1][:, 1:1 + C], y_e, shared, rp, a.bootstrap)
    print(f"  final model: log_loss={final['log_loss']:.4f} "
          f"acc={final['accuracy']:.4f} smECE={final['smece']:.4f}", flush=True)
    gate_against_corner(run, final, int(ev.sum()))

    rows = []
    for li in range(nL):
        row = {"layer": li}
        row["param_lens"] = eval_logits(psi_e[li][:, 1:1 + C], y_e, shared, rp, a.bootstrap)
        for name, Xf, Xe in (("tuned_param_lens", psi_f[li], psi_e[li]),
                             ("tuned_80d_lens", h_f[li], h_e[li])):
            fitted, kl = fit_probe(Xf, q_fit, C, a.probe_epochs, a.probe_lr, device)
            r = eval_logits(apply_probe(fitted, Xe, device), y_e, shared, rp, a.bootstrap)
            r["probe_fit_kl"] = kl
            r["probe_in_dim"] = int(Xf.shape[1])
            row[name] = r
        rows.append(row)
        print(f"  layer {li}: param={row['param_lens']['log_loss']:.4f} "
              f"(smECE {row['param_lens']['smece']:.4f})  "
              f"tunedP={row['tuned_param_lens']['log_loss']:.4f}  "
              f"tuned80={row['tuned_80d_lens']['log_loss']:.4f}  "
              f"[fit KL {row['tuned_param_lens']['probe_fit_kl']:.4f}/"
              f"{row['tuned_80d_lens']['probe_fit_kl']:.4f}]", flush=True)

    out = dict(
        run=str(run), cell=run.name, ckpt=a.ckpt,
        num_layers=int(len(model.blocks)), model_dim=int(model.feature_dim + model.param_dim),
        param_dim=int(model.param_dim), num_classes=int(C),
        n_probe_fit=int(fit.sum()), n_corner=int(len(y_e)),
        labels=[int(v) for v in y_e],
        probe=dict(epochs=a.probe_epochs, lr=a.probe_lr, optimizer="AdamW",
                   objective="KL(q_final || p_probe)", standardized=True,
                   fit_population="train missing"),
        architecture={k: train_cfg.get(k) for k in
                      ("max_item", "masking_rate", "transductive_learning")},
        final_model=final, layers=rows,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
