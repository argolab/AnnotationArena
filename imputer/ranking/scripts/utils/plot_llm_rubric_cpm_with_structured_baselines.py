#!/usr/bin/env python3
"""
Plot LLM Rubric test missing log loss vs training item size:

  - CPM SharedThreshold STAN (from predictive_metrics.json)
  - Naive Bayes IJK (transductive count pool)
  - Structured Naive Bayes (relation-aware; fit on LOO train+val plates by default)
  - Structured log-linear (same features; PyTorch)

Unlike ``run_structured_baselines.py`` (console metrics only), this script **writes PNG(s)**.

Run from imputer/ranking:

  python scripts/utils/plot_llm_rubric_cpm_with_structured_baselines.py

  python scripts/utils/plot_llm_rubric_cpm_with_structured_baselines.py \\
      --ll-epochs 25 --train-instances train,val

Outputs (default paths under PLOTS/TALK/LLM_RUBRIC/):
  - llm_rubric_cpm_structured_baselines_log_loss.png
  - llm_rubric_cpm_structured_baselines_rmse.png  (if CPM has rating_probabilities.csv)

NB (gold) add-λ: ``--nb-gold-lambda`` for the main figure.
Optional sweeps: ``--nb-gold-lambda-sweep`` (NB gold only plot) and ``--nb-gold-augment-lambda-sweep``
(NB gold-augment only plot), each with separate output paths. Use ``--no-nb-gold-augment`` to hide
NB (gold-augment) from main and sweep plots.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# imputer/ranking/scripts/utils -> parents[2] == ranking
_RANKING_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RANKING_ROOT / "BASELINES"))

from structured_baselines.dataset_adapter import (
    build_test_examples,
    build_training_examples,
    bundle_dims,
    load_bundle_dict,
    ratings_for_ijk_fit,
)
from structured_baselines.log_linear_structured import StructuredLogLinear
from structured_baselines.feature_utils import RelationKind, RELATION_NAMES, relation_label
from structured_baselines.naive_bayes_ijk import NaiveBayesIJK
from structured_baselines.naive_bayes_structured import StructuredNaiveBayes

SIZE_RE = re.compile(r"LLMRubric_225_25_9_(\d+)_eval$")


def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _parse_lambda_sweep(s: str) -> list[float]:
    """Parse comma-separated positive floats for NB (gold) λ sweep."""
    out: list[float] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        lam = float(p)
        if lam <= 0.0:
            raise ValueError(f"NB (gold) λ must be positive, got {lam!r}")
        out.append(lam)
    return out


def _extract_size(eval_dir_name: str) -> int | None:
    m = SIZE_RE.match(eval_dir_name)
    return int(m.group(1)) if m else None


def _rmse_from_proba(examples, probs: np.ndarray) -> float | None:
    """1-based Likert truth vs E[Y] under probs (rows align with examples)."""
    if len(examples) == 0:
        return None
    c = probs.shape[1]
    classes = np.arange(1, c + 1, dtype=np.float64)
    pred = probs @ classes
    truth = np.array([ex.y + 1 for ex in examples], dtype=np.float64)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def _cpm_rmse(
    data_root: Path,
    size: int,
    eval_dir: Path,
) -> float | None:
    """Same construction as plot_llm_rubric_new_stan_curve.py for CPM."""
    probs_path = eval_dir / "rating_probabilities.csv"
    bundle_path = data_root / f"LLMRubric_225_25_9_{size}" / "data_bundle.json"
    if not probs_path.exists() or not bundle_path.exists():
        return None
    bundle = _read_json(bundle_path)
    missing = bundle.get("missing_ratings", [])
    test_idxs = [i for i, row in enumerate(missing) if row.get("instance") == "test"]
    if not test_idxs:
        return None
    labels = np.asarray([missing[i]["value"] - 1 for i in test_idxs], dtype=np.int64)
    df = pd.read_csv(probs_path)
    prob_cols = ["prob_cat_1", "prob_cat_2", "prob_cat_3", "prob_cat_4"]
    if not all(c in df.columns for c in prob_cols):
        return None
    grouped = (
        df[df["missing_rating_idx"].isin(test_idxs)]
        .groupby("missing_rating_idx")[prob_cols]
        .mean()
        .reindex(test_idxs)
    )
    if grouped.isnull().any().any():
        return None
    probs = grouped.to_numpy(dtype=np.float64)
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    pred_expected = probs @ classes
    truth = labels.astype(np.float64) + 1.0
    return float(np.sqrt(np.mean((pred_expected - truth) ** 2)))


def _unigram_subset_log_loss(
    bundle: dict,
    subset: str,
    *,
    transductive: bool = False,
    alpha: float = 1.0,
) -> float | None:
    """
    Add-one smoothed pooled unigram baseline on test-missing rows.

    subset in {"none","i","j","k","ij","ik","jk","ijk"} controls pooling key.
    Uses observed rows from train+val by default (non-transductive).
    Set transductive=True to include test-observed rows in the fit pool.
    """
    fit_instances = {"train", "val", "test"} if transductive else {"train", "val"}
    observed = [r for r in bundle.get("observed_ratings", []) if r.get("instance") in fit_instances]
    test_missing = [r for r in bundle.get("missing_ratings", []) if r.get("instance") == "test"]
    if not observed or not test_missing:
        return None

    c = max(int(r["value"]) for r in (observed + test_missing))
    if c <= 0:
        return None

    def _key(r: dict) -> tuple[int, ...]:
        i = int(r["attribute"])
        j = int(r["annotator"])
        k = int(r["item"])
        if subset == "i":
            return (i,)
        if subset == "none":
            return tuple()
        if subset == "j":
            return (j,)
        if subset == "k":
            return (k,)
        if subset == "ij":
            return (i, j)
        if subset == "ik":
            return (i, k)
        if subset == "jk":
            return (j, k)
        if subset == "ijk":
            return (i, j, k)
        raise ValueError(f"Unsupported subset: {subset}")

    pool_counts: dict[tuple[int, ...], list[float]] = {}
    for row in observed:
        key = _key(row)
        if key not in pool_counts:
            pool_counts[key] = [0.0] * c
        idx = int(row["value"]) - 1
        if 0 <= idx < c:
            pool_counts[key][idx] += 1.0

    xent = 0.0
    for row in test_missing:
        key = _key(row)
        counts = pool_counts.get(key, [0.0] * c)
        denom = sum(counts) + alpha * c
        idx = int(row["value"]) - 1
        if not (0 <= idx < c):
            return None
        prob = (counts[idx] + alpha) / denom
        xent -= math.log(prob + 1e-12)
    return xent / len(test_missing)


def _build_logreg_features(examples, I: int, J: int, K: int, C: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build compact dense features for multinomial logistic regression:
      - target one-hot: i, j, k
      - pooled source value-count histograms for groups:
        all, same_i, same_j, same_k, same_ij, same_ik, same_jk
    """
    feat_dim = I + J + K + 7 * C
    X = np.zeros((len(examples), feat_dim), dtype=np.float32)
    y = np.zeros((len(examples),), dtype=np.int64)
    for t, ex in enumerate(examples):
        off_i = 0
        off_j = off_i + I
        off_k = off_j + J
        off_h = off_k + K
        X[t, off_i + ex.target_i] = 1.0
        X[t, off_j + ex.target_j] = 1.0
        X[t, off_k + ex.target_k] = 1.0
        y[t] = ex.y

        h_all = np.zeros(C, dtype=np.float32)
        h_si = np.zeros(C, dtype=np.float32)
        h_sj = np.zeros(C, dtype=np.float32)
        h_sk = np.zeros(C, dtype=np.float32)
        h_sij = np.zeros(C, dtype=np.float32)
        h_sik = np.zeros(C, dtype=np.float32)
        h_sjk = np.zeros(C, dtype=np.float32)
        for (i_s, j_s, k_s, v_s) in ex.sources:
            if not (0 <= v_s < C):
                continue
            h_all[v_s] += 1.0
            same_i = i_s == ex.target_i
            same_j = j_s == ex.target_j
            same_k = k_s == ex.target_k
            if same_i:
                h_si[v_s] += 1.0
            if same_j:
                h_sj[v_s] += 1.0
            if same_k:
                h_sk[v_s] += 1.0
            if same_i and same_j:
                h_sij[v_s] += 1.0
            if same_i and same_k:
                h_sik[v_s] += 1.0
            if same_j and same_k:
                h_sjk[v_s] += 1.0
        h = np.concatenate([h_all, h_si, h_sj, h_sk, h_sij, h_sik, h_sjk], axis=0)
        X[t, off_h : off_h + 7 * C] = h
    return X, y


def _fit_logreg_torch(
    train_examples,
    test_examples,
    I: int,
    J: int,
    K: int,
    C: int,
    *,
    epochs: int = 120,
    lr: float = 0.08,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    device: str | None = None,
) -> tuple[float, float] | None:
    """Return (test_mean_nll, test_rmse) for an experimental logistic baseline."""
    if not train_examples or not test_examples:
        return None
    Xtr, ytr = _build_logreg_features(train_examples, I, J, K, C)
    Xte, yte = _build_logreg_features(test_examples, I, J, K, C)
    feat_dim = Xtr.shape[1]

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = nn.Linear(feat_dim, C).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr_t = torch.from_numpy(Xtr).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    Xte_t = torch.from_numpy(Xte).to(dev)

    n = Xtr_t.shape[0]
    for _ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for st in range(0, n, batch_size):
            idx = perm[st : st + batch_size]
            logits = model(Xtr_t[idx])
            loss = F.cross_entropy(logits, ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        probs = torch.softmax(model(Xte_t), dim=-1).cpu().numpy()
    nll = float(-np.log(probs[np.arange(len(yte)), yte] + 1e-12).mean())
    classes = np.arange(1, C + 1, dtype=np.float64)
    pred_exp = probs @ classes
    truth = yte.astype(np.float64) + 1.0
    rmse = float(np.sqrt(np.mean((pred_exp - truth) ** 2)))
    return nll, rmse


def _structured_nb_i_only_log_loss(
    train_examples,
    eval_examples,
    num_attrs: int,
    num_classes: int,
    *,
    alpha_prior: float = 1.0,
    alpha_emit: float = 1.0,
) -> float | None:
    r"""
    Pink (plot): p(y_{ijk} | i, sources) ∝ p(y_{ijk} | i) · ∏_{i'j'k'≠ijk}
        p(y_{i'j'k'} | i', y_{ijk}, i, 1[j=j'], 1[k=k']).

    Emission is multinomial over source rating v only; i' = i_s and (j=j', k=k') are fixed 4-way rel_jk.
    """
    if not train_examples or not eval_examples:
        return None
    I, C = num_attrs, num_classes
    R = 4
    prior = np.zeros((I, C), dtype=np.float64)
    emit = np.zeros((I, C, R, I, C), dtype=np.float64)

    def _rel_jk(j_s: int, k_s: int, j_t: int, k_t: int) -> int:
        same_j = j_s == j_t
        same_k = k_s == k_t
        if same_j and same_k:
            return 0
        if same_j and not same_k:
            return 1
        if same_k and not same_j:
            return 2
        return 3

    for ex in train_examples:
        it = ex.target_i
        y = ex.y
        prior[it, y] += 1.0
        for (i_s, j_s, k_s, v_s) in ex.sources:
            rel = _rel_jk(j_s, k_s, ex.target_j, ex.target_k)
            emit[it, y, rel, i_s, v_s] += 1.0

    total = 0.0
    for ex in eval_examples:
        it = ex.target_i
        y_true = ex.y
        row = prior[it]
        prior_denom = float(row.sum()) + alpha_prior * C
        scores = np.zeros(C, dtype=np.float64)
        for y in range(C):
            s = math.log((row[y] + alpha_prior) / prior_denom)
            sl = emit[it, y]
            for (i_s, j_s, k_s, v_s) in ex.sources:
                rel = _rel_jk(j_s, k_s, ex.target_j, ex.target_k)
                col = sl[rel, i_s]
                denom_v = float(col.sum()) + alpha_emit * C
                s += math.log((float(col[v_s]) + alpha_emit) / denom_v)
            scores[y] = s
        m = float(scores.max())
        log_norm = m + math.log(float(np.sum(np.exp(scores - m))))
        total += -(scores[y_true] - log_norm)
    return total / len(eval_examples)


def _structured_nb_notype_log_loss(
    train_examples,
    eval_examples,
    num_attrs: int,
    num_classes: int,
    *,
    alpha_prior: float = 1.0,
    alpha_emit: float = 1.0,
) -> float | None:
    r"""
    Orange (plot): p(y_{ijk} | sources) ∝ p(y_{ijk}) · ∏_{i'j'k'≠ijk}
        p(y_{i'j'k'} | i', y_{ijk}, 1[i=i'], 1[j=j'], 1[k=k']).

    rel uses ``feature_utils.relation_label`` (7-way refinement).

    Emission (add-α on each factor), full chain rule on (y, rel):
      log P(v | i', y, rel) + log P(i' | y, rel).
    """
    if not train_examples or not eval_examples:
        return None
    I, C = num_attrs, num_classes
    R = 7
    prior = np.zeros((C,), dtype=np.float64)
    emit = np.zeros((C, R, I, C), dtype=np.float64)

    for ex in train_examples:
        y = ex.y
        prior[y] += 1.0
        for (i_s, j_s, k_s, v_s) in ex.sources:
            rel = relation_label(i_s, j_s, k_s, ex.target_i, ex.target_j, ex.target_k)
            emit[y, rel, i_s, v_s] += 1.0

    total = 0.0
    for ex in eval_examples:
        y_true = ex.y
        prior_denom = float(prior.sum()) + alpha_prior * C
        scores = np.zeros(C, dtype=np.float64)
        for y in range(C):
            s = math.log((prior[y] + alpha_prior) / prior_denom)
            sl = emit[y]
            for (i_s, j_s, k_s, v_s) in ex.sources:
                rel = relation_label(i_s, j_s, k_s, ex.target_i, ex.target_j, ex.target_k)
                rel_mat = sl[rel]
                n_i = float(rel_mat[i_s].sum())
                rel_total = float(rel_mat.sum())
                denom_v = n_i + alpha_emit * C
                s += math.log((float(rel_mat[i_s, v_s]) + alpha_emit) / denom_v)
                denom_ip = rel_total + alpha_emit * I
                s += math.log((n_i + alpha_emit) / denom_ip)
            scores[y] = s
        m = float(scores.max())
        log_norm = m + math.log(float(np.sum(np.exp(scores - m))))
        total += -(scores[y_true] - log_norm)
    return total / len(eval_examples)


def _orange_factorized_preprocess(train_examples, num_attrs: int, num_classes: int):
    """Counts for the orange factorized model: emit[y, rel, i_src, v_src]."""
    if not train_examples:
        return None
    I, C = num_attrs, num_classes
    R = 7
    prior = np.zeros((C,), dtype=np.float64)
    emit = np.zeros((C, R, I, C), dtype=np.float64)
    for ex in train_examples:
        y = ex.y
        prior[y] += 1.0
        for (i_s, j_s, k_s, v_s) in ex.sources:
            rel = relation_label(i_s, j_s, k_s, ex.target_i, ex.target_j, ex.target_k)
            emit[y, rel, i_s, v_s] += 1.0
    return prior, emit, I, C, R


def _orange_factorized_eval(
    pre,
    eval_examples,
    *,
    alpha_prior: float = 1.0,
    alpha_emit: float = 1.0,
    drop_pi_rel: set[int] | None = None,
    marginal_pi_rel: set[int] | None = None,
    shuffle_rel: int | None = None,
    rng: np.random.Generator | None = None,
) -> float | None:
    """
    Evaluate orange factorized model:
      log P(y|x) = log P(y) + Σ_r [log P(v_r | i_r, y, rel_r) + log P(i_r | y, rel_r)].
    Optional diagnostics:
      - drop_pi_rel: drop log P(i|y,rel) for selected rel buckets
      - marginal_pi_rel: for those buckets, replace log P(i|y,rel) with marginal log P(i|y)
          (counts pooled over relation buckets / v via emit sum axis (rel, v))
      - shuffle_rel: permute i among sources in that rel bucket per example

    marginal_pi_rel and drop_pi_rel are treated as disjoint; if both flag the same bucket,
        marginal replaces the term (preferred over dropping).
    """
    if pre is None or not eval_examples:
        return None
    prior, emit, I, C, _R = pre
    drop_pi_rel = drop_pi_rel or set()
    marginal_pi_rel = marginal_pi_rel or set()
    rng = rng or np.random.default_rng(0)

    marginal_counts = None
    marginal_totals = None
    if marginal_pi_rel:
        marginal_counts = emit.sum(axis=(1, 3)).astype(np.float64, copy=False)
        marginal_totals = marginal_counts.sum(axis=1)

    total = 0.0
    for ex in eval_examples:
        prior_denom = float(prior.sum()) + alpha_prior * C
        src_rows: list[tuple[int, int, int]] = []
        for (i_s, j_s, k_s, v_s) in ex.sources:
            rel = relation_label(i_s, j_s, k_s, ex.target_i, ex.target_j, ex.target_k)
            src_rows.append((rel, i_s, v_s))
        if shuffle_rel is not None:
            idxs = [t for t, (rel, _i_s, _v_s) in enumerate(src_rows) if rel == shuffle_rel]
            if len(idxs) > 1:
                orig_i = [src_rows[t][1] for t in idxs]
                perm_i = list(np.array(orig_i, dtype=np.int64)[rng.permutation(len(orig_i))])
                for t, ip in zip(idxs, perm_i):
                    rel, _i_old, v_s = src_rows[t]
                    src_rows[t] = (rel, int(ip), v_s)

        scores = np.zeros(C, dtype=np.float64)
        for y in range(C):
            s = math.log((prior[y] + alpha_prior) / prior_denom)
            sl = emit[y]
            for (rel, i_s, v_s) in src_rows:
                rel_mat = sl[rel]
                n_i = float(rel_mat[i_s].sum())
                rel_total = float(rel_mat.sum())
                denom_v = n_i + alpha_emit * C
                s += math.log((float(rel_mat[i_s, v_s]) + alpha_emit) / denom_v)
                if marginal_counts is not None and rel in marginal_pi_rel:
                    m_i = float(marginal_counts[y, i_s])
                    m_tot = float(marginal_totals[y])
                    denom_m = m_tot + alpha_emit * float(I)
                    s += math.log((m_i + alpha_emit) / denom_m)
                elif rel not in drop_pi_rel:
                    denom_ip = rel_total + alpha_emit * I
                    s += math.log((n_i + alpha_emit) / denom_ip)
            scores[y] = s
        m = float(scores.max())
        log_norm = m + math.log(float(np.sum(np.exp(scores - m))))
        total += -(scores[ex.y] - log_norm)
    return total / len(eval_examples)


def _orange_direct_stats(pre) -> tuple[list[float], list[float]]:
    """Per relation bucket: MI(i; y | rel) and weighted pairwise JS over P(i|y,rel)."""
    if pre is None:
        return [], []
    _prior, emit, I, C, R = pre
    mi_vals: list[float] = []
    js_vals: list[float] = []

    def _js(p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        return 0.5 * float(np.sum(p * (np.log(p + 1e-12) - np.log(m + 1e-12)))) + 0.5 * float(
            np.sum(q * (np.log(q + 1e-12) - np.log(m + 1e-12)))
        )

    for rel in range(R):
        counts_yi = emit[:, rel, :, :].sum(axis=2)  # (C, I)
        total = float(counts_yi.sum())
        if total <= 0.0:
            mi_vals.append(float("nan"))
            js_vals.append(float("nan"))
            continue
        p_yi = counts_yi / total
        p_y = p_yi.sum(axis=1, keepdims=True)
        p_i = p_yi.sum(axis=0, keepdims=True)
        ratio = p_yi / (p_y @ p_i + 1e-12)
        mi = float(np.sum(p_yi * np.log(ratio + 1e-12)))
        mi_vals.append(mi)

        dists = counts_yi + 1e-9
        dists = dists / dists.sum(axis=1, keepdims=True)
        wy = (p_y.reshape(-1)).astype(np.float64)
        js_num = 0.0
        js_den = 0.0
        for a in range(C):
            for b in range(a + 1, C):
                w = float(wy[a] * wy[b])
                if w <= 0.0:
                    continue
                js_num += w * _js(dists[a], dists[b])
                js_den += w
        js_vals.append(js_num / js_den if js_den > 0 else float("nan"))
    return mi_vals, js_vals


def _orange_pi_soft_confusion_matrix(
    pre,
    train_examples,
    rel_bucket: int,
    *,
    alpha_emit: float = 1.0,
) -> tuple[np.ndarray, int] | None:
    """
    Fractional aggregation between observed rubric dimension i_src on an arc (row) and pooled
    P(i' | y, rel_bucket) columns (Orange-style Laplace smoothed categorical over dims i',
    collapsing over emitted v for that bucket).

    Each matching training arc contributes one scaled copy of \\hat{\\mathbf{p}}(\\cdot|\\, y, \\mathrm{rel})
    computed from counts ``emit``.

    Rows / columns span attribute indices ``0 .. I-1`` (nine rubric dims in LLM Rubric bundles).
    """
    if pre is None or not train_examples:
        return None
    _, emit, I, C, _R = pre
    M = np.zeros((I, I), dtype=np.float64)
    n_edges = 0
    for ex in train_examples:
        y = int(ex.y)
        if not (0 <= y < C):
            continue
        sl = emit[y]
        rel_mat = sl[rel_bucket]
        n_ip = rel_mat.sum(axis=1).astype(np.float64, copy=False)
        rel_total = float(n_ip.sum())
        if rel_total <= 0.0:
            p_vec = np.full(I, 1.0 / float(I), dtype=np.float64)
        else:
            p_vec = (n_ip + alpha_emit) / (rel_total + alpha_emit * float(I))

        for (i_s, j_s, k_s, v_s) in ex.sources:
            if relation_label(i_s, j_s, k_s, ex.target_i, ex.target_j, ex.target_k) != rel_bucket:
                continue
            if not (0 <= i_s < I):
                continue
            M[int(i_s), :] += p_vec
            n_edges += 1
    if n_edges == 0:
        return None
    return M, n_edges


def _gold_relation_idx(si: bool, sj: bool, sk: bool) -> int:
    """
    Encode (1[i_s==i_t], 1[j_s==j_t], 1[k_s==k_t]) into 0..6, excluding the impossible
    all-true case (source would equal target cell).
    Order: 000,001,010,011,100,101,110.
    """
    b = (1 if si else 0) + 2 * (1 if sj else 0) + 4 * (1 if sk else 0)
    if b == 7:
        return -1
    return b


def _nb_gold_preprocess(
    train_examples,
    bundle: dict,
    *,
    ijk_transductive: bool,
):
    """
    Accumulate multinomial counts for NB (gold). Call once per bundle; reuse for many λ.

    λ (``alpha``) is applied only in ``_nb_gold_eval_slices`` so sweeps avoid refitting counts.
    """
    if not train_examples:
        return None
    rows = ratings_for_ijk_fit(bundle, transductive=ijk_transductive)
    if not rows:
        return None
    c = max(int(r["value"]) for r in rows)
    max_i = max(int(r["attribute"]) for r in rows)
    max_j = max(int(r["annotator"]) for r in rows)
    max_k = max(int(r["item"]) for r in rows)
    class_counts = np.zeros(c, dtype=np.float64)
    i_counts = np.zeros((c, max_i), dtype=np.float64)
    j_counts = np.zeros((c, max_j), dtype=np.float64)
    k_counts = np.zeros((c, max_k), dtype=np.float64)
    seen_j: set[int] = set()
    seen_k: set[int] = set()
    for r in rows:
        y = int(r["value"]) - 1
        ii = int(r["attribute"]) - 1
        jj = int(r["annotator"]) - 1
        kk = int(r["item"]) - 1
        class_counts[y] += 1.0
        i_counts[y, ii] += 1.0
        j_counts[y, jj] += 1.0
        k_counts[y, kk] += 1.0
        seen_j.add(jj)
        seen_k.add(kk)

    R = 7
    emit = np.zeros((c, R, c), dtype=np.float64)
    for ex in train_examples:
        y = ex.y
        for (i_s, j_s, k_s, v_s) in ex.sources:
            if not (0 <= v_s < c):
                continue
            rel = _gold_relation_idx(i_s == ex.target_i, j_s == ex.target_j, k_s == ex.target_k)
            if rel < 0:
                continue
            emit[y, rel, v_s] += 1.0

    return (
        class_counts,
        i_counts,
        j_counts,
        k_counts,
        seen_j,
        seen_k,
        emit,
        c,
        max_i,
        max_j,
        max_k,
    )


def _nb_gold_eval_slices(
    pre,
    eval_examples,
    *,
    smoothing_lambda: float,
) -> tuple[float, float] | None:
    """Evaluate NB (gold) with add-λ smoothing on all factors (prior, marginals per y, emissions)."""
    (
        class_counts,
        i_counts,
        j_counts,
        k_counts,
        seen_j,
        seen_k,
        emit,
        c,
        max_i,
        max_j,
        max_k,
    ) = pre
    if smoothing_lambda <= 0.0 or not eval_examples:
        return None

    lam = float(smoothing_lambda)
    alpha = lam
    n_class = float(class_counts.sum()) + alpha * c
    total_nll = 0.0
    sum_sq = 0.0
    classes = np.arange(1, c + 1, dtype=np.float64)

    for ex in eval_examples:
        it, jt, kt, y_true = ex.target_i, ex.target_j, ex.target_k, ex.y
        j_ok = jt in seen_j
        k_ok = kt in seen_k
        scores = np.zeros(c, dtype=np.float64)
        for y in range(c):
            s = math.log((class_counts[y] + alpha) / n_class)
            denom_i = class_counts[y] + alpha * max_i
            s += math.log((i_counts[y, it] + alpha) / denom_i)
            if j_ok:
                denom_j = class_counts[y] + alpha * max_j
                s += math.log((j_counts[y, jt] + alpha) / denom_j)
            if k_ok:
                denom_k = class_counts[y] + alpha * max_k
                s += math.log((k_counts[y, kt] + alpha) / denom_k)
            for (i_s, j_s, k_s, v_s) in ex.sources:
                if not (0 <= v_s < c):
                    continue
                rel = _gold_relation_idx(i_s == it, j_s == jt, k_s == kt)
                if rel < 0:
                    continue
                sl = emit[y, rel]
                denom_e = float(sl.sum()) + alpha * c
                s += math.log((float(sl[v_s]) + alpha) / denom_e)
            scores[y] = s
        m = float(scores.max())
        log_norm = m + math.log(float(np.sum(np.exp(scores - m))))
        log_p_true = scores[y_true] - log_norm
        total_nll -= log_p_true
        exp_y = float(np.dot(np.exp(scores - log_norm), classes))
        sum_sq += (exp_y - (y_true + 1.0)) ** 2

    n = len(eval_examples)
    return total_nll / n, math.sqrt(sum_sq / n)


def _nb_gold_augment_preprocess(
    train_examples,
    bundle: dict,
    *,
    ijk_transductive: bool,
):
    """
    NB (gold-augment): marginals identical to NB (gold); per-source naive product over
    P(v|y) plus optional edge factors P(v|y,same-i), P(v|y,same-j), P(v|y,same-k).
    Fit counts on LOO train_examples.
    """
    if not train_examples:
        return None
    rows = ratings_for_ijk_fit(bundle, transductive=ijk_transductive)
    if not rows:
        return None
    c = max(int(r["value"]) for r in rows)
    max_i = max(int(r["attribute"]) for r in rows)
    max_j = max(int(r["annotator"]) for r in rows)
    max_k = max(int(r["item"]) for r in rows)
    class_counts = np.zeros(c, dtype=np.float64)
    i_counts = np.zeros((c, max_i), dtype=np.float64)
    j_counts = np.zeros((c, max_j), dtype=np.float64)
    k_counts = np.zeros((c, max_k), dtype=np.float64)
    seen_j: set[int] = set()
    seen_k: set[int] = set()
    for r in rows:
        y = int(r["value"]) - 1
        ii = int(r["attribute"]) - 1
        jj = int(r["annotator"]) - 1
        kk = int(r["item"]) - 1
        class_counts[y] += 1.0
        i_counts[y, ii] += 1.0
        j_counts[y, jj] += 1.0
        k_counts[y, kk] += 1.0
        seen_j.add(jj)
        seen_k.add(kk)

    emit_base = np.zeros((c, c), dtype=np.float64)
    emit_si = np.zeros((c, c), dtype=np.float64)
    emit_sj = np.zeros((c, c), dtype=np.float64)
    emit_sk = np.zeros((c, c), dtype=np.float64)
    for ex in train_examples:
        y = ex.y
        for (i_s, j_s, k_s, v_s) in ex.sources:
            if not (0 <= v_s < c):
                continue
            emit_base[y, v_s] += 1.0
            if i_s == ex.target_i:
                emit_si[y, v_s] += 1.0
            if j_s == ex.target_j:
                emit_sj[y, v_s] += 1.0
            if k_s == ex.target_k:
                emit_sk[y, v_s] += 1.0

    return (
        class_counts,
        i_counts,
        j_counts,
        k_counts,
        seen_j,
        seen_k,
        emit_base,
        emit_si,
        emit_sj,
        emit_sk,
        c,
        max_i,
        max_j,
        max_k,
    )


def _nb_gold_augment_eval_slices(
    pre,
    eval_examples,
    *,
    smoothing_lambda: float,
) -> tuple[float, float] | None:
    """
    Score augment model: ∏_sources [ P(v|y) · (P(v|y,si=i) if si=i) · (P(v|y,sj=j) ...) · ... ]
    Each factor Laplace-smoothed with same λ over C ratings.
    """
    (
        class_counts,
        i_counts,
        j_counts,
        k_counts,
        seen_j,
        seen_k,
        emit_base,
        emit_si,
        emit_sj,
        emit_sk,
        c,
        max_i,
        max_j,
        max_k,
    ) = pre
    if smoothing_lambda <= 0.0 or not eval_examples:
        return None
    lam = float(smoothing_lambda)
    alpha = lam
    n_class = float(class_counts.sum()) + alpha * c
    total_nll = 0.0
    sum_sq = 0.0
    classes = np.arange(1, c + 1, dtype=np.float64)

    def _log_p_table(row: np.ndarray, v: int) -> float:
        denom = float(row.sum()) + alpha * c
        return math.log((float(row[v]) + alpha) / denom)

    for ex in eval_examples:
        it, jt, kt, y_true = ex.target_i, ex.target_j, ex.target_k, ex.y
        j_ok = jt in seen_j
        k_ok = kt in seen_k
        scores = np.zeros(c, dtype=np.float64)
        for y in range(c):
            s = math.log((class_counts[y] + alpha) / n_class)
            denom_i = class_counts[y] + alpha * max_i
            s += math.log((i_counts[y, it] + alpha) / denom_i)
            if j_ok:
                denom_j = class_counts[y] + alpha * max_j
                s += math.log((j_counts[y, jt] + alpha) / denom_j)
            if k_ok:
                denom_k = class_counts[y] + alpha * max_k
                s += math.log((k_counts[y, kt] + alpha) / denom_k)
            for (i_s, j_s, k_s, v_s) in ex.sources:
                if not (0 <= v_s < c):
                    continue
                s += _log_p_table(emit_base[y], v_s)
                if i_s == it:
                    s += _log_p_table(emit_si[y], v_s)
                if j_s == jt:
                    s += _log_p_table(emit_sj[y], v_s)
                if k_s == kt:
                    s += _log_p_table(emit_sk[y], v_s)
            scores[y] = s
        m = float(scores.max())
        log_norm = m + math.log(float(np.sum(np.exp(scores - m))))
        log_p_true = scores[y_true] - log_norm
        total_nll -= log_p_true
        exp_y = float(np.dot(np.exp(scores - log_norm), classes))
        sum_sq += (exp_y - (y_true + 1.0)) ** 2

    n = len(eval_examples)
    return total_nll / n, math.sqrt(sum_sq / n)


def _nb_gold_eval(
    train_examples,
    eval_examples,
    bundle: dict,
    *,
    ijk_transductive: bool,
    alpha: float = 1.0,
) -> tuple[float, float] | None:
    """
    Gold NB (advisor spec):

        P(y | i,j,k, obs) ∝ P(y) P(i|y) P(j|y) P(k|y)
                            ∏_{sources r} P(v_r | y, 1[i_r=i], 1[j_r=j], 1[k_r=k])

    - Add-λ (Laplacian) smoothing on every factor; ``alpha`` is λ (historical name in this codebase).
    - Drop P(j|y) factor if target annotator j never appears in the IJK fit pool.
    - Drop P(k|y) factor if target item k never appears in the IJK fit pool.
    - Skip a source term if its value is unobserved (no such cell in our adapter).

    Marginals use the same flat pool as ``NaiveBayesIJK`` (transductive flag).
    Emission counts are fit on LOO ``train_examples`` only.
    """
    if not train_examples or not eval_examples:
        return None
    pre = _nb_gold_preprocess(train_examples, bundle, ijk_transductive=ijk_transductive)
    if pre is None:
        return None
    return _nb_gold_eval_slices(pre, eval_examples, smoothing_lambda=alpha)


def _structured_nb_true_notype_log_loss(
    train_examples,
    eval_examples,
    num_classes: int,
    *,
    alpha_prior: float = 1.0,
    alpha_emit: float = 1.0,
) -> float | None:
    r"""
    Purple (plot): p(y_{ijk} | sources) ∝ p(y_{ijk}) · ∏_{i'j'k'≠ijk}
        p(y_{i'j'k'} | y_{ijk}, 1[i=i'], 1[j=j'], 1[k=k']).

    No explicit i' in the emission; rel is the 7-way (i,j,k) equality pattern (same cell excluded).
    Multinomial over source rating v given (y, rel).
    """
    if not train_examples or not eval_examples:
        return None
    C = num_classes
    R = 7
    prior = np.zeros((C,), dtype=np.float64)
    emit = np.zeros((C, R, C), dtype=np.float64)

    for ex in train_examples:
        y = ex.y
        prior[y] += 1.0
        for (i_s, j_s, k_s, v_s) in ex.sources:
            rel = _gold_relation_idx(i_s == ex.target_i, j_s == ex.target_j, k_s == ex.target_k)
            if rel < 0:
                continue
            emit[y, rel, v_s] += 1.0

    total = 0.0
    for ex in eval_examples:
        y_true = ex.y
        prior_denom = float(prior.sum()) + alpha_prior * C
        scores = np.zeros(C, dtype=np.float64)
        for y in range(C):
            s = math.log((prior[y] + alpha_prior) / prior_denom)
            sl = emit[y]
            for (i_s, j_s, k_s, v_s) in ex.sources:
                rel = _gold_relation_idx(i_s == ex.target_i, j_s == ex.target_j, k_s == ex.target_k)
                if rel < 0:
                    continue
                col = sl[rel]
                denom_v = float(col.sum()) + alpha_emit * C
                s += math.log((float(col[v_s]) + alpha_emit) / denom_v)
            scores[y] = s
        m = float(scores.max())
        log_norm = m + math.log(float(np.sum(np.exp(scores - m))))
        total += -(scores[y_true] - log_norm)
    return total / len(eval_examples)


def _load_marformer_log_loss(size: int) -> float | None:
    """
    Try common Marformer result layouts and return test missing log-loss if found.
    """
    candidates = [
        Path("RESULTS/MARFORMER/LLM_RUBRIC") / f"LLMRubric_225_25_9_{size}" / "TEST_RESULTS",
        Path("RESULTS/MARFORMER_HARD_MASK/LLM_RUBRIC") / f"LLMRubric_225_25_9_{size}" / "TEST_RESULTS",
        Path("RESULTS/MARFORMER") / f"LLMRubric_225_25_9_{size}" / "TEST_RESULTS",
    ]
    for run_dir in candidates:
        if not run_dir.exists():
            continue
        jsons = sorted(run_dir.glob("best-*.json"))
        if not jsons:
            jsons = sorted(run_dir.glob("*.json"))
        for jp in jsons:
            try:
                d = _read_json(jp)
            except Exception:
                continue
            miss = d.get("missing", {})
            if isinstance(miss, dict):
                if "log_loss" in miss:
                    return float(miss["log_loss"])
                rating = miss.get("rating", {})
                if isinstance(rating, dict) and "xent" in rating:
                    return float(rating["xent"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot CPM + structured baselines vs train size (LLM Rubric)")
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path("RESULTS/STAN/LLM_RUBRIC/CPM_SHARED_THRESHOLD"),
        help="Directory containing LLMRubric_225_25_9_*_eval folders",
    )
    ap.add_argument("--data-root", type=Path, default=Path("DATA/STAN/LLM_RUBRIC"))
    ap.add_argument(
        "--output-logloss",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_cpm_structured_baselines_log_loss.png"),
    )
    ap.add_argument(
        "--output-rmse",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_cpm_structured_baselines_rmse.png"),
    )
    ap.add_argument(
        "--output-logloss-exp",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_cpm_structured_baselines_log_loss_logreg_exp.png"),
        help="Experimental output path (used when --experimental-logreg is set).",
    )
    ap.add_argument(
        "--output-rmse-exp",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_cpm_structured_baselines_rmse_logreg_exp.png"),
        help="Experimental RMSE output path (used when --experimental-logreg is set).",
    )
    ap.add_argument(
        "--train-instances",
        default="train,val",
        help="Comma-separated instances for LOO structured training (default: train,val)",
    )
    ap.add_argument("--ll-epochs", type=int, default=35)
    ap.add_argument("--ll-lr", type=float, default=0.05)
    ap.add_argument("--ll-batch", type=int, default=256)
    ap.add_argument("--device", default=None, help="cpu / cuda / cuda:0 for log-linear")
    ap.add_argument(
        "--no-ijk-transductive",
        action="store_true",
        help="Fit IJK NB without test-observed in the count pool",
    )
    ap.add_argument("--skip-log-linear", action="store_true", help="Faster plot without log-linear curve")
    ap.add_argument(
        "--no-ll-tqdm",
        action="store_true",
        help="Disable tqdm progress bar during log-linear training",
    )
    ap.add_argument(
        "--ll-tqdm-batches",
        action="store_true",
        help="Also show a per-epoch batch tqdm (noisy but finer-grained)",
    )
    ug = ap.add_mutually_exclusive_group()
    ug.add_argument(
        "--unigram-nontransductive",
        dest="unigram_transductive",
        action="store_false",
        help="Fit unigram baselines on train+val observed only (default).",
    )
    ug.add_argument(
        "--unigram-transductive",
        dest="unigram_transductive",
        action="store_true",
        help="Fit unigram baselines on train+val+test observed.",
    )
    ap.set_defaults(unigram_transductive=False)
    ap.add_argument(
        "--experimental-logreg",
        action="store_true",
        help="Add experimental multinomial logistic-regression baseline and save to *-exp outputs.",
    )
    ap.add_argument("--logreg-epochs", type=int, default=120)
    ap.add_argument("--logreg-lr", type=float, default=0.08)
    ap.add_argument("--logreg-batch", type=int, default=256)
    ap.add_argument(
        "--no-nb-baselines",
        action="store_true",
        help="Do not draw any NB baselines (keeps CPM/unigram/log-linear only).",
    )
    ap.add_argument(
        "--emphasize-nb",
        action="store_true",
        help="Fade non-NB curves and emphasize NB baselines (IJK/T, gold + structured NBs).",
    )
    ap.add_argument(
        "--no-nb-gold-augment",
        action="store_true",
        help="Hide NB (gold-augment) from main and sweep plots.",
    )
    ap.add_argument(
        "--nb-gold-lambda",
        type=float,
        default=1.0,
        help="Add-λ (Laplace) for NB (gold) and NB (gold-augment) in the main figure (λ>0, default 1).",
    )
    ap.add_argument(
        "--nb-gold-lambda-sweep",
        type=str,
        default="",
        help="Comma-separated λ for an extra plot: NB (gold) only vs CPM (empty skips).",
    )
    ap.add_argument(
        "--nb-gold-augment-lambda-sweep",
        type=str,
        default="",
        help="Comma-separated λ for an extra plot: NB (gold-augment) only vs CPM (empty skips).",
    )
    ap.add_argument(
        "--output-nb-gold-sweep-logloss",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_nb_gold_lambda_sweep_log_loss.png"),
    )
    ap.add_argument(
        "--output-nb-gold-sweep-rmse",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_nb_gold_lambda_sweep_rmse.png"),
    )
    ap.add_argument(
        "--output-nb-gold-augment-sweep-logloss",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_nb_gold_augment_lambda_sweep_log_loss.png"),
    )
    ap.add_argument(
        "--output-nb-gold-augment-sweep-rmse",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_nb_gold_augment_lambda_sweep_rmse.png"),
    )
    ap.add_argument(
        "--orange-diagnostics",
        action="store_true",
        help="Write extra diagnostic plots isolating P(i'|y,rel) contribution bucket-by-bucket.",
    )
    ap.add_argument(
        "--orange-diagnostics-seed",
        type=int,
        default=7,
        help="RNG seed for per-bucket shuffle diagnostic.",
    )
    ap.add_argument(
        "--output-orange-ablation",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_orange_rel_ablation_log_loss_delta.png"),
    )
    ap.add_argument(
        "--output-orange-ablation-marginal",
        type=Path,
        default=Path(
            "PLOTS/TALK/LLM_RUBRIC/llm_rubric_orange_rel_ablation_marginal_pi_y_log_loss_delta.png"
        ),
    )
    ap.add_argument(
        "--output-orange-shuffle",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_orange_rel_shuffle_log_loss_delta.png"),
    )
    ap.add_argument(
        "--output-orange-directstats",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_orange_rel_direct_stats.png"),
    )
    ap.add_argument(
        "--output-orange-pi-confusion",
        type=Path,
        default=Path(
            "PLOTS/TALK/LLM_RUBRIC/"
            "llm_rubric_orange_pi_soft_confusion_same_item_same_attr_diff_annot.png",
        ),
    )
    ap.add_argument(
        "--orange-pi-confusion-rel",
        type=int,
        choices=list(range(7)),
        default=int(RelationKind.SAME_ITEM_SAME_ATTR_DIFF_ANNOT),
        help=(
            "RelationKind code (default: SAME_ITEM_SAME_ATTR_DIFF_ANNOT=1) "
            "for P(i|y,rel) soft confusion under --orange-diagnostics."
        ),
    )
    args = ap.parse_args()
    if args.nb_gold_lambda <= 0.0:
        raise SystemExit("--nb-gold-lambda must be positive.")

    train_inst = {s.strip() for s in args.train_instances.split(",") if s.strip()}

    try:
        nb_gold_sweep_lambdas = _parse_lambda_sweep(args.nb_gold_lambda_sweep)
        nb_gold_augment_sweep_lambdas = _parse_lambda_sweep(args.nb_gold_augment_lambda_sweep)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    nb_gold_sweep_ll: dict[float, list[tuple[int, float]]] = {lam: [] for lam in nb_gold_sweep_lambdas}
    nb_gold_sweep_rmse: dict[float, list[tuple[int, float]]] = {lam: [] for lam in nb_gold_sweep_lambdas}
    nb_gold_augment_sweep_ll: dict[float, list[tuple[int, float]]] = {
        lam: [] for lam in nb_gold_augment_sweep_lambdas
    }
    nb_gold_augment_sweep_rmse: dict[float, list[tuple[int, float]]] = {
        lam: [] for lam in nb_gold_augment_sweep_lambdas
    }

    cpm_ll: list[tuple[int, float]] = []
    marformer_ll: list[tuple[int, float]] = []
    ijk_ll: list[tuple[int, float]] = []
    snb_ijk_ll: list[tuple[int, float]] = []
    snb_notype_ll: list[tuple[int, float]] = []
    snb_true_notype_ll: list[tuple[int, float]] = []
    snb_i_ll: list[tuple[int, float]] = []
    nb_gold_ll: list[tuple[int, float]] = []
    nb_gold_augment_ll: list[tuple[int, float]] = []
    ll_ll: list[tuple[int, float]] = []
    logreg_ll: list[tuple[int, float]] = []
    unigram_keys = ["none", "i", "ij"]
    unigram_ll: dict[str, list[tuple[int, float]]] = {k: [] for k in unigram_keys}
    cpm_rmse: list[tuple[int, float]] = []
    ijk_rmse: list[tuple[int, float]] = []
    nb_gold_rmse: list[tuple[int, float]] = []
    nb_gold_augment_rmse: list[tuple[int, float]] = []
    snb_rmse: list[tuple[int, float]] = []
    ll_rmse: list[tuple[int, float]] = []
    logreg_rmse: list[tuple[int, float]] = []
    orange_ablation_delta: dict[int, list[tuple[int, float]]] = {r: [] for r in range(7)}
    orange_ablation_marginal_delta: dict[int, list[tuple[int, float]]] = {r: [] for r in range(7)}
    orange_shuffle_delta: dict[int, list[tuple[int, float]]] = {r: [] for r in range(7)}
    orange_pi_conf_store: tuple[int, tuple, list] | None = None
    orange_mi: dict[int, list[tuple[int, float]]] = {r: [] for r in range(7)}
    orange_js: dict[int, list[tuple[int, float]]] = {r: [] for r in range(7)}

    for metrics_path in sorted(args.results_root.glob("LLMRubric_225_25_9_*_eval/predictive_metrics.json")):
        size = _extract_size(metrics_path.parent.name)
        if size is None:
            continue
        size_stan_plot = size
        size_baseline_plot = size
        metrics = _read_json(metrics_path)
        rll = metrics.get("rating_missing_log_likelihood")
        if rll is not None:
            cpm_ll.append((size_stan_plot, float(-rll)))
        mf = _load_marformer_log_loss(size)
        if mf is not None:
            marformer_ll.append((size_stan_plot, mf))

        bundle_path = args.data_root / f"LLMRubric_225_25_9_{size}" / "data_bundle.json"
        if not bundle_path.exists():
            print(f"[skip baselines] no bundle for size {size}: {bundle_path}")
            r = _cpm_rmse(args.data_root, size, metrics_path.parent)
            if r is not None:
                cpm_rmse.append((size, r))
            continue

        print(f"size={size}  fitting baselines…")
        bundle = load_bundle_dict(bundle_path)
        I, _J, C = bundle_dims(bundle, bundle_path)
        K = max(
            int(r["item"]) for r in (bundle.get("observed_ratings", []) + bundle.get("missing_ratings", []))
        )
        train_ex = build_training_examples(bundle, instances=train_inst)
        test_ex = build_test_examples(bundle)

        for key in unigram_keys:
            u = _unigram_subset_log_loss(
                bundle,
                key,
                transductive=args.unigram_transductive,
                alpha=1.0,
            )
            if u is not None:
                unigram_ll[key].append((size_baseline_plot, float(u)))

        nb_ijk = NaiveBayesIJK.fit_from_bundle(bundle, transductive=not args.no_ijk_transductive)
        ev = nb_ijk.evaluate(test_ex)
        ijk_ll.append((size_baseline_plot, ev["mean_nll"]))
        ijk_rmse.append((size_baseline_plot, _rmse_from_proba(test_ex, nb_ijk.predict_proba(test_ex)) or float("nan")))

        pre_nb_gold = _nb_gold_preprocess(
            train_ex,
            bundle,
            ijk_transductive=not args.no_ijk_transductive,
        )
        if pre_nb_gold is not None and test_ex:
            gl = _nb_gold_eval_slices(pre_nb_gold, test_ex, smoothing_lambda=float(args.nb_gold_lambda))
            if gl is not None:
                nb_gold_ll.append((size_baseline_plot, float(gl[0])))
                nb_gold_rmse.append((size_baseline_plot, float(gl[1])))
            for lam in nb_gold_sweep_lambdas:
                gsl = _nb_gold_eval_slices(pre_nb_gold, test_ex, smoothing_lambda=lam)
                if gsl is not None:
                    nb_gold_sweep_ll[lam].append((size_baseline_plot, float(gsl[0])))
                    nb_gold_sweep_rmse[lam].append((size_baseline_plot, float(gsl[1])))

        pre_aug = _nb_gold_augment_preprocess(
            train_ex,
            bundle,
            ijk_transductive=not args.no_ijk_transductive,
        )
        if pre_aug is not None and test_ex:
            ga = _nb_gold_augment_eval_slices(pre_aug, test_ex, smoothing_lambda=float(args.nb_gold_lambda))
            if ga is not None:
                nb_gold_augment_ll.append((size_baseline_plot, float(ga[0])))
                nb_gold_augment_rmse.append((size_baseline_plot, float(ga[1])))
            for lam_a in nb_gold_augment_sweep_lambdas:
                gsa = _nb_gold_augment_eval_slices(pre_aug, test_ex, smoothing_lambda=lam_a)
                if gsa is not None:
                    nb_gold_augment_sweep_ll[lam_a].append((size_baseline_plot, float(gsa[0])))
                    nb_gold_augment_sweep_rmse[lam_a].append((size_baseline_plot, float(gsa[1])))

        snb = StructuredNaiveBayes.fit(
            train_ex,
            num_attrs=I,
            num_classes=C,
            num_anns=_J,
            num_items=K,
        )
        evs = snb.evaluate(test_ex)
        # Keep RMSE computation for optional diagnostics, but do not draw this curve on log-loss plot.
        snb_ijk_ll.append((size_baseline_plot, evs["mean_nll"]))
        snb_rmse.append((size_baseline_plot, _rmse_from_proba(test_ex, snb.predict_proba(test_ex)) or float("nan")))
        i_only = _structured_nb_i_only_log_loss(
            train_ex,
            test_ex,
            num_attrs=I,
            num_classes=C,
            alpha_prior=1.0,
            alpha_emit=1.0,
        )
        if i_only is not None:
            snb_i_ll.append((size_baseline_plot, float(i_only)))
        notype = _structured_nb_notype_log_loss(
            train_ex,
            test_ex,
            num_attrs=I,
            num_classes=C,
            alpha_prior=1.0,
            alpha_emit=1.0,
        )
        if notype is not None:
            snb_notype_ll.append((size_baseline_plot, float(notype)))
        if args.orange_diagnostics:
            pre_orange = _orange_factorized_preprocess(train_ex, num_attrs=I, num_classes=C)
            base_orange = _orange_factorized_eval(
                pre_orange,
                test_ex,
                alpha_prior=1.0,
                alpha_emit=1.0,
            )
            if base_orange is not None:
                for rel in range(7):
                    abl = _orange_factorized_eval(
                        pre_orange,
                        test_ex,
                        alpha_prior=1.0,
                        alpha_emit=1.0,
                        drop_pi_rel={rel},
                    )
                    if abl is not None:
                        orange_ablation_delta[rel].append((size_baseline_plot, float(abl - base_orange)))
                    abl_m = _orange_factorized_eval(
                        pre_orange,
                        test_ex,
                        alpha_prior=1.0,
                        alpha_emit=1.0,
                        marginal_pi_rel={rel},
                    )
                    if abl_m is not None:
                        orange_ablation_marginal_delta[rel].append(
                            (size_baseline_plot, float(abl_m - base_orange))
                        )
                    shf = _orange_factorized_eval(
                        pre_orange,
                        test_ex,
                        alpha_prior=1.0,
                        alpha_emit=1.0,
                        shuffle_rel=rel,
                        rng=np.random.default_rng(args.orange_diagnostics_seed + 1000 * size + rel),
                    )
                    if shf is not None:
                        orange_shuffle_delta[rel].append((size_baseline_plot, float(shf - base_orange)))
                mi_vals, js_vals = _orange_direct_stats(pre_orange)
                if mi_vals and js_vals:
                    for rel in range(7):
                        if rel < len(mi_vals) and not np.isnan(mi_vals[rel]):
                            orange_mi[rel].append((size_baseline_plot, float(mi_vals[rel])))
                        if rel < len(js_vals) and not np.isnan(js_vals[rel]):
                            orange_js[rel].append((size_baseline_plot, float(js_vals[rel])))
                if orange_pi_conf_store is None or size > orange_pi_conf_store[0]:
                    orange_pi_conf_store = (size, pre_orange, train_ex)
        true_notype = _structured_nb_true_notype_log_loss(
            train_ex,
            test_ex,
            num_classes=C,
            alpha_prior=1.0,
            alpha_emit=1.0,
        )
        if true_notype is not None:
            snb_true_notype_ll.append((size_baseline_plot, float(true_notype)))

        if not args.skip_log_linear:
            ll = StructuredLogLinear.fit(
                train_ex,
                num_attrs=I,
                num_classes=C,
                epochs=args.ll_epochs,
                lr=args.ll_lr,
                batch_size=args.ll_batch,
                device=args.device,
                verbose=False,
                show_progress=not args.no_ll_tqdm,
                tqdm_batches=args.ll_tqdm_batches,
                tqdm_desc=f"Log-linear | train_items={size}",
            )
            evl = ll.evaluate(test_ex)
            ll_ll.append((size_baseline_plot, evl["mean_nll"]))
            ll_rmse.append((size_baseline_plot, _rmse_from_proba(test_ex, ll.predict_proba(test_ex)) or float("nan")))
        if args.experimental_logreg:
            lg = _fit_logreg_torch(
                train_ex,
                test_ex,
                I=I,
                J=_J,
                K=K,
                C=C,
                epochs=args.logreg_epochs,
                lr=args.logreg_lr,
                batch_size=args.logreg_batch,
                device=args.device,
            )
            if lg is not None:
                logreg_ll.append((size_baseline_plot, float(lg[0])))
                logreg_rmse.append((size_baseline_plot, float(lg[1])))

        r = _cpm_rmse(args.data_root, size, metrics_path.parent)
        if r is not None:
            cpm_rmse.append((size_stan_plot, r))

    if not cpm_ll:
        raise SystemExit(f"No CPM predictive_metrics.json under {args.results_root}")

    def _sort(pts: list[tuple[int, float]]) -> list[tuple[int, float]]:
        return sorted(pts, key=lambda x: x[0])

    cpm_ll = _sort(cpm_ll)
    xs = [p[0] for p in cpm_ll]
    plt.figure(figsize=(9.0, 5.4))
    bg_alpha = 0.25 if args.emphasize_nb else 1.0
    bg_lw = 1.4 if args.emphasize_nb else 2.2
    nb_alpha = 1.0
    nb_lw = 2.6 if args.emphasize_nb else 2.0

    plt.plot(
        xs,
        [p[1] for p in cpm_ll],
        marker="o",
        color="#1b9e77",
        linewidth=bg_lw,
        alpha=bg_alpha,
        label="CPM SharedThreshold STAN",
    )
    if marformer_ll:
        marformer_ll = _sort(marformer_ll)
        plt.plot(
            [p[0] for p in marformer_ll],
            [p[1] for p in marformer_ll],
            marker="o",
            linestyle="-",
            color="#1f6fba",
            linewidth=bg_lw,
            alpha=bg_alpha,
            label="Marformer",
        )
    if ijk_ll and not args.no_nb_baselines:
        ijk_ll = _sort(ijk_ll)
        plt.plot(
            [p[0] for p in ijk_ll],
            [p[1] for p in ijk_ll],
            marker="*",
            linestyle="-.",
            color="#111111",
            linewidth=nb_lw,
            alpha=nb_alpha,
            label="Naive Bayes (i,j,k, T)",
        )
    if nb_gold_ll and not args.no_nb_baselines:
        nb_gold_ll = _sort(nb_gold_ll)
        plt.plot(
            [p[0] for p in nb_gold_ll],
            [p[1] for p in nb_gold_ll],
            marker="P",
            linestyle="-",
            color="#c9a227",
            linewidth=nb_lw,
            alpha=nb_alpha,
            label=f"NB (gold), λ={args.nb_gold_lambda:g}",
        )
    if nb_gold_augment_ll and not args.no_nb_baselines and not args.no_nb_gold_augment:
        nb_gold_augment_ll = _sort(nb_gold_augment_ll)
        plt.plot(
            [p[0] for p in nb_gold_augment_ll],
            [p[1] for p in nb_gold_augment_ll],
            marker="X",
            linestyle="-",
            color="#8b6914",
            linewidth=nb_lw,
            alpha=nb_alpha,
            label=f"NB (gold-augment), λ={args.nb_gold_lambda:g}",
        )
    unigram_style = {
        "none": ("#495057", "D"),
        "i": ("#2e8b57", "x"),
        "ij": ("#0b7285", ">"),
    }
    for key in unigram_keys:
        pts = unigram_ll[key]
        if not pts:
            continue
        pts = _sort(pts)
        color, marker = unigram_style[key]
        plt.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker=marker,
            linestyle=":",
            color=color,
            linewidth=1.3 if args.emphasize_nb else 1.6,
            alpha=bg_alpha,
            label=("Unigram (pool none)" if key == "none" else f"Unigram (pool {key})"),
        )
    if snb_i_ll and not args.no_nb_baselines:
        snb_i_ll = _sort(snb_i_ll)
        plt.plot(
            [p[0] for p in snb_i_ll],
            [p[1] for p in snb_i_ll],
            marker="s",
            linestyle="--",
            color="#f781bf",
            linewidth=nb_lw,
            alpha=nb_alpha,
            label="Structured NB (i-only)",
        )
    if snb_notype_ll and not args.no_nb_baselines:
        snb_notype_ll = _sort(snb_notype_ll)
        plt.plot(
            [p[0] for p in snb_notype_ll],
            [p[1] for p in snb_notype_ll],
            marker="d",
            linestyle="--",
            color="#ff7f0e",
            linewidth=nb_lw,
            alpha=nb_alpha,
            label="Structured NB (hybrid)",
        )
    if snb_true_notype_ll and not args.no_nb_baselines:
        snb_true_notype_ll = _sort(snb_true_notype_ll)
        plt.plot(
            [p[0] for p in snb_true_notype_ll],
            [p[1] for p in snb_true_notype_ll],
            marker="h",
            linestyle="--",
            color="#bc5090",
            linewidth=nb_lw,
            alpha=nb_alpha,
            label="Structured NB (true no type)",
        )
    if ll_ll:
        ll_ll = _sort(ll_ll)
        plt.plot(
            [p[0] for p in ll_ll],
            [p[1] for p in ll_ll],
            marker="^",
            linestyle="-",
            color="#7570b3",
            linewidth=1.3 if args.emphasize_nb else 2.0,
            alpha=bg_alpha,
            label="Structured log-linear",
        )
    if logreg_ll:
        logreg_ll = _sort(logreg_ll)
        plt.plot(
            [p[0] for p in logreg_ll],
            [p[1] for p in logreg_ll],
            marker="h",
            linestyle="-",
            color="#2a9d8f",
            linewidth=1.3 if args.emphasize_nb else 2.0,
            alpha=bg_alpha,
            label="Logistic Regression (exp.)",
        )

    plt.xlabel("Training items (+25 test item observed only LLM rating)")
    plt.ylabel("Test Missing Log Loss")
    plt.title("LLM Rubric: CPM vs structured baselines")
    plt.xticks(sorted(set(xs)))
    plt.grid(alpha=0.3)
    plt.legend()
    output_logloss = args.output_logloss_exp if args.experimental_logreg else args.output_logloss
    output_rmse = args.output_rmse_exp if args.experimental_logreg else args.output_rmse
    output_logloss.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_logloss, dpi=300)
    plt.close()
    print(f"Saved: {output_logloss}")

    if args.orange_diagnostics:
        rel_labels = list(RELATION_NAMES)
        rel_colors = mpl_cm.tab10(np.linspace(0.0, 0.9, 7))
        # Distinct linestyle + marker so overlaps stay readable when alpha < 1
        rel_linestyles = ("-", "--", "-.", ":", (0, (8, 3)), (0, (4, 2, 1, 2)), (0, (1.2, 2)))
        rel_markers = ("o", "s", "^", "v", "D", "X", "P")
        rel_alpha = 0.62

        def _marker_edge(rgb):
            rgb = np.asarray(rgb[:3], dtype=np.float64)
            return tuple(np.clip(rgb * 0.55 + 0.06, 0.0, 1.0))

        plt.figure(figsize=(9.2, 5.4))
        for rel in range(7):
            pts = _sort(orange_ablation_delta[rel])
            if not pts:
                continue
            plt.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                linestyle=rel_linestyles[rel],
                marker=rel_markers[rel],
                linewidth=2.1,
                color=rel_colors[rel],
                alpha=rel_alpha,
                markerfacecolor=rel_colors[rel],
                markeredgecolor=_marker_edge(rel_colors[rel]),
                markeredgewidth=0.6,
                label=rel_labels[rel],
            )
        plt.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
        plt.xlabel("Training items (+25 test item observed only LLM rating)")
        plt.ylabel("Δ log loss (drop P(i'|y,rel) for one rel)")
        plt.title("Orange Diagnostic 1: Leave-One-Relation-Out Ablation")
        plt.xticks(sorted(set(xs)))
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7, ncol=2)
        args.output_orange_ablation.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.output_orange_ablation, dpi=300)
        plt.close()
        print(f"Saved: {args.output_orange_ablation}")

        plt.figure(figsize=(9.2, 5.4))
        for rel in range(7):
            pts = _sort(orange_ablation_marginal_delta[rel])
            if not pts:
                continue
            plt.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                linestyle=rel_linestyles[rel],
                marker=rel_markers[rel],
                linewidth=2.1,
                color=rel_colors[rel],
                alpha=rel_alpha,
                markerfacecolor=rel_colors[rel],
                markeredgecolor=_marker_edge(rel_colors[rel]),
                markeredgewidth=0.6,
                label=rel_labels[rel],
            )
        plt.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
        plt.xlabel("Training items (+25 test item observed only LLM rating)")
        plt.ylabel(
            r"$\Delta$ log loss (subst.\ $P(i\mid y,\mathrm{rel})\to P(i\mid y)$ in one rel)",
        )
        plt.title(
            "Orange Diagnostic 2: Replace P(i'|y,rel) with marginal P(i'|y) per bucket",
        )
        plt.xticks(sorted(set(xs)))
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7, ncol=2)
        args.output_orange_ablation_marginal.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.output_orange_ablation_marginal, dpi=300)
        plt.close()
        print(f"Saved: {args.output_orange_ablation_marginal}")

        plt.figure(figsize=(9.2, 5.4))
        for rel in range(7):
            pts = _sort(orange_shuffle_delta[rel])
            if not pts:
                continue
            plt.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                linestyle=rel_linestyles[rel],
                marker=rel_markers[rel],
                linewidth=2.1,
                color=rel_colors[rel],
                alpha=rel_alpha,
                markerfacecolor=rel_colors[rel],
                markeredgecolor=_marker_edge(rel_colors[rel]),
                markeredgewidth=0.6,
                label=rel_labels[rel],
            )
        plt.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
        plt.xlabel("Training items (+25 test item observed only LLM rating)")
        plt.ylabel("Δ log loss (shuffle i' within one rel bucket)")
        plt.title("Orange Diagnostic 3: Per-Bucket Shuffle Test")
        plt.xticks(sorted(set(xs)))
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7, ncol=2)
        args.output_orange_shuffle.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.output_orange_shuffle, dpi=300)
        plt.close()
        print(f"Saved: {args.output_orange_shuffle}")

        fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True)
        for rel in range(7):
            pts_mi = _sort(orange_mi[rel])
            if pts_mi:
                ax[0].plot(
                    [p[0] for p in pts_mi],
                    [p[1] for p in pts_mi],
                    linestyle=rel_linestyles[rel],
                    marker=rel_markers[rel],
                    linewidth=1.95,
                    color=rel_colors[rel],
                    alpha=rel_alpha,
                    markerfacecolor=rel_colors[rel],
                    markeredgecolor=_marker_edge(rel_colors[rel]),
                    markeredgewidth=0.55,
                    label=rel_labels[rel],
                )
            pts_js = _sort(orange_js[rel])
            if pts_js:
                ax[1].plot(
                    [p[0] for p in pts_js],
                    [p[1] for p in pts_js],
                    linestyle=rel_linestyles[rel],
                    marker=rel_markers[rel],
                    linewidth=1.95,
                    color=rel_colors[rel],
                    alpha=rel_alpha,
                    markerfacecolor=rel_colors[rel],
                    markeredgecolor=_marker_edge(rel_colors[rel]),
                    markeredgewidth=0.55,
                    label=rel_labels[rel],
                )
        ax[0].set_title("I(i'; y | rel)")
        ax[1].set_title("Weighted pairwise JS of P(i'|y,rel)")
        ax[0].set_ylabel("Information")
        for a in ax:
            a.set_xlabel("Training items (+25 test item observed only LLM rating)")
            a.set_xticks(sorted(set(xs)))
            a.grid(alpha=0.3)
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7)
        fig.suptitle("Orange Diagnostic 4: Direct Statistics by Relation")
        args.output_orange_directstats.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(rect=(0, 0.08, 1, 0.96))
        plt.savefig(args.output_orange_directstats, dpi=300)
        plt.close()
        print(f"Saved: {args.output_orange_directstats}")

        if orange_pi_conf_store is not None:
            pi_sz, pi_pre, pi_train = orange_pi_conf_store
            rrel = args.orange_pi_confusion_rel
            rel_nm = RELATION_NAMES[rrel]
            pi_conf = _orange_pi_soft_confusion_matrix(
                pi_pre,
                pi_train,
                rrel,
                alpha_emit=1.0,
            )
            if pi_conf is None:
                print(
                    f"[orange pi confusion] skipped: no arcs for rel={rel_nm} ({rrel}) "
                    f"train_items≈{pi_sz}",
                    file=sys.stderr,
                )
            else:
                Mat, arc_n = pi_conf
                Ipi = Mat.shape[0]
                row_sums = Mat.sum(axis=1, keepdims=True)
                with np.errstate(invalid="ignore", divide="ignore"):
                    row_norm = np.divide(Mat, np.maximum(row_sums, 1e-15))
                row_norm = np.nan_to_num(row_norm, nan=0.0, posinf=0.0, neginf=0.0)
                vmax = float(np.max(Mat)) if Mat.size > 0 else 0.0

                fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.95))
                tick_lab = [str(ii) for ii in range(Ipi)]
                pcm0 = axes[0].imshow(row_norm, origin="upper", vmin=0.0, vmax=1.0, cmap="magma_r")
                axes[0].set_title(f"Rows normalized ({arc_n:g} arcs, rel={rel_nm})")
                axes[0].set_xlabel("Column: mass under $\\hat{p}(i'|\\, y, \\mathrm{rel})$ (pooled dims)")
                axes[0].set_ylabel("Row: observed source rubric dimension $i_{\\mathrm{src}}$")
                axes[0].set_xticks(np.arange(Ipi))
                axes[0].set_xticklabels(tick_lab, fontsize=9)
                axes[0].set_yticks(np.arange(Ipi))
                axes[0].set_yticklabels(tick_lab, fontsize=9)
                fig.colorbar(pcm0, ax=axes[0], shrink=0.82, fraction=0.046, label="row sums to 1")

                vmin_m = 0.0
                vmax_m = vmax if vmax > 0 else 1.0
                pcm1 = axes[1].imshow(Mat, origin="upper", vmin=vmin_m, vmax=vmax_m, cmap="inferno")
                axes[1].set_title("Aggregated soft mass (each arc adds one full $\\hat{\\mathbf{p}}$)")
                axes[1].set_xlabel("Mass to column j (same as left)")
                axes[1].set_ylabel("Observed $i_{\\mathrm{src}}$ index")
                axes[1].set_xticks(np.arange(Ipi))
                axes[1].set_xticklabels(tick_lab, fontsize=9)
                axes[1].set_yticks(np.arange(Ipi))
                axes[1].set_yticklabels(tick_lab, fontsize=9)
                fig.colorbar(pcm1, ax=axes[1], shrink=0.82, fraction=0.046, label="fractional Σ")

                fig.suptitle(
                    f"Largest train split ({pi_sz} items): pooled P(i|y,rel) soft confusion "
                    f"({rel_nm}); Laplace α_emit=1, collapsed over emitted rating v.",
                    fontsize=10,
                )
                args.output_orange_pi_confusion.parent.mkdir(parents=True, exist_ok=True)
                plt.tight_layout(rect=(0, 0.02, 1, 0.91))
                plt.savefig(args.output_orange_pi_confusion, dpi=300)
                plt.close()
                print(f"Saved: {args.output_orange_pi_confusion}")

    if nb_gold_sweep_lambdas:
        lam_sorted = sorted(nb_gold_sweep_lambdas)
        sweep_colors = mpl_cm.plasma(np.linspace(0.12, 0.92, len(lam_sorted)))
        sweep_has_data = any(nb_gold_sweep_ll.get(lam) for lam in lam_sorted)
        if sweep_has_data:
            plt.figure(figsize=(9.0, 5.4))
            plt.plot(
                [p[0] for p in cpm_ll],
                [p[1] for p in cpm_ll],
                marker="o",
                color="#1b9e77",
                linewidth=2.2,
                alpha=0.38,
                label="CPM SharedThreshold STAN",
            )
            for lam, color in zip(lam_sorted, sweep_colors):
                pts = nb_gold_sweep_ll.get(lam) or []
                if not pts:
                    continue
                pts = _sort(pts)
                plt.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker="P",
                    linestyle="-",
                    color=color,
                    linewidth=2.0,
                    label=f"NB (gold), λ={lam:g}",
                )
            plt.xlabel("Training items (+25 test item observed only LLM rating)")
            plt.ylabel("Test Missing Log Loss")
            plt.title("LLM Rubric: NB (gold) add-λ smoothing sweep")
            plt.xticks(sorted(set(xs)))
            plt.grid(alpha=0.3)
            plt.legend(ncol=2, fontsize=8)
            Path(args.output_nb_gold_sweep_logloss).parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(args.output_nb_gold_sweep_logloss, dpi=300)
            plt.close()
            print(f"Saved: {args.output_nb_gold_sweep_logloss}")

            if cpm_rmse and any(nb_gold_sweep_rmse.get(lam) for lam in lam_sorted):
                cpm_sorted = _sort(cpm_rmse)
                plt.figure(figsize=(9.0, 5.4))
                plt.plot(
                    [p[0] for p in cpm_sorted],
                    [p[1] for p in cpm_sorted],
                    marker="o",
                    color="#d55e00",
                    linewidth=2.2,
                    alpha=0.38,
                    label="CPM SharedThreshold STAN",
                )
                for lam, color in zip(lam_sorted, sweep_colors):
                    pts = nb_gold_sweep_rmse.get(lam) or []
                    if not pts:
                        continue
                    pts = _sort(pts)
                    plt.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        marker="P",
                        linestyle="-",
                        color=color,
                        linewidth=2.0,
                        label=f"NB (gold), λ={lam:g}",
                    )
                plt.xlabel("Training items (+25 test item observed only LLM rating)")
                plt.ylabel("Test Missing RMSE")
                plt.title("LLM Rubric: NB (gold) add-λ smoothing sweep")
                xt = sorted(
                    set(p[0] for lam in lam_sorted for p in (nb_gold_sweep_rmse.get(lam) or []))
                    | {p[0] for p in cpm_sorted}
                )
                plt.xticks(xt)
                plt.grid(alpha=0.3)
                plt.legend(ncol=2, fontsize=8)
                Path(args.output_nb_gold_sweep_rmse).parent.mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(args.output_nb_gold_sweep_rmse, dpi=300)
                plt.close()
                print(f"Saved: {args.output_nb_gold_sweep_rmse}")

    if nb_gold_augment_sweep_lambdas and not args.no_nb_gold_augment:
        lam_aug = sorted(nb_gold_augment_sweep_lambdas)
        aug_colors = mpl_cm.plasma(np.linspace(0.12, 0.92, len(lam_aug)))
        aug_has = any(nb_gold_augment_sweep_ll.get(lam) for lam in lam_aug)
        if aug_has:
            plt.figure(figsize=(9.0, 5.4))
            plt.plot(
                [p[0] for p in cpm_ll],
                [p[1] for p in cpm_ll],
                marker="o",
                color="#1b9e77",
                linewidth=2.2,
                alpha=0.38,
                label="CPM SharedThreshold STAN",
            )
            for lam, color in zip(lam_aug, aug_colors):
                pts = nb_gold_augment_sweep_ll.get(lam) or []
                if not pts:
                    continue
                pts = _sort(pts)
                plt.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker="X",
                    linestyle="-",
                    color=color,
                    linewidth=2.0,
                    label=f"NB (gold-augment), λ={lam:g}",
                )
            plt.xlabel("Training items (+25 test item observed only LLM rating)")
            plt.ylabel("Test Missing Log Loss")
            plt.title("LLM Rubric: NB (gold-augment) add-λ smoothing sweep")
            plt.xticks(sorted(set(xs)))
            plt.grid(alpha=0.3)
            plt.legend(ncol=2, fontsize=8)
            Path(args.output_nb_gold_augment_sweep_logloss).parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(args.output_nb_gold_augment_sweep_logloss, dpi=300)
            plt.close()
            print(f"Saved: {args.output_nb_gold_augment_sweep_logloss}")

            if cpm_rmse and any(nb_gold_augment_sweep_rmse.get(lam) for lam in lam_aug):
                cpm_sorted = _sort(cpm_rmse)
                plt.figure(figsize=(9.0, 5.4))
                plt.plot(
                    [p[0] for p in cpm_sorted],
                    [p[1] for p in cpm_sorted],
                    marker="o",
                    color="#d55e00",
                    linewidth=2.2,
                    alpha=0.38,
                    label="CPM SharedThreshold STAN",
                )
                for lam, color in zip(lam_aug, aug_colors):
                    pts = nb_gold_augment_sweep_rmse.get(lam) or []
                    if not pts:
                        continue
                    pts = _sort(pts)
                    plt.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        marker="X",
                        linestyle="-",
                        color=color,
                        linewidth=2.0,
                        label=f"NB (gold-augment), λ={lam:g}",
                    )
                plt.xlabel("Training items (+25 test item observed only LLM rating)")
                plt.ylabel("Test Missing RMSE")
                plt.title("LLM Rubric: NB (gold-augment) add-λ smoothing sweep")
                xt = sorted(
                    set(p[0] for lam in lam_aug for p in (nb_gold_augment_sweep_rmse.get(lam) or []))
                    | {p[0] for p in cpm_sorted}
                )
                plt.xticks(xt)
                plt.grid(alpha=0.3)
                plt.legend(ncol=2, fontsize=8)
                Path(args.output_nb_gold_augment_sweep_rmse).parent.mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(args.output_nb_gold_augment_sweep_rmse, dpi=300)
                plt.close()
                print(f"Saved: {args.output_nb_gold_augment_sweep_rmse}")

    if cpm_rmse and (
        ijk_rmse
        or nb_gold_rmse
        or (nb_gold_augment_rmse and not args.no_nb_gold_augment)
        or snb_rmse
        or ll_rmse
        or logreg_rmse
    ):
        plt.figure(figsize=(9.0, 5.4))
        cpm_rmse = _sort(cpm_rmse)
        plt.plot(
            [p[0] for p in cpm_rmse],
            [p[1] for p in cpm_rmse],
            marker="o",
            color="#d55e00",
            linewidth=bg_lw,
            alpha=bg_alpha,
            label="CPM SharedThreshold STAN",
        )
        if ijk_rmse and not args.no_nb_baselines:
            ijk_rmse = _sort(ijk_rmse)
            plt.plot(
                [p[0] for p in ijk_rmse],
                [p[1] for p in ijk_rmse],
                marker="*",
                linestyle="-.",
                color="#111111",
                linewidth=nb_lw,
                alpha=nb_alpha,
                label="Naive Bayes (i,j,k, T)",
            )
        if nb_gold_rmse and not args.no_nb_baselines:
            nb_gold_rmse = _sort(nb_gold_rmse)
            plt.plot(
                [p[0] for p in nb_gold_rmse],
                [p[1] for p in nb_gold_rmse],
                marker="P",
                linestyle="-",
                color="#c9a227",
                linewidth=nb_lw,
                alpha=nb_alpha,
                label=f"NB (gold), λ={args.nb_gold_lambda:g}",
            )
        if nb_gold_augment_rmse and not args.no_nb_baselines and not args.no_nb_gold_augment:
            nb_gold_augment_rmse = _sort(nb_gold_augment_rmse)
            plt.plot(
                [p[0] for p in nb_gold_augment_rmse],
                [p[1] for p in nb_gold_augment_rmse],
                marker="X",
                linestyle="-",
                color="#8b6914",
                linewidth=nb_lw,
                alpha=nb_alpha,
                label=f"NB (gold-augment), λ={args.nb_gold_lambda:g}",
            )
        if snb_rmse and not args.no_nb_baselines:
            snb_rmse = _sort(snb_rmse)
            plt.plot(
                [p[0] for p in snb_rmse],
                [p[1] for p in snb_rmse],
                marker="s",
                linestyle="--",
                color="#e7298a",
                linewidth=nb_lw,
                alpha=nb_alpha,
                label="Structured NB (i,j,k)",
            )
        if ll_rmse:
            ll_rmse = _sort(ll_rmse)
            plt.plot(
                [p[0] for p in ll_rmse],
                [p[1] for p in ll_rmse],
                marker="^",
                linestyle="-",
                color="#7570b3",
                linewidth=1.3 if args.emphasize_nb else 2.0,
                alpha=bg_alpha,
                label="Structured log-linear",
            )
        if logreg_rmse:
            logreg_rmse = _sort(logreg_rmse)
            plt.plot(
                [p[0] for p in logreg_rmse],
                [p[1] for p in logreg_rmse],
                marker="h",
                linestyle="-",
                color="#2a9d8f",
                linewidth=1.3 if args.emphasize_nb else 2.0,
                alpha=bg_alpha,
                label="Logistic Regression (exp.)",
            )
        plt.xlabel("Training items (+25 test item observed only LLM rating)")
        plt.ylabel("Test Missing RMSE")
        plt.title("LLM Rubric: CPM vs structured baselines (RMSE)")
        plt.xticks(sorted(set(p[0] for p in cpm_rmse)))
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_rmse, dpi=300)
        plt.close()
        print(f"Saved: {output_rmse}")


if __name__ == "__main__":
    main()
