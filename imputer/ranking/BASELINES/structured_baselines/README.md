# Structured baselines

Fast categorical baselines for **missing rating imputation** on `data_bundle.json` (same task as Marformer / CPM STAN: predict held-out cells on each item).

**Three closed-form models:** pooled unigram, naive Bayes IJK, structured naive Bayes (factorized per-attr-pair + shared CHANGEJ/CHANGEK).  
**Optional:** **structured log-linear** (`StructuredLogLinear` in `log_linear_structured.py`) — softmax linear model over the same factorized features as SNB, trained with PyTorch (Adam); supervision on train-missing rows when present, otherwise on train-observed all-source examples; **validation early stopping** on val-missing mean NLL when val splits exist (`--log-linear` on the CLI).

---

## Quick start

From `imputer/ranking`:

```bash
# Metrics on one bundle (test missing)
python BASELINES/run_structured_baselines.py \
  --bundle DATA/LLMRubric_225_25_8_175/data_bundle.json

# Also print val missing metrics
python BASELINES/run_structured_baselines.py \
  --bundle path/to/data_bundle.json \
  --eval-val

# Tune structured NB smoothing
python BASELINES/run_structured_baselines.py \
  --bundle path/to/data_bundle.json \
  --snb-alpha-sweep 0.5,1,2,5,10,20

# Save JSON
python BASELINES/run_structured_baselines.py \
  --bundle path/to/data_bundle.json \
  --out RESULTS/metrics.json
```

## Plots (calibration and learning curves)

Full command reference (structured-only and **all baselines** including Marformer, STAN, ReMasker, MIWAE) → **[../README.md](../README.md)**.

**Calibration** (reliability diagram + smECE; one bundle):

```bash
python scripts/utils/plot_structured_baselines_calibration.py \
  --bundle path/to/data_bundle.json \
  --output PLOTS/my_calibration.png \
  --split test
```

**Learning curves** (log loss / RMSE vs train size; folder of bundles):

```bash
python scripts/utils/plot_structured_baselines_learning_curve.py \
  --data-root DATA/STAN/DOMAIN3-FINAL/ItemSplits/Transductive \
  --size-regex 'DOMAIN3-FINAL_Item_T_(\d+)$' \
  --xlabel 'Training items' \
  --title 'DOMAIN3-FINAL structured baselines' \
  --output-logloss PLOTS/TALK/DOMAIN3-FINAL/item_T_log_loss.png \
  --output-rmse PLOTS/TALK/DOMAIN3-FINAL/item_T_rmse.png
```

**LLM Rubric** (structured + CPM STAN curves):

```bash
python scripts/utils/plot_llm_rubric_cpm_with_structured_baselines.py
```

**Smoke test:**

```bash
python BASELINES/structured_baselines/test_smoke.py
```

### Structured log-linear (optional; PyTorch)

```bash
python BASELINES/run_structured_baselines.py \
  --bundle path/to/data_bundle.json \
  --log-linear \
  --log-linear-progress

# Fixed epoch budget (no val early stopping)
python BASELINES/run_structured_baselines.py \
  --bundle path/to/data_bundle.json \
  --log-linear \
  --log-linear-patience 0
```

Training minimizes cross-entropy on **train-missing** examples when those exist; otherwise on **train-observed** rows (see `build_train_observed_examples`—needed for LLM Rubric–style bundles with no train missing). In both cases, sources for each training example are all transductive observed cells except the target. If the bundle has **val-missing** rows, each epoch evaluates val mean NLL and restores the best weights; stopping triggers after `--log-linear-patience` epochs without improvement (default 5; `0` disables). Hyperparameter defaults: `cli_defaults.py` (`DEFAULT_LOG_LINEAR_*`).

---

## Data format

Bundles live under e.g. `DATA/.../data_bundle.json` with:

- `observed_ratings` — known labels  
- `missing_ratings` — held-out labels (used only for evaluation metrics)

Each row has `attribute`, `annotator`, `item`, `value` (1-based integers), and `instance` ∈ `{train, val, test}`.

Optional `configs.json` in the same folder sets `datagen.C` (number of Likert classes).

---

## Training and evaluation (one rule)

| Phase | What is used |
|--------|----------------|
| **Fit** | Every **observed** row in `train`, `val`, and `test` (transductive pool). Missing rows are never used for fitting. |
| **Predict** | Each **missing** row in the split you evaluate (`test` or `val`). **Sources** = **all** transductive observed cells (train + val + test), excluding only the target cell itself. Sources are not restricted to same-item or same-split neighbors. |

Indices `(i, j, k)` are attribute, annotator, and item (0-based inside the code).

---

## Models and formulas

Let `y` be the latent class of the **target** cell `(i*, j*, k*)`.  
Let each **source** cell be `(i', j', k', y')` (observed label).  
All **count-based** models use **add-α Laplace smoothing** (defaults α = 1) on multinomial factors. The optional **log-linear** model does not use those counts at fit time; it is trained with gradient descent on train-missing labels.

### 1. Pooled unigram — `PooledUnigramIJ`

Pools only on **(attribute, annotator)**; ignores item at fit time.

\[
P(y \mid i^*, j^*) \propto \text{Count}(y, i^*, j^*) + \alpha
\]

Prediction for a missing row uses its `(attribute, annotator)` bucket.  
No source cells at predict time.

### 2. Naive Bayes IJK — `NaiveBayesIJK`

Standard naive Bayes over the three slot indices:

\[
P(y \mid i^*, j^*, k^*) \propto
P(y)\, P(i^* \mid y)\, P(j^* \mid y)\, P(k^* \mid y)
\]

Each factor is a smoothed multinomial fit from the transductive pool (one count per observed rating).  
No source cells at predict time.

### 3. Structured naive Bayes — `StructuredNaiveBayes`

IJK slot factors plus three classes of pairwise source factors:

\[
P(y \mid i^*, j^*, k^*, \text{sources})
\propto
P(y)\, P(i^* \mid y)\, P(j^* \mid y)\, P(k^* \mid y)
\prod_{i' \neq i^*} P_{i',i^*}(y_{i'j^*k^*} \mid y)
\prod_{j' \neq j^*} P_{\text{CHANGEJ}}(y_{i^*j'k^*} \mid y)
\prod_{k' \neq k^*} P_{\text{CHANGEK}}(y_{i^*j^*k'} \mid y)
\]

**Source routing** (each source cell fires at most one factor):

| Factor | Condition on source `(i', j', k')` | Parameters |
|--------|-------------------------------------|------------|
| **ATTR_PAIR** | `j'=j*`, `k'=k*`, `i'≠i*` | Per `(i', i*)` pair: `n_attr[i', i*, y_target, y_source]` |
| **CHANGEJ** | `i'=i*`, `k'=k*`, `j'≠j*` | Shared: `n_change_j[y_target, y_source]` |
| **CHANGEK** | `i'=i*`, `j'=j*`, `k'≠k*` | Shared: `n_change_k[y_target, y_source]` |
| *ignored* | all other cases | — |

**Multiplicity:** multiple source cells mapping to the same CHANGEJ/CHANGEK entry multiply the factor (log domain: add log-prob for each occurrence). Example: CHANGEJ sources with ratings `[3, 3, 4, 5]` vs target `y=4` contributes `2·log P(3|4) + log P(4|4) + log P(5|4)`.

**Fit (global pool):**

- Slot tables: one increment per observed cell.
- Pair tables: for every ordered distinct pair `(target, source)` in the transductive pool, route and increment the matching table.

**Predict:** same formula; sources = all transductive observed cells except the target cell.

**Implementation:** `plate_graph_factorized.py` (`FactorizedPlateCounts`), routing via `factor_routing.py`.

### 4. Structured log-linear — `StructuredLogLinear`

Same **feature structure** as SNB but with **free parameters** fit by minimizing cross-entropy:

| Parameter | Shape | Contribution |
|-----------|-------|--------------|
| `w_y` | `(C,)` | prior log-odds |
| `w_i` | `(I, C)` | attribute unigram |
| `w_j` | `(J, C)` | annotator unigram |
| `w_k` | `(K, C)` | item unigram |
| `w_attr` | `(I, I, C, C)` | per `(i', i*)` attr-pair; `w_attr[i', i*, v', y]` |
| `w_change_j` | `(C, C)` | shared CHANGEJ; `count × w_change_j[v', y]` |
| `w_change_k` | `(C, C)` | shared CHANGEK; `count × w_change_k[v', y]` |

Supervision: **train-missing** cells when present, else **train-observed** pseudo-tasks (see `build_train_observed_examples`). With val-missing data, training uses **early stopping** on val NLL and restores the best checkpoint.

---

## Python API

```python
from pathlib import Path
from structured_baselines.runner import load_and_fit, evaluate_split

bundle, fitted = load_and_fit(Path("data_bundle.json"), snb_alpha=1.0)
metrics = evaluate_split(fitted, bundle, "test")
# metrics["unigram_ij"], metrics["ijk"], metrics["snb"], optional metrics["log_linear"]

bundle, fitted = load_and_fit(
    Path("data_bundle.json"),
    fit_log_linear=True,
    log_linear_epochs=64,
    log_linear_early_stopping_patience=5,
)
```

---

## Layout

| File | Role |
|------|------|
| `runner.py` | `fit_baselines`, `evaluate_split` |
| `dataset_adapter.py` | Bundle I/O, transductive pool, eval examples |
| `unigram_pooled.py` | Unigram (ij) |
| `naive_bayes_ijk.py` | IJK naive Bayes |
| `naive_bayes_structured.py` | Structured NB wrapper |
| `plate_graph_factorized.py` | Counts + log-posterior for SNB |
| `factor_routing.py` | Route source cells to ATTR_PAIR / CHANGEJ / CHANGEK factors |
| `log_linear_structured.py` | Optional softmax log-linear (train missing; val early stopping) |
| `feature_utils.py` | Legacy relation labels (not used by SNB/log-linear) |
| `cli_defaults.py` | Default α values |
| `../run_structured_baselines.py` | CLI |

---

## Defaults

| Setting | Value |
|---------|--------|
| Fit pool | `train`, `val`, `test` observed |
| `unigram_alpha`, `ijk_alpha`, `snb_alpha` | `1.0` |

Flags: `--unigram-alpha`, `--ijk-alpha`, `--snb-alpha`, `--snb-alpha-sweep`, `--eval-val`, `--out`, and log-linear: `--log-linear`, `--log-linear-epochs`, `--log-linear-lr`, `--log-linear-batch`, `--log-linear-patience`, `--log-linear-progress`.
