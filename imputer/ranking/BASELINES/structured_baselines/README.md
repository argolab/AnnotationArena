# Structured baselines

Fast categorical baselines for **missing rating imputation** on `data_bundle.json` (same task as Marformer / CPM STAN: predict held-out cells on each item).

**Three models:** pooled unigram, naive Bayes IJK, structured naive Bayes (relation-aware).

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

**LLM Rubric figures** (log loss, RMSE, and **calibration** vs train size + CPM STAN):

```bash
# All three PNGs (default)
python scripts/utils/plot_llm_rubric_cpm_with_structured_baselines.py

# Calibration reliability diagram only
python scripts/utils/plot_llm_rubric_cpm_with_structured_baselines.py --calibration-only

# Use val missing instead of test for calibration panels
python scripts/utils/plot_llm_rubric_cpm_with_structured_baselines.py --calibration-split val
```

Outputs (defaults under `PLOTS/TALK/LLM_RUBRIC/`):

- `llm_rubric_cpm_structured_baselines_log_loss.png`
- `llm_rubric_cpm_structured_baselines_rmse.png`
- `llm_rubric_cpm_structured_baselines_calibration.png` (smECE reliability panels)

#### What is a “calibration plot” here?

Each panel is a **reliability diagram** (confidence vs. accuracy, with a smooth calibration curve).
The title reports **smECE** — lower is better calibrated. This is separate from log-loss or RMSE curves.

**Calibration on any single bundle:**

```bash
python scripts/utils/plot_structured_baselines_calibration.py \
  --bundle path/to/data_bundle.json \
  --output PLOTS/my_calibration.png \
  --split test

# Optional CPM panel
python scripts/utils/plot_structured_baselines_calibration.py \
  --bundle DATA/LLMRubric_225_25_9_175/data_bundle.json \
  --cpm-eval-dir RESULTS/STAN/LLM_RUBRIC/CPM_SHARED_THRESHOLD/LLMRubric_225_25_9_175_eval \
  --output PLOTS/cal_with_cpm.png
```

**Together with Marformer / neural baselines** (LLM Rubric or SummEval grid):

```bash
python scripts/utils/plot_realdata_calibration.py --dataset LLMRubric --sizes 175
```

**Smoke test:**

```bash
python BASELINES/structured_baselines/test_smoke.py
```

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
| **Predict** | Each **missing** row in the split you evaluate (`test` or `val`). **Sources** = other **observed** ratings on the **same item** in that split (e.g. test-observed neighbors for test missing). |

Indices `(i, j, k)` are attribute, annotator, and item (0-based inside the code).

---

## Models and formulas

Let `y` be the latent class of the **target** cell `(i*, j*, k*)`.  
Let each **source** cell be `(i', j', k', y')` (observed label).  
All models use **add-α Laplace smoothing** (defaults α = 1) on multinomial factors.

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

Same IJK factors on the target, plus one factor per **source** cell, keyed by the **structural relation** between source and target.

\[
P(y \mid i^*, j^*, k^*, \text{sources})
\propto
P(y)\, P(i^* \mid y)\, P(j^* \mid y)\, P(k^* \mid y)
\prod_{\text{source } (i',j',k',y')}
P\bigl(y' \mid y,\; r(i',j',k' \to i^*,j^*,k^*)\bigr)
\]

**Fit (global pool):**

- Slot tables: one increment per observed cell.  
- Relation tables: for every **ordered** pair of distinct observed cells `(target, source)`, increment the table for parent label `y` (target) and child label `y'` (source) at relation `r`.

**Predict:** same formula, but sources are only same-item neighbors in the eval split (see table above).

#### Relation label `r` (7 types)

Compare source indices to target indices `(i*, j*, k*)`:

| Code | Name | When |
|------|------|------|
| 0 | `SAME_ITEM_SAME_ANNOT_DIFF_ATTR` | same `j`, `k`; different `i` |
| 1 | `SAME_ITEM_SAME_ATTR_DIFF_ANNOT` | same `i`, `k`; different `j` |
| 2 | `SAME_ANNOT_SAME_ATTR_DIFF_ITEM` | same `i`, `j`; different `k` |
| 3 | `SAME_ITEM_ONLY` | same `k` only |
| 4 | `SAME_ANNOT_ONLY` | same `j` only |
| 5 | `SAME_ATTR_ONLY` | same `i` only |
| 6 | `UNRELATED` | all three differ |

Defined in `feature_utils.relation_label` (first matching row wins).

---

## Python API

```python
from pathlib import Path
from structured_baselines.runner import load_and_fit, evaluate_split

bundle, fitted = load_and_fit(Path("data_bundle.json"), snb_alpha=1.0)
metrics = evaluate_split(fitted, bundle, "test")
# metrics["unigram_ij"], metrics["ijk"], metrics["snb"]
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
| `feature_utils.py` | Relation labels |
| `cli_defaults.py` | Default α values |
| `../run_structured_baselines.py` | CLI |

---

## Defaults

| Setting | Value |
|---------|--------|
| Fit pool | `train`, `val`, `test` observed |
| `unigram_alpha`, `ijk_alpha`, `snb_alpha` | `1.0` |

Flags: `--unigram-alpha`, `--ijk-alpha`, `--snb-alpha`, `--snb-alpha-sweep`, `--eval-val`, `--out`.
