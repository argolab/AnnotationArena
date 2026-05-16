# Baselines (`imputer/ranking/BASELINES`)

Run all commands from **`imputer/ranking`**.

| Goal | Script |
|------|--------|
| Structured metrics (one bundle) | `BASELINES/run_structured_baselines.py` |
| Structured calibration (one bundle) | `scripts/utils/plot_structured_baselines_calibration.py` |
| Structured learning curves (folder of bundles) | `scripts/utils/plot_structured_baselines_learning_curve.py` |
| LLM Rubric: structured + CPM curves | `scripts/utils/plot_llm_rubric_cpm_with_structured_baselines.py` |
| **All methods** calibration (LLM Rubric / SummEval) | `scripts/utils/plot_realdata_calibration.py` |
| **All methods** log-loss curves (LLM Rubric / SummEval) | `scripts/utils/plot_realdata_test_loss.py` |

Model details and formulas → **[structured_baselines/README.md](structured_baselines/README.md)**.

---

## Calibration plots (reliability diagrams)

Each panel is a **reliability diagram** (predicted confidence vs. empirical accuracy, all classes flattened). The title shows **smECE** (smooth expected calibration error); lower is better. This measures probability calibration, not log loss or RMSE.

### Structured baselines only (any `data_bundle.json`)

By default three panels: pooled unigram **P(y|i,j)**, **NB IJK**, **structured NB**. Add **`--log-linear`** for a fourth panel (PyTorch softmax over the same structured features; uses val early stopping when you enable it via `fit_baselines` / CLI—see [structured_baselines/README.md](structured_baselines/README.md)).

```bash
python scripts/utils/plot_structured_baselines_calibration.py \
  --bundle DATA/STAN/DOMAIN3-FINAL/ItemSplits/Transductive/DOMAIN3-FINAL_Item_T_1000/data_bundle.json \
  --output PLOTS/TALK/DOMAIN3-FINAL/calibration_item_T_1000.png \
  --split test
```

Optional STAN panel if you have an eval dir with `rating_probabilities.csv`:

```bash
python scripts/utils/plot_structured_baselines_calibration.py \
  --bundle DATA/STAN/LLM_RUBRIC/LLMRubric_225_25_9_175/data_bundle.json \
  --cpm-eval-dir RESULTS/STAN/LLM_RUBRIC/CPM_SHARED_THRESHOLD/LLMRubric_225_25_9_175_eval \
  --output PLOTS/cal_with_stan.png \
  --split test
```

Use `--split val` for val-missing cells instead of test.

### All baselines (Marformer, STAN, structured, ReMasker, MIWAE)

**Datasets:** LLM Rubric and SummEval only (hard-coded paths in the script).

**Panels per dataset:**

| Method | LLM Rubric | SummEval |
|--------|------------|----------|
| Marformer | yes | yes |
| STAN | CPM SharedThreshold | Factor + Normal |
| Unigram (ij), NB IJK, Structured NB | yes (fit on bundle) | yes |
| ReMasker | yes | yes |
| MIWAE | yes | yes |

```bash
# All configured sizes (can take a while — refits structured models per size)
python scripts/utils/plot_realdata_calibration.py

# One size
python scripts/utils/plot_realdata_calibration.py --dataset LLMRubric --sizes 175
python scripts/utils/plot_realdata_calibration.py --dataset SummEval --sizes 1280
```

**Outputs:** `PLOTS/TALK/LLMRubric/ece_reliability_llm_rubric_size{SIZE}.png`, `PLOTS/TALK/SummEval/ece_reliability_summeval_size{SIZE}.png`.

Missing result trees produce empty panels (script prints `[empty panel] …`).

#### Prerequisites for all-baseline calibration

| Method | Expected paths (LLM Rubric example) |
|--------|-------------------------------------|
| Bundles | `DATA/STAN/LLM_RUBRIC/LLMRubric_225_25_9_{size}/data_bundle.json` |
| Marformer | `RESULTS/MARFORMER/LLM_RUBRIC/LLMRubric_225_25_9_{size}/` (checkpoint + `train_config.json`) |
| CPM STAN | `RESULTS/STAN/LLM_RUBRIC/CPM_SHARED_THRESHOLD/LLMRubric_225_25_9_{size}_eval/rating_probabilities.csv` |
| ReMasker | `RESULTS/BASELINES/REMASKER/LLMRUBRIC/LLMRubric_225_25_9_{size}/test_predictions.json` |
| MIWAE | `RESULTS/BASELINES/MIWAE/LLMRUBRIC/LLMRubric_225_25_9_{size}/test_predictions.json` |

Structured panels need only the bundle (models are fit on the fly).

---

## Learning curves (log loss / RMSE vs training size)

### Structured baselines only (any folder of bundles)

Scans `--data-root/*/data_bundle.json`, fits unigram / IJK / SNB at each size, plots test-missing NLL and RMSE. Optional STAN overlay via `--stan-results-root`.

**DOMAIN3-FINAL — item expansion (transductive):**

```bash
mkdir -p PLOTS/TALK/DOMAIN3-FINAL

python scripts/utils/plot_structured_baselines_learning_curve.py \
  --data-root DATA/STAN/DOMAIN3-FINAL/ItemSplits/Transductive \
  --size-regex 'DOMAIN3-FINAL_Item_T_(\d+)$' \
  --xlabel 'Training items' \
  --title 'DOMAIN3-FINAL: structured baselines (item, transductive)' \
  --output-logloss PLOTS/TALK/DOMAIN3-FINAL/item_T_structured_log_loss.png \
  --output-rmse PLOTS/TALK/DOMAIN3-FINAL/item_T_structured_rmse.png \
  --output-calibration PLOTS/TALK/DOMAIN3-FINAL/item_T_structured_calibration.png
```

**DOMAIN3-FINAL — annotator expansion:**

```bash
python scripts/utils/plot_structured_baselines_learning_curve.py \
  --data-root DATA/STAN/DOMAIN3-FINAL/AnnotSplits/Transductive \
  --size-regex 'DOMAIN3-FINAL_Annot_T_(\d+)$' \
  --xlabel 'Training annotators' \
  --title 'DOMAIN3-FINAL: structured baselines (annot, transductive)' \
  --output-logloss PLOTS/TALK/DOMAIN3-FINAL/annot_T_structured_log_loss.png \
  --output-rmse PLOTS/TALK/DOMAIN3-FINAL/annot_T_structured_rmse.png
```

Non-transductive splits: use `ItemSplits/NonTransductive` or `AnnotSplits/NonTransductive` and `--size-regex 'DOMAIN3-FINAL_Item_NT_(\d+)$'` (or `Annot_NT_…`).

**Optional STAN curve** (when eval dirs exist):

```bash
python scripts/utils/plot_structured_baselines_learning_curve.py \
  --data-root DATA/STAN/DOMAIN3-FINAL/ItemSplits/Transductive \
  --size-regex 'DOMAIN3-FINAL_Item_T_(\d+)$' \
  --stan-results-root RESULTS/STAN/TENSOR/DOMAIN3-FINAL/ITEM \
  --stan-eval-regex 'DOMAIN3-FINAL_Item_T_(\d+)_TENSOR_eval$' \
  --stan-label 'Oracle Tensor STAN' \
  --xlabel 'Training items' \
  --title 'DOMAIN3-FINAL: STAN + structured baselines' \
  --output-logloss PLOTS/TALK/DOMAIN3-FINAL/item_T_all_log_loss.png \
  --output-rmse PLOTS/TALK/DOMAIN3-FINAL/item_T_all_rmse.png \
  --no-calibration
```

Calibration at largest size: `--output-calibration …` (default). Pick another size with `--calibration-size 500`.

### LLM Rubric: structured baselines + CPM STAN

```bash
# Log loss, RMSE, and calibration (largest train size) — needs CPM eval dirs
python scripts/utils/plot_llm_rubric_cpm_with_structured_baselines.py

# Curves only or calibration only
python scripts/utils/plot_llm_rubric_cpm_with_structured_baselines.py --no-plot-calibration
python scripts/utils/plot_llm_rubric_cpm_with_structured_baselines.py --calibration-only --calibration-size 175
```

Defaults: `--data-root DATA/STAN/LLM_RUBRIC`, `--results-root RESULTS/STAN/LLM_RUBRIC/CPM_SHARED_THRESHOLD`.

Outputs under `PLOTS/TALK/LLM_RUBRIC/`:

- `llm_rubric_cpm_structured_baselines_log_loss.png`
- `llm_rubric_cpm_structured_baselines_rmse.png`
- `llm_rubric_cpm_structured_baselines_calibration.png`

### All baselines (LLM Rubric / SummEval)

**`plot_realdata_test_loss.py`** — one figure per dataset with curves for:

- Marformer  
- STAN (CPM for LLM Rubric; Factor + Normal for SummEval)  
- ReMasker, MIWAE  
- Global empirical unigram (pooled over all observed cells — **not** the structured IJK/SNB models)

```bash
python scripts/utils/plot_realdata_test_loss.py
```

Outputs: `PLOTS/TALK/LLMRubric/llm_rubric_test_loss_by_size.png`, per-size MBR-L2 snapshots, runtime plot; same pattern under `PLOTS/TALK/SummEval/`.

For **structured** unigram (ij) + IJK + SNB curves on LLM Rubric, use `plot_llm_rubric_cpm_with_structured_baselines.py` (above). There is no single script yet that overlays Marformer, neural baselines, and all three structured models on one learning-curve figure.

#### Prerequisites for all-baseline learning curves

Same bundle and result layout as the calibration table. Neural baselines must be trained first (see below). Marformer metrics are read from `TEST_RESULTS/best.json`; STAN from `predictive_metrics.json` under the eval run dirs.

---

## Structured baselines (categorical)

Missing-cell prediction on `data_bundle.json`: **unigram (ij)**, **NB IJK**, **structured NB**, and optional **structured log-linear** (``--log-linear``; val early stopping, train-observed fallback when train-missing is empty).

```bash
python BASELINES/run_structured_baselines.py \
  --bundle DATA/STAN/DOMAIN3-FINAL/ItemSplits/Transductive/DOMAIN3-FINAL_Item_T_100/data_bundle.json

python BASELINES/run_structured_baselines.py \
  --bundle path/to/data_bundle.json \
  --log-linear \
  --eval-val \
  --out RESULTS/structured_baselines/metrics.json
```

---

## Neural baselines (ReMasker / MIWAE)

Train on one bundle; writes `test_predictions.json` for plotting scripts.

```bash
python BASELINES/run_baselines.py \
  --method remasker \
  --data-bundle DATA/STAN/LLM_RUBRIC/LLMRubric_225_25_9_50/data_bundle.json \
  --output-dir RESULTS/BASELINES/REMASKER/LLMRUBRIC/LLMRubric_225_25_9_50

python BASELINES/run_baselines.py \
  --method miwae \
  --data-bundle DATA/STAN/LLM_RUBRIC/LLMRubric_225_25_9_50/data_bundle.json \
  --output-dir RESULTS/BASELINES/MIWAE/LLMRUBRIC/LLMRubric_225_25_9_50 \
  --epochs 500
```

Repeat for each train size, keeping the directory name aligned with `baseline_run` in `plot_realdata_*.py` (e.g. `LLMRubric_225_25_9_{size}` under `RESULTS/BASELINES/REMASKER/LLMRUBRIC/`).

See `run_baselines.py --help` for training options.

---

## Quick reference: DOMAIN3-FINAL synthetic data

Data lives under `DATA/STAN/DOMAIN3-FINAL/` (from the Marformer repo). Sizes are listed in `domain3_metadata.json`.

| Task | Structured calibration | Structured learning curve |
|------|------------------------|---------------------------|
| Item / transductive | `plot_structured_baselines_calibration.py` on one bundle | `plot_structured_baselines_learning_curve.py` on `ItemSplits/Transductive` |
| Annot / transductive | same | `AnnotSplits/Transductive` |

All-baseline Marformer/STAN/neural grids are **not** wired to DOMAIN3-FINAL in `plot_realdata_*.py`; use the structured scripts above, or extend those plotters once `RESULTS/` exist for DOMAIN3-FINAL runs.

**Recurrent Marformer** (weight-shared core + prelude/coda) uses separate entrypoints and results: `python -m imputer.entity_mf.recurrent.train`, output under `RESULTS/RECURRENT_MARFORMER/`, scripts in `scripts/DOMAIN3-FINAL/RecurrentMarformer/`. See `imputer/entity_mf/recurrent/README.md`.
