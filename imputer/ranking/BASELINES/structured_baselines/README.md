# Structured baselines for domain-3 missing ratings

Three baselines for the same **per-item missing-cell** task used by Marformer on `data_bundle.json`:

1. **Naive Bayes IJK** (`naive_bayes_ijk.py`) — classic factorized model  
   \(P(y \mid i,j,k) \propto P(y)\,P(i\mid y)\,P(j\mid y)\,P(k\mid y)\) with Laplace smoothing.  
   Fitted on a **flat pool** of observed ratings (transductive: train+val+test observed by default).

2. **Structured Naive Bayes** (`naive_bayes_structured.py`) — relation-aware **conditional** model  
   \[
   \log P(y \mid \text{sources}) =
     \log P(y \mid i^\*) +
     \sum_r \log P(i_r, v_r, \text{rel}_r \mid y, i^\*)
   \]
   where \(i^\*\) is the **target attribute index** (0-based), \(v_r\) is the source value (class), and `rel_r` comes from `feature_utils.relation_label`.  
   The conditional is a smoothed multinomial over the joint \((\text{relation}, \text{source attribute}, \text{source value})\) with **\(7 \times I \times C\)** bins per \((i^\*, y)\) slice — fixed in graph size.

3. **Structured log-linear** (`log_linear_structured.py`) — softmax linear classifier with the same feature templates:  
   \(\text{score}(y) = w_{\text{uni}}[i^\*, y] + \sum_r w_{\text{bi}}[i_r, v_r, i^\*, y, \text{rel}_r]\).  
   Trained with PyTorch Adam on leave-one-out **train** plates (default).

## Relation labels

`feature_utils.relation_label(i_s,j_s,k_s, i_t,j_t,k_t)` returns one of:

| ID | Name | Pattern |
|----|------|---------|
| 0 | SAME_ITEM_SAME_ANNOT_DIFF_ATTR | same j,k; i differs |
| 1 | SAME_ITEM_SAME_ATTR_DIFF_ANNOT | same i,k; j differs |
| 2 | SAME_ANNOT_SAME_ATTR_DIFF_ITEM | same i,j; k differs |
| 3 | SAME_ITEM_ONLY | k only |
| 4 | SAME_ANNOT_ONLY | j only |
| 5 | SAME_ATTR_ONLY | i only |
| 6 | UNRELATED | all differ |

Indices `i,j,k` are **0-based** (bundle JSON is 1-based; the adapter converts).

## Training / test examples

`dataset_adapter.py`:

- **Training (default):** for each `(item, instance)` plate with `instance ∈ train_instances`, every cell is a target once; sources are the other cells on that plate (leave-one-out). No test plates are used unless you pass e.g. `--train-instances train,val`.

- **Test:** each `missing_ratings` row with `instance == "test"` is a target; sources are **`observed_ratings` on the same test item** only (no leakage from other items or from missing values).

## How to run

From `imputer/ranking`:

```bash
python BASELINES/run_structured_baselines.py \
  --bundle DATA/STAN/DOMAIN3-ITEM/Tensor_400_25_9_ItemTest_SharedThreshold/Tensor_400_25_9_ItemTest_SharedThreshold_10/data_bundle.json
```

### LLM Rubric: curve vs CPM (PNG)

`run_structured_baselines.py` only prints JSON. For **line plots** over train-item sizes together with **CPM SharedThreshold** STAN, use (from `imputer/ranking`):

```bash
python scripts/utils/plot_llm_rubric_cpm_with_structured_baselines.py
```

Writes `PLOTS/TALK/LLM_RUBRIC/llm_rubric_cpm_structured_baselines_log_loss.png` and `_rmse.png`. It discovers sizes from `RESULTS/STAN/LLM_RUBRIC/CPM_SHARED_THRESHOLD/LLMRubric_225_25_9_*_eval/`. Options: `--ll-epochs`, `--train-instances`, `--skip-log-linear`, `--results-root`, `--data-root`, `--output-logloss`.

Options: `--train-instances train,val`, `--no-ijk-transductive`, `--ll-epochs`, `--eval-val`.

## Smoke test

```bash
python BASELINES/structured_baselines/test_smoke.py
```

## Optional extensions (hooks)

The design keeps **\(i,j,k\)** only inside `relation_label` and the adapter. Nonparametric identity features (per-item or per-annotator biases) can be added later by extending the score with extra lookup tables without changing relation semantics.
