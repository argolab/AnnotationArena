# Real Data: LLMRubric and SummEval

Both datasets are annotation datasets where human (and sometimes LLM) annotators rate text
summaries on multiple quality criteria. They are stored in the common bundle format
(`data_bundle.json`) and converted from raw sources via `scripts/convert-*/`.

---

## LLMRubric

**Source data:** Human evaluation of summarization quality using a rubric-based approach,
augmented with LLM annotations providing full distributions over rating categories.

**Dimensions:**
- Items (K): 225 summaries
- Annotators (J): 25 (24 human + 1 LLM annotator)
- Attributes (I): 9 quality criteria
- Rating scale: 10-point Likert (C=10)

**Split type:** ItemTest — test entities are unseen items. Annotators are fixed (all 25 rate all items).

**Data directory:** `DATA/LLM_RUBRIC/`

**Available splits (by K_train — number of training items):**
```
LLMRubric_225_25_9_10     (10 train items)
LLMRubric_225_25_9_20
LLMRubric_225_25_9_30
LLMRubric_225_25_9_40
LLMRubric_225_25_9_50
LLMRubric_225_25_9_75
LLMRubric_225_25_9_100
LLMRubric_225_25_9_125
LLMRubric_225_25_9_150
LLMRubric_225_25_9_175
```

**Key feature — LLM soft distributions:**
The LLM annotator does not give a single vote; it provides a full probability distribution
over the 10 rating classes. This is stored as `rating_dist` in the bundle. When
`--llm-input-dist` is enabled, these distributions are encoded in the param stream as
log-probabilities instead of a one-hot spike. Human ratings have one-hot `rating_dist`
(the `_is_one_hot` check in `types.py:268` ensures they still use the hard-label spike).

**Missingness pattern:** MCAR-ish — annotations are fairly dense. Not all human annotators
rate all items; the missingness is what makes imputation necessary.

**Conversion:** `scripts/convert-llm-rubric/convert.py`

---

## SummEval

**Source data:** Human evaluation of neural summarization systems on the CNN/DailyMail dataset.
Standard benchmark with multiple quality dimensions rated by crowdworkers.

**Dimensions:**
- Items (K): 1600 summaries
- Annotators (J): 8
- Attributes (I): 4 quality criteria (coherence, consistency, fluency, relevance)
- Rating scale: 5-point Likert (C=5)

**Split type:** ItemTest — test entities are unseen items. All 8 annotators are shared.

**Data directory:** `DATA/SUMMEVAL/`

**Available splits (by K_train):**
```
SummEval_1600_8_4_50      (50 train items)
SummEval_1600_8_4_100
SummEval_1600_8_4_500
SummEval_1600_8_4_750
SummEval_1600_8_4_1000
SummEval_1600_8_4_1280
```

**Note:** SummEval is much larger than LLMRubric in item count (1600 vs 225) but has fewer
annotators (8 vs 25) and attributes (4 vs 9). No LLM annotator — all annotations are human
hard-label ratings. `--llm-input-dist` is not meaningful here.

**Conversion:** `scripts/convert-summeval/`

---

## Bundle Format

Both datasets are stored as `data_bundle.json` with the following structure:

```json
{
  "observed_ratings": [...],        // ratings available as context
  "missing_ratings":  [...],        // ratings to predict
  "all_ratings":      [...],        // union of the above
  "observed_pairwise": [...],       // pairwise rankings (if any)
  "missing_pairwise":  [...],
  "missing_ratings_indexes_in_test_instance": [...],  // indices into missing_ratings for test split
  "train_posterior_rating_probs": [...],  // shape [I*J_train, K, C]
  "val_posterior_rating_probs":   [...],
  "test_posterior_rating_probs":  [...],
  "embeddings":           [...],    // item embeddings (from STAN for synthetic; from model for real)
  "annotator_embeddings": [...],    // annotator embeddings
  "stats": { "K": ..., "J": ..., "I": ..., ... }
}
```

Each rating entry:
```json
{
  "item":       1,            // 1-indexed
  "annotator":  3,            // 1-indexed
  "attribute":  2,            // 1-indexed
  "value":      4,            // 1-indexed rating value
  "instance":   "train",      // "train" | "val" | "test"
  "rating_dist": [0,0,0,1,0,0,0,0,0,0]  // one-hot for human; soft for LLM
}
```

The `DataConverter` in `imputer/data.py` reads bundles and converts to `RankingData` objects
(0-indexed internally) for model consumption.
