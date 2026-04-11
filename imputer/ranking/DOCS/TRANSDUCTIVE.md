# Transductive vs Non-Transductive Training

---

## The Core Problem

In annotation imputation, we have a matrix of ratings over (annotators × items × attributes).
Some cells are observed, some are missing and must be predicted. The key question is: **what
entities does the model see at training time?**

Two regimes exist depending on the test split type.

---

## Non-Transductive (Inductive) — Default

**Flag:** no flag required (default mode)

**How it works:**
- Training uses only `train_observed` as context.
- Validation evaluates on `val_missing` conditioned on `val_observed` + `train_observed`.
- Test evaluates on `test_missing` conditioned on `test_observed` + `train_observed`.
- At test time, the model encounters **new entities** it has never seen during training
  (new items in ItemTest, new annotators in AnnotatorTest).

**Masking during training:**
All train_observed ratings are the maskable pool. A random subset is masked each step;
the model predicts those from the remaining observed context.

**When to use:**
- Always for ItemTest (test items are by definition unseen).
- As the standard baseline for AnnotatorTest as well.

---

## Transductive — `--transductive-learning`

**Flag:** `--transductive-learning`

**How it works:**
- Training uses `train_observed + val_observed + test_observed` all as potential context.
- The model sees all entities (including val/test ones) at train time.
- Masking can still be applied to any portion of the combined observed set.
- Evaluation remains the same: predict `*_missing` conditioned on `*_observed`.

**Masking during training (standard transductive):**
All of `train_observed + val_observed + test_observed` form the maskable pool. At each step,
a random subset is masked across all splits.

**When to use:**
- When test entities can be observed at train time (e.g., transductive node classification analogy).
- For AnnotatorTest when test annotators are present in training context.
- For ItemTest when test items can be partially observed during training.

---

## Transductive with Val/Test Masking — `--transductive-valtest-mask`

**Flag:** `--transductive-learning --transductive-valtest-mask`

**How it works:**
Splits the observed ratings into two pools:
- **Fixed context:** `train_observed` — always visible, never masked.
- **Maskable pool:** `val_observed + test_observed` — subject to masking.

```python
if self.transductive_valtest_mask:
    maskable_sources = list(self.val_observed) + list(self.test_observed)
    fixed_sources    = list(self.train_observed)
else:
    maskable_sources = train_observed + val_observed + test_observed
    fixed_sources    = []
```

**Why:**
Standard transductive masking randomly masks any observed rating, including train ones. But at
inference time, the model only needs to predict val/test missing ratings — it always has full
train context. Val/test masking directly simulates this: model always sees complete train context
and learns to predict val/test from partial val/test context. This makes training distribution
match inference distribution more closely.

**Current use:**
All `MARFORMER_HARD` (ItemTest) and `MARFORMER_ANNOT_DROP` (AnnotatorTest) experiments use
this mode with `MASKING_RATE=0.5`.

---

## Summary Table

| Mode | Train Context | Maskable Pool | Use Case |
|---|---|---|---|
| Non-transductive | train_observed | train_observed | Inductive generalization (new items/annotators) |
| Transductive | all observed | all observed | All entities visible at train time |
| Transductive + VTM | all observed | val+test observed only | Directly simulates inference task |

---

## ItemTest vs AnnotatorTest

**ItemTest** (`K_train/K_val/K_test` in stats):
- Train/val/test entities are **items**. Test items are entirely unseen.
- All annotators are shared across all splits.
- `rating_dist` is absent for test-instance ratings (STAN has no posterior for unseen test items).
  This is a no-op in the code: `build_param` handles `None` gracefully via `_is_one_hot` check.

**AnnotatorTest** (`J_train/J_val/J_test` in stats):
- Train/val/test entities are **annotators**. Test annotators are entirely unseen.
- All items are shared across all splits.
- All observed ratings have `rating_dist` present.

Both split types are supported by the same training code with the same flags. The `DataConverter`
reads `instance` labels from the bundle to route ratings to the correct split.
