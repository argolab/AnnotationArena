# A validation protocol for Domain 3

Domain 3 ships with an empty validation split. `K_val = 0` on the item axis and
`J_val_split = 0` on the annotator axis, so `val_ratings` is 0 in every bundle,
`val/missing_ce` is never logged, and Lightning's `ModelCheckpoint` -- which
monitors exactly that metric -- **never saved a `best-*.ckpt` in any run**. The
"best" checkpoint the paper's figures use is produced afterwards by
`plot_domain3_sigmoid._write_best_json_from_test_results`: an argmin over every
scored checkpoint, keyed on **test** log-loss.

That is discrepancy D14, and on the 3-seed grid it is worth this much:

| K (item, transductive) | last epoch | argmin on test | optimism | epoch chosen |
|---|---|---|---|---|
| 100 | 1.0966 | 0.9627 | **+0.134** | 99 |
| 300 | 0.9107 | 0.8840 | +0.027 | 241 |
| 500 | 0.8639 | 0.8521 | +0.012 | 274 |
| 2000 | 0.7875 | 0.7863 | +0.001 | 332 |
| 4000 | 0.7611 | 0.7610 | +0.000 | 374 |

It is a small-data problem specifically: as K grows the optimism vanishes and
the chosen epoch converges on the last one.

## The protocol

Three phases, chained in `RERUN/submit_validation.sh`.

**Search.** Carve part of the training partition off as validation
(`imputer/entity_mf/validation.py`), train the full 400 epochs, and record the
validation curve. The carve is done at load time on the in-memory bundle, so
every arm still trains on byte-identical files on disk.

**Select.** `analysis/select_epochs.py` takes the argmin of the validation
missing-rating cross-entropy, per run. Per run rather than per size, because
that is what early stopping actually is, and it lets the seed spread in the
chosen epoch be reported rather than hidden.

**Refit.** Retrain on the **whole** training partition -- no carve -- for
exactly that many epochs. The reported model is the last checkpoint of the
refit, so nothing is ever selected on test, and the final model's data budget
equals Stan's. That is the reason for refitting rather than shipping the
early-stopped search model.

## Two carves, because they ask different questions

| | What is held out | What it measures |
|---|---|---|
| **`block`** (primary) | whole items (item axis) or whole annotators (annotator axis) | the same question the test set asks: the test block *is* a block of unseen items or annotators |
| **`mcar`** (comparison) | random observed training cells | matrix completion: fill in a cell of an item you have already seen |

A carved validation set has to be kept out of the training graph the same way
`test_missing` is. Otherwise it is an easier problem than the test set and the
epoch it selects is biased late; `val_missing_in_graph=False` does that.

**A third signal, free and already there.** The *already-missing* training cells
are scored every epoch and never contribute to the loss -- they are held-out
data sitting unused, and the `combined_missing` curve in every training history
already reports them. They are a legitimate stopping signal at zero cost. But on
the item axis they sit on training items, so like `mcar` they answer the
completion question, not the generalisation one. Worth knowing; not a substitute
for the block carve.

## Edge cases, handled explicitly rather than silently

`Item_T_K` holds out a fixed 100-item test block, so training items are
K &minus; 100 -- and **`Item_T_100` has zero training items**. Likewise
`Annot_T_10` has no training annotators. Those cells cannot select on data at
all; they take their axis's median selected epoch and are flagged as fallbacks
in the epoch table and on the report page.

## Status

Runs in flight:

| Run | Protocol | Tasks |
|---|---|---|
| `val-block-01` | block | 114 search + 114 refit |
| `val-mcar-01` | mcar | 114 search + 114 refit |

Both over seeds 42/43/44 on the `replicate-01` data, so they are directly
comparable to `seeds-01`.

Early evidence, `Item_T_500` seed 42, block carve (34 of 229 training items held
out): validation bottoms out at **epoch 145** with CE 0.9708 and rises to 1.2084
by epoch 399 -- a **+0.238 nat** overfit gap, detected without touching test.
The test-argmin for the same cell is epoch 274, so validation stops
substantially earlier than test-set selection would. Whether that costs test
performance is the question the refit answers.
