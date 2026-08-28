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

## What the search phase found

All 114 block-carve search runs completed. Selected budgets, and the validation
CE at the chosen epoch versus at epoch 399:

| arm | axis | size | refit epochs | per-run argmin spread | val CE @ best | val CE @ end |
|---|---|---|---|---|---|---|
| Marformer | annot | 15 | 59 | 78&ndash;385 | 0.884 | 1.303 |
| Marformer | annot | 25 | 149 | 126&ndash;163 | 1.032 | **1.791** |
| Marformer | item | 300 | 128 | 118&ndash;174 | 0.812 | 0.964 |
| Marformer | item | 500 | 156 | 145&ndash;278 | 0.974 | 1.163 |
| Marformer | item | 2000 | 390 | 352&ndash;399 | 0.800 | 0.804 |
| Marformer | item | 4000 | 389 | 342&ndash;386 | 0.806 | 0.818 |
| MarformerNT | annot | 15 | 105 | 63&ndash;125 | 1.065 | **3.087** |

Three things worth reading off it.

**The overfitting is severe where it exists.** Marformer on 25 annotators ends
0.76 nats above its own validation optimum; Marformer NT on 15 annotators ends
**2.0 nats** above. That is not a tuning detail.

**And absent where it does not.** At item K=2000 and above the chosen epoch is
390 of 400 and the validation CE barely moves across the whole run
(0.800 &rarr; 0.804). Validation correctly declines to stop early when there is
nothing to stop.

**Pooling the seeds was necessary, not cosmetic.** The per-run argmins span
78&ndash;385 within a single cell. At Marformer item K=400 the three seeds bottom
at 284, 305 and 314 while the pooled curve chooses 185 -- which looks wrong until
you look at the basin: the pooled mean is 1.0518 at epoch 185 and 1.0495 at 314,
a 0.002-nat difference across 130 epochs. Each seed's own argmin is the bottom of
its own noise; the pooled curve says the basin is flat and the early end of it is
the conservative choice. A one-standard-error rule would formalise this and is
the obvious refinement if the protocol is kept.

**Six cells had no validation signal** -- `Item_T_100` and `Annot_T_10`, both
with no training block to carve from, times three seeds. They take their axis
median (181 and 176 epochs) and are flagged as fallbacks in the epoch table and
on the report page.

## Result

Both artifacts, side by side:

- No validation: https://claude.ai/code/artifact/96812eeb-4c0c-4c9a-a413-67bd321e6282
- Block validation: https://claude.ai/code/artifact/5bae9038-9624-4a46-ae8b-d9faa1f8f6b6

Across all 38 settings x 3 seeds, test-missing log-loss:

```
val-block vs last epoch      -0.0343 nats   (28 of 38 cells better)
val-block vs best-on-test    +0.0233 nats
mean refit budget            191 epochs (of 400)
```

**This is the outcome the protocol was built for.** Test-set selection was
buying 0.057 nats of optimism on average; block validation recovers 60% of it
legitimately, and the remaining +0.023 is the honest price of not looking at
test. It also halves training: 191 epochs instead of 400.

Where it works best is where D14 was worst -- the annotator axis and small item
sizes:

| setting | last | best on test | **val-block** |
|---|---|---|---|
| Marformer annot J=10 | 1.0984 | 0.9646 | **0.9881** |
| Marformer annot J=20 | 1.0213 | 0.9067 | **0.9509** |
| Marformer item K=100 | 1.0966 | 0.9627 | **1.0026** |
| MarformerNT annot J=10 | 1.1433 | 0.9157 | **0.9665** |
| MarformerNT annot J=30 | 0.9003 | 0.8152 | **0.8292** |

At J=10 on the NT arm it recovers 78% of a 0.23-nat gap.

### Corrected: on the held-out cells alone it is much better than that

The numbers above are the *pooled* metric `test.py` reports, and that metric is
mostly training-partition cells (discrepancy **D22** -- 24% of it is genuinely
held-out at K=500, 2.9% at K=4000). Scoring on the 414 genuinely held-out cells
instead:

```
val-block vs last epoch, POOLED metric      -0.0343 nats   (28/38 cells better)
val-block vs last epoch, TEST cells only    -0.0635 nats   (35/38 cells better)
```

The pooled number understates the protocol by roughly half, and for a coherent
reason: early stopping costs a little on the training partition's own missing
cells -- more epochs do fit those better -- and gains a lot on the held-out
cells, which is where the overfitting actually shows. Pooling the two nets them
against each other.

On test cells alone the annotator axis is transformed:

| setting | no validation | **block validation** | delta |
|---|---|---|---|
| MarformerNT annot J=10 | 1.2515 | **0.9731** | **-0.278** |
| MarformerNT annot J=15 | 1.2085 | **0.9250** | **-0.284** |
| Marformer annot J=25 | 1.0793 | **0.9318** | -0.148 |
| Marformer annot J=30 | 1.0266 | **0.9143** | -0.112 |
| Marformer item K=100 | 1.0966 | **1.0026** | -0.094 |

### Where it does not work, and why

On the pooled metric ten cells got worse, clustered on the non-transductive item
axis between K=300 and K=700. **On test cells alone only three cells are worse,
and all three by 0.006 nats or less** -- K=2000 (+0.005), K=4000 (+0.000) and
non-transductive K=400 (+0.006), which are exactly the settings where there is
no overfitting to remove and validation correctly declines to act. The
"failure band" was an artifact of the pooled metric.

What survives is a milder version: at the large item sizes the protocol has
nothing to gain and adds a little variance. That is the right behaviour, not a
defect.

The one genuine fragility remains the width of the search: at MarformerNT
K=500 the three seeds' own argmins were 67, 226 and 82, on a validation set of
~146 target cells over ~60 held-out items. Pooling and smoothing absorbed it
here, but the margin is thin.

Two things would harden it, in order of cost: a one-standard-error rule (pick
the earliest epoch within 1 SE of the minimum, which formalises the flat-basin
handling already visible at K=400), and a larger validation fraction on the item
axis.

### The MCAR arm had a bug; it is re-running

The first MCAR run came out catastrophically worse (+0.278 nats, stopping at 39
epochs on average). That was not a property of MCAR validation, it was an
implementation error: the MCAR carve moves observed cells *out* of the observed
list, so `val_observed` is empty and the validation graph was nothing but
targets -- no context at all, a task on which the least-trained model wins.
`val_eval_vars` now scores MCAR cells against `train_observed`, which is the
context they actually came from. Verified on a 40-epoch run: the curve descends
like the block arm's. Re-running as `val-mcar-02`.

## Status

Runs in flight:

| Run | Protocol | Tasks |
|---|---|---|
| `val-block-01` | block | **complete** -- 114 search + 114 refit, evaluated |
| `val-mcar-02` | mcar | running (the first attempt hit the context bug above) |

Both over seeds 42/43/44 on the `replicate-01` data, so they are directly
comparable to `seeds-01`.

Early evidence, `Item_T_500` seed 42, block carve (34 of 229 training items held
out): validation bottoms out at **epoch 145** with CE 0.9708 and rises to 1.2084
by epoch 399 -- a **+0.238 nat** overfit gap, detected without touching test.
The test-argmin for the same cell is epoch 274, so validation stops
substantially earlier than test-set selection would. Whether that costs test
performance is the question the refit answers.
