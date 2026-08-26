# Why Marformer does not overfit the way heteroskedastic regression does

Answer to Jason's thread (2026-08-25). Everything below is from the current
code and from a fresh Domain 3 replication run (`RERUN/runs/replicate-01`),
trained on newly generated synthetic data rather than any cached results.

---

## Short answer

Jason asked four questions. Taking them in order:

0. **Do we overfit in the same way?** **Yes, at small training sizes — in both
   arms.** On Domain 3 synthetic at K = 100 the test curve rises 0.21 nats past
   its minimum (§4a). On LLM-Rubric at K = 10 it is far worse: validation
   log-loss 1.35 at its minimum, 3.69 by epoch 300, while accuracy keeps
   improving (§4b). The reason this is not visible in the paper is that the
   reported checkpoint is chosen after the fact — on test for the synthetic
   arm, on validation for the real one.
1. **Is it the masking procedure inflating the training sample?** Partly, but
   that is not the main thing.
2. **Is it weight decay?** Marginally.
3. **Is it early stopping on validation data?** In Domain 3 synthetic, no —
   there is no validation set, no early stopping, and no validation-based
   selection at all. On LLM-Rubric there *is*, and it is doing essentially all
   of the work.
4. **Which split is calibration measured on?** Test — but for the synthetic
   figures the *checkpoint was also chosen on test*, so the reported calibration
   is optimistic. The real-data (LLM-Rubric) figures do not have this problem.

The reason the failure mode is *weaker* in Domain 3 synthetic than on real data
is more structural than any of the first three: **in the item-generalization setting
the model has no per-item parameters to overfit with.** It cannot put mass on
the training $y_i$ because it has no way to represent "item $i$" — the item
deviation table is dropped on every training step and ends at exactly zero.
See §3. On LLM-Rubric that dropout is 0.7 rather than 1.0, items keep some
identity, and the failure mode returns.

There is one place where the model *does* put near-total mass on the training
values, and it is worth knowing about: §5.

---

## 1. There is no validation set, so options (3) is ruled out

The Domain 3 synthetic bundles contain only `train` and `test`. Counting the
`instance` field in a generated bundle:

| split | observed | missing |
|---|---|---|
| `Item_T_500` | train 1365, test 360 | train 1317, **test 414** |
| `Item_NT_500` | train 1707, test 360 | train 1677, test 414 |
| `Annot_T_30` | train 712, test 360 | train 701, test 414 |

There is no `val` key at all. (414 test-missing ratings is exactly the number
the Logit Lens appendix quotes, which is a good sign the regenerated data
matches what the paper used.)

Consequences, all confirmed in the run logs:

- `val_all` is empty, so every epoch prints `No val variables to evaluate on`
  and `val/missing_ce` is never logged.
- Lightning warns:
  `ModelCheckpoint(monitor='val/missing_ce') could not find the monitored key`.
- **No `best-*.ckpt` is ever written.** A finished run's `checkpoints/` holds
  `last.ckpt` and 16 `periodic-epoch=*.ckpt`, nothing else.
- Training therefore runs all 400 epochs with no early stopping, which matches
  Prabhav's recollection and contradicts the paper's current wording
  (`DISCREPANCIES.md` D2).

So Jason's third hypothesis is not what is protecting us — there is nothing
there to do the protecting.

By contrast **LLM-Rubric does have a validation split** (observed
`{train 6201, val 225, test 225}`), so there `best-*.ckpt` is genuine.

## 2. The masking procedure does enlarge the sample, but not by memorisable repetition

`MCARMasking.mask` draws `int(n * 0.15)` variables **without replacement**, so
exactly 15% are masked per step, freshly resampled every step (not an i.i.d.
coin flip — `DISCREPANCIES.md` D15). An "epoch" is `mask_augmentations = 5`
steps, so 400 epochs is 2000 distinct masks.

For `Item_T_100` the graph holds 360 variables, 54 masked per step. Each
observed variable is thus a prediction target roughly 2000 x 0.15 = 300 times,
and appears as *context* the other ~1700 times.

The important part is not the count but that **the conditioning set differs
every time**. The model never sees the same prediction problem twice: the same
$y$ must be predicted from many different subsets of its neighbours. Driving
the predictive entropy to zero would require being right from every context,
including ones that genuinely do not determine $y$.

## 3. The real reason: in item generalization there are no per-item parameters

This is the substantive answer.

`run_marformer_group.sh` sets, for the item axis:

```
ITEM_DROPOUT_RATE=1.0
ANNOTATOR_DROPOUT_RATE=0.0
```

and in `model.py::_build_initial_streams` the entity deviation is dropped with
that probability during training:

```python
if self.training and t.variation.dropout_rate > 0:
    if torch.rand(1).item() < t.variation.dropout_rate:
        dev = torch.zeros_like(dev)
```

`torch.rand(1) < 1.0` is always true, so **item deviations are zeroed on every
training step**. They start at zero (`torch.zeros`) and never receive gradient,
so they stay at exactly zero. Checked directly in a trained checkpoint
(`Item_T_500`, epoch 399):

```
model.deviation_tables.item        shape=(500, 75)  |w|max = 0.000000
model.deviation_tables.annotator   shape=(10, 75)   |w|max = 0.023612
model.deviation_tables.attribute   shape=(9, 75)    |w|max = 0.030682
```

Every item is represented by the *same* type centroid. The model has no
parameter that says "this is item 137", so the heteroskedastic-regression
failure — memorise $y_i$, then predict low $\sigma$ everywhere — is not
available to it. To predict a masked rating it has to use the other observed
ratings of that item in the same graph, which is exactly the test-time task.

On the annotator axis (`ANNOTATOR_DROPOUT_RATE=0.8`) the same pressure is
weaker but the outcome is similar: annotator deviations end at
`|w|max = 0.0015`, essentially unused.

**This also predicts where overfitting *should* appear.** Anything with
per-entity free parameters that are not dropped — attributes here, and any
future setting where item identity is kept (`item_dropout_rate = 0`) — is where
to look. Haojun's real-valued Gaussian experiments are worth checking on
exactly this axis: if the variance head reads off a per-entity embedding that
is never dropped, the mechanism from the paper is back in play.

## 4. Measured train/test gaps: no overfitting signature

From `runs/replicate-01`, final-epoch training masked CE against test
missing log-loss (both in nats, 4-way Likert, chance = 1.386):

| run | train masked CE | test log-loss | gap |
|---|---|---|---|
| `Item_T_100` | 0.697 | 0.932 | +0.235 |
| `Item_T_200` | 0.776 | 0.895 | +0.119 |
| `Item_T_300` | 0.809 | 0.868 | +0.060 |
| `Item_T_400` | 0.844 | 0.863 | +0.019 |
| `Item_T_500` | 0.809 | 0.830 | +0.021 |
| `Item_T_600` | 0.809 | 0.829 | +0.020 |
| `Item_T_700` | 0.828 | 0.823 | **-0.005** |
| `Item_T_800` | 0.866 | 0.816 | **-0.050** |
| `Annot_T_10` | 0.698 | 0.934 | +0.236 |
| `Annot_T_30` | 0.716 | 0.868 | +0.152 |
| `Annot_T_50` | 0.740 | 0.777 | +0.037 |

Three things to note.

- **Training CE never collapses.** It plateaus around 0.7-0.87 and *rises* with
  training-set size, which is what you expect from a model fitting a
  conditional distribution rather than memorising targets. There is no
  train-CE-to-zero behaviour anywhere.
- **The gap shrinks with data and goes negative** past ~700 items. Part of that
  is a genuine generalisation gain and part is an artefact: training CE is
  measured in `model.train()` mode, so attention dropout (0.1) and entity
  dropout are active, while test is measured in `eval()` mode. So the training
  number is slightly pessimistic and these gaps are, if anything, upper bounds.
- **These test numbers are the test-argmin over 16 checkpoints**, which is what
  the pipeline reports (§7). That hides the shape of the curve — see §4a, which
  is the more important table.

### 4a. But the test curve *does* turn back up at small K

The table above uses `best.json`, i.e. the best checkpoint as scored on test.
Looking at the full per-checkpoint test curve instead (from the 27 fully
evaluated ablation runs, variant `all`, seed 42):

| K_train | test log-loss by epoch (24 → 399) | min | final | acc at min → final |
|---|---|---|---|---|
| 100 | 1.298, 0.966, 0.933, 0.958, 0.972, 1.012, 1.054, 1.171 | **0.933** @124 | **1.138** | 0.560 → **0.488** |
| 300 | 1.360, 0.927, 0.901, 0.868, 0.879, 0.895, 0.895, 0.883 | 0.868 @174 | 0.884 | 0.585 → 0.597 |
| 500 | 1.389, 0.880, 0.846, 0.840, 0.836, 0.844, 0.846, 0.863 | 0.833 @149 | 0.844 | 0.623 → 0.615 |

**So Domain 3 synthetic does overfit at K = 100** — test log-loss rises 0.21
nats past its minimum, and accuracy *falls* from 0.560 to 0.488. That is plain
overfitting, degrading both the distribution and the point prediction. I had
this wrong initially: selecting on test flattens the curve to its minimum and
makes the problem invisible in the reported numbers.

At K = 300 and 500 the rise is small (+0.016, +0.011) and accuracy is flat or
improving, so those are close to converged rather than overfit.

Averaged over all 27 ablation runs, choosing the checkpoint on test rather than
taking the last one is worth:

| K_train | last | test-argmin | what selection buys |
|---|---|---|---|
| 100 | 1.1598 | 0.9641 | **+0.1957** |
| 300 | 0.9060 | 0.8778 | +0.0281 |
| 500 | 0.8643 | 0.8432 | +0.0211 |

At K = 100 that 0.196 nats is comparable to the entire reported margin between
Marformer and its baselines. This is D14 quantified, and it is the strongest
argument for fixing it.

## 4b. On real data at small K, we overfit *exactly* the way Jason describes

The synthetic results above make Marformer look immune. It is not. On
LLM-Rubric, where items keep their identity (`item_dropout_rate = 0.7`, not
1.0) and the training set can be genuinely tiny, the failure mode appears in
textbook form.

Validation log-loss over 300 epochs, `LLMRubric_225_25_9_10` (10 training
items):

| epoch (approx) | val log-loss | val accuracy |
|---|---|---|
| 20 | 1.52 | 0.23 |
| 40 | 1.73 | 0.39 |
| 120 | 1.64 | 0.43 |
| 180 | 2.32 | 0.41 |
| 240 | 2.64 | 0.36 |
| 300 | 3.69 | 0.39 |

Log-loss bottoms out around epoch 71 and then climbs to **2.7x chance**
(chance = 1.386), while **accuracy keeps improving**. That is precisely the
signature from the heteroskedastic paper: the point prediction is still getting
better while the predicted distribution becomes overconfident, so log-loss
blows up even as the argmax improves. It also answers Jason's parenthetical
directly — if we stop early, we are *not* stopping while the mean is still in
error; we are stopping while the mean is still improving and the variance has
already collapsed. The two are trading off against each other.

The effect decays with training size, exactly as Jason predicted. All ten
sizes, 300 epochs each:

| K_train | min val log-loss | epoch of min | final val log-loss | inflation |
|---|---|---|---|---|
| 10 | 1.350 | 71 | 3.693 | **+2.343** |
| 20 | 1.069 | 83 | 1.608 | +0.539 |
| 30 | 1.002 | 115 | 1.536 | +0.533 |
| 40 | 1.009 | 121 | 1.388 | +0.379 |
| 50 | 1.064 | 125 | 1.287 | +0.223 |
| 75 | 1.012 | 99 | 1.236 | +0.224 |
| 100 | 0.992 | 120 | 1.182 | +0.190 |
| 125 | 1.022 | 155 | 1.114 | +0.093 |
| 150 | 1.005 | 153 | 1.193 | +0.188 |
| 175 | 1.052 | 174 | 1.190 | +0.138 |

Note that the minimum sits around epoch 70-175 out of 300 at every size, so the
last third of training is actively harmful on this dataset regardless of K.

So the honest overall picture is:

- **Domain 3 synthetic**: mild at K >= 300, but real at K = 100 (§4a), where
  the test curve rises 0.21 nats past its minimum and accuracy falls. §3 makes
  it *harder* than it would otherwise be, not impossible. There is no
  validation set to stop on, so the run just trains through it and the reported
  number is rescued by test-set selection instead.
- **LLM-Rubric**: strong overfitting at small K, and the reported numbers are
  saved *entirely* by validation-based checkpoint selection. `best.json` there
  comes from the validation-selected checkpoint, so the paper's 1.52 at
  K_train = 10 is roughly the epoch-71 minimum, not the epoch-300 value of 3.69
  the model actually ends at.

This is not a replication artefact: the LLM-Rubric arm reproduces the paper's
reported numbers closely once evaluated correctly (see the note on D20 below) --
1.4743 vs 1.52 at K=10, 1.0494 vs 1.050 at K=50, 0.9775 vs 0.998 at K=175, with
the Unigram baseline matching to three decimals. So the overfitting above is a
property of the published setup, not of my rerun.

That is worth stating plainly in the paper. As written, the methods section
implies early stopping is a minor detail; on the real-data arm it is the
difference between 1.5 and 3.7 nats. It is also the reason the paper's own
text has Marformer *losing* to Stan CPM and Best Unigram at K_train = 10 —
that is the regime where this bites hardest.

## 5. The one place we *do* put a ton of mass on the training answer

The objective is not purely masked autoencoding. `compute_trainable_loss`
optimises

```
15.0 * masked_loss  +  1.0 * observed_loss
```

and the observed term asks the model to predict tokens **whose values are
visible in its own input**. That is a copy task, and the model solves it
essentially perfectly:

| run | final `train_observed_ce` |
|---|---|
| `Item_T_100` | 0.0028 |
| `Item_T_500` | 0.0002 |
| `Annot_T_30` | 0.0002 |

0.0002 nats is as close to a point mass as makes no difference. So the honest
statement is: *the model does put near-total probability on the training
values it can see* — but through a copying channel keyed on the observed value
in the input, not by memorising which $y$ went with which entity. It is a
different mechanism from the one in the heteroskedastic paper.

Whether it is harmless is a separate question, and I do not think it is
obviously harmless: it is a standing gradient pressure toward low predictive
entropy, applied at every step, on 85% of the tokens. It is worth an ablation
at `observed_loss_weight = 0` to see whether calibration changes. Logged as
`DISCREPANCIES.md` D16. The paper currently describes the objective as masked
autoencoding and does not mention this term.

## 6. Weight decay

`weight_decay = 0.01` on AdamW over a 453K-parameter model. It is doing
something — the near-zero annotator deviations in §3 are partly its work — but
it is not the load-bearing mechanism. The item-axis result holds with the
deviations *structurally* removed, independent of any penalty.

## 7. Which data measures calibration — and a problem

Metrics are computed on the **test** split (`test.py` calls
`evaluate_entity_marformer_split(..., split="test")`). So Jason's instinct in
the last message is right that this is what we want.

But for the synthetic figures the **checkpoint is also selected on test**:

- `test.py` scores *every* checkpoint on test;
- `run_marformer_eval.sh` passes `--checkpoint all`, sorts by
  `missing.log_loss`, and writes the winner as `best.json`;
- `plot_domain3_sigmoid.py` re-does the same selection if `best.json` is absent.

Every plotted Marformer point is a **min over 16 checkpoints of test
log-loss**. In our replication the selected epoch ranges from 74 to 299 and
varies non-monotonically with training size, which is what selection noise
looks like. The Stan baselines get no equivalent per-size selection, so the
comparison is tilted, and the smECE numbers quoted in the text inherit the same
optimism.

**LLM-Rubric does not have this problem** — there `best.json` comes from the
validation-selected checkpoint. So the two halves of the paper's Domain 3
results are not selected the same way. This is `DISCREPANCIES.md` D14 and it is
the one thing here I would fix before the paper goes out; it needs no
retraining, only re-running eval with validation selection.

---

## What I would do next

1. **Fix the selection** (D14): report the validation-selected checkpoint, or
   a fixed epoch, for the synthetic figures. Cheap — checkpoints are on disk.
   This requires adding a validation split to the synthetic generator, which
   currently produces none.
2. **Ablate `observed_loss_weight`** (§5, D16): 0 vs 1, watching smECE rather
   than log-loss.
3. **Check Haojun's Gaussian setting on the §3 axis**: does the variance head
   depend on a per-entity embedding that survives dropout? That is where the
   paper's failure mode would actually bite.
4. If we want the heteroskedastic paper's problem to be *impossible* rather
   than *unavailable*, the entity-dropout trick generalises: it is a structural
   way of denying the model the parameters it would need to be overconfident,
   and it is not Gaussian-specific.
