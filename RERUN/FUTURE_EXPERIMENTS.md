# Future experiments

This document is organised in two parts.

**Part 1 — side-branch experiments.** These revisit assumptions baked into the
current pipeline and try to invalidate them or simplify away things we may not
need. Each suite is standalone: run it on a feature branch, keep the history
there, and revert the branch when finished. What comes back to `main` is not the
code but the *takeaway*, recorded here. Runs and result files stay untracked.
Each suite below states what to run, and what the resulting change to the main
experiment would be under each outcome.

**Part 2 — the final rerun checklist.** Before we kick off the definitive
end-to-end run, everything we played with in Part 1 gets finalised, and anything
not-well-designed gets replaced with a standard version. The bar for folding a
Part 1 result into the final run: we should be confident the rerun will be at
least as good as the current one. Part 2 tracks those changes and their status.

Companion documents: [DISCREPANCIES.md](DISCREPANCIES.md) (paper ↔ code
mismatches) and [FIGURES.md](FIGURES.md) (figure → script map).

---



# Part 0 — execution plan

Written 2026-09-04, before any suite was run. This section is the plan; each
suite in Part 1 gets a **Result** block appended as it finishes.


## The measurement protocol (identical for every suite)

Every suite below is a small-effect comparison, and Domain 3's noise is large
enough to manufacture any conclusion we like if we are careless. So the rules
are fixed once, here, and no suite deviates.

1. **Metric.** Corner log-loss, `corner_whole=true`. Never the pooled metric
   (that is D22), never accuracy.

2. **No selection.** Arms are compared at a *fixed* epoch (400), the same for
   every arm. This removes D14 from the comparison entirely: nothing is chosen
   on the test corner, so the corner is an honest held-out number. Suites that
   are *about* stopping (S7, and S3's schedule arm) additionally report the
   validation-selected refit, and say which is which.

3. **Pairing.** Arms share the same (data draw, model seed) pairs, and we report
   the *paired* mean difference. Unpaired comparison is hopeless here — see the
   noise floor below.

4. **Replication.** 2 data draws x 3 model seeds = 6 paired samples per
   (arm, K), and at least 2 values of K. n = 12 paired samples per arm.

### Noise floor, measured (not assumed)

Computed 2026-09-04 over the 328 existing corner-scored evaluation rows:

| source of variation | sd (nats, corner log-loss) |
| --- | --- |
| generative data draw, fixed K/arm/protocol (D28) | **0.085** |
| model seed, fixed draw (64 groups, n>=3) | 0.014 median, 0.023 mean |
| paired difference, same checkpoint, eval knob changed | 0.012 |
| **paired difference between two ARMS, same draw/seed/K** | **0.035** median, 0.058 mean |

**The last row is the one that matters, and it is the one that was nearly got
wrong.** The first three describe re-scoring the same weights. Comparing two
*separately trained* arms is a different and much noisier measurement, because
each arm lands at its own point in weight space. Measured on the residual
ablation (`resid-01/02/03`, 6 arms x 3 model seeds x 3 sizes), the paired
between-arm difference has sd:

| K | paired sd across model seeds |
| --- | --- |
| 100 | 0.012 - **0.193** |
| 300 | **0.015 - 0.048** |
| 500 | 0.032 - 0.039 |

K=100 is unusably noisy; K=300 is the quietest size measured, and is also the
cheapest of the three. Sizes below 300 should not be used for arm comparison at
all.

**What this implies for sample size.** With a paired sd of 0.035, resolving a
0.015-nat effect at two standard errors needs **n = 22 per arm**; at the mean sd
of 0.058 it needs n = 61. The original plan of n = 12 would have returned
"inconclusive" for every suite while spending 25 GPU-hours to do it -- too broad
and too shallow at the same time.

**Revised decision rule.** The threshold moves to **0.03 nats**, which n = 6
resolves at two standard errors (SE = 0.014). This is not a weakening: every
Part 1 suite asks *"is the simpler configuration acceptable?"*, which is a
one-sided tolerance test, not the detection of a small effect. "Within 0.03
nats of the shipped setting" is a sufficient licence to adopt the simpler
option, and 0.03 on a ~0.85 baseline is 3.5%. Where a suite genuinely needs
0.015 resolution, it needs n >= 22 and should be costed as such rather than run
underpowered.


### Shortlisting: how the sweep was cut from 232 runs to 56

Revised 2026-09-04 after the between-arm sd was measured and the GPU queues
filled up. The original Wave 2 plan queued 232 runs (~25 GPU-hours); the
shortlist is **56 runs (~2 GPU-hours)** and answers the same questions. Nothing
had started when it was cancelled, so nothing was wasted. Five levers, in
descending order of what they saved:

1. **One screening size, K=300, instead of {300, 1000}.** K=1000 costs 3.5x
   K=300 per run. The residual-ablation data shows arm effects *shrink* as K
   grows (`inner=none` vs `all`: +0.236 at K=100, +0.115 at K=300, +0.023 at
   K=500), so K=300 is where a difference is visible if there is one, and it is
   simultaneously the quietest size measured. An arm that is indistinguishable
   at K=300 is very unlikely to separate at K=1000. Confirmation at the reported
   sizes happens once, for the winner, not for every arm.

2. **Drop arms the document itself marks conditional.** S2's delta=1.0 on the
   annotator axis is explicitly "if 0.7 fails"; S5's lr-scaled arms are
   explicitly the fallback if the same-lr arm fails. Running a fallback before
   knowing whether it is needed is pure cost. -18 runs.

3. **S5 becomes a trajectory-overlay test, n=1 per seed instead of n=6.**
   The claim under test is that AdamW makes a global loss rescale a near-no-op.
   That predicts more than "the two arms land in the same place" -- it predicts
   the two runs *track each other epoch by epoch* from the same seed. A
   trajectory that overlays is far more evidence than a landing point that
   matches, and it needs 2 runs, not 12. 48 runs -> 4.

4. **S2's item axis drops the NT arm for screening.** The transductive arm is
   what Figure 6 mostly reports; if delta=0.7 matches 1.0 there, the NT check is
   a confirmation rather than a discovery. 72 -> 18.

5. **S1 stays at n=2 per arm, on purpose.** S1's question is not "do these two
   arms differ by 0.03" but "does feature-only norm collapse the *spread* of the
   four residual variants", and that spread is ~0.1 nats wide at K=300 -- an
   order of magnitude above the noise. n=2 resolves a 0.1-nat question fine. It
   would not resolve a 0.03-nat one, and the write-up must not claim it does.

**The general principle**: match precision to the question. Three of these five
suites are asking a coarse question (does this collapse / is this a no-op / is
this within tolerance) and were about to be run at fine-question cost.

### Free precision: averaging the tail checkpoints works (1.8x, and 2.1x at K=300)

Job 300293 scored all 16 periodic checkpoints of every `resid-01` run (3 arms x
3 seeds x 3 sizes) on the corner. Paired between-arm sd across model seeds, as
a function of how many tail checkpoints are averaged:

| checkpoints averaged | K=100 | K=300 | K=500 | mean |
| --- | --- | --- | --- | --- |
| 1 (the final one) | 0.0721 | 0.0336 | 0.0376 | 0.0478 |
| 3 | 0.0304 | 0.0193 | 0.0395 | 0.0297 |
| **5** | 0.0253 | **0.0160** | 0.0368 | **0.0260** |
| 8 | 0.0268 | 0.0106 | 0.0378 | 0.0251 |

**Averaging the last 5 periodic checkpoints (epochs ~300-400) cuts the paired sd
by 1.8x overall and 2.1x at K=300**, for no GPU at all -- variance scales as
1/n, so that is worth roughly **4x the replicates**. Most of the seed-to-seed
spread really is SGD endpoint jitter rather than a real difference in where the
arms converge.

**This restores the tighter threshold at the shortlist's cost.** With a paired
sd of 0.016 at K=300, n = 6 resolves **0.015 nats** at two standard errors, not
just 0.03. The screening wave is therefore no less sensitive than the 232-run
plan would have been, at a twelfth of the GPU.

**Two limits, both of which matter:**

1. **It only works where the corner curve has flattened.** The K=4000
   transductive run is still descending at epoch 399 (see the large-K findings
   below), so averaging its tail averages over a trend and biases the number
   upward. Use `--tail 1` at K=4000 until a longer run shows it has converged.
   `analyse_arms.py` documents this at the point of use.
2. **It does not help at K=500** (0.0376 -> 0.0368). Whatever the noise is
   there, it is not endpoint jitter. Do not assume the 2x carries to every size.

Note this changes the *estimand* slightly: "mean corner loss over epochs
300-400" rather than "corner loss at epoch 400". For comparing arms that is
fine and arguably a better summary of where an arm converges. The **reported**
figure still uses the protocol's selected checkpoint -- tail-averaging is a
measurement tool for design decisions, not a change to what gets published.

### Cost model (mi300, 400 epochs, measured from `runtime-01`)

| K | 300 | 500 | 1000 | 2000 | 4000 |
| --- | --- | --- | --- | --- | --- |
| train (s) | ~140 | ~230 | 446 | 1003 | 2060 |

So a 12-run arm at K in {300, 1000} costs about 1 GPU-hour. All of Part 1 is
well under 100 GPU-hours; the constraint is queue latency, not compute.


## Branch policy

`patches` stays shippable. Three kinds of change, handled differently:

- **New knobs that are pure generalisations** (make an already-hardcoded
  constant env-overridable, defaults unchanged) land directly on `patches`.
  They are the same class as the architecture knobs already there, they cannot
  change any existing result, and every suite needs them. This covers S2
  (`delta_tau`), S5 (loss weights), and S3's schedule selector.
- **New behaviour behind a default-off flag** gets its own branch and its own
  git worktree, so no job ever has its scripts edited underneath it. One
  worktree per suite, created when the suite starts, deleted when it ends.
  This covers **S1** (`side/s1-featnorm`), **S7** (`side/s7-carve`), and
  **S3**'s WSD scheduler (`side/s3-wsd`).
- **Nothing else.** Result files and run trees stay untracked. What comes back
  to `patches` is the winning flag plus the Result block in this document.

Each side worktree symlinks its `RERUN/runs` at the main tree's, so the six
existing data draws (s42, s101-s105) are reused rather than regenerated, and
each suite writes under its own `RUN_ID`.


## Two large-K findings that came free (2026-09-04)

Both from existing periodic-checkpoint data, no new runs.

**1. Wider TRAINING chunks do not help at large K -- the plateau is not context
starvation.** Training chunk 600 vs 300, evaluation chunk held fixed, 28 paired
cells per size:

| K | train chunk 600 vs 300 |
| --- | --- |
| 1000 | -0.0056 +/-0.0035 |
| 4000 | **+0.0094 +/-0.0049** (worse) |

So the obvious mechanistic story for the plateau -- "each prediction only ever
conditions on 299 other items, so effective context stops growing with K" -- is
wrong, or at least is not fixed by widening the training chunk. Note the
asymmetry with S4: more context at *evaluation* helps at large K (-0.018),
more at *training* does not. That rules out one hypothesis for free.

**2. The K=4000 transductive run is still improving when the budget ends.**
Corner log-loss at the final periodic checkpoints, two independent runs:

| K (T arm) | epoch of corner minimum | trend over last 5 checkpoints |
| --- | --- | --- |
| 1000 | 149 | -0.0003 (flat) |
| 2000 | 124 | +0.0040 (rising) |
| **4000** | **399 = the last one** | **-0.017 (still descending)** |

K=1000 and K=2000 converge well inside the budget; K=4000 does not. **The
"Marformer plateaus after K~1000 while Stan keeps improving" story may be
partly an epoch-budget artifact**: 400 epochs is enough at the sizes where the
curve flattens and not enough at the size where it would keep going. Stan is
run to convergence; Marformer is cut off. This is the cheapest available
candidate for closing the gap in the limit and it is now Wave 3a.


## Waves

Ordered by what blocks what.

### Wave 0 — free (no training at all)

| task | cost | status |
| --- | --- | --- |
| Noise-floor calibration from existing rows | 0 | **done**, table above |
| **S4**: re-score the `seeds-01` checkpoints at eval chunk 100/200/300/400/500/600 | ~1 CPU-hour | **done**, 684 rows |

S4 needs no training: it is the same checkpoints scored under a different
evaluation chunking, and `grid_eval.py` already sweeps that axis. Spending a
training run on it would be waste.

### Wave 1 — S7, which gates the rest

S7 is first because F4, F5 and F6 all wait on it, and because every other
suite's "and here is the validation-selected version" line is only worth
printing if the carve is trustworthy. Branch `side/s7-carve`.

### Wave 2 — three independent suites, run concurrently

**S1** (`side/s1-featnorm`), **S2** and **S5** (both on `patches`, config-only
once the knobs are overridable). None depends on another.

### Wave 3 — the large-K wave: what could close the Stan gap

Waves 1 and 2 screen at K=300, which is where arm differences are largest and
cheapest to see. But an arm that is neutral at K=300 and helpful at K=4000 is
exactly what would close the Stan gap, and a K=300 screen cannot see it by
construction. So Wave 3 runs a deliberately small set of arms at K=1000 and
K=4000, chosen by mechanism rather than by covering the grid.

| # | arm | question | runs |
| --- | --- | --- | --- |
| 3a | `epochs=1200` at K=4000, T, 3 seeds | Is the plateau an epoch-budget artifact? See finding 2 above. One arm suffices: at constant lr the 400-epoch run is an exact prefix of the 1200-epoch run at the same seed, so the periodic checkpoints give both budgets from one run. | 3 |
| 3b | `delta_item` in {0.0, 0.7, 1.0} at K in {1000, 4000}, 2 seeds | On the item axis delta_item=1.0 *always* drops the per-item embedding, so the model is forced to be fully amortised and can never use per-item parameters. Stan learns per-item parameters and keeps improving past K=1000. If that discarded channel is the plateau, lowering delta helps at large K and not at small K. | 12 |
| 3c | WSD at K=4000 (`side/s3-wsd`) | Only worth running if 3a/3d say length matters -- a decay phase is how you spend a longer budget well. Held until 3d reports. | 0 for now |
| 3d | **epoch budget as a function of K** -- see below | Is the 400-epoch budget simply too small at large K, and can *early stopping* find the better epoch rather than the corner being read directly? | 12 |


### Wave 3b result (2026-09-05) — delta_item is NOT the plateau; hypothesis refuted

Run `w3delta`, 12 runs (3 arms x 2 model seeds x K in {1000, 4000}, draw s101,
T arm), paired against the shipped delta_item = 1.0. Reported at `--tail 1`
because the K=4000 curve has not flattened; the `--tail 5` column is shown too
and agrees.

| arm | K=1000 (tail 1) | K=4000 (tail 1) | K=4000 (tail 5) |
| --- | --- | --- | --- |
| delta_item = 0.7 | +0.0001 +/-0.0006 | **-0.0008 +/-0.0004** | -0.0005 +/-0.0003 |
| delta_item = 0.0 | +0.0121 +/-0.0101 | +0.0033 +/-0.0052 | -0.0001 +/-0.0001 |

**The hypothesis this wave was built to test is dead.** The idea was that
delta_item = 1.0 always drops the per-item embedding, so the model is forced to
be fully amortised and can never use per-item parameters -- while Stan learns
per-item parameters and keeps improving past K=1000. If that discarded channel
were the plateau, lowering delta would help at K=4000 and not at K=300. It does
neither: every arm is within 0.003 nats of every other at K=4000, two orders of
magnitude short of the 0.11-nat gap.

**In hindsight the mechanism explains itself.** On the item axis the *test*
items are new, so their deviation embeddings are untrained (zero) no matter what
delta was during training -- at test time the model is always operating in the
delta=1 regime. Training with delta<1 only changes how *training* items are
represented internally, which is not the bottleneck.

**Two things this settles beyond the Stan question:**

1. **delta_item = 0.7 is adoptable on the item axis at every size tested**
   (K=300, 1000, 4000), which is what S2 needed.
2. **delta_tau is close to inert on the item axis.** Turning it off entirely
   costs 0.006 nats at K=300 and nothing measurable at K=4000. The paper
   presents delta_tau as a substantive mechanism; on this axis it is not, and
   the text should say so rather than implying the value was carefully chosen.
   The annotator axis may still differ -- `s2annot` is the arm that would show
   it, and the NT arm (where items really are unseen and unseen-ness is the
   whole question) has not been checked.

**Remaining live candidates for the Stan gap:** the epoch budget (Wave 3d,
running) and **capacity**, which is still completely untested.

### Wave 3d — the epoch budget, and whether early stopping can validate it

**This is now the most promising open lead on the Stan gap**, so it gets its own
section. The evidence that opened it (all from existing checkpoints, no new
compute):

| K (T arm) | epoch of corner minimum | trend over last 5 checkpoints |
| --- | --- | --- |
| 1000 | 149 | -0.0003 (flat) |
| 2000 | 124 | +0.0040 (rising) |
| **4000** | **399 = the last one** | **-0.017 (still descending)** |

400 epochs is comfortably enough at K<=2000 and demonstrably not enough at
K=4000. Stan is run to convergence; Marformer is cut off. If the budget is the
binding constraint at large K, then the reported plateau is partly an artifact
of a hyperparameter nobody revisited when the sizes grew.

**Two arms, because the honest version of the claim needs both.**

| arm | run id | what it establishes |
| --- | --- | --- |
| full data, 1200 epochs, K in {1000, 2000, 4000}, 3 seeds | `w3long` (K=4000, done), `w3budget` (K=1000/2000) | Where the corner *actually* bottoms out as a function of K. This is a measurement, not a protocol -- reading the stopping epoch off it would be D14 all over again. |
| block carve, 2000 epochs, K in {1000, 2000, 4000}, 2 seeds | `w3budgetval` | Whether **early stopping selects the later epoch on its own**, without seeing the corner. Longer budget than the full-data arm because the carve removes 15% of the training data, which pushes the optimum later. |

One long run per (K, seed) suffices for both: at constant lr the periodic
checkpoints of a 1200- or 2000-epoch run contain every shorter budget for that
seed exactly, so the budget curve comes free rather than needing an arm per
budget.

### Wave 3d result, criterion 1 (2026-09-05) — MET. The K=4000 budget is real, and worth ~0.02 nats

Run `w3long`: K=4000, T arm, full training data, **1200 epochs**, 3 model
seeds, draw s101. Mean corner log-loss across seeds:

| epoch | 99 | 199 | 399 | 599 | 799 | 999 | 1099 | 1199 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| corner | 0.7601 | 0.7531 | **0.7323** | 0.7196 | 0.7127 | 0.7115 | **0.7061** | 0.7145 |

Three ways to read the gain, in decreasing order of honesty:

| comparison | gain vs epoch 400 | contaminated? |
| --- | --- | --- |
| fixed budget of 1200 epochs, chosen in advance | **-0.018** | no -- this is the defensible number |
| best periodic checkpoint (849 / 924 / 1199 by seed) | -0.034 | **yes**, this is an argmin on the corner = D14 |

**Criterion 1 is met on the honest number** (-0.018 < -0.015), and the D14
figure sets an upper bound of about -0.034 on what a perfect stopping rule
could recover. The curve descends steadily and monotonically in the mean from
epoch 400 to ~1100 before turning up, so this is a trend, not a lucky
checkpoint.

**What it does and does not buy.** Stan Proj at K=4000 on this draw is 0.619;
Marformer T was 0.731 at 400 epochs and is 0.715 at 1200. So the budget closes
the gap from **0.112 to about 0.096 -- roughly 15% of it.** This is a real
effect that is worth having and worth correcting in the paper, and it is *not*
the explanation for the Stan gap. Anyone reading this section later should not
come away thinking the plateau has been explained; 85% of it is still
unaccounted for, and **capacity remains the only untested candidate**.

**Consequence for the paper regardless of the final figure:** the published
large-K Marformer points are **undertrained by ~0.02 nats**, because a
400-epoch budget chosen when the sizes were small was never revisited when they
grew. That is a correction to state, not just a setting to change.

Criterion 2 -- can early stopping find this epoch *without seeing the corner* --
is `w3budgetval`, still running. Until it reports, the honest statement is "the
headroom exists"; the claim that our protocol can capture it is not yet earned.

### Wave 3d, second data point (2026-09-05) — the budget must SCALE, not simply rise

`w3budget`, K=1000, T arm, full data, 1200 epochs, seed 42 (1 of 3 seeds so far):

| epoch | 99 | 299 | 399 | 499 | 699 | 899 | 1099 | 1199 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| corner | 0.7608 | 0.7374 | **0.7401** | 0.7268 | 0.7464 | 0.8072 | 0.8097 | 0.8452 |

**At K=1000 a fixed 1200-epoch budget is worse than 400 by +0.105 nats.** The
corner bottoms out around epoch 500-670 and then degrades badly. Compare the
same arm at K=4000, where 1200 epochs is *better* than 400 by -0.018.

**This inverts the obvious fix and is the most important thing W3d has produced
so far.** "The budget is too small, raise it" would be actively harmful: setting
1200 epochs everywhere would cost 0.105 nats at K=1000 to gain 0.018 at K=4000.
The correct statement is that **a flat budget is wrong in both directions** --
too long at K=1000, too short at K=4000 -- and the only safe way to use a larger
budget is to pair it with a stopping rule that actually stops.

**So criterion 2 is not a nicety, it is the whole mechanism.** The plan changes
from "raise the epoch budget" to:

> raise the epoch **cap** to something generous (1200-2000), and let the
> validation carve choose where to stop within it, per size.

That only works if the carve reliably stops early at small K and late at large
K. S7 gives grounds for optimism -- on fresh draws the carve tracks the corner
optimum to within 3-50 epochs and errs conservative -- but S7 tested it inside a
400-epoch budget, where there was little room to run away. `w3budgetval` tests
it inside 2000, which is the regime that matters and a genuinely harder test.

**Revised success criterion for F13** (superseding the two-part one below, which
assumed the fix was a bigger flat number):

1. the carve's selected epoch **grows with K** -- materially past 400 at
   K=4000, and materially *below* 1200 at K=1000; and
2. refitting at the selected epoch beats the flat-400 baseline at K=4000
   **without losing at K=1000**.

If the carve cannot do this, the honest outcome is to keep the flat 400 and
state in the paper that the large-K points are budget-limited -- which is a
weaker but defensible position, and much better than shipping a uniform 1200
that quietly wrecks the middle of the curve.

**Second seed confirms it, and sharpens the picture (2026-09-05).**

| K | seed | corner @400 | corner @ best epoch | corner @1200 | best epoch |
| --- | --- | --- | --- | --- | --- |
| 1000 | 42 | 0.7401 | 0.7241 | 0.8452 | 674 |
| 1000 | 43 | 0.7304 | 0.7087 | 0.7918 | 749 |
| 4000 | 42-44 (mean) | 0.7323 | 0.6980 | 0.7145 | 849-1199 |

The flat-1200 penalty at K=1000 reproduces: +0.105 and +0.061, mean **+0.083**.

**But the more useful reading is the middle column, and it revises what I said
above.** At K=1000 the corner does *not* simply flatten after epoch 400 -- it
keeps descending to about epoch 700 (0.735 -> 0.716, **-0.019**) and only then
collapses. An earlier note in this document said K=1000 "bottoms out around
epoch 149 and flattens"; that was read off a 400-epoch run, where the tail is
noisy and the continued descent is not visible. With a 1200-epoch budget the
optimum is clearly at ~700.

So the corrected summary is:

| K | optimal epoch | gain over flat-400 at the optimum | cost of a flat 1200 |
| --- | --- | --- | --- |
| 1000 | ~700 | **-0.019** | +0.083 |
| 4000 | ~1100 | **-0.034** | -0.018 |

**Both sizes are undertrained at 400, and both are wrecked by a uniform cap.**
That is a stronger case for the raise-the-cap-and-stop-properly plan than the
K=4000 result alone made: it is worth ~0.02-0.03 across the whole large-K range,
not just at the endpoint. It also makes the stopping rule non-optional, since
the epoch that is optimal at K=4000 is catastrophic at K=1000.

**Still one draw (s101) and two seeds at K=1000.** The direction is unambiguous
(4 of 4 runs) but the optimal epochs, 674 / 749 / 849 / 924 / 1199, are pinned
only to within a couple of hundred epochs -- which is fine, because the point is
that a stopping rule has to choose them, not that we hardcode them.

### Wave 3d, criterion 2 — first data point (2026-09-05, K=1000 seed 42)

| K | seed | cap | carve-selected epoch | val CE at the pick | val CE at epoch 2000 |
| --- | --- | --- | --- | --- | --- |
| 1000 | 42 | 2000 | **545** | 0.8426 | 1.6119 |

Full-data corner optimum on the same draw and seed: **674**.

Three things this one run establishes, none of which are the whole criterion:

1. **The carve does not run away.** Given four times the budget it needed, it
   selected epoch 545, not 2000. The validation curve nearly doubles from 0.843
   to 1.612 by the end of the cap, so the overfitting the corner also sees is
   plainly visible to the carve. This was the main risk of raising the cap.
2. **It selects past 400**, so it would in fact recover part of the headroom the
   flat budget is leaving on the table.
3. **It errs conservative again** -- 545 against a corner optimum of 674, i.e.
   129 epochs early. That matches S7's finding on fresh draws inside a 400-epoch
   budget, and it is the right direction to be wrong in. Part of the gap is
   expected rather than error: the carve trains on 85% of the data, so its own
   optimum genuinely arrives earlier than the full-data optimum does.

**Both K=1000 seeds now in, and the news is worse than the first run suggested.**

| K | seed | carve picked | full-data corner optimum | corner at the carve's epoch vs at 400 |
| --- | --- | --- | --- | --- |
| 1000 | 42 | 545 | 674 | **+0.0135** |
| 1000 | 43 | 412 | 749 | +0.0037 |

(Corner values are a local 3-checkpoint average around each target epoch. On raw
single checkpoints the gap reads +0.0357 / -0.0019, mean +0.0169; roughly half of
that is the checkpoint-to-checkpoint wobble the tail-averaging result already
documented, so the smoothed numbers are the fair ones.)

**At K=1000 the carve does not capture the headroom that is there.** The corner
optimum sits at 674-749 and is worth -0.019 against a flat 400; the carve picks
412-545, and the corner at its pick is **+0.009 worse than simply stopping at
400**. So on this size the stopping rule is not merely conservative -- it is
currently adding nothing, and if anything costing a little.

Two candidate reasons, not yet separated:

1. **The carve trains on 85% of the data**, so its own optimum genuinely arrives
   earlier than the full-data optimum. Some of the 130-340 epoch gap is correct
   behaviour, not error, and the refit-on-full-data step is supposed to absorb
   it -- but there is no reason the *ratio* should be constant across sizes.
2. **The corner curve at K=1000 is not smooth.** It descends to ~700 through a
   local bump around 500-550, which is precisely where the carve stopped on seed
   42. A selector reading a different (validation) curve has no way to know the
   corner has a second, lower minimum later.

**What this does and does not decide.** Criterion 2 has two parts: the selected
epoch must *grow with K*, and the refit must beat flat-400 at K=4000 *without
losing at K=1000*. The second part is now marginal at K=1000 (+0.009, n=2). The
first part is untested -- both points so far are the same size. **K=4000 is
where the flat budget is unambiguously too short, and it is the run that
decides whether the rule is usable at all.**

**This is one run at one size, and the criterion is about how the selected epoch
CHANGES with K.** A single point cannot show growth. The K=4000 runs are what
decide it; if they also select ~545 then the carve is not tracking size and
criterion 2 fails despite this encouraging start.

**Success criterion, stated in advance so it cannot be moved afterwards.**
The budget change is adopted for the final run only if **both** hold:

1. the full-data corner curve at K=4000 keeps descending materially past epoch
   400 (more than 0.015 nats below the epoch-400 value), **and**
2. the carve's validated stopping epoch at K=4000 lands materially past 400 --
   i.e. early stopping finds it *without being shown the corner*.

If (1) holds but (2) does not, the budget is real but our protocol cannot claim
it honestly, and that becomes an S7 problem rather than a budget one.

**If both hold, F13 lands: the final run's epoch budget becomes a function of
K** (fit from the validated epochs) instead of a flat 400 at every size, and
every large-K point in Figure 6 is re-run under it. Note this would also mean
the *published* large-K Marformer points are undertrained, which is a
correction to make in the paper regardless of what the final figure shows.

### Wave 3d, the oracle budget curve (2026-09-05) — the optimum SCALES with K

`w3budget-s101`, 6/6 runs: K=1000 and K=2000, 3 model seeds each, 1200-epoch
cap with periodic checkpoints, full training partition (no carve). Every
checkpoint scored on the corner, so this is the *oracle* epoch -- the best a
budget rule could possibly do, not what any protocol can honestly deliver.

| K | seed | corner @400 | oracle epoch | corner there | oracle gain vs flat 400 |
| --- | --- | --- | --- | --- | --- |
| 1000 | 42 | 0.7401 | 674 | 0.7241 | -0.0159 |
| 1000 | 43 | 0.7304 | 749 | 0.7087 | -0.0217 |
| 1000 | 44 | 0.7443 | 524 | 0.7200 | -0.0243 |
| 2000 | 42 | 0.7534 | 949 | 0.7223 | -0.0311 |
| 2000 | 43 | 0.7314 | 799 | 0.7045 | -0.0269 |
| 2000 | 44 | 0.7189 | 599 | 0.7108 | -0.0082 |

Mean oracle epoch **649 at K=1000, 782 at K=2000**, against ~1100 at K=4000
from the long-run arm. Mean raw-argmin gain over a flat 400: **-0.021, -0.022,
-0.034** respectively.

**Half of that is winner's curse, and the corrected number is what matters.**
The corner curve is noisy checkpoint to checkpoint -- paired sd of adjacent
checkpoints is 0.009-0.018, i.e. the *same order as the gain being claimed* --
so taking a raw argmin over ~48 checkpoints is picking the low point of a noisy
curve and reporting the noise as signal. Smoothing each curve with a 5-checkpoint
moving average before taking the argmin gives the defensible version:

| K | smoothed optimum epoch (per seed) | mean | smoothed gain vs flat 400 (per seed) | mean |
| --- | --- | --- | --- | --- |
| 1000 | 649 / 699 / 499 | 616 | -0.0099 / -0.0093 / -0.0097 | **-0.0096** |
| 2000 | 624 / 749 / 549 | 641 | -0.0065 / -0.0256 / -0.0140 | **-0.0154** |
| 4000 | 849 / 749 / 1199 | 932 | -0.0240 / -0.0312 / -0.0318 | **-0.0290** |

At K=1000 the raw gain of -0.021 is really **-0.010**; at K=2000, -0.022 is
really **-0.015**. K=4000 barely moves (-0.034 -> **-0.029**), which is itself
informative: the larger the training set, the smoother the curve (checkpoint sd
falls from 0.018 to 0.008) and the more of the apparent gain is real.

**The corrected picture is still the F13 picture, and is cleaner than the raw
one.** The headroom grows monotonically with K -- 0.010, 0.015, 0.029 -- and at
K=1000 it is now *below* the 0.015 decision threshold this document uses
everywhere else. So the budget matters at large K and is close to irrelevant at
small K, which is exactly the shape a K-dependent rule should have. Any earlier
sentence in this document quoting -0.021 or -0.022 as the K=1000/2000 headroom
is superseded by this table.

**Two things this settles.**

*The premise of F13 is sound.* The optimal epoch is not a constant that we
happened to set too low -- it **grows monotonically with K** across three sizes
(649 -> 782 -> ~1100), and the payoff for reaching it grows too. A flat budget
is the wrong shape of rule at any value, which is why raising 400 to a flat 1200
made K=1000 *worse* (the inversion recorded above) while helping K=4000.

*A flat long budget is not a fallback.* At epoch 1200 the K=1000 runs sit at
0.8452 / 0.7918 / 0.8727 -- 0.08 to 0.15 nats **worse** than their own optimum,
and worse than flat-400. The late curve does not plateau, it climbs. Whatever
budget rule ships must stop, and stop at a K-dependent place.

**What this does NOT settle, and the reason criterion 2 still gates F13.** Every
epoch in the table is an argmin taken on the corner, i.e. exactly the D14 error
we corrected elsewhere in this document. These numbers are the *ceiling*, and
quoting them as achievable would be dishonest. The live question is unchanged:
does the block carve, which never sees the corner, land near these epochs and
track their growth with K? K=1000 already said "picks late, but gains nothing"
(+0.009 vs flat 400). The K=2000 and K=4000 validated runs are in flight, and
K=4000 is where the oracle gain is largest (-0.034) and therefore where a
working selector has the most room to prove itself.

### Wave 3d, criterion 2 — K=1000 and K=2000 complete (2026-09-05)

The selected epoch comes from the carve run's own validation curve; it is then
applied to the *full-data* curve from `w3budget`, which is the protocol (select
on the carve, refit on the full training partition) minus refit noise. Both
endpoints are read off the 5-checkpoint smoothed curve, for the reason given in
the winner's-curse note above: comparing two single checkpoints of a curve whose
adjacent-checkpoint sd is 0.009-0.018 measures the bumps, not the budget.

| K | seed | carve-selected epoch | smoothed corner there | smoothed @400 | vs flat 400 |
| --- | --- | --- | --- | --- | --- |
| 1000 | 42 | 545 | 0.7460 | 0.7422 | +0.0038 |
| 1000 | 43 | 412 | 0.7342 | 0.7328 | +0.0014 |
| 2000 | 42 | 705 | 0.7401 | 0.7387 | +0.0013 |
| 2000 | 43 | 588 | 0.7197 | 0.7363 | **-0.0165** |

Mean: **+0.0026 at K=1000, -0.0076 at K=2000**.

**First half of criterion 2: PASSES, and at K=2000 it is precise.** The selected
epoch tracks K -- mean **478 at K=1000, 646 at K=2000** -- and never runs away
inside a 2000-epoch cap. At K=2000 the carve's mean selection of 646 sits
essentially on the smoothed oracle of 641. At K=1000 it is conservative (478 vs
616), which is the expected direction: the carve trains on 85% of the data, so
its own optimum genuinely arrives earlier than the refit's.

**Second half: the rule captures about half the available headroom at K=2000 and
none at K=1000.** Against smoothed headroom of -0.010 (K=1000) and -0.015
(K=2000), the rule delivers +0.003 and -0.008. That is a coherent result rather
than a failure: at K=1000 there is essentially nothing to win and the rule wins
nothing; at K=2000 there is 0.015 to win and it takes half.

**Correction to the first reading of these runs.** An earlier version of this
section reported +0.0357 at K=1000 seed 42 and called the second half a failure.
That number compared two *single* checkpoints -- epoch 549 happened to sit on a
0.03 bump. Smoothing both endpoints puts the same run at +0.0038. The bump was
the measurement, not the protocol. The conclusion changes accordingly: the carve
is not losing to flat 400, it is roughly breaking even at small K and gaining at
larger K.

**K=4000 still decides F13, for a sharper reason than before.** Extrapolating
"captures about half", K=4000's -0.029 of smoothed headroom would yield about
-0.015 -- exactly at this document's decision threshold, and the first size where
the answer is not "too small to care". K=4000 is also the only size where
headroom exceeds the curve's own checkpoint noise by a real factor (0.029 vs
0.008, a factor of 3.6, against 0.5-1.7 at the smaller sizes). Both K=4000 runs
are in flight. If they land near half the headroom, F13 ships as a K-dependent
cap; if they come back at a wash like K=1000, flat 400 stays and the paper says
the large-K points are budget-limited.

**Caveat carried from the tail-averaging result:** tail-averaged corner numbers
are invalid where the curve has not flattened, which is exactly the K=4000 case
here. Wave 3d must be read with `--tail 1` until it converges.


Ruled out already, at zero cost: wider *training* chunks (finding 1).
Not yet addressed and still the largest untested candidate: **capacity**.

### Wave 3e — capacity (screen submitted 2026-09-05)

The remaining untested candidate, and the only one measured at the scale of the
Stan gap. Screened first at reduced replication; see the sequential note below.

**Why it has to wait for W3d, not run beside it.** A bigger model has a
different optimal epoch than a smaller one, almost certainly a later one. Run at
a flat 400 epochs, a larger model would be *more* undertrained than the
baseline, and the sweep would report "capacity does not help" when what it
actually measured was "capacity plus a budget that was already too short".
The published capacity intuition may well be wrong for exactly this reason.
**Capacity must be measured with the stopping rule W3d settles on, not before
it.**

**Design.**

| | setting | why |
| --- | --- | --- |
| sizes | **K=4000 only** | Capacity is not the question at K=300; the gap is at large K. This is the opposite of the Wave 2 screen and deliberately so. |
| arms | base (80/8/4/128), wide128, wide160, deep12, and one big (128 dim x 12 layers) | Width and depth separated, plus one arm that moves both. The earlier sweep's 10 arms were more than the noise floor can resolve. |
| replicates | **screen first: 1 draw x 2 seeds**, escalate to 2 draws x 3 seeds only if a signal appears | see the sequential note below |
| budget | whatever W3d lands on, applied per arm via the carve | see above |
| chunk | train 300 / eval 500 | S4 |

**Cost.** The full design (5 arms x 2 draws x 3 seeds) is 30 runs at K=4000,
roughly 60-80 GPU-hours -- larger than everything in Part 1 combined.

**Sequential instead: screen at 1 draw x 2 seeds (10 runs, ~25 GPU-hours),
escalate only on a signal.** Submitted 2026-09-05 as `capscreen`.

**The screen is independent of the unfinished budget question, by construction.**
It trains to a 1200-epoch cap and saves a checkpoint every 25 epochs. Since
every shorter budget is an exact prefix of a constant-lr run, the screen can be
**re-read at whatever epoch W3d's stopping rule eventually selects, without
retraining anything**. That removes what would otherwise be a hard ordering
constraint -- capacity measured at a flat 400 epochs would have penalised the
larger arms, which need longer, and returned a false "capacity does not help".

**What n=2 can and cannot do, stated plainly.** With a paired sd around 0.035 --
and worse at K=4000, where tail-averaging is invalid until the curve flattens --
two seeds give a standard error near 0.025. That **cannot establish
significance** for the 0.03 bar, and no result from this screen should be
written up as if it had.

What n=2 *can* do is rank five arms and rule out the possibility that one of them
is dramatically better, which is the only outcome that would change the paper.
The screen is also read on the **whole learning curve**, not a single endpoint:
48 checkpoints per run, and a genuinely better architecture sits lower across the
whole curve rather than at one epoch. That is the same trick that let S5 settle
parity in 4 runs instead of 48.

**The screen's decision rule is deliberately permissive, because the errors are
not symmetric.** Capacity is the last live hypothesis for the Stan gap, so a
false negative here closes the investigation wrongly and is far more expensive
than a false positive, which only costs the escalation runs.

> **Escalate to the full 2 draws x 3 seeds if any arm's 2-seed mean beats base by
> more than 0.015** -- half the real decision bar -- **or if any arm's curve sits
> visibly below base's over the second half of training.** Declare capacity dead
> only if every arm is flat or worse on both seeds.

**Success criterion:** an arm that beats the base model by more than 0.03 nats
at K=4000 under an honest stopping rule. Anything smaller does not change the
Marformer-vs-Stan story and would not justify changing the architecture the
paper describes.

**If nothing beats base:** that is a publishable result in itself -- it says the
plateau is not a capacity limit, which combined with the budget and chunking
results would leave the generative-model mismatch (Marformer is amortised, Stan
is not) as the explanation, and that is a much more interesting sentence than
"we did not tune enough".

### Wave 4 — S3 proper, after S7 and S5

lr interacts with the loss normalisation (S5) and can only be tuned honestly on
a carve we trust (S7), so it goes last of the domain-3 suites.

### Wave 4 — S6, off the critical path

S6 is a different domain on real data; it does not gate the Domain 3 final run
and should not be allowed to. Descoped to two steps, the first of which is
nearly free — see the S6 note in Part 1.

### Wave 5 — the final rerun

Freeze every knob, regenerate on N draws, run the full grid.


## Is every suite worth running? — recommendation

| suite | verdict | why |
| --- | --- | --- |
| S1 | **run, scoped down** | The four residual variants are already known to be indistinguishable after honest early stopping. So S1's real question is narrower than written: does feature-only norm change the *ranking or spread* of the four? Two norm settings x four variants, not a full re-sweep. |
| S2 | **run in full** | Cheap, and it directly rewrites a paragraph of the paper that currently reads as "we tried a few values". |
| S3 | **run** | The WSD arm is the valuable half: removing a tuned hyperparameter is worth more to the paper than finding a better lr. |
| S4 | **done** | Answered in Wave 0 with no training at all; see the S4 Result block. |
| S5 | **run, but the stated hypothesis needs correcting** | See the note below — this changes what the arms should be. |
| S6 | **descope** | Step 1 (audit and record what actually ran) closes F7 and costs an afternoon. Step 2 (re-tune on a fresh carve) is a genuine improvement but is a second paper-section's worth of work; recommend it as optional and strictly off the critical path. |
| S7 | **run first** | Gates F4, F5, F6 and the early-stopping claim. |

### Correction to S5's premise

S5 says the 15:1 weighting "is part of why the learning rate looks so small --
multiplying lr by 15 and using 1:1/15 should be roughly equivalent". Under
**AdamW that is not right**, and the experiment as written would test the wrong
thing. Adam divides by a running estimate of the gradient magnitude, so
multiplying the whole loss by a constant leaves the update essentially
unchanged (up to `eps`), and AdamW's weight decay is decoupled so it does not
rescale either. The prediction is therefore:

- `(1, 1/15)` at **the same lr** should already be a parity match. That is the
  arm that lets us rewrite the config to match \cref{eq:trainreg} for free.
- `(1, 1/15)` at **15x lr** is a real change to the optimisation, not a
  restatement of the same one, and should be expected to differ.

The arms below are set up to test both readings, so whichever is right we get a
usable answer rather than a confound. The user's preference for round numbers
(20x first, 10x if too high) is kept for the second reading.

### The thing that is *not* on this list

Worth stating plainly before we spend a hundred GPU-hours: **none of S1-S7
operates at the scale of the gap we actually have.** On both fresh draws, Stan
with the projection prior ends about **0.11 nats ahead of Marformer at
K >= 1000**, and Marformer plateaus after K ~ 1000 while Stan keeps improving.
S1-S7 are 0.02-nat questions about uniformity and simplification, and they are
worth doing for exactly that -- a cleaner, more defensible design. They will not
close 0.11 nats. The only knobs measured at that scale are **capacity** (the
sweep that was cancelled, and whose earlier single-seed table was lost to the
seed collision) and **training length / schedule** (S3). Those two should be
understood before the final rerun is launched, or the final figure will be a
carefully-tuned version of the same plateau.


---



# Part 1 — side-branch experiment suites



## S1 — Residual stream variants, with LayerNorm off the parameter stream

- **Assumption under test:** the four-variant residual-stream comparison
(`dual_outer_residual` × `inner_residual`) was run with `norm_1`/`norm_2`
applied to the *whole* `combined = [features; params]` stream. So every
variant normalised the parameter slice, and the outer residual may simply
have been compensating for that.
- **Run:** redo the 4-variant sweep with LayerNorm restricted to the feature
slice, leaving $\vpsi$ un-normalised.
- **If the variants converge** (the gap between them shrinks or disappears):
the outer residual is a workaround for normalising $\vpsi$, not an
architectural finding. Drop the extra stream and simplify §Architecture.
- **If the gap survives:** the double residual stream is doing real work;
keep it and say explicitly in the paper that $\vpsi$ is normalised.


### Result (2026-09-05) — the gap SURVIVES, and moving the norm does not shrink it

Run `s1fn-s101`, 14/16 runs scored on the `side/s1-featnorm` branch. K=300,
item axis, transductive, 2 model seeds, tail-averaged, paired against the
shipped arm (`cn_all`: combined-stream norm, `dual_outer_residual=all`).
`cn` = norm over the whole `[features; params]` stream (shipped);
`fn` = norm over the feature slice only, leaving psi un-normalised.

| arm | norm | residual | paired vs shipped | n |
| --- | --- | --- | --- | --- |
| `s1_cn_all` | combined | all (shipped) | 0.0000 (reference) | 2 |
| `s1_fn_all` | feature-only | all | +0.0234 +/-0.0156 | 2 |
| `s1_cn_none` | combined | none | +0.0525 +/-0.0514 | 2 |
| `s1_fn_none` | feature-only | none | +0.0715 +/-0.0255 | 2 |
| `s1_cn_none_id` | combined | none + identity | +0.0614 +/-0.0130 | 2 |
| `s1_fn_none_id` | feature-only | none + identity | +0.0467 +/-0.0601 | 2 |
| `s1_cn_innone` | combined | inner none | +0.1432 +/-0.0789 | 2 |

Reference arm's absolute corner log-loss: 0.7969 (n=2, sd 0.0050).

**The residual variants do not converge under feature-only norm.** That was the
whole question, and the answer is no. Matched pair by pair, the shipped
combined norm is not the thing holding the variants apart:

| residual setting | cn | fn | fn - cn |
| --- | --- | --- | --- |
| all | +0.0000 | +0.0234 | +0.023 |
| none | +0.0525 | +0.0715 | +0.019 |
| none + identity | +0.0614 | +0.0467 | -0.015 |

Two of three pairings get *worse* under feature-only norm and one gets better,
all by about 0.02, with standard errors of the same size. Meanwhile the
residual setting itself moves the number by up to **0.143** -- an order of
magnitude more, and in the same rank order as the original ablation.

**Verdict: keep the shipped architecture (combined norm, `dual_outer_residual =
all`), and say in the paper that psi is normalised.** The S1 hypothesis was
that the outer residual is a workaround for normalising psi; if that were true,
removing psi from the norm would have collapsed the gap between residual
variants. It did not move it.

**What n=2 can and cannot support here.** It cannot resolve 0.02 -- every
fn-vs-cn number above is inside its own error bar, so "feature-only norm is
mildly worse" is *not* a claim this sweep establishes. What two seeds do settle
is the question that mattered: an effect large enough to matter for the Stan
gap (0.11 nats) or to justify simplifying the architecture section would have
been unmissable at this n, and the largest fn-vs-cn difference is 0.023. There
is no large win hiding in the normalisation placement.

**One arm was dropped, deliberately.** `s1_fn_innone` (jobs 300290/300291) sat
priority-blocked on a100 for over a day and was cancelled rather than waited
for. Its `cn` counterpart is the worst arm in the sweep by a factor of two
(+0.143); the `fn` version could only have told us how bad a configuration we
are not going to ship is. That is 2 runs of GPU not spent, recorded here so the
gap in the table is a decision rather than an omission.

**An eval-path bug this sweep exposed, now fixed.** Scoring failed three times
(jobs 315450, 315714, 315720) before producing any of the above.
`imputer/entity_mf/test.py::_reconstruct` rebuilds the model from
`train_config.json` but never read `feature_only_norm`, so every S1 checkpoint
(LayerNorm width = feature_dim = 75) was loaded into a model with model_dim-wide
norms (80) and failed with a size mismatch. Only the `fn` arms were affected,
which is why the run looked half-working. Fixed on `side/s1-featnorm`
(commit 4dae20e). The flag is branch-local and defaults to False, so no other
checkpoint in the project reconstructs any differently. Scoring the branch's
checkpoints also needs the branch on the import path, which
`side/eval_arms_s1.sbatch` does -- note that repointing the *cwd* is not enough,
because `config/cluster.sh` exports `PYTHONPATH=<main>/imputer/ranking` and that
is what wins.



## S2 — A simple rule of thumb for $\delta_\tau$

- **Assumption under test:** current values are ad hoc and axis-specific —
item axis $1.0/0.0$, annotator axis $0.0/0.8$, real LLM-Rubric $0.7$. We want
one rule we can state in one sentence.
- **Run, in this order:**
  1. **$0.7/0$ on both axes** (do this first — it is the candidate most likely
    to work everywhere, and cheapest to state).
  2. If that fails, $1.0/0$ on both axes, i.e. raise the annotator axis from
    0.8 to 1.0.
- **If 0.7/0 works on both axes:** rule of thumb becomes "$\delta_\tau=0.7$ for
the type you must generalise over, 0 for the rest", and it also applies to
domains 1 and 2. Cleanest outcome.
- **If only 1.0/0 works:** rule becomes "1 for the generalisation type, 0
otherwise", which is also clean but forces the deviation table to stay at
exactly zero — worth saying that is deliberate.
- **If neither:** keep per-axis values and state them as tuned, not principled.
- **Extends to domains 1 and 2**, which currently use $\delta_\tau=0$
throughout; whichever rule wins should be checked there too.
- Related: D17 (dropout is not rescaled at eval, so the deviation contributes
$(1-p)\times$ in training and $1\times$ at test).


### Result, item axis at K=300 (2026-09-05) — 0.7 and 1.0 are the same model

Run `s2item`, 18 runs (3 arms x 2 draws x 3 model seeds, K=300, T arm), corner
log-loss averaged over the last 5 periodic checkpoints, paired against the
shipped delta_item = 1.0:

| arm | paired mean vs 1.0 | n |
| --- | --- | --- |
| delta_item = 0.7 | **+0.0004 +/-0.0002** | 6 |
| delta_item = 0.0 | +0.0060 +/-0.0024 | 6 |

Reference arm's absolute corner log-loss: 0.7831.

**delta_item 0.7 and 1.0 produce the same model to four decimal places**, in
all six (draw, seed) cells -- far tighter than the 0.015 threshold and tighter
than the measurement noise on a single arm. The configs were checked to confirm
the arms really did differ (0.0 / 0.7 / 1.0 in `train_config.json`); they did.

**Why so tight, mechanically:** on the item axis the per-item deviation table
gets gradient only on the fraction of updates where the embedding is *not*
dropped, and weight decay pulls it back toward zero in between. At delta=0.7 it
gets gradient 30% of the time and still ends up near zero, so the two arms
converge to nearly the same function. At delta=0.0 the table is trained every
step, and that is the arm that moves -- though only by 0.006.

**The second row is the more interesting one, and it is not what the paper
implies.** delta_tau is presented as a substantive mechanism, but turning it
off entirely costs **0.006 nats** at K=300. On this evidence embedding dropout
is close to free either way at this size, rather than load-bearing. Whether
that holds at K=4000 -- where there is far more per-item signal for the
deviation table to actually learn -- is Wave 3b, and it is the version of this
question that matters for the Stan gap.

**Read on the rule of thumb:** delta = 0.7 is adoptable on the item axis. The
annotator axis (`s2annot`, 18/18) and Wave 3b since agreed; see below.


### Result, annotator axis (2026-09-05, both draws -- SETTLED)

Run `s2annot`, 18/18 runs (draws s101 and s105). J=30, transductive, 3 model
seeds per draw, tail-averaged, paired against the shipped delta_annot = 0.8:

| arm | draw s101 | draw s105 | both draws (n=6) |
| --- | --- | --- | --- |
| delta_annot = 0.7 | -0.0005 +/-0.0008 | +0.0010 +/-0.0014 | **+0.0002 +/-0.0008** |
| delta_annot = 0.0 | +0.0092 +/-0.0135 | -0.0111 +/-0.0070 | **-0.0009 +/-0.0082** |

Reference arm's absolute corner log-loss: 0.7949 over both draws (n=6, sd
0.0514 -- the spread is between draws, not between seeds).

**delta = 0.7 is indistinguishable from the shipped 0.8 on the annotator axis**,
at +0.0002 +/-0.0008 -- two orders of magnitude inside the 0.015 threshold, and
the tightest paired estimate in the whole sweep. Combined with the item-axis
result (0.7 matches 1.0 at K=300, 1000 and 4000), the rule of thumb

> "0.7 for the type you must generalise over, 0 for the rest"

can be adopted for all axes and domains. Domains 1 and 2 use 0 throughout and
are already consistent with it. The conditional delta=1.0 arm never needed to be
run.

**Say why we are adopting it, though.** The sweep licenses the simplification;
it does not show a gain. Every delta the sweep touched -- 0.0, 0.7, 0.8, 1.0 --
lands within noise of every other on both axes. The case for 0.7 is uniformity
across three domains and one sentence instead of a per-axis table, not
performance. Claiming a performance benefit would be unsupported by these
numbers.

**The one hypothesis the second draw killed.** After draw s101 it looked like
turning delta off entirely was mildly harmful on the annotator axis (+0.009),
which would have been consistent with a real mechanism: on the item axis test
items are new, so their embeddings are untrained whatever delta was, but the
transductive annotator arm *does* see the test annotators. Draw s105 came back
at -0.011, the opposite sign and the same size. Pooled, -0.0009 +/-0.0082. The
+0.009 was draw-to-draw noise; there is no asymmetry to report.



## S3 — Learning rate: tune it, or remove the need to tune it

- **Assumption under test:** lr was chosen by hand and frozen ($10^{-4}$ in
domains 1–2, $2\times10^{-4}$ in domain 3). There was no validation data at
the time. There is now.
- **Run:**
  1. Smoke sweep on one setup (K=500, 2 seeds) to see whether a tuned lr
    converges faster, stays stable, and reaches a better loss.
  2. If convergence is much faster, promote lr to a hyperparameter tuned on the
    validation carve before the full train+val refit.
  3. Broader sweep for stability across sizes and seeds in domain 3.
  4. Try a **WSD schedule** (warmup–stable–decay): if a decay phase at the end
    stabilises training regardless of the starting lr, we can stop tuning lr
     altogether.
- **If WSD removes the sensitivity:** that is the preferred outcome — one fewer
hyperparameter to tune before each retrain. Keeping the tuned-hyperparameter
count minimal matters for the practical efficiency of the whole system.
- **If lr must stay tuned:** document it as tuned on validation, and report the
chosen value per domain.



## S4 — Evaluation chunk size: 300/300 vs 300/500

- **Assumption under test:** we train at `max_item=300` (training GPU memory)
but evaluate at 500 (more GPU available at eval, and empirically a little
better). That is a train/test mismatch we currently have to explain.
- **Run:** evaluate the same trained checkpoints at 300 and at 500.
- **If 300/300 is no worse:** switch to 300/300 everywhere and simplify the
paper to a single chunk size.
- **If 300/500 is genuinely better:** keep it, and keep the explanation.
- See D1 and D24 in DISCREPANCIES.md.


### Result (2026-09-04) — 300/300 is free below K=500 and costs ~0.015 nats above it

Run: `side/s4_eval.sbatch`, job 299893. No training. Re-scored all 114
`seeds-01` checkpoints (the shipped configuration: `max_item=300`, 400 epochs,
model seeds 42/43/44, both axes, both arms) at evaluation chunk 100 through
600. 684 corner-scored rows, `corner_whole=true`. Numbers are **paired**
differences against evaluation at 300, item axis.

| K | 500 vs 300 | 600 vs 300 | 100 vs 300 |
| --- | --- | --- | --- |
| 100-400 | 0.0000 | 0.0000 | +0.009 to +0.073 |
| 500 | -0.0059 +/-0.0033 | -0.0059 +/-0.0033 | +0.063 |
| 700 | -0.0051 +/-0.0036 | -0.0051 +/-0.0036 | +0.055 |
| 900 | -0.0138 +/-0.0062 | -0.0096 +/-0.0048 | +0.061 |
| 1000 | -0.0157 +/-0.0118 | -0.0095 +/-0.0118 | +0.089 |
| 2000 | **-0.0180 +/-0.0028** | -0.0131 +/-0.0013 | +0.109 |
| 4000 | **-0.0175 +/-0.0042** | -0.0115 +/-0.0054 | +0.113 |

Three things the sweep settles:

1. **Below K=500 the two are bit-identical** (0.0000, not "small"). With
   `corner_whole` the corner is one chunk and the remaining items fit either
   way, so there is literally nothing to choose. Every Stan-comparable point in
   the paper is at K <= 500.
2. **Above K=500 the larger evaluation chunk wins, and the gap grows with K**,
   reaching -0.018 nats at K=2000 and K=4000 -- above the 0.015 decision
   threshold. So 300/300 is not free where the curve is longest.
3. **The optimum is interior, near 500.** It is not "more context is always
   better": on the 8 cells where chunk 1200 was also scored, 1200 is *worse*
   than 600 by +0.0098 and no better than 300. Evaluating with chunks far
   larger than training used costs something too.

The annotator axis is completely insensitive (0.0000 at every chunk from 100 to
600): its bundles never have enough items to split, so chunk size is not a
parameter there at all.

**Recommendation: keep the larger evaluation chunk, and stop describing it as a
mismatch.** It is not one. Chunking exists only because the backward pass does
not fit in memory; inference has no backward pass, so it can afford more
context, and Stan is given the whole tensor at once regardless. Giving Marformer
the largest chunk it can afford at inference is the *fair* comparison, not a
thumb on the scale. One sentence in the paper -- "training uses item chunks of
300 for memory; inference, which has no backward pass, uses 500" -- covers it,
and it is worth 0.015 nats at the large-K end where we are already behind Stan.

Pick **500**, not 600 or 1200: 500 and 600 are indistinguishable below K=800 and
500 is better above it, and 1200 is worse than both.

**Correction to an earlier reading.** An estimate quoted earlier in this
project put the 600-vs-300 gap at -0.016 +/- 0.012 over 24 paired checkpoints.
That sample came from the `grid-01` tree, which mixes shuffle variants and
several training chunk sizes. On the shipped configuration with 66 paired cells
the item-axis average is -0.0047, and the effect is concentrated at K >= 900
rather than being uniform. The table above supersedes it.

**F1 resolves to:** two chunk sizes, documented, not one.



## S5 — Normalise the loss weights to $1 : 1/15$

- **Assumption under test:** the code uses masked weight 15, observed weight 1.
That is a 15× scaling of the whole masked objective, which is part of why the
learning rate looks so small — multiplying lr by 15 and using $1 : 1/15$
should be roughly equivalent, and matches how \cref{eq:trainreg} is written.
- **Run:** one arm at $1 : 1/15$ with lr scaled by 15, confirm parity.
- **If parity holds:** adopt $1 : 1/15$ for the final run so the code matches the paper's normalisation and $\lambda_\mathrm{copy}=1/15$ reads directly off the config.(we mshould scale lr not by 15 but by 10 or 20, for sake of lr being clean enough with one digit of accuracy, we shoudl try with 20x first and use 10x if that is too high.)
- Related: D16 (the observed-token copy term is a standing pressure toward low
predictive entropy, and is not currently described in the paper).


### Result (2026-09-05) — parity holds at the SAME lr; no rescaling needed

Run `s5par`, 4 runs (2 arms x 2 model seeds, K=300, draw s101, T arm). Corner
log-loss at every periodic checkpoint, `s5_base` = (15 : 1) at lr 2e-4 versus
`s5_norm_lr1x` = (1 : 1/15) at **the same** lr 2e-4:

| seed | mean diff over the whole trajectory | max abs diff | diff at the last checkpoint |
| --- | --- | --- | --- |
| 42 | -0.0001 | 0.0031 | -0.0012 |
| 43 | -0.0002 | 0.0027 | +0.0018 |

The two runs do not merely land in the same place -- they **track each other
epoch by epoch for 400 epochs**, never diverging by more than 0.003 nats, which
is float32-and-Adam-epsilon noise. This is much stronger evidence than a
matched endpoint would be, and it is why the suite needed 4 runs rather than 48.

**The suite's original premise was wrong, and acting on it would have changed
the optimisation.** S5 as written says the 15:1 weighting "is part of why the
learning rate looks so small -- multiplying lr by 15 and using 1:1/15 should be
roughly equivalent". Under AdamW it is not: Adam divides by a running estimate
of the gradient magnitude, so scaling the whole loss by a constant leaves the
update essentially unchanged, and AdamW's weight decay is decoupled so it does
not rescale either. Multiplying lr by 15 (or 10, or 20) would therefore have
been a **real** change to training dressed up as a re-parameterisation.

**Adopt (1 : 1/15) at lr 2e-4, unchanged.** The config then reads directly off
\cref{eq:trainreg} with lambda_copy = 1/15, the learning rate keeps its familiar
value, and nothing about the optimisation moves. The lr-scaled arms were
conditional on this failing and are not needed -- 24 runs saved.

**F3 resolves to:** normalise the weights, do **not** touch the lr.



## S6 — Replicate LLM-Rubric and pin down its hyperparameters

- **Assumption under test:** our local LLM-Rubric config may not be what
produced the paper's numbers. Known divergences from the paper text: 300
epochs (paper says 400), $\delta_\mathrm{item}=0.7$, `max_item=10`, neither
stated.
- **Run:** replicate the LLM-Rubric results end to end and record the hyperparameters that actually ran. (Or use the new validation set to produce a better results)
- **Also:** carve a validation set from the real-data training split and tune
hyperparameters on it, rather than inheriting domain 3's synthetic settings.
- **Outcome either way:** update the paper's LLM-Rubric paragraph to the
recorded values.


### Result, step 1 (2026-09-04) — the config is now recorded; F7 no longer needs a rerun

No compute. Read `RERUN/jobs/llmrubric_train.sbatch` and confirmed against the
`train_config.json` actually written by three of the `replicate-01` LLM-Rubric
runs (sizes 10, 40, 100): all three record `epochs=300, max_item=10,
item_dropout_rate=0.7, lr=2e-4, embedding_dim=80, num_layers=8`. So the script
is what ran; there is no third, unrecorded configuration.

**Full recorded configuration** (real LLM-Rubric, Marformer arm, all 10 sizes
10-175):

| group | value |
| --- | --- |
| architecture | `embedding_dim 80, num_layers 8, attention_heads 4, d_ff 128, num_ffn_layers 1, dropout 0.1` |
| optimisation | `AdamW lr 2e-4, weight_decay 0.01, lr_schedule none, lr_min 1e-5` |
| objective | `masking_rate 0.15, mask_augmentations 5, masked_loss_weight 15.0, observed_loss_weight 1.0` |
| dropout rates | `item_dropout_rate 0.7`, `annotator_dropout_rate` left at its default **0.0** |
| chunking | `max_item 10` |
| budget | `epochs 300`, `seed 42`, single seed |
| init / attention | `type_embedding_init kaiming, --no-per-head-rel, --scale-shared-rel, --use-pointer` |
| real-data only | `--llm-annotator-id 24, --human-observed-rate 0.0, --llm-input-dist` |

**Divergences from the paper's text** (\S "We train with AdamW...", main.tex
line 1520), now pinned down rather than suspected:

| paper says | code does |
| --- | --- |
| 400 epochs | **300** |
| (silent) | `item_dropout_rate` = **0.7** |
| (silent) | `max_item` = **10** |

Everything else in that paragraph -- lr 2e-4, weight decay 0.01, masking 0.15,
five mask augmentations, masked loss weight 15, observed 1, dropout 0.1 -- is
exactly what ran, and the architecture matches Domain 3's item axis exactly.
So the LLM-Rubric arm differs from Domain 3 item in only three knobs (epochs,
delta_item, max_item) plus the three real-data flags, which is a much smaller
divergence than the paper's silence suggests.

**F7 can be closed by editing three numbers in the paper.** No rerun is
required for that, which was the expensive part of S6 as originally scoped.

**Two things the audit exposes that are worth stating rather than fixing here:**

1. **`max_item=10` is not a memory setting, it is a modelling choice.** The
   LLM-Rubric bundle is 225 items; a chunk of 10 means the model conditions on
   at most 9 other items when predicting. Domain 3 uses 300. Nobody has measured
   whether 10 is better than 225 on this data -- and S4 has just shown that on
   Domain 3 the chunk size is worth up to 0.11 nats when it is set too small.
   **This is the single most likely place where the LLM-Rubric numbers are
   leaving performance on the table**, and it is a cheap eval-only sweep of
   exactly the kind S4 turned out to be.
2. **There is still no validation split here.** 300 epochs is a fixed budget,
   not a validated one, and no checkpoint is selected -- so the arm inherits
   D14's problem in the milder "arbitrary budget" form rather than the
   "selected on test" form.

**Recommended remaining S6 scope (optional, off the critical path):** the
eval-chunk sweep in (1), which is free; and only then the validation carve.



## S7 — Reshape the validation carve (blocking S3 and the early-stopping claim)

- **Assumption under test:** the current `block` carve holds out whole items,
and its signal tracks the *training* cells rather than the test corner. On
the non-transductive arm it selects a stopping epoch ~190 epochs too late,
and at K=2000 the refit is worse than not stopping at all.
- **Run:** carve validation blocks shaped like the corner (100 items × 10
annotators) instead of whole items, and re-measure selected epoch vs corner
optimum.
- **If the reshaped carve tracks the corner:** the validation protocol becomes
trustworthy, and S3's lr tuning can rest on it.
- **If it still does not:** NT points at K ≥ 1000 stay protocol-limited and
must be labelled as such in the paper.
- Full evidence in D26. **This one gates the early-stopping story**, so it
should run before we commit paper text claiming validation-based stopping.

### Result (2026-09-05) — the carve is fine. D26 was a property of base seed 42.

Runs `s7-block-s101` and `s7-block-s105`: NT item axis, block carve, K in
{1000, 2000, 4000}, 3 model seeds, corner scored at every periodic checkpoint.

| draw | K | corner optimum epoch | val-selected epoch | val - corner | cost vs a corner oracle | cost vs not stopping |
| --- | --- | --- | --- | --- | --- | --- |
| **s42** (D26) | 1000 | **~124** | 208-312 | **+84 to +188** | +0.086 | -0.026 |
| s101 | 1000 | 324 | 283 | -41 | +0.0152 | -0.0026 |
| s101 | 2000 | 357 | 322 | -35 | +0.0219 | +0.0156 |
| s101 | 4000 | 374 | 371 | **-3** | +0.0056 | +0.0010 |
| s105 | 1000 | 349 | 299 | -50 | +0.0201 | -0.0046 |
| s105 | 2000 | 374 | 358 | -16 | +0.0116 | +0.0015 |
| s105 | 4000 | 357 | 342 | -15 | +0.0067 | -0.0065 |

**The mismatch does not reproduce, and the reason is not the carve.** D26
recorded the carve selecting ~190 epochs *later* than the corner optimum and
left the mechanism open. The answer is that **the corner optimum moved, not the
carve**: on seed 42 the NT corner bottoms out at epoch ~124, while on both fresh
draws it bottoms out at 324-374. The carve selects in much the same place on all
three draws; seed 42 is the draw where the corner turns up early.

That is D28 explaining D26. Seed 42's corner is an anomalously hard sample
(z = +2.0 to +3.8 across six arm/K cells), and a hard corner degrades sooner,
which reads as "validation stopped too late" when the comparison is to that
corner.

**On fresh draws the carve is slightly conservative, never late** -- it selects
3 to 50 epochs *before* the corner optimum, costs 0.006-0.022 nats against a
corner oracle, and is within noise of not stopping at all. Conservative is the
right direction for a stopping rule to err in.

**Consequences:**

1. **Keep the `block` carve.** The corner-shaped carve proposed as S7 step 2 is
   not needed and was never built -- no new machinery, no new protocol to
   explain. **F5 resolves to "keep the block carve"**, and **F6 (standardise
   early stopping across all three domains) is unblocked.**
2. **The NT points at K >= 1000 are not protocol-limited.** D26's closing
   instruction to label them as such should be withdrawn; they can be read as
   the model's capability.
3. **D26 needs rewriting, not extending.** Its measurements were right; its
   conclusion generalised a single draw.
4. **A third, unlooked-for corroboration of the budget finding.** On both fresh
   draws the NT corner optimum sits at 324-374 out of a 400-epoch budget --
   right at the end, exactly as the transductive K=4000 arm does. Two arms and
   three sizes now independently say 400 epochs is close to binding.

**Scope caveat:** NT item axis only, which is where D26 found the problem. The
transductive arm never showed the symptom (D25: no inflation) and the annotator
axis was not re-tested.


---



# Part 2 — before the final rerun

Changes to land before kicking off the definitive end-to-end run. Status is one
of: **pending** (not started), **blocked** (waiting on a Part 1 suite),
**ready** (decided, not yet applied), **done**.


| #   | Change                                                                                                                                                            | Depends on | Status  |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------- |
| F1  | Single chunk size across train and eval, or a documented reason for two                                                                                           | S4         | **ready** — S4 done: keep 300 train / **500** eval, document the reason (inference has no backward pass). Identical below K=500, worth 0.015 nats above it. |
| F2  | One $\delta_\tau$ rule applied to all axes and domains | S2 | **ready** — δ=0.7 is within noise of the shipped value on BOTH axes (item at K=300/1000/4000; annotator at J=30, two draws, +0.0002 ±0.0008). Rule: δ=0.7 for the type generalised over, 0 for the rest. Adopted for uniformity, not for a measured gain — δ_τ is close to inert everywhere we probed it. |
| F3  | Loss weights normalised to $1 : 1/15$, **lr left at 2e-4** (not rescaled)                                                                                        | S5         | **ready** — S5 done 2026-09-05: parity holds at the same lr, trajectories overlay within 0.003 nats. Rescaling lr would have been a real change, not a re-parameterisation. |
| F4  | lr either tuned on validation or made scheduler-insensitive (WSD)                                                                                                 | S3, S7     | blocked |
| F5  | Validation carve — **keep `block` as shipped**, no reshape | S7 | **ready** — S7 done 2026-09-05: the D26 mismatch is a seed-42 artifact; on two fresh draws the carve is slightly conservative, never late. |
| F6  | Early stopping standardised across all three domains (carve → select → refit everywhere) | S7 | **unblocked** — the carve is trustworthy; proceed. |
| F7  | LLM-Rubric hyperparameters recorded and paper text updated                                                                                                        | S6         | **ready** — audited 2026-09-04, full config in the S6 Result block; paper needs three numbers changed (400→300 epochs, state δ_item=0.7 and max_item=10). |
| F8  | `test.py` hardcodes `max_item=500` — **do NOT change it to inherit from `train_config.json`** | S4 | **reversed** — see note below |
| F9  | Domain 1 validation set carved from *training* data, not the first 50 test instances                                                                              | —          | ready   |
| F10 | Domain 1: confirm what the committed training objective actually is (currently KL to exact BN posteriors, not CMLE on masked values) and reconcile with the paper | —          | pending |
| F11 | Domain 1/2 model sizes, epochs, batch sizes reconciled with the paper table                                                                                       | —          | pending |
| F12 | Runtime figures read elapsed time from run metadata instead of hardcoded numbers (D18)                                                                            | —          | pending |
| F13 | Epoch **cap** raised with validation-chosen stopping, instead of a flat 400 | W3d | **reframed 2026-09-05** — a flat 1200 is +0.105 WORSE at K=1000 and -0.018 better at K=4000, so the budget must scale. Now depends entirely on the carve stopping correctly inside a 2000-epoch cap (`w3budgetval`). |



## The final rerun, specified

The Part 1 suites decide individual knobs. This section is about the shape of
the run itself, and it contains one decision that costs more than all of Part 1
put together.

### The decision that sets the budget: how many generative draws

Figure 6 today is **one generative draw with no error bars**. D28 measured the
draw-to-draw sd of the corner metric at **0.085 nats**. The Marformer-vs-Stan
gap we are trying to report is ~0.11. So a single-draw figure cannot support
the comparison it is drawn to support -- one unlucky draw moves a curve by most
of the effect, which is exactly what happened on seed 42, where the corner is
anomalously hard in all six (arm, K) cells at z = +2.0 to +3.8.

**Recommendation: the final figure plots the mean over 3 generative draws, with
the draw-to-draw spread as the error band, and says so in the caption.** Model
seeds are averaged within each draw (their sd is 0.014, six times smaller, so 2
per draw is plenty). That is the difference between a figure that shows a
result and a figure that shows one sample.

Cost: 3x the Marformer grid (cheap -- the whole item axis is ~10 GPU-hours per
draw) and 3x the Stan grid (**not** cheap -- Stan is the pole). Mitigation: we
already have three draws generated (s42, s101, s105) and Stan fits in progress
on two of them, so the marginal cost is smaller than 3x from here.

If 3 draws proves unaffordable on the Stan side, the fallback is to plot Stan on
the draws we can afford and **label the Marformer band as the draw spread**,
rather than plotting both single-draw and implying the gap is precise.

### Grid

| axis | arm | K / J |
| --- | --- | --- |
| item | T | 100-1000 by 100, then 2000, 3000, 4000 |
| item | NT | 100-1000 by 100, then 2000, 3000, 4000 (currently stops at 900; the bundles already exist on the fresh draws) |
| annot | T | 10-50 by 5 |
| annot | NT | 10-40 by 5 |
| Stan | all four variants | to K=500 as published, plus 1000-4000 on the item axis, where Stan overtakes |

Extending Stan past 500 is not optional any more: on both fresh draws Stan Proj
is ~0.11 nats ahead of Marformer at K >= 4000. Stopping the Stan curve at 500,
as the paper does, hides the crossover. Whatever the figure ends up showing, it
should not be able to be read as "Marformer wins" because the comparison was
truncated before it stopped winning.

### Frozen configuration for the final run

Updated 2026-09-05 as each suite reported. **Everything marked SETTLED is
decided and needs no further work**; the two marked otherwise are the only
things still gating a launch.

| # | knob | final-run setting | status | authority |
| --- | --- | --- | --- | --- |
| 1 | metric | corner log-loss, corner held whole in a chunk carrying training context | SETTLED | D22, D25 |
| 2 | training chunk | `max_item = 300` | SETTLED | memory ceiling |
| 3 | evaluation chunk | **500** (deliberately different from training; document why) | SETTLED | **S4** |
| 4 | loss weights | **1 : 1/15**, matching \cref{eq:trainreg} | SETTLED | **S5** |
| 5 | learning rate | **2e-4, unchanged** -- do *not* rescale with the weights | SETTLED | **S5** |
| 6 | delta_tau | **0.7** for the type being generalised over, **0** for the rest, both axes, all domains | SETTLED (both annotator draws in; +0.0002 ±0.0008) | **S2** |
| 7 | validation carve | **`block`, as shipped** -- no reshape | SETTLED | **S7** |
| 8 | stopping | carve -> select on validation -> refit on the full training partition | SETTLED | D14, **S7** |
| 9 | checkpoint reported | the refit's **last** checkpoint, never an argmin on test | SETTLED | D14 |
| 10 | replicates | **3 generative draws x 2 model seeds**, draw spread as the error band | SETTLED (a decision, not an experiment) | D28 + Part 0 noise table |
| 11 | residual variant | `all` (shipped) | SETTLED | resid ablation, **S1** |
| 12 | **epoch budget** | **a function of K, not a flat 400** | **GATING** -- criterion 1 met, criterion 2 running | **W3d** |
| 13 | **capacity** | unknown | **GATING** -- never measured | -- |

**The two gates, restated plainly.**

*Budget (12).* Criterion 1 is met: at K=4000 a fixed 1200-epoch budget is worth
-0.018 nats, and two further arms (NT item on both fresh draws) independently
show the corner optimum sitting at the very end of a 400-epoch budget. What is
not yet earned is the claim that our stopping rule finds that epoch **without
being shown the corner** -- `w3budgetval` decides it. If it fails, the headroom
is real but unclaimable, and that is an S7 problem rather than a budget one.

*Capacity (13).* Still the only untested knob measured at the scale of the Stan
gap. Everything settled above moves the number by 0.02 or less; the gap is 0.11.
Launching the final run without knowing whether 80 dims and 8 layers are enough
at K=4000 means the final figure is a carefully-tuned version of whatever
plateau we already have. **This should be the next thing that gets GPU.**

### What changed from the pre-suite plan

Worth recording, because three of these would have been applied as tidy-ups by
someone who had not measured them:

| was going to | measurement said | now |
| --- | --- | --- |
| unify train and eval chunk size at 300 | 500 is near-optimal at eval; 300 costs 0.018 at large K | keep both, document why |
| normalise loss weights **and** rescale lr by 15-20x | AdamW makes the rescale a no-op; changing lr would have been a real change in disguise | normalise weights only |
| make `test.py` inherit `max_item` from training | that *is* the 300-vs-500 regression, applied silently | name the constant, keep 500 |
| build a corner-shaped validation carve | the existing carve was never broken; seed 42 was | build nothing |
| keep a flat 400-epoch budget | binding at K=4000 on three independent arms | scale with K |

### Order of operations

1. Finish Part 1; fold each winner into the runner defaults on `patches`.
2. Re-run the capacity sweep. It is the only knob measured at the scale of the
   Stan gap and its earlier result was destroyed by the seed collision, so the
   final run should not be launched while it is unknown.
3. Regenerate the third draw's missing bundles; verify byte-identical
   regeneration on s42 as the determinism control (this has passed once).
4. Launch Marformer (all axes, all arms, 3 draws x 2 seeds), then Stan.
5. Rebuild every figure from the new tree; re-derive DISCREPANCIES entries that
   quote numbers.

### What would make me not launch

- S7 unresolved: without a trustworthy carve the NT arm at K >= 1000 is
  protocol-limited and the points are not the model's capability (D26).
- Capacity unknown: see 2 above.
- Stan not extended past K=500 on at least two draws: the figure would be
  truncated exactly where the interesting thing happens.


### F8 was going to be a regression

`test.py:241` hardcodes `max_item=500` while training uses 300, and F8 was
written to make the evaluator inherit the training value so the two agree. **S4
says that would make the reported numbers worse**, by up to 0.018 nats at
K=4000, because 500 is not an accident -- it is close to the optimum, and the
optimum is genuinely different at evaluation time than at training time. See
the S4 Result block.

The real defect is that 500 is an undocumented magic number sitting in the
middle of a function, not that it disagrees with training. **The fix is to name
it, default it to 500, make it settable, and say in the paper why inference uses
a larger chunk than training** (inference has no backward pass, so it can afford
more context; Stan is given the whole tensor regardless).

This is worth stating plainly because the original F8 would have been applied as
a tidy-up by someone who had not measured it, and it would have quietly cost
performance at exactly the sizes where we are already behind Stan.

F9–F12 do not need a side experiment — they are corrections, and can be applied
directly once someone confirms the intended behaviour.