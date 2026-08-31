# Plan: a corrected Figure 6(a)–(b)

Goal: reproduce Figure 6(a) log-loss/item and 6(b) log-loss/annotator under the
protocol the paper actually states, extend the Marformer NT curve to the full
item sweep, and — separately and only if the above lands — pilot larger item
counts than the paper reports.

Run id: `fig6-01`. Pilot (separate): `pilot-10k`.

---

## 0. What the paper specifies, and what already complies

Paper, §Domain 3 experimental setup (main.tex:1269) and the Fig. 6 caption:

> a fixed **test corner**: K_test = 100 items x J_test = 10 annotators x I criteria.
> **All reported log-losses are evaluated on missing entries within this corner**,
> providing a consistent target across all training sizes.

| arm | trains on | evaluated on |
|---|---|---|
| Marformer (T) | train observed + **test-corner observed** | the 414 corner cells |
| Marformer NT | train observed only (main.tex:1584) | the 414 corner cells |

**Training already complies, in both arms.** `train.py` builds
`maskable_sources = train_observed + val_observed + test_observed` when
`--transductive-learning` is set, and `train_observed` alone when it is not;
`test_missing` is never in the training graph either way. Domain 3 has no
validation split, so "train + val observed" and "train observed" are the same
set. Nothing about the trained checkpoints is wrong.

**Only the evaluation is wrong** (D22/D23): `test.py::_reconstruct` returns
`train_obs + test_obs + test_miss + train_miss` and the metric averages every
missing token in it, while Stan is scored on the corner alone.

Consequence for this plan: **the existing dots need re-evaluation, not
retraining.** New training is required only where a dot does not exist yet.

---

## 1. Scope

### 1a. Re-evaluate every existing dot on the test corner  *(no retraining)*

38 settings x 3 seeds, already on disk in `runs/replicate-01` and `runs/seeds-01`.
`analysis/split_missing_metrics.py` already does exactly this and has been
verified to reproduce the stored pooled number to six decimals, so the corner
metric it reports is a decomposition of the published one, not a different
measurement. Cost: ~30 s per pass on one H100.

### 1b. Fill in the missing dots  *(new training)*

| axis | arm | have | need | new cells |
|---|---|---|---|---|
| item | T | 100–1000, 2000, 3000, 4000 | — | 0 |
| item | NT | 100–900 | **1000, 2000, 3000, 4000** | 4 x 3 seeds = 12 |
| annot | T | 10–50 | — | 0 |
| annot | NT | 10–40 | **45, 50** | 2 x 3 seeds = 6 |

18 new trainings. Data for those NT sizes does not exist and must be generated
first (§3).

### 1c. Pilot: beyond the paper's range  *(only after 1a+1b land)*

Item axis, T and NT, at K = 10 000. Requires generating a 10 000-item dataset,
which is a new generator invocation, not an extension of the existing one.
Kept in its own tree (§6) so it cannot contaminate the figure.

### Out of scope

- **Stan.** No Stan results exist anywhere on this filesystem — nothing
  generated survived the migration, and the only Stan outputs present are for
  other domains (`Factor_*`, `TENSOR_*`), not `TENSOR-SIG_eval`. Per Tom, we are
  not rerunning Stan. The figure will therefore carry Marformer T and NT only,
  with the Stan series left as a documented gap to be filled when the original
  outputs are located.
- **The empirical baselines** (unigram / NB / log-linear) — they run from
  `BASELINES/structured_baselines`, are cheap, and can be added later; they are
  not part of 6(a)/(b) `_clean_nt` panels.

---

## 2. Hyperparameters

Held at the replication's values so new dots are comparable with existing ones.
Read back from `train_config.json` of `replicate-01`, not from the runner
defaults (the runner's `MAX_ITEM` default is 100; the runs used 300).

| | value |
|---|---|
| embedding_dim / layers / heads / d_ff | 80 / 8 / 4 / 128 |
| epochs | 400 |
| optimizer | AdamW, lr 2e-4, weight_decay 0.01 |
| lr schedule | none |
| masking_rate | 0.15 |
| mask_augmentations | 5 |
| masked / observed loss weight | 15 / 1 |
| dropout | 0.1 |
| **max_item (training chunk)** | **300** |
| item_dropout_rate | 1.0 (item axis), 0.0 (annot axis) |
| annotator_dropout_rate | 0.0 (item axis), 0.8 (annot axis) |
| seeds | 42, 43, 44 |
| batch_augmentations | **false** — changes the optimisation, so off for figure runs |

Speed flags that do **not** change semantics are on (`--tf32`, tensorised graph
path, cached eval graphs, fused AdamW). `--batch-augmentations` and any `--lr`
change are off: they were measured to shift the result and belong to the speed
report, not here.

**Evaluation chunk size:** `max_item = 300`, matching training, not `test.py`'s
hardcoded 500 (D20). Applied through `split_missing_metrics.py`, which takes it
as a parameter.

---

## 3. Data generation for the new NT sizes

The NT splits are shifted by one step relative to T by construction —
`Item_NT_K` has the same composition as `Item_T_(K+100)`:

| bundle | obs train | miss train | equals |
|---|---|---|---|
| Item_NT_100 | 312 | 282 | Item_T_200 |
| Item_NT_200 | 634 | 599 | Item_T_300 |
| Item_NT_500 | 1707 | 1677 | Item_T_600 |

That is deliberate and correct: it makes the x-axis mean "items available for
training" in both arms — T_K trains on K items *including* the 100-item corner,
NT_K trains on K items *excluding* it. The new NT sizes must preserve this, so
they are generated by the same script with the same base seed, only extending
`ITEM_NT_SIZES` and `ANNOT_NT_SIZES`.

Generation is deterministic given the base seed and each size is built
independently, so extending the size list cannot perturb existing bundles. This
will be **verified by checksum** on the pre-existing bundles before any training
is launched.

```
ITEM_NT_SIZES  = 100,...,900,1000,2000,3000,4000     (was ...,900)
ANNOT_NT_SIZES = 10,...,40,45,50                     (was ...,40)
```

---

## 4. Checkpoint selection

Selection is a separate defect (D14: the reported point is an argmin over ~18
checkpoints scored on test). It is not what this task is fixing, but it cannot
be ignored, because the figure needs *some* policy.

- **Primary: last checkpoint.** No test leakage, and it is the honest analogue
  of Stan, which is fit to convergence with no selection step.
- **Also computed: test-argmin**, so the change from the published figure can be
  attributed between the two causes (wrong eval set vs. selection).
- **Also computed where available: val-block selected**, from `val-block-01`,
  which is the legitimate protocol built earlier.

All three go into the payload; the figure plots the first and the report shows
the spread.

---

## 5. Sequence

| phase | what | compute |
|---|---|---|
| P0 | checksum existing bundles; extend size lists; generate new NT data | CPU, minutes |
| P1 | re-evaluate all existing dots on the corner (T and NT, both axes, 3 seeds) | 1 GPU x minutes |
| P2 | train the 18 new NT cells | 18 jobs; est. 20 min–1 h each |
| P3 | evaluate the new cells on the corner | 1 GPU x minutes |
| P4 | build the corrected 6(a)/(b) from the corner payload | CPU |
| P5 | report + artifact | — |
| P6 | pilot at K=10 000, separate tree | 2 arms x 3 seeds |

Runtime estimate for P2 comes from `replicate-01`'s recorded
`elapsed_seconds` (1377 s at K=500, 12 923 s at K=4000) divided by the measured
post-optimisation speed-up. K=4000 is the long pole.

---

## 6. File layout — extension work kept separate

```
RERUN/
  FIG6_PLAN.md                  <- this file
  fig6/                         <- everything for the corrected figure
    submit_fig6.sh
    jobs/nt_extend.sbatch
    jobs/corner_eval.sbatch
    build_fig6.py               <- reads corner payload, writes the panels
    out/                        (gitignored)
  runs/fig6-01/                 <- new NT trainings + corner evaluations
  pilot10k/                     <- 10k pilot, self-contained
    generate_10k.sbatch
    submit_pilot.sh
  runs/pilot-10k/
```

Nothing under `fig6/` or `pilot10k/` writes into `runs/replicate-01`,
`runs/seeds-01` or `imputer/ranking/{DATA,RESULTS,PLOTS}`. The pilot never
feeds the figure.

---

## 7. Commit points

One commit per phase boundary, each independently revertible:
P0 data + checksums · P1 corner re-evaluation of existing dots · P2 launch
scripts · P3 new-cell results · P4 the figure · P5 report · P6 pilot.

---

## 8. Risks

| risk | mitigation |
|---|---|
| Regenerating data perturbs existing bundles | checksum every pre-existing bundle before/after; abort on mismatch |
| NT at K=4000 is slow or OOMs | K=4000 T exists and trains inside 4 h pre-optimisation; NT is strictly smaller (no corner). Chunked at max_item=300 |
| Figure not comparable without Stan | stated on the figure and in the report; the Marformer-vs-Marformer comparison (T vs NT) is unaffected |
| Editing scripts under running jobs | never; new files only, and launches wait for the previous phase |
