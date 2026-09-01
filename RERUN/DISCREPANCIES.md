# Paper ↔ code discrepancies

A running register of places where `Marformer-Paper/main.tex` and this
repository disagree, or where the paper describes something the code does not
do. Revisit this before each rerun and again when finalising the paper: some
entries change the reported numbers, others only change the prose.

Status values: **open** (unresolved), **decided** (resolved, action noted),
**stale** (no longer applies).

Started 2026-08-25 while tracing the figure→script map ([FIGURES.md](FIGURES.md)).

---

## D1 — Item chunking: 500 (paper) vs 300 (code)

- **Status:** open
- **Affects:** numbers, for every Domain 3 item-expansion figure at size > 300
- **Paper:** §Architecture for the Annotation Domain, footnote: *"For training
  sets beyond 500 items, we restrict the item graph to at most 500 items per
  forward pass (chunking) to control memory."*
- **Code:** `scripts/DOMAIN3-SIGMOID/run_marformer_group.sh` sets
  `DOMAIN3_MAX_ITEM=300`, consumed as `--max-item` by
  `scripts/DOMAIN3/run_marformer_group.sh`.
- **Why it matters:** the footnote also claims *"empirically the gap is small"*
  between chunked and full passes. If the runs were actually done at 300, that
  claim was measured at a different chunk size than the one stated.
- **Update (2026-08-25):** both numbers are in the code, on different sides of
  the pipeline. *Training* chunks at 300 (`DOMAIN3-SIGMOID/run_marformer_group.sh:22`
  → `--max-item`). *Evaluation* chunks at 500 — `imputer/entity_mf/test.py:238`
  hardcodes `max_item=500`, ignoring the trained value. So a model trained on
  300-item graphs is tested on 500-item graphs: the paper's "500" describes the
  eval path, and train/test see different graph sizes. That is a train/test
  mismatch, not just a documentation error.
- **To resolve:** make eval inherit `max_item` from `train_config.json` rather
  than hardcoding, then decide on a single value and rerun the item axis.
  `RERUN` exposes the training side as `DOMAIN3_MAX_ITEM`, so a comparison is
  `DOMAIN3_MAX_ITEM=500 ./submit_all.sh --run-id chunk500 --data-from baseline --stage marformer`.

## D2 — Early stopping: described but not implemented

- **Status:** open
- **Affects:** numbers, and a methods claim
- **Paper:** same section: *"Training runs for up to 400 epochs with early
  stopping (patience 20 epochs on validation loss"* — with Jason's inline note
  *"why validation loss here and training loss before?"* and a response that
  *"Prabhav thinks there's no early stopping actually"*.
- **Code:** the runner passes `--epochs 400` and no patience/early-stopping
  flag. Every run trains the full 400 epochs — Prabhav is right that nothing
  stops early.
- **But there is model selection**, which is the thing the paper's clause was
  probably reaching for. `train.py:1092` registers a `ModelCheckpoint` with
  `monitor="val/missing_ce", mode="min", save_top_k=1` alongside a periodic
  checkpoint every 25 epochs. So a validation-selected checkpoint *is* written
  (`best-*.ckpt`). See D14 for what the eval pipeline then does with it.
- **To resolve:** describe what actually happens — 400 epochs, no early
  stopping, checkpoint selection after the fact — and fix D14 first, since the
  two together decide what "the model" even means in the reported numbers.

## D3 — Early stopping is described inconsistently across domains

- **Status:** open
- **Affects:** prose only
- **Paper:** §Marformer Training, Jason: Domain 1 discusses an "early stopping"
  rule that actually stops on *training* loss (a convergence diagnostic, not a
  hyperparameter), while Domain 3 discusses stopping on *validation* loss.
- **Open question in the text:** whether the validation set counts toward the
  reported training-set size, and whether competing methods get it.
- **To resolve:** standardise the treatment of convergence and early stopping
  across all three domains. Depends on D2.

## D4 — Masking rate differs by domain, stated in one place

- **Status:** open
- **Affects:** prose
- **Paper:** the masking paragraph says each observed attribute is masked with
  15% probability. Tom's inline note: correct for Domain 3, but Domains 1 and 2
  use 40%.
- **Code:** `--masking-rate 0.15` in the Domain 3 runner — consistent for
  Domain 3. Domains 1/2 cannot be checked against committed code (see D7).
- **To resolve:** state the per-domain rates explicitly rather than a single
  number.

## D5 — Domain 1: is any data actually missing?

- **Status:** open
- **Affects:** a correctness claim about the training setup
- **Paper:** Tom's inline note on the masking paragraph: *"@Prabhav I was
  checking code for domain 1 it seems data is all observed? There is no missing
  observations that we don't train on?"*
- **Why it matters:** the paper's framing is training on *incomplete* data. If
  Domain 1 trains on complete instances with only masking applied, that is a
  different (weaker) claim than the one the setup section makes.
- **To resolve:** needs someone to read `imputer/graphical_synthetic_modeling/`
  and confirm. Blocked by D7 in practice.

## D6 — MCAR mechanism: i.i.d. coin flip or exact half?

- **Status:** open
- **Affects:** prose, possibly data generation
- **Paper:** §Datasets (Domain 1): *"reveals each variable with i.i.d.
  probability 1/2"*, followed by a `\fixme`: *"JASON WILL REWRITE: prabhav says
  at least domains 1 and 3 sample exactly half the variables [rounded to nearest
  integer] on each complete instance, without replacement. Not sure about
  domain 2 because he doesn't have Haojun's scripts."*
- **Code:** Domain 3 generation uses `--mcar-missing-rate 0.5` with
  `--observation-protocol mcar`; whether that is i.i.d. or exact-half needs
  checking in `STAN/stan_code/pipeline/data_gen.py`.
- **Note:** these are different mechanisms. Both are MCAR, so ignorability holds
  either way, but the description should match what ran.

## D7 — Domains 1 and 2 have no runnable code here

- **Status:** open
- **Affects:** reproducibility of 10 figures
- **Detail:** `imputer/gaussian/` (Domain 2) and
  `imputer/graphical_synthetic_modeling/` (Domain 1) entered the repo in one
  commit on 2026-04-11 and were never touched again. They are byte-identical to
  the copies in `AA_new`. Neither has a runner script; neither mentions SMAR;
  Domain 2's run scripts are Haojun's and were never committed — `main.tex`
  says so. See the "Orphaned" table in [FIGURES.md](FIGURES.md).
- **To resolve:** get Haojun's Domain 2 scripts; establish whether Domain 1's
  committed code is the version that produced the paper's figures.

## D8 — Logit Lens appendix has no code

- **Status:** open
- **Affects:** reproducibility of 2 figures and ~10 quoted numbers
- **Detail:** the appendix reports Param Lens / Tuned Param Lens / Tuned 80d-Lens
  results at Domain 3, item generalization, T=500, L=8, on 414 held-out
  test-missing ratings, with specific values (log-loss 1.386 → 0.566 at layer 1;
  smECE 0.163 → 0.027). No `lens`, `layerwise`, or tuned-probe code exists in
  either repository, so the analysis ran from a script that was never committed.
- **To resolve:** recover the script, or reimplement — the appendix describes
  the method in enough detail (linear probes, 300 epochs, AdamW, KL to the final
  model's predictions, on a held-out set of training-missing ratings).

## D9 — `scripts/README.md` says LLM-Rubric data cannot be regenerated

- **Status:** decided (2026-08-25)
- **Affects:** reproducibility of 5 figures
- **Was:** `scripts/README.md`: *"There is no generation script — if a size you
  need isn't present, ask the repo owner for it rather than trying to synthesize
  it."*
- **Actually:** `scripts/convert-llm-rubric/` (brought over from `AA_new` on
  branch `patches`) contains the raw TSVs and
  `make_llmrubric_splits.py`, which rebuilds
  `DATA/LLM_RUBRIC/LLMRubric_225_25_9_{10,50,100,150,175}/` deterministically
  (seed 42). Verified working. Wrapped as
  `RERUN/jobs/make_llmrubric_data.sh --run-id <ID>`.
- **Update (2026-08-25, after the Globus re-transfer):** the original bundles
  came back in `AA_new` — all ten sizes (10/20/30/40/50/75/100/125/150/175),
  18 MB — and are now staged at `RERUN/shared/DATA/LLM_RUBRIC/`.
  `make_llmrubric_data.sh` links those by default and only converts with
  `--rebuild`. So the converter is a reproducibility path, not the only source.
- **Remaining:** the converter's `TRAIN_SIZES` produces only 10/50/100/150/175,
  so a from-scratch rebuild is missing 20/30/40/75/125. If the paper's real-data
  curves use those points, extend that list so the two sources agree.

## D10 — LLM-Rubric Stan results path mismatch

- **Status:** open (worked around)
- **Affects:** whether the real-data figures build at all
- **Detail:** `scripts/LLMRubric/CPMSigmoid/run_size*.sh` writes to
  `RESULTS/STAN/LLM_RUBRIC/CPMSigmoid`; `plot_llmrubric_cpm_sigmoid.py` reads
  `RESULTS/STAN/CPMSigmoid`. Nothing reconciles them.
- **Workaround:** `RERUN/make_figures.sh` creates a compatibility symlink.
- **To resolve:** pick one path and fix the other side.

## D11 — `BASELINES/` was gitignored but is imported by the plot script

- **Status:** decided (2026-08-25)
- **Affects:** reproducibility of 12 figures from a fresh clone
- **Detail:** `plot_domain3_sigmoid.py` imports
  `BASELINES.structured_baselines`, but `.gitignore` listed
  `imputer/ranking/BASELINES`, so the module was never committed —
  `ModuleNotFoundError` on any fresh clone.
- **Resolved:** the ignore rule is removed and the directory is tracked on
  branch `patches`.

## D12 — Duplicated figure includes in `main.tex`

- **Status:** open
- **Affects:** the PDF only
- **Detail:** the six main-text Domain 3 figures are included twice — once in
  §Experiments and Results (~lines 1592–1639) and again in §"PLACEHOLDER for old
  material" (~lines 1755–1795), which is marked for deletion once its text is
  merged.

## D13 — Open asks in `main.tex` that need code that does not exist

- **Status:** open
- **Affects:** planned additions
- **Detail:** MALA curves for Domain 2 (and ideally Domain 1); extending the
  Domain 1/2 x-axis the way Domain 3 was extended, since the Marformer curves
  are still improving at 2000; renaming baseline legends to `MAP (EM)` /
  `MAP (SGD)` / `MAP (SGD -- true init)`. All blocked on D7.

## D14 — Reported checkpoints are selected on the **test** set

- **Status:** open
- **Affects:** numbers, in every Domain 3 figure; and a methodological claim
- **Detail:** training writes ~18 checkpoints per run (`best-*` selected on
  validation, `last`, and `periodic-*` every 25 epochs). Evaluation then
  ignores the validation-selected one and re-ranks *all* of them on test:
  - `imputer/entity_mf/test.py:229` evaluates each checkpoint with
    `split="test"`;
  - `scripts/DOMAIN3/run_marformer_eval.sh:97` passes `--checkpoint all`, then
    sorts by `missing.log_loss` and writes the winner as `best.json` with
    `"selection_metric": "missing.log_loss"`;
  - `plot_domain3_sigmoid.py:322` (`_write_best_json_from_test_results`)
    does the same thing again if `best.json` is absent.
  Every plotted Marformer point is therefore a **min over ~18 checkpoints of
  test log-loss**, not a fixed model evaluated once.
- **Why it matters:** this is test-set leakage through model selection. It
  biases Marformer's log-loss and calibration favourably relative to the Stan
  baselines, which get no such per-size selection. It also directly answers
  Jason's question about which split calibration is measured on: the *metrics*
  are computed on test, but the *checkpoint* was chosen using the same test
  split, so the calibration numbers are optimistic.
- **The real-data pipeline differs**, which makes the inconsistency worse.
  `plot_llmrubric_cpm_sigmoid.py` reads `TEST_RESULTS/best.json` and
  `last.json`, which `test.py --checkpoint both` writes from the
  *validation*-selected `best-*.ckpt` and the final epoch. So the LLM-Rubric
  numbers are nearly honest — a 2-way min over best/last — while the synthetic
  Domain 3 numbers are a min over ~18 checkpoints. The two sets of figures in
  the paper are therefore not selected the same way, and are not directly
  comparable as evidence.
- **Magnitude, measured** (27 fully-evaluated runs, `RERUN/runs/resid-01`):
  choosing the checkpoint on test rather than taking the last one is worth
  **+0.196 nats at K=100**, +0.028 at K=300, +0.021 at K=500. At K=100 that is
  comparable to the entire reported margin between Marformer and its baselines.
  The reason it is so large at K=100 is that the model genuinely overfits there
  — the test curve bottoms at 0.933 (epoch 124) and rises to 1.138 by epoch 399,
  with accuracy falling 0.560 → 0.488. Test-set selection hides that entirely.
- **To resolve:** select on validation (the `best-*.ckpt` already exists) and
  report test metrics for that one checkpoint, in both pipelines. Re-running
  eval is cheap — the checkpoints are already on disk — so this can be
  quantified as "test-selected vs validation-selected" without retraining.
  `RERUN/analysis/checkpoint_selection.py` does exactly that.

## D15 — Masking is exactly-k, not an i.i.d. coin flip (Domain 3 training)

- **Status:** decided (2026-08-25), prose fix
- **Affects:** prose
- **Detail:** `imputer/entity_mf/masking.py` (`MCARMasking.mask`) draws
  `num_to_mask = int(len(observed_vars) * masking_rate)` without replacement,
  so exactly 15% of observed variables are masked per step, not each
  independently with probability 0.15. Same distinction Jason flags for data
  generation in D6, but this one is about the *training* mask.

## D16 — Observed tokens carry a loss term too

- **Status:** open
- **Affects:** how the objective is described; relevant to the overfitting thread
- **Detail:** the paper describes masked auto-encoding. The code optimises
  `masked_loss_weight * masked_loss + observed_loss_weight * observed_loss`
  with weights 15.0 and 1.0 (`eval.py:121`, `run_marformer_group.sh`). The
  observed term asks the model to predict tokens whose values are visible in
  its own input — a copy task, achievable at near-zero entropy.
- **Why it matters:** it is a standing pressure toward low predictive entropy,
  which is exactly the mechanism Jason worries about. It needs either a mention
  in the paper or an ablation showing it does not affect calibration.

## D17 — Entity-deviation dropout is not rescaled at eval

- **Status:** open
- **Affects:** numbers on the annotator axis; a train/test mismatch
- **Detail:** `model.py::_build_initial_streams` zeroes an entity's deviation
  with probability `dropout_rate` while `self.training`, and uses the full
  deviation at eval. There is no `1/(1-p)` inverted-dropout rescaling, so the
  expected contribution of the deviation is `(1-p)x` during training and `1x`
  at evaluation. On the annotator axis `p = 0.8`, a 5x scale change between
  train and test.
- **Why it does not blow up:** the learned deviations end up tiny anyway
  (`|w|max = 0.0015` for annotators after 400 epochs), so 5x of nearly nothing
  is still nearly nothing. But that is a consequence, not a design — and it
  partly explains why the deviations stay small.
- **On the item axis `p = 1.0`**, so the deviations receive no gradient at all
  and remain exactly zero. That is load-bearing for the overfitting argument
  (see `reports/OVERFITTING.md` §3) and should be stated in the paper rather
  than left implicit in a runner script.
- **To resolve:** either rescale, or document that this is deliberate
  structural regularisation rather than dropout in the usual sense.

## D18 — LLM-Rubric runtime figure uses hardcoded numbers

- **Status:** open
- **Affects:** one appendix figure
- **Detail:** `plot_llmrubric_cpm_sigmoid.py` carries
  `marformer_total_runtime_seconds` as a literal dict of hand-entered values
  (e.g. `10: (4*60+47) * 70/300.0`), rather than reading elapsed time from the
  runs. The same is true of `stan_runtime_seconds`. So the runtime curve cannot
  be regenerated from a rerun — it will silently show the old machine's timings.
- **Why it matters:** the paper quotes "18 minutes versus 162 minutes ... a
  roughly 9x speedup" from this figure. Those numbers describe the old cluster.
- **To resolve:** read elapsed seconds from each run's launcher metadata, which
  `run_marformer_group.sh` already records as `elapsed_seconds`.

## D19 — The RERUN matrix was missing the largest item sizes

- **Status:** decided (2026-08-25)
- **Affects:** 3 points on every item-generalization figure
- **Detail:** the paper sweeps item generalization over
  `{100,...,1000} u {2000,3000,4000}` for Marformer (Stan stops at 500), but
  `RERUN/jobs/generate_data.sbatch` and `submit_all.sh` stopped at 1000.
- **Resolved:** both extended. The base bundle already holds `K_TRAIN=4800`
  items, so the large sizes are subsets of it — `SUBSPLIT_ONLY=1` re-derives
  the splits from an existing base without re-running HMC.

## D20 — `test.py` evaluates at a hardcoded `max_item=500`, ignoring the trained value

- **Status:** open
- **Affects:** every reported Marformer number, and LLM-Rubric especially
- **Detail:** `imputer/entity_mf/test.py:238` passes `max_item=500` literally,
  regardless of what the run trained with (`train_config.json ->
  training.max_item`). Domain 3 synthetic trains at 300; LLM-Rubric trains at
  **10**. So a model trained on 10-item graphs is scored on a 225-item graph.
- **And the two sides of the pipeline disagree.**
  `plot_llmrubric_cpm_sigmoid.py::_marformer_probs_and_labels` re-scores with
  `max_item=10`, matching training. So the *checkpoint* is selected using
  `max_item=500` metrics from `test.py`, and the *number that gets plotted* is
  then recomputed at `max_item=10`. Selection and reporting use different graph
  constructions.
- **Measured** (LLM-Rubric K=40, `periodic-epoch=0024.ckpt`):

  | eval `max_item` | test log-loss | accuracy |
  |---|---|---|
  | 10 (trained value) | 1.2373 | 0.4099 |
  | 25 | 1.2268 | 0.4099 |
  | 50 | 1.2231 | 0.4550 |
  | 225 / 500 | 1.3020 | 0.4444 |

  At K=40 that is only ~0.065 nats, but **the error grows sharply with training
  size**, because a larger training set means a larger single graph and so a
  bigger departure from the 10-item graphs the model was trained on:

  | K_train | eval @10 (trained) | eval @50 | eval @500 (`test.py`) | paper |
  |---|---|---|---|---|
  | 50  | **1.0494** | 1.0336 | 1.2798 | 1.050 |
  | 175 | **0.9775** | 1.0751 | 1.7828 | 0.998 |

  So `best.json` for LLM-Rubric is wrong by up to **0.8 nats** at K=175, and
  wrong in a size-dependent way that would invert the trend if anyone plotted
  it. The published figures are unaffected only because the plot script
  re-scores at `max_item=10` — at which point the numbers match the paper
  closely (1.0494 vs 1.050; 0.9775 vs 0.998). Checkpoint *selection*, however,
  still uses the `max_item=500` numbers.
- **To resolve:** read `max_item` from `train_config.json` in `test.py` instead
  of hardcoding, and use the same value on both sides. Related to D1, which is
  the same class of bug on the training side.

## D21 — the per-epoch evaluation runs with dropout on

- **Where:** `imputer/entity_mf/train.py`, `on_train_epoch_end`.
- **What the code does:** it calls `evaluate_entity_marformer_split` without
  ever putting the model in eval mode. `test.py` does call `model.eval()`
  (line 170), and it is the only place in the training/eval path that does.
  So every number in `training_history.json` — including the
  `combined_missing` / `val_missing` cross-entropies the training curves are
  drawn from, and the `val/missing_ce` the "best" checkpoint is selected on —
  is computed with `nn.Dropout(0.1)` active in every block and with entity
  deviations dropped at their training rates (1.0 on the item axis, 0.8 on the
  annotator axis).
- **Does it move numbers?** Not the reported ones: every published metric comes
  from `test.py`, which evaluates correctly. It does move the *curves*, which
  are noisier and biased high, and it makes the `best` checkpoint selection
  partly a lottery over dropout masks. On the annotator axis, where the
  annotator deviations are dropped 80% of the time, the effect is larger than
  on the item axis.
- **Found:** while profiling (`reports/TRAINING_SPEED.md`); the evaluation also
  ran with autograd enabled, which was fixed there because it is purely a
  speed matter. The train/eval mode question is not, so it was left alone.
- **To resolve:** wrap the epoch-end evaluation in `model.eval()` /
  `model.train()`. That changes every recorded training curve, including the
  ones `reports/OVERFITTING.md` reads, so it wants a deliberate rerun rather
  than a quiet fix. Related to D17 (dropout is not rescaled at eval either).

## D22 — the reported "test" metric is mostly *training*-partition cells

- **Where:** `imputer/entity_mf/test.py`, `_reconstruct` (line ~77) feeding
  `evaluate_entity_marformer_split`.
- **What the code does:** it builds one evaluation graph from
  `train_obs + test_obs + test_miss + train_miss`, and then reports metrics over
  **every** status-0 token in that graph. So the number labelled "test" averages
  the held-out test cells together with the training partition's own missing
  cells. The pooling is not a rounding detail; the test cells are a minority
  almost everywhere, and their share shrinks as the sweep grows, because the
  test block is a fixed 100 items (414 cells) while the training partition grows
  with K:

  | split | test cells | training cells | test share of the reported number |
  |---|---|---|---|
  | `Item_T_500` | 414 | 1 317 | 24% |
  | `Item_T_1000` | 414 | 3 002 | 12% |
  | `Item_T_4000` | 414 | 13 836 | **2.9%** |
  | `Annot_T_30` | 414 | 701 | 37% |
  | `Annot_T_50` | 414 | 1 424 | 23% |

- **Does it move numbers?** Yes, and it moves the shape of the headline figure.
  Marformer, item axis, last checkpoint, mean over 3 seeds:

  | K | published (pooled) | test cells only |
  |---|---|---|
  | 100 | 1.0966 | 1.0966 |
  | 200 | 0.9805 | 1.0127 |
  | 500 | 0.8639 | 0.9179 |
  | 1000 | 0.8681 | 0.9338 |
  | 2000 | 0.7874 | 0.8730 |
  | 4000 | **0.7691** | **0.8590** |

  The published curve falls 0.33 nats from K=100 to K=4000 and is monotone. The
  test-only curve falls 0.24 nats, is **not** monotone (it flattens after
  K≈500 and moves within noise thereafter), and sits 0.09 nats higher at the
  right-hand end. The item-generalization claim is therefore weaker than the
  figure shows, and the part of the curve beyond K≈1000 is mostly measuring
  in-domain matrix completion on training items -- which the transductive model
  has already seen the rest of -- rather than generalization to unseen items.

- **It also interacts with the validation work.** Scoring on the pooled metric,
  block validation beats no-validation in 28 of 38 cells (mean -0.034 nats).
  Scoring on the test cells alone it wins **35 of 38** (mean **-0.064**). Early
  stopping costs a little on the training-partition cells and gains a lot on the
  test cells, so the pooled number understates it by roughly half, and the
  apparent "failure band" at non-transductive item K=300-700 largely disappears.
- **Verified:** `RERUN/analysis/split_missing_metrics.py` recomputes both from
  the same forward pass; `_verify_split.py` confirms it reproduces the stored
  `last.json` log-loss to six decimals at test.py's own chunk size, so the split
  is a decomposition of the published number, not a different measurement.
- **To resolve:** report the test partition separately. Either restrict the
  metric to `instance == "test"` cells, or report both and say which is which.
  Related to D20, which is the same class of bug (an evaluation setting that
  does not match what was intended) on the chunking side.

## D23 — Marformer and Stan are scored on different cell sets in the same figure

- **Where:** `scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py`.
  `_marformer_probs_all_missing` (line ~383) vs `_stan_probs_from_eval_dir`
  (line ~510).
- **What the code does:** the two series in the item/annotator generalization
  figures are built from different populations.
  - **Stan** restricts to the test block:
    `test_idxs = bundle["missing_ratings_indexes_in_test_instance"]`, then
    `df[df["missing_rating_idx"].isin(test_idxs)]`. Test cells only, n=414.
  - **Marformer** takes `eval_vars` from `test.py::_reconstruct` — which is
    `train_obs + test_obs + test_miss + train_miss` (D22) — and keeps *every*
    `is_missing` token. Train and test pooled. The plotted log-loss is then
    overridden by `best.json`'s `missing.log_loss`, which is pooled as well.
- **Why it matters:** this is not the same measurement. Training-partition
  missing cells are in-domain completion on items the transductive model has
  seen most of; test cells are unseen items. The training cells are easier —
  Marformer's train log-loss is below its test log-loss in **all 38 runs** of
  `replicate-01`. So Marformer is scored on a progressively easier mixture
  (97% training cells at K=4000) while Stan is scored purely on the hard task.
- **Magnitude** (Marformer, item axis, transductive, seed 42, last checkpoint —
  the plotted value against its own test-only value):

  | K | plotted (pooled) | test cells only | unearned advantage |
  |---|---|---|---|
  | 200 | 0.9234 | 0.9570 | 0.034 |
  | 500 | 0.8454 | 0.9051 | 0.060 |
  | 1000 | 0.8380 | 0.9160 | 0.078 |
  | 2000 | 0.7840 | 0.8775 | 0.094 |
  | 4000 | 0.7640 | 0.8492 | 0.085 |

  Correcting it moves the Marformer curve *up* by 0.03-0.09 nats with Stan
  unchanged. `ITEM_STAN_CUTOFF = 500` means the exact-comparison region carries
  a 0.03-0.06 nat distortion. The calibration figures inherit it too, since
  smECE is computed from the same payloads.
- **Unresolved:** whether this changes the ranking or only the margin. Stan was
  not rerun, so there are no Stan results in this tree to compare against. The
  Marformer side is already corrected for all 38 settings in
  `RERUN/_d22/split.json`; this needs Stan's per-size log-losses to close.
- **To resolve:** score Marformer on `missing_ratings_indexes_in_test_instance`
  too, which is the same fix as D22 applied at the plotting layer. No
  retraining — the checkpoints are on disk.

---

## How to add an entry

Keep the shape: what the paper says, what the code does, whether it moves
numbers or only prose, and what would resolve it. Anything that changes numbers
should say which figures are affected, so a rerun can be scoped.

## D24 — item chunking puts a visible discontinuity in the generalization curve

- **Where:** `DOMAIN3_MAX_ITEM=300` in
  `scripts/DOMAIN3-SIGMOID/run_marformer_group.sh`, consumed as `--max-item`,
  with chunking done by `_chunk_variables_by_item_limit`.
- **What happens:** the item-generalization curve jumps at exactly the first
  training size whose graph no longer fits in one chunk, in *both* arms:

  | bundle | items in graph | chunks @300 | corner log-loss |
  |---|---|---|---|
  | `Item_T_500` | 295 | 1 | 0.9179 |
  | **`Item_T_600`** | **356** | **2** | **1.0934** |
  | `Item_T_700` | 417 | 2 | 0.9748 |
  | `Item_NT_400` | 295 | 1 | 0.9365 |
  | **`Item_NT_500`** | **356** | **2** | **1.0786** |
  | `Item_NT_600` | 417 | 2 | 0.9898 |

  Roughly +0.17 nats at the boundary, decaying as further items are added. It is
  not seed noise -- all three seeds show it -- and it is not a bad item draw:
  `Item_T_600` and `Item_NT_500` have byte-identical training item sets (290
  items), which is exactly why the two curves spike 100 apart on the x-axis.
- **Why it matters:** the published figure inherits a discontinuity that is an
  artifact of an implementation detail, not a property of the model. It falls in
  the middle of the range where the paper compares against Stan.
- **Relation to D1:** the paper states chunking at **500**, not 300. At 500 the
  first two-chunk size would be ~900 items instead of ~600, so the artifact
  should move. `chunk500-01` tests exactly that at sizes 600-900.
- **CONFIRMED (chunk500-01, 24 runs, 3 seeds).** The prediction was that at
  `max_item=500` the first two-chunk size moves from ~600 items to ~900, taking
  the spike with it. It does, exactly:

  | K | items | chunks@300 | LL @300 | chunks@500 | LL @500 |
  |---|---|---|---|---|---|
  | 500 | 295 | 1 | 0.9179 | 1 | 0.9201 |
  | **600** | 356 | **2** | **1.0934** | **1** | **0.9384** |
  | 700 | 417 | 2 | 0.9748 | 1 | 0.9064 |
  | 800 | 474 | 2 | 0.9517 | 1 | 0.9038 |
  | **900** | 527 | 2 | 0.9375 | **2** | **1.0264** |
  | 1000 | 578 | 2 | 0.9338 | 2 | 1.0293 |
  | 2000 | 1189 | 4 | 0.8730 | 3 | 0.9113 |
  | 4000 | 2375 | 8 | 0.8590 | 5 | 0.8566 |

  The spike is wherever the graph first fails to fit in one chunk, and nowhere
  else. Raising the chunk size does not remove it, it relocates it.
- **Note on the footnote's claim.** "Empirically the gap is small" holds at the
  large end -- 0.8590 vs 0.8566 at K=4000 -- so chunk size is nearly free once
  many chunks are in play. The cost is a *boundary* effect, roughly +0.15 nats
  at the single size where the split first happens, not a monotone penalty.
- **To resolve:** the fix is not a bigger chunk. Chunking must keep the test
  corner whole and give it training context in the same chunk, so the corner is
  never evaluated in a starved final chunk. Until then any item-axis figure
  carries a spike whose position is set by `max_item`.

## D25 — the non-transductive arm overfits badly past K≈900, and the paper never looks

- **Where:** the item-axis Marformer NT sweep. The published figure stops at
  K=500 for NT (`plot_domain3_sigmoid.py` caps it lower still,
  `ITEM_MARFORMER_NT_MAX = 500`), so nothing beyond that has been plotted.
- **What happens:** extending NT to the transductive arm's full range, the
  corner log-loss *reverses* — it improves to K≈900 and then gets steadily
  worse, while the transductive arm keeps improving:

  | K | Marformer T (corner) | Marformer NT (corner) |
  |---|---|---|
  | 900 | 0.9375 | 0.9529 |
  | 1000 | 0.9338 | 1.1082 |
  | 2000 | 0.8730 | 1.1265 |
  | 4000 | 0.8590 | 1.1349 |

- **It is overfitting, and this is measured, not inferred.** Scoring every
  periodic checkpoint on the corner (`fig6/corner_curve.py`):

  | run | corner min | at epoch | final (ep 399) | inflation |
  |---|---|---|---|---|
  | NT K=1000 | 0.9661 | 124 | 1.1082 | **+0.142** |
  | NT K=4000 | 0.9742 | 99 | 1.1081 | **+0.134** |
  | T K=4000 | 0.8516 | **399** | 0.8516 | +0.000 |

  The NT curves bottom out around epoch 100–125 and climb monotonically to the
  end. The transductive run at the same size has its minimum at the *last*
  epoch and no inflation at all.
- **Why the two arms differ:** transductive training puts the test corner's
  observed cells in the training graph, so the model is anchored to the
  evaluation distribution throughout. NT has no such anchor and overfits its
  training items. The effect grows with K because more items mean more chunks,
  hence more optimizer steps per epoch at a fixed 400 epochs.
- **What it means for the paper.** main.tex:747 says the non-transductive
  variant "has comparable accuracy at large training sizes". That claim is
  **untested** — NT is never plotted past K=500 — and on the last checkpoint it
  is false by ~0.28 nats at K=4000. Early-stopped it is much closer (0.974 vs
  0.852) but still a real gap, and the gap no longer shrinks with K.
- **To resolve:** NT needs early stopping to be reported honestly at all. The
  block-validation refit protocol supplies it; `fig6-val-01` runs it for
  K=1000–4000. Either extend the NT curve with that protocol, or state
  explicitly that the comparability claim only covers K ≤ 500.

## D26 — the block-validation carve is shaped unlike the test corner, and under-stops the NT arm

- **Where:** `imputer/entity_mf/validation.py`, the `block` carve, as driven by
  `RERUN/submit_validation.sh`.
- **What the carve does:** it holds out whole **items** from the training
  partition, moving both their observed and their missing cells out
  (`validation.py:94`), so validation items really are unseen. The carve is
  principled; the problem is that its signal does not track the corner.

  **CORRECTION.** An earlier version of this entry blamed annotator shape --
  "the carve attaches all 50 annotators, the corner has 10". That is wrong on
  the item axis: the item-split bundles contain only **10 annotators in total**,
  the same 10 as the corner, with ~12 ratings per item in both blocks. Corner
  and training block are structurally alike, so annotator density is not the
  explanation.
- **Consequence:** on the non-transductive arm the validation curve peaks much
  later than the corner does, so the selected stopping epoch is too late.
  NT item, K=1000, seed 42:

  | signal | epoch |
  |---|---|
  | corner optimum (ground truth) | ~124 |
  | block-val, this run's own argmin | 208 |
  | block-val, pooled over seeds + smoothed | 312 |
  | refit budget used | 313 |

  The protocol is not blind — `best_val_ce` 0.880 rises to `last_val_ce` 0.950,
  so it *does* see overfitting — it just calls it ~190 epochs late. The pooling
  and smoothing step, added because per-run argmins scattered 78-385, pushes the
  choice later still.
- **What the evidence actually says.** The validation curve tracks the
  *training* cells, not the corner. At NT K=1000: train cells 0.8392, val CE
  0.880-0.950, corner 1.0778. Validation sits with the training distribution
  while the corner is 0.24 nats away. Since NT's failure is transfer to the
  corner rather than a fit failure (D25a: its training cells reach 0.7659 at
  K=4000, matching the transductive arm's 0.7664), a validation set carved from
  the training partition cannot see it. **The mechanism for why a held-out block
  still behaves like training data is not yet established** -- val target counts
  (400) and block sizes (~87 items vs the corner's 66) are comparable, so the
  obvious size explanation does not hold either. This is open.
- **What it costs on the corner** (3 seeds, item axis, NT):

  | K | plain 400 ep | block-val refit | corner-optimal |
  |---|---|---|---|
  | 1000 | 1.0778 | 1.0519 | **0.9661** |
  | 2000 | 1.1265 | **1.1408** | — |
  | 3000 | 1.0782 | 1.0740 | — |
  | 4000 | 1.1349 | 1.0632 | **0.9742** |

  The refit recovers roughly a quarter to a half of the achievable gap, and at
  K=2000 it is *worse* than not stopping early at all.
- **Scope.** This bites the NT arm only. The transductive arm does not overfit
  at these sizes (D25: corner minimum at epoch 399, zero inflation), so a late
  stop costs it nothing, and the existing `val-block-01` results stand.
- **To resolve:** carve validation blocks shaped like the corner — 100 items x
  10 annotators — instead of whole items, so the selection signal and the
  reported metric measure the same regime. Until then, NT points at K >= 1000
  are protocol-limited and should be labelled as such rather than read as the
  model's capability.
