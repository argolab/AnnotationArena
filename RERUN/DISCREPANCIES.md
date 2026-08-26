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

---

## How to add an entry

Keep the shape: what the paper says, what the code does, whether it moves
numbers or only prose, and what would resolve it. Anything that changes numbers
should say which figures are affected, so a rerun can be scoped.
