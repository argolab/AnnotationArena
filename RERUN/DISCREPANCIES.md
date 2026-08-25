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
- **To resolve:** decide which is correct, then either change the constant and
  rerun the item axis, or fix the footnote. `RERUN` exposes this as
  `DOMAIN3_MAX_ITEM`, so a comparison run is
  `DOMAIN3_MAX_ITEM=500 ./submit_all.sh --run-id chunk500 --data-from baseline --stage marformer`.

## D2 — Early stopping: described but not implemented

- **Status:** open
- **Affects:** numbers, and a methods claim
- **Paper:** same section: *"Training runs for up to 400 epochs with early
  stopping (patience 20 epochs on validation loss"* — with Jason's inline note
  *"why validation loss here and training loss before?"* and a response that
  *"Prabhav thinks there's no early stopping actually"*.
- **Code:** the runner passes `--epochs 400` and no patience/early-stopping
  flag. Every run trains the full 400 epochs.
- **To resolve:** if there is no early stopping, delete the clause and say so
  explicitly (reviewers will ask about regularisation). If there should be,
  it needs implementing and a full rerun.

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
- **Remaining:** the old README lists sizes 20, 30, 40, 75 and 125 as available;
  the converter's `TRAIN_SIZES` produces only 10/50/100/150/175. Extend that
  list if the paper's curves need the intermediate points. Both sizes used by
  the calibration figures (10 and 175) are covered.

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

---

## How to add an entry

Keep the shape: what the paper says, what the code does, whether it moves
numbers or only prose, and what would resolve it. Anything that changes numbers
should say which figures are affected, so a rerun can be scoped.
