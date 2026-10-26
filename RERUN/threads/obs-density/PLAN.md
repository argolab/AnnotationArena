# Observation-density thread (Phase 1)

**Question.** Domain 3's synthetic data is generated at one observation rate
(50%: `MCAR_MISSING_RATE=0.5` in `jobs/generate_data.sbatch`). Does Marformer
still impute when the data is much sparser or much denser, and does it hold its
position relative to Stan at every density?

**Phase 1 scope** (agreed 2026-10-17): the diagonal only -- three densities,
training and test block at the same density, one training size, item axis,
transductive. No 2D train-density x test-density table. The off-diagonal is a
distribution-shift question, costs a code change (see "Why not the 2D table"),
and its differences would sit inside the across-draw noise floor.

## Design

| | |
|---|---|
| densities | 15% / 50% / 85% observed |
| size | K=500 (`Item_T_500`: K_train=400, K_test=100, J=10, I=9) |
| axis / protocol | item, transductive |
| generative draws | 6 (`dataseed-s42, s101..s105`) |
| model seeds | 42, 43 |
| Marformer | search (cap 2000 epochs) -> select -> refit -> eval |
| validation carve | `mcar`, frac 0.15, split seed 7 |
| Stan | all four arms (Oracle TENSOR-SIG, Bin, Proj, Both), transductive |
| scored target | test-instance missing cells (the fixed 100x10 corner) |
| cells | 6 draws x 3 densities = 18 |

**Headline number is the Marformer - Stan gap within each density cell.** We do
not compare log-loss across densities: the three columns are means over
different target sets with different error bars. Within a column both families
see exactly the same cells, so the gap is clean.

Measured cell contents, draw s101, `Item_T_500` (3492 existing cells):

| observed rate | realised | train obs/missing | test obs/missing |
|---|---|---|---|
| 0.15 | 0.1558 | 444 / 2409 | 100 / **539** |
| 0.50 | 0.5017 | 1445 / 1408 | 307 / **332** |
| 0.85 | 0.8411 | 2398 / 455 | 539 / **100** |

Bold is the scored corner. The realised rate drifts from the nominal because
the split keeps only the 10 test annotators, a non-random slice of the cells the
rate was applied to; the rate is exact on the base bundle (0.1500 / 0.5000 /
0.8500 of 180000).

The nested masks give a common core of exactly the 100 cells missing at 85%
observed -- same cells, three amounts of context. Kept as a paired secondary
read now that the headline is the within-column Stan comparison.

## Nested masks

`domain3subsplit.py --observed-rate R --mask-seed S` redraws the
observed/missing partition MCAR before splitting. The permutation depends only
on `(S, n_ratings)`, so with `S` held fixed across the three rates the observed
sets nest:

    observed(0.15) subset observed(0.50) subset observed(0.85)

Three consequences, all wanted:

1. The three cells share one generative draw -- identical latent scores,
   embeddings, annotator parameters. They differ *only* in which cells are
   revealed, so the across-draw variance (~0.09 nats on a corner score,
   `DISCREPANCIES.md` D23) cancels instead of being paid three times.
2. The missing sets nest the other way, so the ~96 cells held out in all three
   conditions form a common core: same cells, three amounts of context. Kept as
   a paired secondary read, not the headline.
3. No Stan regeneration. The six 300 MB base bundles are reused as-is.

The 50% column is therefore a *re-draw* of the mask on the same latent data,
not the published `Item_T_500` mask. It should land within mask-draw noise of
the published number -- that is the first thing to check when results land.

## Validation carve: why `mcar`, not the frozen `block`

The two carves in `imputer/entity_mf/validation.py` scale in **opposite**
directions with density, and the frozen `block` carve is the one that gets
thinner as more is observed:

| carve | what is scored as `val/missing_ce` | size at K=500, 15/50/85% observed |
|---|---|---|
| `block` | the val item block's **already-missing** cells | 361 / 211 / 68 |
| `mcar` | observed training cells **hidden** by the carve | 67 / 217 / 360 |

(From the measured cell contents below, times `--val-frac 0.15`.)

`block` picks a block of training items and relabels everything on them,
including their missing cells -- and those missing cells are what the metric
averages over (`train.py:668-691`, `val_eval_vars = val_all` when
`val_observed` is non-empty). The count of already-missing cells falls as the
observed rate rises, hence the inverse scaling.

`mcar` instead takes observed training cells out of the observed list and
scores those (`validation.py:97-117`); `val_observed` ends up empty so
`val_eval_vars = train_observed + val_missing`, i.e. training context plus the
hidden cells. Its size tracks the observed count, which is the behaviour this
thread wants, and its question -- hide a cell that was observed, predict it --
is exactly what the scored corner cells are under MCAR generation. Either way
the carved cells are absent from the training graph entirely
(`val_missing_in_graph=(val_protocol == "none")`, `train.py:1474`), so selection
is honest in both.

One further reason, specific to the transductive arm: `block`'s val items are
**still trained on** there, because `maskable_sources = train_observed +
val_observed + test_observed` (`train.py:282-284`). So in the transductive arm
`block` is not unseen-item validation at all -- it is clustered held-out cells
on seen items, which is the same question `mcar` asks, less evenly spread. The
docstring's rationale for preferring `block` holds only in the
non-transductive arm.

**Bridge run.** Because the frozen final config uses `block`, the 50% cell is
run under *both* carves on draw s101. If the selected epoch and the corner
score agree, the switch to `mcar` is a free choice; if not, that is itself worth
knowing and the thread reports it.

**Thin-carve risk.** At 15% observed the carve is only ~64 cells. Mitigations:
`--smooth 5` on the selection curve (already the frozen recipe) and a
`--val-frac 0.30` sensitivity run on the 15% cell of draw s101. If selected
epochs come back wild at 15%, switch to `--select per-size` (pools the two model
seeds) and say so.

## Missingness mechanism (for the write-up)

Three separate things, worth keeping straight in the text:

1. **Which cells exist** -- structured, not MCAR. Each item is rated by exactly
   `num_annotate_annotator = 4` annotators drawn uniformly without replacement
   from all 50 (`tensor_generation_sigmoid.stan:272-287`). An item-split bundle
   keeps only annotators 41-50, so an item contributes a cell only if one of its
   4 draws landed in that window: ~0.78 of the 10 per item, and 206 of the 500
   items in `Item_T_500` have no cells at all. The support is a random 4-regular
   bipartite design, column-sliced. Structured, but ignorable -- it does not
   depend on the ratings' values.
2. **Observed vs missing among existing cells** -- pure MCAR, and this is the
   knob this thread turns. `apply_mcar_protocol` shuffles all existing ratings
   in one list and flags a fixed fraction missing, uniformly across
   train/val/test (`data_gen.py:570-572`); `_remask_nested` is the same
   mechanism with the rate as an argument.
3. **Training-time masking** -- MCAR over the observed set, redrawn every step,
   5 augmentations per epoch (`masking.py:48-66`). Held at 0.15 throughout.

So this thread varies observation *density* under a fixed, ignorable
mechanism. It is not a missing-not-at-random result. (LLM-Rubric is the
structured-missingness case: LLM annotator observed everywhere, humans only on
train items.)

## Known confounds, to state rather than fix

- **Target-set size moves inversely with density** (~543 scored cells at 15%,
  ~96 at 85%). Handled by comparing within-column against Stan, plus the
  common-core read.
- **Supervised tokens per step move ~6x.** Masking rate stays 0.15, so the
  masked-token count per step is 0.15 x observed: ~79 at 15% vs ~445 at 85%.
  This is the honest "same hyperparameters" choice. A probe arm at 15% with a
  raised masking rate would show whether the supervision budget is what binds;
  deferred to Phase 2.

## Why not the 2D table

`apply_mcar_protocol` applies one rate to the whole rating list at once, so
train-rate != test-rate needs the MCAR draw stratified by instance. Beyond
that: in the transductive arm the test block's observed half is part of the
training graph, so every off-diagonal cell needs its own training run -- the
"train once, evaluate at three densities" shortcut only works
non-transductively. And the one genuinely interesting row, train-on-mixed, is
better implemented as a masking-rate augmentation (draw the rate per
augmentation instead of fixing it at 0.15) than as data mixing: one run, no new
data, and it tests density-invariance directly. That is Phase 2.

## Scoring: corner, not pooled

The reported Marformer number comes from `analysis/split_missing_metrics.py
--corner-whole` (via `jobs/corner_score.sbatch`), **not** from `tree_eval`'s
`TEST_RESULTS/*.json`. `tree_eval`'s "missing" metric pools the held-out corner
with the training partition's own missing cells (D22), while Stan is scored on
the test block alone -- putting the two in one table is D23, one table over two
populations. `collect.py` additionally *asserts* that every Marformer row's
`test.n` equals every Stan arm's `n_missing_ratings` within a cell, so a
population mismatch fails loudly instead of being averaged.

Expected corner sizes per draw: ~539 cells at 15% observed, ~332 at 50%, ~100
at 85%.

### smECE (follow-on, route scouted)

The corner scorer emits log-loss/accuracy/rmse only, so log-loss is Phase 1's
headline. The paper also reports smECE, and the trusted path for it is
`analysis/smece_vs_stan_grid.py`, which loads
`scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py` as a module and calls
`_marformer_probs_all_missing` / `_collect_{oracle,misspec}_payloads`, then
`shared.rp.prepare_rel_diagram(..., report_CE=True)`.

Two things to get right when adding it:

- **Do not rebind the repo symlinks.** That script reaches its data through the
  plotter's module-level `ROOT`, which is why `make_figures.sh` symlinks
  `imputer/ranking/{DATA,RESULTS}` at a run. Doing that 18 times is serial,
  stateful, and would fight whatever is currently bound. `DatasetConfig` is a
  dataclass, so the right move is `dataclasses.replace(pd3.ITEM_CFG,
  data_root=..., marformer_root=..., stan_oracle_root=..., misspec_root=...)`
  per cell.
- **Gate the output, not the component.** `plot_domain3_sigmoid.py` has carried
  its own copies of D14/D22/D23 (it re-derives evaluation rather than reading
  the scorers). `METRIC_POPULATION = "corner"` is now its module default, which
  is the D22 fix, but that must not be taken on trust: compute log-loss through
  pd3 for one cell and check it equals `corner_metrics/corner.json` for that
  same cell. Only if they agree is its smECE on the right population.

## Layout

    threads/obs-density/
      PLAN.md            this file
      make_data.sbatch   array over 18 cells: remask + subsplit
      submit_phase1.sh   driver: data -> search -> select -> refit -> eval
                         -> corner, plus four Stan arms; one array per stage
                         with the CELL as the array axis
      verify_remask.sh   nesting + no-flag-regression check for the remask
      collect.py         assemble the density table from corner.json and the
                         Stan predictive_metrics.json, with the D23 gate
      out/               collect.py output

Run ids: `obsden-s<draw>-o<rate>`, e.g. `obsden-s101-o15`. `OBSDEN_PREFIX`
overrides the prefix, which is how the one-cell smoke test runs through the same
code path without writing into the real cells.

Stages are independently re-runnable: `./submit_phase1.sh --stage corner`.

## Trap: --mem on `med` silently buys CPUs

`med` has `MaxMemPerCPU=4000`, so a memory request is also a core request:
`--mem=120G` becomes a **31-core** allocation, and the task then waits for a
node with 31 free cores. Two jobs here sat PENDING for half an hour for that
reason while the partition had hundreds of idle cores, just not 31 on one node.

    --mem=120G -> 31 cpus      --mem=32G -> 9 cpus
    --mem=64G  -> 17 cpus      --mem=24G -> 7 cpus

`jobs/corner_score.sbatch`'s own header asks for 120G because it was sized for
K=4000. At K=500 the bundle is 2.4 MB and 24G is ample, so this thread overrides
it (`OBSDEN_CORNER_MEM`, default 24G). Check the CPU count in `squeue -o %C`
whenever a CPU job will not start -- it is often not the number you passed to
`-c`.

## Trap: one conf file per STAGE, not per submission

`submit_phase1.sh`'s `write_conf` names the conf after the stage
(`conf/OBSDEN-SELECT.env`). Submitting the same stage twice in quick succession
therefore makes the second invocation **overwrite the first's conf before its
job starts**, and both jobs then run the second cell list. This happened on
2026-10-17: the two bridge selects were submitted back to back and both scored
`obsden-s101-o15-vf30`, so the "block bridge" would have compared vf30 against
itself and reported agreement. Nothing errored.

Fixed by `OBSDEN_TAG`, which suffixes the job, conf and log names. The main run
submits each stage once and never hit this, so the tag is empty by default.
**Set `OBSDEN_TAG` on any submission that reuses a stage**, e.g.

    OBSDEN_TAG=blk OBSDEN_CELLS=obsden-s101-o50-blk \
      OBSDEN_VAL_PROTOCOL=block ./submit_phase1.sh --stage select

## Known wart: the shared runner log

`scripts/DOMAIN3/common.sh:95` sets `DOMAIN3_LOG_DIR` unconditionally to
`imputer/ranking/logs` and tees with `-a`, and the file is named after
(family, axis, protocol, size) only. Every cell of this thread shares one name,
so all 18 append to `domain3_sigmoid_marformer_item_transductive_500.out` and
the content interleaves. It also still holds output from *older* runs, which is
actively misleading -- it is what made the carve look like it had not fired
during the smoke test.

Nothing is lost: the runner's `exec > >(tee -a ...)` also writes to the job's own
stdout, so each cell's `threads/obs-density/logs/*_<jobid>_<task>.out` has its
own complete copy. **Read the per-job log, never the shared one.**

Not fixed here on purpose: making the log dir overridable means editing
`common.sh` under running jobs, and bash reads scripts by byte offset.

## Upstream edits this thread required

All insertions only -- no existing line was modified, and a no-flag
`domain3subsplit.py` run was verified byte-identical to the shipped
`Item_T_500` bundle, so the frozen pipeline is provably unaffected.

- `STAN/stan_code/scripts/domain3subsplit.py`: `--observed-rate`, `--mask-seed`,
  `_remask_nested`.
- `jobs/{marformer_train,select_epochs,tree_eval,stan_fit,corner_score}.sbatch`:
  a `CELL_RUN_IDS` block making the run id the array axis. Needed because 18
  cells at one size would otherwise be 18 arrays of one task, and the account
  runs ~10 jobs at a time.
- `submit_validation.sh`: an `EXTRA_DEP` hook (unused by the final driver, which
  calls the job scripts directly, but kept -- it is the general fix for gating
  the search stage on a caller's own job).
