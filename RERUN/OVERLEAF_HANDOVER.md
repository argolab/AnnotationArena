# Overleaf handover — Domain 3 numbers and figures refreshed, prose untouched

**For the agent that updates the paper.** Everything here has been applied to the
branch `claude/numbers-refresh` in `Marformer-Paper/`. **Nothing is pushed.**
Pushing to Overleaf needs Tom's explicit approval, per change — approval for one
push never carries to the next.

```
cd Marformer-Paper && git diff main..claude/numbers-refresh
```

## What was done, and what was deliberately not

**Done:** numeric literals, figure files, and one figure-generation setting.

**Not done:** no sentence was rewritten, no caption reworded, no section moved,
nothing in Domains 1–2. Where a number could not move without a sentence moving
with it, the number was left alone and a `\Claude{...}` margin note was attached
instead. There are **nine** such notes; §4 below lists every one with its evidence
and the decision it needs. They are the actual work left to do.

One partial exception, flagged as note 8: changing "4 chains" to "1 chain" forces
`chains`→`chain` and drops `each` / `per chain` / `total`. That is grammar dragged
along by a number, not a rewrite, and it is the only place any prose word changed.
Verify with:

```
git diff main..claude/numbers-refresh -- main.tex   # only numerals, macros, \claude{} calls
```

## 1. Where the numbers come from

Authoritative source: `RERUN/runs/final-dual/`.

| artifact | what it holds |
|---|---|
| `reports/tables.txt` | the three result tables, test-corner log-loss |
| `corner_metrics/corner.json` | per-cell test-corner metric, 42 cells, the thing everything is checked against |
| `PLOTS/*/[*_]summary.json` | exactly what each figure plots |
| `reports/smece.json` | smECE, newly serialised (see §3) |
| `paper_figures/` | the 17 PNGs installed into `PLOTS/` |

**One command re-verifies that the figures and the tables are the same quantity:**

```
python RERUN/analysis/check_plot_corner.py     # exits 0; prints all 42 cells
```

Run it after any figure rebuild. It compares the summary JSON the build writes
against `corner.json` cell by cell and fails on a mismatch *or* a missing cell. It
needs no GPU. This check exists because the figures silently disagreed with the
tables for months: `plot_domain3_sigmoid.py` re-derives evaluation rather than
reading the pipeline's conclusions, so it carried its own copy of every eval bug.

## 2. Numeric edits applied

### Domain 3 synthetic

| what | was | now | source |
|---|---|---|---|
| `δ_item`, item axis | 0.7 | **1.0** | `train_config.json` of every item refit |
| `δ_judge`, judge axis | 0.7 | **0.8** | `train_config.json` of every judge refit |
| Stan runtime at K=500 | ≈100 min | **≈52 min** | `item_generalization_summary.json` |
| runtime caption | "≈100 min at 500" | **"≈52 min at 500, rising to ≈757 min at 4000"** | same |
| Marformer judge-axis runtime | below 15 min | **below 4 min** | `annot_generalization_summary.json` (max 3.68 at J=50) |
| Stan chains (two places) | 4 | **1** | every Stan fit, including the published ones |
| calibration figure, item | `item_calibration_size500.png` | **`..._size4000.png`** | `make_figures.sh` spec field |
| calibration figure, judge | `annot_calibration_size30.png` | **`..._size40.png`** | same |
| subcaption, item | $K_\text{train}=500$ | **4000** | matches the new panel |
| subcaption, judge | $J_\text{train}=30$ | **40** | matches the new panel |
| smECE item, Marformer / NT / Stan | 0.055 / 0.044 / 0.019 | **0.026 / 0.026 / 0.025** | the K=4000 panel's own titles |
| smECE judge, Marformer / NT / Stan | 0.035 / 0.029 / 0.027 | **0.034 / 0.032 / 0.027** | the J=40 panel's own titles |

### LLM-Rubric (only the three models that were rerun)

| quantity | was | now |
|---|---|---|
| Marformer @10 | 1.52 | **1.95** |
| Stan CPM @10 | 1.44 | **1.39** |
| Best Unigram @10 | 1.38 | 1.38 (unchanged) |
| Marformer @50 | 1.05 | **1.12** |
| Stan CPM @50 | 1.21 | **1.20** |
| Marformer @175 | 0.998 | **1.007** |
| Stan CPM @175 | 1.077 | **1.093** |
| Best Unigram @175 | 1.112 | 1.112 (unchanged) |
| MBR-L2 Marformer @175 | 0.52 | **0.59** |
| MBR-L2 Stan CPM @175 | 0.74 | **0.78** |
| runtime Marformer @175 | 18 min | **22 min** |
| speedup vs Stan CPM | ≈9× | **≈7×** |

ReMasker and MIWAE were **not** rerun; their numbers are untouched. See note 4.

## 3. Figure changes

**The Stan cutoff is gone.** `ITEM_STAN_CUTOFF` was `500`: beyond it the Stan
series were drawn as a **flat dotted extension** of their K=500 value — a line
that is not data. The rerun fits Stan Oracle *and* all three misspecified variants
at every item size to 4000.

The published figures were internally inconsistent about this: the accuracy panels
extended K=500 flat, while the **runtime** panel plotted the real 1000–4000 fits,
because the runtime row keyed on the display size and the metric rows keyed on the
cutoff-mapped source size. So one figure showed Stan costing 757 min at K=4000
while the panel beside it implied Stan had never been run there. All panels now use
the real values.

Consequence, and it is the most consequential thing in this handover:

| K | Marformer | Stan Oracle | gap |
|---|---|---|---|
| 500 | 0.913 | 0.679 | 0.234 |
| 1000 | 0.767 | 0.642 | 0.125 |
| 2000 | 0.728 | 0.615 | 0.113 |
| 3000 | 0.706 | 0.607 | 0.099 |
| 4000 | 0.738 | 0.600 | **0.138** |

The gap narrows and then widens again. See notes 1 and 2.

### The calibration panels now report K=4000 and J=40, not K=500 and J=30

**This is a change of reported size, decided deliberately, and it carries a prose
obligation.** `main.tex` lines 1693/1700 now include
`item_calibration_size4000.png` and `annot_calibration_size40.png`; the subcaptions
and the quoted smECE values in the Calibration paragraph were updated to match.
`make_figures.sh` was changed so a rebuild stages those sizes
(the `cal_size` field of each `spec`).

**Why the size moved.** At the old sizes the paragraph's claim was not merely stale,
it was contradicted:

| | Marformer | Marformer NT | Stan Oracle | verdict |
|---|---|---|---|---|
| item K=500 (old) | 0.055 | 0.044 | 0.019 | **Marformer worse, beyond its CI** |
| item K=4000 (new) | 0.026 ± 0.017 | 0.026 ± 0.018 | 0.025 ± 0.016 | on par |
| annot J=30 (old) | 0.035 | 0.029 | 0.027 | on par |
| annot J=40 (new) | 0.034 ± 0.020 | 0.032 ± 0.018 | 0.027 ± 0.015 | on par |

"On par" here is the convention the captions already use: the gap is smaller than
the Marformer estimate's bootstrap half-width. The new gaps are +0.001, +0.001,
+0.006 and +0.005.

**The prose still has to change, and the numbers alone do not fix it.** The sentence
says the two Marformer variants are "both lower than" Stan. At 4000/40 they are
*not lower* — on the item axis Marformer is nominally 0.001 **above** Stan Oracle.
The supportable claim is **on par with**, never **better than**. The numerals were
updated (in scope); the clause was left alone (prose) and carries a `\Claude` note
spelling out the required surgery. **Do not ship the paragraph with the new numbers
and the old clause** — that combination is worse than either alone.

**What this costs the argument, stated plainly.** The paper chose 500/30 as
"representative" sizes partly to show the model stays calibrated in *small* domains.
That reading does not survive. Full grid in
`RERUN/analysis/smece_table.py` (`--ci` for intervals); on the item axis Marformer
is worse than the best Stan variant beyond its CI at **K=200, 300, 400, 500**, and
on par only at K=100 and K>=1000. The judge axis is on par from J=20 up. So the
calibration claim is **asymptotic, not uniform**, and the figure caption's phrase
"at representative training sizes" is now doing misleading work — consider
"largest training sizes". A sentence conceding the small-K item band is probably
needed; that is a writing decision, not a number.

K=100 is technically "on par" and should **not** be used to rescue the small-data
reading: at 100 items the model's log-loss is 0.939 against Stan's 0.789, i.e. it is
nearly uninformative, and a near-uniform predictor is well calibrated almost by
construction. A reviewer will say so.

**J=50 was asked for and is not usable.** Stan Oracle has no judge fit past J=40
(`ORACLE/ANNOT` contains 10–40 only) and Marformer NT has no judge refit past 40, so
a J=50 panel would be missing both the oracle and the NT arm — the two series the
"on par" claim is about. Only Stan Bin/Proj/Both exist at 45 and 50. **J=40 is the
largest judge size at which every series in the panel exists**, which is why it was
chosen over 50. Same logic picks K=4000 on the item axis.

**Both PNGs were opened and checked**, not just copied: each shows six panels with
all series present and the smECE values in the panel titles matching the quoted
numbers.

### smECE moved a lot, and the cause is measured

The regenerated panels give smECE far above the published values and **reverse the
ordering against Stan**: at item K=500 Stan is 0.019 against Marformer 0.055 and
NT 0.044, where the paper says Marformer and NT are "both lower". At annot J=30 all
three overlap (0.035 / 0.029 / 0.027).

I first guessed the population fix (pooled -> corner) was the cause. **It is not.**
`RERUN/analysis/smece_decompose.py` runs the full 2x2 — (pooled, corner) x
(argmin-on-test, F13 last) — on the rerun models *and* on the pre-rerun ones in
`runs/replicate-01`. The published numbers are reproducible from the pre-rerun
models under the old policy, so the attribution is established, not inferred:

| cell | paper | old model, old policy | new model, old policy | current | retraining | corrections |
|---|---|---|---|---|---|---|
| item/500 Marformer | 0.016 | 0.0145 | 0.0586 | 0.0548 | **+0.044** | −0.004 |
| item/500 NT | 0.016 | 0.0125 | 0.0522 | 0.0439 | **+0.040** | −0.008 |
| annot/30 Marformer | 0.022 | 0.0194 | 0.0365 | 0.0351 | **+0.017** | −0.001 |
| annot/30 NT | 0.014 | 0.0178 | 0.0242 | 0.0287 | **+0.006** | +0.005 |

Retraining is 109%, 126%, 109% and 59% of the total change — over 100% because the
corner correction pushes the *other* way in three of four cells. **Scoring on the
test corner makes calibration look slightly better, not worse**, so the metric fix
cannot explain the shift.

What remains is what retraining changed: the standard Transformer block, the dual
value stream, and F13 stopping the refit after a few hundred epochs instead of
2000. One suggestive datum: on the pre-rerun models the *checkpoint* dominated
calibration (annot/30 is 0.0194 at the test-argmin epoch against 0.0825 at epoch
2000), so late training was wrecking it there; the F13 refits stop early and are
still worse, which points at the architecture rather than the schedule. Not chased
further — flagged in the paper note.

**smECE is now serialised.** It was computed inside the calibration plotting code
and rendered only into a plot title, so after a rerun the prose and the panel could
drift with nothing to catch it. `RERUN/analysis/extract_smece.py` reads the same
payloads the panels are drawn from and calls the same `relplot` path, so the
numbers are the plotted ones by construction; `make_figures.sh` now runs it inside
the run binding, so one command produces both. Values in
`runs/final-dual/reports/smece.json`.

**A y-axis bug the numeric gate could not see.** Making Stan's values real pushed
Stan Oracle to 0.600 at K=4000, below a **hardcoded** axis floor (0.65 in the
linear path, 0.67 in the compressed one), so the Stan curves were drawn *off the
bottom of the plot*. Every number in that figure was correct, which is why
`check_plot_corner.py` passed it — value-level verification and rendering are
separate failure modes, and only opening the PNG caught this. Both floors now
follow the data, and `_plot_metric` prints a loud `WARNING ... will be CLIPPED`
naming any series that falls outside the limits.

That guard currently fires **twice, expected**: `Structured Log-Linear` sits at
2.1–2.3 nats in the two *full* figures, above the compressed scale's 1.68 top.
That baseline is off-scale by design and predates this work — do not chase it.

**Figure install:** `RERUN/make_figures.sh --run-id final-dual --install` copies
`paper_figures/` into `Marformer-Paper/PLOTS/{DOMAIN3-SYNTH,REAL-LLMRUBRIC}`, which
is where `main.tex` already points.

## 4. The ten `\Claude` notes — the work that is left

Each is a margin note in `main.tex` (magenta). Grep `\claude{` to find them.

1. **"Stan results are reported only through K=500"** — false; fits exist to 4000.
   The cost argument survives (757 min vs 13), the claim does not.
2. **"approaching near-oracle performance by K=4000"** — contradicted by the
   corrected figure. Marformer 0.738 vs Stan 0.600, and the gap *widens* from 3000
   to 4000. This is the ≈0.13-nat gap nothing explains: capacity, epoch budget and
   forced amortisation were each tested and ruled out (`FUTURE_EXPERIMENTS.md`
   waves 3b/3e). Either state the residual gap and leave it open, or drop the
   convergence claim.
3. **"non-transductive … with a small performance penalty"** — it is a crossover,
   not a penalty: NT is better at K=2000 and 4000, worse at K=100. Two figure bugs
   hid it (NT scored on the pooled population; item NT truncated at K=500).
4. **LLM-Rubric paragraph** — (a) mixed provenance: three models rerun, ReMasker
   and MIWAE not; (b) the Marformer curve is not monotone (bumps at K=30 and 150),
   so "continues to improve" describes a curve that rises twice. The bumps are
   overfitting, not the degeneracy failure mode — training loss is lowest where
   test loss is worst — so no restart rule applies.
5. **Relational attention, `main.tex:841`** — the key-dimensionality sentence
   describes `use_per_head_rel=true`; Domain 3 ships the other design. Also, for
   Jason's note: the learned key embedding is a **rank-constrained** special case
   of the frozen-bit readout, so the argument for Shaw is parameterisation and
   learning dynamics, not power. The value term is the half that widens the
   function class. Both are implemented and measured on `side/shaw-rel`.
6. **Runtime figure is non-monotone** — wall-clock is scaled by the *selected*
   epoch count, so K=4000 looks cheapest. Right for "time to the reported model",
   misleading as "cost of scaling".
7. **No Stan fit at J=50** — the judge sweep stops at J=40 (55 min). Either the
   size or the number in that sentence is wrong; I could not tell which. This is
   also why the judge calibration panel is J=40 and not J=50 (§3, note 10).
8. **1 chain vs the R̂ table** — I changed the chain count because 1 chain is what
   every reported fit used. But split-R̂ from one chain is the weaker split-half
   statistic, and a 4-chain diagnostic run exists
   (`RERUN/side/stan_4chain.sbatch`). Say the diagnostics are 4-chain while the
   fits are 1-chain, or recompute the table.
9. **LLM-Rubric cannot inherit `δ_item` by reference** — it genuinely uses 0.7
   while the synthetic item axis uses 1.0, so the cross-reference in §1898 has to
   state 0.7 explicitly.
10. **Calibration paragraph, `main.tex:1709`** — the one note that is *blocking*,
   because the numerals around it have already been changed. The reported size moved
   to K=4000 / J=40 and the three series are now statistically indistinguishable
   (gaps +0.001, +0.001, +0.006, +0.005, all inside the bootstrap intervals), but
   the sentence still says the two Marformer variants are **"both lower than"**
   Stan. That is false — on the item axis Marformer is nominally 0.001 *above*
   Stan Oracle. Replace the clause with an on-par statement. Also decide whether to
   concede the small-K item band (worse than the best Stan beyond its CI at
   K=200–500), and whether the caption's "representative training sizes" should
   become "largest training sizes". Full detail in §3.

## 5. Things this handover does *not* cover

- **`RERUN/FINAL_RUN_PAPER_NOTES.md`** is the fuller set of paper changes Tom is
  drafting himself — the F13 protocol, the standard block, why the codec does not
  ship, the restart-on-degenerate rule, the LLM-Rubric protocol. This document is
  only the numbers-and-figures slice of it.
- `baseline_cutoff=1000` was left alone. The empirical baselines genuinely stop
  there; whether to extend them is a separate question.
- The epoch budget: 2000 is a **search** budget, not a training length. If the
  paper states a number of epochs for Domain 3, check it against
  `FINAL_RUN_PAPER_NOTES.md` §4.
- Domain 3's §952 "Validation and early stopping" paragraph already describes F13
  correctly — someone wrote it before this pass. Nothing was changed there.

## 6. If you rebuild the figures yourself

```
RERUN/make_figures.sh --run-id final-dual --submit --cache --device cuda
python RERUN/analysis/check_plot_corner.py          # must exit 0
RERUN/make_figures.sh --run-id final-dual --install
```

Two traps, both of which have silently produced wrong figures before:

- `TEST_RESULTS/best.json` is reused if present, and `--cache` reuses
  `*_cache.npz`. Both can hold values computed under an older policy, so a policy
  change can do nothing until they are deleted. (`--cache` is safe for the Stan
  change specifically, because Stan payloads are always re-read from CSVs.)
- `plot_domain3_sigmoid.py` has **separate code paths for the transductive and
  non-transductive arms**, and the same decision can appear in several readers. The
  pooled-metric override lived in four places; gating one left the whole NT arm
  wrong. Check the output with `check_plot_corner.py`, not a component.
