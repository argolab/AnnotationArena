# Final run — what changed, and what the paper needs to say

Notes only. Nothing here has been applied to the Overleaf source; the LaTeX is
yours to write. Each item says what the paper currently claims, what the final
run actually did, and what evidence backs the change, so you can decide the
wording without re-deriving the result.

Companion file: `PAPER_EDITS_SWEEP.md` (B1–B12), which is the earlier sweep and
still stands. This file covers only what the final run changed.

Run of record: **`RERUN/runs/final-dual`** — generative draw `s101`, model seed
`42`, one of each. Numbers are the **test corner** (`corner_whole=true`, 414
cells), never the pooled missing-cell metric: the pooled number mixes the corner
with the training partition's own held-out cells, so it is not comparable with
Stan (D22/D23).

---

## 1. Architecture: the blocks are now standard Transformer blocks  [rewrite]

**Paper now:** each block carries an extra outer residual add around the whole
block *and* a trailing $d \times d$ linear on the block output, on top of the
ordinary pre-norm sublayer residuals.

**Final run:** both are gone. `dual_outer_residual=none`,
`block_output_projection=none`. The block is the textbook one.

**Why both at once, which is the part worth a sentence in the paper.** The two
are entangled and each looks load-bearing alone: removing the outer add by
itself costs **+0.150 nats at K=300**. The trailing projection carries no
identity component (alignment −0.003, effective rank 57.5/80), so without the
outer add every block's skip path runs through a dense near-random rotation,
eight times over. Remove the rotation too and the skip path is clean. Taken
together they cancel.

| arm | K=300 (dev-selected) | K=2000 (dev-selected) | K=2000 (tail) |
|---|---|---|---|
| `proj_keep_add` — as published | 0 (ref) | 0 (ref) | 0 (ref) |
| `proj_keep_rot` — linear only | +0.150 | −0.014 | −0.013 |
| **`proj_drop_rot` — standard block** | +0.052 (+0.007 at best epoch) | **−0.021** | **−0.023** |
| `proj_frozen_add` — linear frozen | +0.035 | −0.008 | +0.001 |

Draw-controlled check at K=2000 (thread C2, same draw both arms): dual + old
block 0.7045, dual + standard block 0.6812 → **−0.0233 for the standard block.**

**The one caveat to carry.** At K=300 the standard block is at parity only at its
*best* epoch (+0.007); under the real dev-selected protocol it costs +0.052,
because these variants are harder to early-stop against a ~110-target dev block.
Honest phrasing: *equal architecture, worse stoppability*. K=100 has no dev block
at all.

## 2. The typed value codec is NOT in the paper's architecture  [no change — but do not claim it]

The encoder/decoder codec for value types with $N(0,1)$ init was built and
evaluated, and it does **not** ship. It is at parity on average but it
destabilises training:

- **9 collapsed cells in 72 with the codec; 0 in 42 without.**
- Escape time from the uniform plateau: codec median 80 epochs, range 50–310,
  **sd 77.5**. Dual stream median 32, range 28–40, **sd 3.2**.

Attribution is clean in both directions — codec + *old* block still collapses
(1/30), dual + *standard* block does not (0/4) — so this is the codec, not the
block simplification.

If you want the codec mentioned at all, the defensible claim is: *it matches the
hand-built value stream wherever there is enough data, and it removes the
hand-set `logit_high` constant from the value path.* It is not the configuration
the reported numbers come from.

## 3. Model selection: the paper's protocol was selecting on test  [rewrite — this is the substantive one]

**Paper now:** Domain 3 has no validation split, so the reported checkpoint is
chosen on test (D14). This is the thing Jason's overfitting thread was pointing
at.

**Final run (protocol F13):** for each cell,

1. carve a 15% **block** out of the training partition (`val_frac=0.15`,
   `val_split_seed=7`);
2. search 2000 epochs and pick a stopping epoch per run — smooth over 5 epochs,
   argmin, floor at 25;
3. merge the block back and **refit from scratch on 100% of the training
   partition** for exactly that many epochs;
4. report the refit's **last** checkpoint. Never an argmin on test.

Two facts worth stating because they pre-empt the obvious objections:

- **The second stage is nearly free in accuracy terms.** Skipping the refit and
  just reporting the search run costs a median **+0.005 nats** over 29 usable
  cells (large-$K$ mean −0.0004) — i.e. the extra 15% of data buys essentially
  nothing. The refit is for protocol cleanliness, not for the number.
- **Two cells cannot carve a block at all** — item/T $K=100$ and annot/T $J=10$
  are too small — so they have no dev-selected epoch and are reported at a fixed
  budget. Say so rather than letting the reader assume uniform treatment.

## 4. Epoch budget: 2000, not 400  [rewrite]

The paper's 400 epochs leaves the large-$K$ cells undertrained (this is B3 in
the sweep). The search stage now runs 2000.

**But the curves do not grok — they overfit, on both architectures.** Validation
bottoms out at epoch 280–880 while training loss keeps falling to convergence at
1300–1700, and the gap widens monotonically after that. So 2000 is a *search*
budget that the dev block then cuts back; it is not the reported training length.
Do not describe it as "trained for 2000 epochs."

Matched-window comparison (0–400 epochs, same draw): the final architecture
overfits *less* than the published one (+0.120 vs +0.184 train/val gap).

## 5. Chunking: `max_item=500` at train and eval  [rewrite]

`max_item` is the number of items per forward-pass graph — a chunking knob, not
a training-set size. Train and eval now agree at 500. The published LLM-Rubric
figure was re-scored at `max_item=10` while `test.py` used 500 (D20/D30), so the
published number and the code did not match. Both are 500 now; for LLM-Rubric
the domain's own value (10) is retained, consistently, at train and eval.

## 6. Training instability and random restarts  [insert — new text needed]

Some cells never leave the uniform plateau: the model starts at $\ln 4 = 1.3863$
and has to escape it, and the dev-selected stopping epoch is chosen on the search
run while the *refit* is a fresh initialisation whose escape time is independent.
When the selected epoch falls below the refit's own escape epoch, the refit
reports the uniform distribution.

The sentence you wanted: *training is occasionally unstable, and we restart from
a different model seed when the training performance is degenerate.* The
important half, which makes it defensible rather than a fudge:

> **Degeneracy is detectable from the training curve alone — 8 of 8 cases caught,
> no false alarms — so the restart criterion never looks at test data.**

### What the dual stream buys, measured on the final run itself

Escape epoch across all 42 search runs (first epoch with training loss below
$\ln 4 - 0.05$):

| | min | median | max | **sd** |
|---|---|---|---|---|
| typed codec (`final-s101`) | 46 | 80 | 310 | **77.5** |
| **dual stream (`final-dual`)** | **27** | **34** | **43** | **4.1** |

The spread falls by a factor of 19. That is the whole mechanism: the stopping
epoch is chosen on the search run and applied to a *fresh* refit whose escape
time is independent, so what matters is not the mean escape time but its
variance. With escape confined to 27–43, any selected epoch above ~43 is safe by
construction.

On this run that leaves **exactly one cell exposed**: `Marformer/annot/20`, whose
selected epoch is 36. Empirically 15 of the 42 observed escape epochs exceed 36,
so that cell has roughly a **36% chance** of coming back degenerate. The other 41
cells have a selected epoch above every escape epoch observed and **cannot**
collapse. This is worth a sentence in the paper: the failure mode is not
eliminated, it is confined to cells whose dev-selected epoch lands inside the
escape window, which only happens at the smallest annotator counts.

`RERUN/analysis/curve_qc.py` implements the check and reports any cell within
0.02 of $\ln 4$ as DEGENERATE; it also flags cells more than 0.06 nats above
their neighbour median.

**Open issue worth one clause.** The selector can pick an epoch *before its own
search run escaped the plateau* — item $K=400$ escaped at epoch 190 and the
selector chose 148 — because argmin on a flat, noisy plateau is arbitrary. The
one-line fix is to constrain selection to epochs where validation has already
dropped below $\ln 4 - \epsilon$. **Not implemented in the final run.** Either
implement it and re-select, or do not claim the selector is plateau-aware.

## 7. Stan: say what was actually run  [rewrite — this is B12]

The fits are **1 chain**, 300 warmup, 500 sampling, `adapt_delta` 0.85,
`max_treedepth` 12 — including the published ones. The paper should say 1 chain
and footnote that a 4-chain refit lands in the same place, rather than implying a
4-chain budget.

### The LLM-Rubric Stan rerun agrees with the published curve

Rerun vs the curve digitised from the paper figure (±0.005):

| K | published | rerun | diff |
|---|---|---|---|
| 10 | 1.435 | 1.3902 | −0.045 |
| 20 | 1.475 | 1.2748 | **−0.200** |
| 30 | 1.230 | 1.2303 | +0.000 |
| 40 | 1.212 | 1.2091 | −0.003 |
| 50 | 1.205 | 1.2039 | −0.001 |
| 75 | 1.160 | 1.1610 | +0.001 |
| 100 | 1.128 | 1.1178 | −0.010 |
| 125 | 1.093 | 1.1125 | +0.020 |
| 150 | 1.088 | 1.0847 | −0.003 |
| 175 | 1.077 | 1.0925 | +0.016 |

Excluding K=20: median **−0.001** over 9 sizes, both signs, 4 of 9 positive — digitisation
error plus 1-chain MCMC noise. **No systematic shift, so the rerun numbers can
replace the published ones.**

**K=20 deserves a look before you quote it.** The published curve *rises* from
K=10 (1.435) to K=20 (1.475) — its only non-monotone step — and the rerun removes
that kink (1.3902 → 1.2748). A 1-chain fit on 20 items mixing badly is the
likeliest explanation, which would mean the rerun is *fixing* a bad published
fit rather than disagreeing with a good one. Your call whether to say anything;
the safe move is to plot the rerun and say nothing.

## 8. LLM-Rubric: retrained on `dual`, and what the arm's protocol actually is  [rewrite]

The LLM-Rubric Marformer runs had been trained with the **typed codec** (verified
from the written `train_config.json`, not the submit command). They are retrained
on `dual` + standard block, so both domains now report the same architecture.
The typed-codec outputs are preserved at `RESULTS/MARFORMER/LLM_RUBRIC_codec_typed`.

Test log-loss, Marformer vs Stan CPM, on the 25 held-out test items:

| K | 10 | 20 | 30 | 40 | 50 | 75 | 100 | 125 | 150 | 175 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Marformer** | 1.953 | 1.151 | 1.250 | **1.043** | 1.124 | 1.078 | **1.036** | **1.017** | 1.215 | **1.007** |
| Stan CPM | 1.390 | 1.275 | 1.230 | 1.209 | 1.204 | 1.161 | 1.118 | 1.112 | 1.085 | 1.093 |

Marformer wins from K=20 on except at K=30 and K=150, and loses badly at K=10.

### The arm has a real validation split — state it, because it differs from Domain 3

I went looking for test contamination here and did not find it. The bundle splits
the 225 items **K_train / K_val = 25 / K_test = 25**, and every held-out rating
carries an explicit `instance: "val"` or `"test"` tag: 666 ratings each, disjoint.
Early stopping selects on the val 666 and the reported number is the test 666. So:

- **LLM-Rubric model selection is legitimate** and always was. If `PAPER_EDITS_SWEEP.md`
  B9 ("the published number is partly selected on test") was about this
  mechanism, it does not apply to the rerun. The two sets merely have the same
  size, which is what made it look suspicious.
- **But the two arms use different protocols**, and the paper should not imply
  one: Domain 3 carves a 15% block out of training and refits (F13); LLM-Rubric
  has a dedicated 25-item val split and no refit. Both are defensible; saying so
  costs one sentence and pre-empts the question.

### A real bug to know about: `last.ckpt` is not the last epoch

For every LLM-Rubric cell, `last.ckpt` is **byte-identical to `best-*.ckpt`** —
same `epoch` and `global_step` in the file, 0 of 138 tensors differing. For K=10
both are epoch **120**, not 299, even though training ran the full 300 epochs and
`periodic-epoch=0299.ckpt` exists. `ModelCheckpoint`'s `save_last` stops
refreshing `last.ckpt` after the monitored metric's last improvement.

Consequences:

- `test.py --checkpoint both` writes two files with **identical** numbers
  (1.9528375115838494 in both, to every digit). `best.json` and `last.json` are
  not two readings.
- Anything describing the LLM-Rubric number as the final epoch is **wrong** — it
  is the validation-selected epoch. The number is fine; the description is not.
- **Domain 3 is unaffected.** Its refits have no `best-*.ckpt` at all, and their
  `last.ckpt` epoch equals the history length exactly (Item_T_2000: 377/377,
  Item_T_400: 148/148, Annot_T_30: 136/136). F13 reports a genuine last epoch.
  Only the LLM-Rubric arm is touched.

### The small-K bumps are overfitting, not degeneracy — do not restart them

K=10 (1.953, worse than $\ln 4$), K=30 and K=150 sit above their neighbours. None
is degenerate: every cell escaped the plateau, and training loss is *lowest*
exactly where test loss is worst (K=10 trains to 0.223). The restart rule in §6
keys on degenerate *training* performance, so it does not fire here and should
not be stretched to cover it. These points are real variance between two
666-rating samples at small K, and the honest treatment is to plot them.

## 9. The final numbers  [swap these in]

Run of record `final-dual`, draw s101, model seed 42. **Test corner** log-loss,
`corner_whole=true`. Rendered by `analysis/result_tables.py --run-id final-dual`;
the live copy is `runs/final-dual/reports/tables.txt`.

### Synthetic — item generalization

| K | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 | 2000 | 3000 | 4000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Marformer (T) | 0.939 | 0.949 | 0.981 | 0.901 | 0.913 | 0.915 | 0.831 | 0.970 | 0.749 | 0.767 | 0.728 | 0.706 | 0.738 |
| Marformer (NT) | 0.964 | 0.931 | 0.908 | 0.961 | 0.874 | 0.929 | 0.757 | 0.764 | 0.748 | 0.798 | 0.721 | 0.724 | 0.722 |
| Stan Oracle | 0.789 | 0.747 | 0.693 | 0.688 | 0.679 | — | — | — | — | 0.642 | 0.615 | 0.607 | 0.600 |
| Stan Bin | 0.856 | 0.833 | 0.823 | 0.814 | 0.820 | — | — | — | — | 0.816 | 0.807 | 0.817 | 0.796 |
| Stan Proj | 0.796 | 0.759 | 0.708 | 0.698 | 0.684 | — | — | — | — | 0.655 | 0.632 | 0.625 | 0.619 |
| Stan Both | 0.854 | 0.848 | 0.840 | 0.833 | 0.828 | — | — | — | — | 0.830 | 0.820 | 0.817 | 0.820 |

### Synthetic — annotator generalization

| J | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| Marformer (T) | 0.977 | 0.923 | 1.023 | 0.918 | 0.828 | 0.803 | 0.780 | 0.788 | 0.773 |
| Marformer (NT) | 1.004 | 0.918 | 0.883 | 0.851 | 0.846 | 0.801 | 0.780 | — | — |
| Stan Oracle | 0.789 | 0.769 | 0.757 | 0.733 | 0.707 | 0.703 | 0.680 | — | — |
| Stan Bin | 0.856 | 0.850 | 0.859 | 0.861 | 0.838 | 0.828 | 0.816 | 0.822 | 0.798 |
| Stan Proj | 0.796 | 0.781 | 0.775 | 0.769 | 0.731 | 0.712 | 0.707 | 0.693 | 0.673 |
| Stan Both | 0.854 | 0.851 | 0.855 | 0.858 | 0.836 | 0.833 | 0.818 | 0.816 | 0.798 |

(Stan was fit at the sizes the paper plots, hence the gaps at K=600–900 and J=45/50.
The Stan fits live under `final-s101` and are linked into `final-dual`; both runs'
`DATA` symlink to the same `dataseed-s101/DATA`, so they are the same draw.)

### LLM-Rubric — see §8 for the table and the three caveats.

### Marformer still trails Stan on the synthetic data — this is the known gap

Marformer does not beat Stan Oracle or Stan Proj at any synthetic size. At K=4000
it is 0.738 (T) / 0.722 (NT) against Stan Oracle's 0.600 — a gap of about **0.12
to 0.14 nats** that widens slightly with K. This is the unexplained gap already
recorded against the frozen config, not something the architecture change caused:
the codec run had the same shape. It needs an answer in the paper, and the final
run does not provide one.

### Degeneracy: none  [this is the headline]

| | typed codec (`final-s101`) | **dual + standard (`final-dual`)** |
|---|---|---|
| degenerate cells | **8 / 42** | **0 / 42** |
| refit escape epoch | 46–310, median 80 | **27–39, median 34** |
| escape **sd** | **77.5** | **3.8** |
| worst final train loss | at ln4 | 1.195, clear of ln4 = 1.3863 |

No restarts were needed, so the random-restart rule in §6 is stated in the paper
but never exercised on these numbers. The one cell predicted to be at risk
(`annot/20`, selected epoch 36) escaped at 29.

### What the dual stream cost, measured

On the 34 cells where both runs are healthy the dual stream is **better** by a
median of 0.028 nats (better in 22 of 34, mean −0.065). Restricted to the large
item cells, `K >= 1000`, the two are at **parity with a slight edge to the codec**
(median +0.031 in the codec's favour, n=8). On the 8 cells the codec collapsed,
dual gains **+0.43 to +0.48 nats** each.

Honest summary for the paper: *the codec matches the hand-built stream on
accuracy wherever there is enough data, and loses because it is unreliable.*

### Six cells sit above their neighbours — report them, do not rerun them

`curve_qc.py` flags 6 cells more than 0.06 nats above their neighbour median:
item/T K=800 (+0.180), NT item K=600 (+0.114), annot/T J=20 (+0.103), NT annot
J=10 (+0.086), NT item K=400 (+0.070), NT item K=1000 (+0.063). **None is
degenerate** — the largest is 1.023 against ln4 = 1.386.

That residual is computed **on the test corner**, so picking those cells to rerun
and keeping the better result is selection on test. At one model seed a bump of
this size is ordinary cell-to-cell variation. Either plot them as they are, or
rerun *every* cell at a second seed and average. The script now says so; it
previously advised rerunning the flagged cells, which was wrong.


---

## 10. The figure script was selecting the checkpoint on test  [this affects the PUBLISHED figures]

Found while verifying the regenerated figures against §9's tables. It is a
**pre-existing bug**, so the figures already in the paper were built this way.

`scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py` ran `test.py --checkpoint all`,
scored every periodic checkpoint on **test**, and reported whichever minimised
`missing.log_loss`. There was no option to disable it. The giveaway is the
summary JSON: `"200": periodic-epoch=0174.ckpt`, `"300": periodic-epoch=0099.ckpt`
where F13 requires `last.ckpt`.

So the tables and the figures disagreed, with the figures the optimistic half.
Across the 34 `final-dual` cells carrying both readings:

- test-argmin better by **median 0.006 nats, max 0.047**, differing in **23 of 34**
- worst: `Marformer/item/1000` 0.7999 (`periodic-epoch=0274`) vs **0.8470** (last);
  `Marformer/item/300` 0.9540 (`periodic-epoch=0099`) vs **1.0000**

**Fixed in code.** `--checkpoint-policy` now defaults to `last`, and
`test-argmin` survives only to reproduce the older figures; `main()` prints which
is in force. The 74 stale `best.json` files and the plot caches held
test-selected values and were deleted before regenerating.

**Why this matters more than 0.006 nats.** §3 tells the reader that selection no
longer touches test. A figure built by argmin on test contradicts that sentence
directly, whatever the magnitude. If any number in the paper's Domain 3 figures
is quoted in the text, it moves slightly when the figures are rebuilt.

### The figures now score the test corner — and the NT arm needed a second pass

Both halves are fixed in code and the figures are rebuilt, so §9's tables and the
figures are finally the same quantity. Recording how the second half was missed,
because it changes how much weight to put on "verified":

`--metric-population` defaults to `corner`, so the Marformer payloads keep only
`var.instance == "test"` — the population Stan was always scored on — and
chunking protects the corner with the same `keep_whole` rule as the scorer. I
verified that and reported D22/D23 closed. **Checking the rebuilt figures against
the tables showed the transductive arm right and the whole non-transductive arm
still wrong**, for two unrelated reasons:

1. The decision to override a curve with `best.json`'s pooled `missing.log_loss`
   lives in **four** places (the T collector, the NT collector, and both halves
   of the cached path). I had gated one. A `--cache` rebuild would also have
   reverted **both** arms silently.
2. `ITEM_MARFORMER_NT_MAX = 500` truncated the item NT curve at K=500, so the
   figure drew five NT points against a thirteen-point table.

My verification passed because it tested the probability path, which both arms
share and which was already correct — the number that reaches a curve is not
always the one that path computes. The check now reads the summary JSON the build
writes and compares **every** cell against `corner.json`.

**What this changes in the published figures.** The transductive curve was
already right after the first pass. The NT curve moves, by more than the
checkpoint fix did, and **not in a consistent direction**:

| cell | published (pooled) | corrected (corner) | change |
|---|---|---|---|
| NT item 300 | 0.957 | **0.908** | −0.049 |
| NT item 500 | 0.912 | **0.874** | −0.038 |
| NT annot 35 | 0.867 | **0.801** | −0.066 |
| NT annot 20 | 0.945 | **0.883** | −0.062 |
| NT annot 10 | 0.976 | **1.004** | **+0.028** |
| NT item 600–4000 | *absent from the figure* | 0.929 … 0.722 | 8 new points |

So the NT curve is mostly *better* once scored on the right population, but
J=10 moves the other way — the same sign instability already recorded for
D23. **Do not describe the pooled metric as flattering or as penalising
Marformer.** It is the wrong population; the direction depends on the cell.

One consequence for the prose: the item NT curve in the current PDF stops at
K=500. With the cap removed it runs to K=4000, where NT (0.722) and T (0.738)
essentially coincide — if the text says anything about the two protocols
diverging at large K, check it against the new figure.

## 11. Relational attention: one sentence that does not match the code, and how to answer Jason  [independent of any experiment]

Jason's margin note at `main.tex:839` asks whether to switch relational attention
to \citet{shaw2018self}. Thread D is measuring that (`RERUN/threads/shaw-rel/`);
these two points hold regardless of how it comes out, so they can go in now.

**(a) The key-dimensionality sentence is wrong for Domain 3.** `main.tex:841`
says we replace $\vec{k}_j$ with $[\mathbf{rel}_{ij}; \vec{k}_j]$ and that this
"means reducing the dimensionality of the key $\vec{k}_j$ by $|\Rho|$ to make
room for these bits." The code has two relational designs and **Domain 3 ships
the other one**: `run_marformer_group.sh` sets `USE_PER_HEAD_REL=false`, which
adds $|\Rho|$ extra *query* outputs and leaves the key at full width. The mode
that reserves key dimensions (`use_per_head_rel=true`) is not what any reported
number was produced with. Same function class, different parameter budget — but
the sentence as written describes a model we did not run. This sits in the very
paragraph Jason suggests shortening, so it is worth fixing either way.

**(b) His point (2) is right about dynamics and should not be stated as
capacity.** The learned key embedding contributes
$\vec{q}^{\,h}_i \cdot \vec{a}^K_\rho = \vx_i^\top \mathbf{W}_Q^\top \vec{a}^K_\rho$,
which is still a **linear** function of $\vx_i$ — and now has to factor through
$\mathbf{W}_Q$. The frozen bits give each head an unconstrained $|\Rho|$-dimensional
linear readout of $\vx_i$. So as a *score* term Shaw's version is a
**rank-constrained** special case of ours, not a more powerful one. What it
genuinely buys is what he says it buys, and none of it is capacity:

- $\vec{a}^K_\rho$ lives in key space, so similar relations can take similar
  embeddings;
- an unimportant relation can go to zero in every head at once;
- no fixed dimension budget is reserved.

If the paper adopts his framing, say *better parameterisation and learning
dynamics*, not *more powerful*. The one place "more powerful" is defensible is
his point (1): adding the relation to the **value**
($\vz_i = \sum_j \alpha_{ij}(\mathbf{V}\vx_j + \vec{a}^V_\rho)$) is a genuinely
wider function class, and the shipped architecture has no analogue for it at all.

**Early read: no arm is distinguishable from the shipped design.** Paired on the
validation block over the 9 cells complete in all four arms, against `bits`:
`shaw_k` −0.006 ±0.019, `shaw_kv` −0.004 ±0.019, `bits_v` −0.001 ±0.001 nats. All
inside the 0.015 threshold. Two readings inside that: `shaw_kv` ≈ `shaw_k`, so the
value term adds nothing on top of learned keys, and `bits_v` ≈ 0, so it adds
nothing on its own either — which is against the prediction above, since the value
term is the one that genuinely widens the function class. Validation, not the test
corner, and partial; the corner numbers decide.

(An earlier note here said the learned-key arm was slower to its optimum, from
`item/2000`. Across all nine cells that is not so: median epochs to the
validation optimum are `bits` 215, `shaw_k` 184, `shaw_kv` 192, `bits_v` 215 — if
anything the learned arms are slightly earlier. The single-cell reading was
over-read.)

## Checklist

- [ ] §1 architecture: standard block, both pieces dropped together, +0.150 note
- [ ] §1 caveat: K=300 stoppability
- [ ] §2 codec: do not claim it
- [ ] §3 protocol F13 replaces test-selection (D14) — biggest change
- [ ] §3 the two cells with no dev block
- [ ] §4 epoch budget 2000 is a search budget; overfitting, not grokking
- [ ] §5 `max_item=500` train and eval
- [ ] §6 restart-on-degenerate sentence + "detected from training curve only"
- [ ] §6 selector-floor: implement or do not claim
- [ ] §7 Stan 1 chain + 4-chain footnote (B12)
- [ ] §7 swap LLM-Rubric Stan numbers to the rerun (all 10 sizes in; K=30 matches to +0.0003)
- [ ] §8 LLM-Rubric retrained on dual; state the arm's own val split
- [ ] §8 do not call the LLM-Rubric number a final epoch — it is val-selected
- [ ] §8 K=10/30/150 bumps are overfitting, not degeneracy
- [ ] Swap all three result tables (§9)
- [ ] §9 the ~0.13-nat Marformer-vs-Stan gap on synthetic still needs an explanation
- [ ] §9 do NOT rerun the 6 flagged cells (that would be selection on test)
- [ ] §10 figures were argmin-on-test; fixed, but the PUBLISHED figures have this bug
- [x] §10 DECIDE: settled — both report the test corner
- [x] §10 figures now score the corner on BOTH arms (the first fix missed the NT arm)
- [ ] §10 the NT curve moves by up to 0.066 and gains 8 points; check any sentence
      about T and NT diverging at large K — they nearly coincide at K=4000
- [ ] §11 fix the key-dimensionality sentence (main.tex:841) — Domain 3 ships the
      extra-query-dims design, not the reduced-key one
- [ ] §11 if adopting Shaw's framing: "better learning dynamics", NOT "more powerful"
      (the key term is rank-constrained); the VALUE term is the one that adds capacity
- [x] Rebuild figures (`make_figures.sh --run-id final-dual`, policy `last`)
