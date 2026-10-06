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
| 40 | 1.212 | 1.2091 | −0.003 |
| 50 | 1.205 | 1.2039 | −0.001 |
| 75 | 1.160 | 1.1610 | +0.001 |
| 100 | 1.128 | 1.1178 | −0.010 |
| 125 | 1.093 | 1.1125 | +0.020 |
| 150 | 1.088 | 1.0847 | −0.003 |
| 175 | 1.077 | 1.0925 | +0.016 |

Excluding K=20: median **−0.002**, both signs, 3 of 8 positive — digitisation
error plus 1-chain MCMC noise. **No systematic shift, so the rerun numbers can
replace the published ones.**

**K=20 deserves a look before you quote it.** The published curve *rises* from
K=10 (1.435) to K=20 (1.475) — its only non-monotone step — and the rerun removes
that kink (1.3902 → 1.2748). A 1-chain fit on 20 items mixing badly is the
likeliest explanation, which would mean the rerun is *fixing* a bad published
fit rather than disagreeing with a good one. Your call whether to say anything;
the safe move is to plot the rerun and say nothing.

<!-- NUMBERS -->

---

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
- [ ] §7 swap LLM-Rubric Stan numbers to the rerun
- [ ] Swap all three result tables
- [ ] Rebuild figures (`make_figures.sh --run-id final-dual`)
