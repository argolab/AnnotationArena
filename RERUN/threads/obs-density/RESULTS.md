# Observation density, Phase 1 — results

Run 2026-10-17. K=500, item axis, transductive. 6 generative draws x 3
densities x 2 model seeds; Marformer searched to a 2000-epoch cap, stopping
epoch selected per run on a 15% `mcar` carve of the observed training cells,
then refit on the whole training partition. Stan transductive, four arms.
Metric is **corner log-loss** (`split_missing_metrics.py --corner-whole`), the
held-out 100x10 block only — never the pooled metric.

Reproduce: `./submit_phase1.sh` then `python collect.py`.

## Headline: paired per-draw difference, Marformer minus Stan

Negative = Marformer better. Draws are paired by the nested-mask design, so the
paired sd is the right error bar; it is 2-3x tighter than the across-draw sd.
"On par" = |diff| < 2 sem, the convention the paper's captions already use.

| observed | vs Oracle | vs Proj | vs Bin | vs Both |
|---|---|---|---|---|
| 15% | **+0.148** ±0.014 | **+0.143** ±0.017 | **+0.071** ±0.020 | **+0.067** ±0.019 |
| 50% | **+0.143** ±0.008 | **+0.139** ±0.010 | −0.006 ±0.021 *(on par)* | −0.030 ±0.022 *(on par)* |
| 85% | **+0.122** ±0.011 | **+0.099** ±0.017 | **−0.096** ±0.016 | **−0.150** ±0.030 |

All rows n=6. Bold = significant at |diff| > 2 sem. `collect.py` reports all 18
cells complete with consistent scored populations and no guard triggered: no
collapse to uniform, no truncated refit, no epoch budget at the cap.

## Absolute corner log-loss

| observed | corner cells/draw | Marformer | Oracle | Proj | Bin | Both |
|---|---|---|---|---|---|---|
| 15% | 539–660 | 1.144 | 0.997 | 1.002 | 1.073 | 1.077 |
| 50% | 332–410 | 0.851 | 0.708 | 0.712 | 0.857 | 0.881 |
| 85% | 100–125 | 0.724 | 0.602 | 0.626 | 0.820 | 0.874 |

Uniform baseline = ln 4 = 1.386. **Do not compare down a column**: the three
densities score different target sets (the corner is what is missing, so it
shrinks as density rises). Within a row both families see identical cells —
verified, `collect.py` asserts Marformer's `n` equals each Stan arm's
`n_missing_ratings` per cell.

## What it says

1. **Marformer handles the whole range.** It beats uniform at every density and
   never collapses. The sparsest cell (15% observed = 444 observed training
   cells, 68 supervised tokens per step) still reaches 1.144 against uniform
   1.386.

2. **The gap to the correctly-specified model is roughly constant**, +0.148 /
   +0.143 / +0.122 against the Oracle, narrowing modestly at the dense end. So
   Marformer tracks the Bayesian model at a near-fixed offset rather than
   degrading disproportionately when data is scarce. This is the strongest form
   of the "deals with different observation ratios" claim.

3. **Against the genuinely misspecified models the ordering flips with
   density**: Marformer loses at 15%, is on par at 50%, and wins clearly at 85%
   (−0.096 vs Bin, −0.150 vs Both). The mechanism is visible in the absolute
   table — Bin and Both *saturate* (0.857→0.820 and 0.881→0.874 from 50% to
   85%) while Marformer and Proj keep improving. Extra observation stops buying
   a misspecified parametric model anything; it keeps buying the amortised
   model something.

4. **The paper's "better or on par" claim is density-dependent.** It holds at
   the published 50% density and strengthens at 85%, but **fails at 15%**, where
   Marformer is significantly worse than all four Stan arms. That belongs in the
   text as a stated limit, not left for a reader to discover.

## Selection behaviour (a second result, arguably as interesting)

Seed 42, all 18 cells, from the search curves:

| observed | optimal epoch | range | best val | val @2000 | degradation |
|---|---|---|---|---|---|
| 15% | 245 | 125–467 | 1.027 | 2.905 | **+1.878** |
| 50% | 331 | 190–546 | 0.833 | 1.560 | +0.727 |
| 85% | 550 | 362–839 | 0.721 | 1.015 | +0.295 |

- The optimal epoch **grows** with density; overfitting severity past it
  **shrinks** with density.
- At 15% observed, training to a flat 2000 epochs ends at 2.905 — **more than
  twice as bad as predicting uniform**. A fixed epoch budget across densities
  would not be merely suboptimal, it would be catastrophic at the sparse end.
  This is a far sharper argument for per-run epoch selection (F13) than the
  ~0.022 nats it was justified on.
- **The 2000 cap never bound**: the largest selected epoch over all 36 runs is
  839.

## The thin carve at 15% turned out not to matter

At 15% observed the `mcar` carve is only ~60-67 cells, and the two model seeds
disagree about the stopping epoch by 2.13x on average (up to 2.93x). That does
**not** propagate into the metric:

| observed | mean epoch ratio between seeds | mean \|seed diff\| in corner log-loss |
|---|---|---|
| 15% | 2.13 (worst) | 0.0185 (smallest) |
| 50% | 1.23 | 0.0474 |
| 85% | 1.32 | 0.0473 |

The val basin at 15% is flat-bottomed — anywhere from epoch ~100 to ~470 scores
the same. So `--select per-run` on a 67-cell carve is fine here, and the
`--select per-size` fallback is not needed. The larger seed spread at 50/85% is
ordinary initialisation variance, not selection noise.

## Bridge 1: the carve choice costs real nats (PROVISIONAL, n=1 -> widening)

Same cell, same seed, same 332 corner cells; the two runs differ ONLY in which
carve selected the stopping epoch.

| carve | selected epoch | corner log-loss |
|---|---|---|
| `mcar` frac .15 (adopted here) | 426 | **0.8600** |
| `block` frac .15 (**the frozen config**) | 66 | **0.9847** |

`block` stops at epoch 66 -- barely past the uniform-escape point at this
density -- and pays **0.125 nats**. That is larger than the whole
Marformer-minus-Bin gap at 50% observed and comparable to the residual gap to
the Oracle.

**If this replicates it is a finding about the main results, not just this
thread**, since the frozen final config selects on `block`. A six-draw version
(draws s42, s101-s105, 50% observed, seed 42) is running as run ids
`obsden-s*-o50-blk`, tag `blkwide`. Until it lands this is one cell and one
seed and should not be quoted.

A plausible mechanism, consistent with the carve analysis in PLAN.md: `block`
scores the val item block's *already-missing* cells, and in the transductive arm
those items' observed cells are still being trained on, so the metric sees the
model fitting items it has already seen and turns early. `mcar`'s targets are
cells pulled out of the observed set entirely, which is the harder and
better-matched question.

## Bridge 2: val-frac 0.15 is the better choice (closed)

15% observed, draw s101, seed 42, same 539 corner cells:

| carve fraction | selected epoch | corner log-loss |
|---|---|---|
| `mcar` frac .15 | 229 | **1.1662** |
| `mcar` frac .30 | 125 | 1.2113 |

Doubling the carve costs 0.045 nats: it strips more training signal out of the
search and then stops earlier. So keeping `--val-frac 0.15` at the sparse end is
right, and the thin-carve worry is closed in favour of the shipped setting. Note
this is slightly larger than the 0.0185 seed spread, so the 15% basin is
flat-ish but not perfectly flat.

## Caveats

- Within-row comparisons only (different target sets per row).
- The **85% column is the noisiest** in absolute cell count (~115 scored cells
  per draw vs ~600 at 15%), the opposite of the natural intuition.
- The 50% column is a **re-draw** of the mask on the published latent data, not
  the published mask, so it will not reproduce published numbers to the digit.
- Missingness is MCAR throughout, on a structured (4-of-50) support. This is a
  density result, not a missing-not-at-random result.
