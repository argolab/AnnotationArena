# Logit Lens, re-run on final-v2

**Question (Tom, 2026-10-07):** do we still have the logit lens experiment, can we
replicate it, and if so should v1 use the new numbers?

**Answer:** the script never existed (D8) and has now been written; it runs, it is
gated, and it reproduces the *method*. It does **not** reproduce the appendix's
*numbers* — not one of them — and it contradicts three of the appendix's claims.

## What was built

| file | what it does |
|---|---|
| `RERUN/analysis/logit_lens.py` | per-layer readouts + probes + metrics -> JSON |
| `RERUN/analysis/plot_logit_lens.py` | the two appendix figures from that JSON |
| `RERUN/jobs/logit_lens.sbatch` | runs the analysis (CPU, ~4 min) |
| `RERUN/jobs/logit_lens_plot.sbatch` | redraws the figures |

Reimplemented from the appendix's own method description, because no `lens`,
`layerwise` or tuned-probe code exists in either repository.

**The gate.** The lens's final layer *is* the model the paper reports, so it must
reproduce the paper's reported number or the lens is reading a different
population. `gate_against_corner()` checks it against `corner_metrics/corner.json`
and aborts on any mismatch:

```
[gate] corner.json n=316 log_loss=0.8272  lens n=316 log_loss=0.8272
[gate] final layer reproduces the reported corner exactly
```

A second gate asserts the replayed block arithmetic equals `model(graph)`
bit-for-bit, so an architecture change under the script fails loudly instead of
producing a plausible wrong curve.

## Setup

`final-v2`, `Item_T_500`, `last.ckpt` — the refit cell the paper reports.
L=8, D=80, P=5, C=4, all matching the appendix's stated geometry.
Probes: linear, AdamW, 300 epochs, KL to the final model's predictions, fitted on
the **1403 training-missing** cells (cells the training loss never scores, so they
are held out from the model and carry no test information), evaluated on the
**316-cell test corner**.

Two recorded deviations: inputs are standardised before the linear map (an affine
reparametrisation — the probe is still exactly linear — without which 300 epochs
does not converge on a stream mixing +-20 logits with O(1) features), and the step
size is 1e-2 because the appendix records none.

## Results

```
 L |   Param    acc  smECE |  TunedP  smECE | Tuned80  smECE
 0 |  1.3863  0.234  0.000 |  1.4169  0.041 |  1.4170  0.041
 1 |  1.2642  0.601  0.070 |  0.9697  0.030 |  0.9459  0.030
 2 |  1.2714  0.560  0.065 |  0.9716  0.029 |  0.9448  0.030
 3 |  1.1752  0.560  0.078 |  0.8181  0.024 |  0.8306  0.042
 4 |  1.2222  0.563  0.073 |  0.8280  0.026 |  0.8308  0.043
 5 |  1.1680  0.601  0.078 |  0.8187  0.027 |  0.8315  0.043
 6 |  1.1170  0.566  0.078 |  0.8191  0.028 |  0.8309  0.043
 7 |  0.9497  0.592  0.063 |  0.8260  0.036 |  0.8309  0.042
 8 |  0.8272  0.633  0.042 |  0.8326  0.039 |  0.8298  0.042
final model  ll=0.8272  acc=0.6329  MBR-L2=0.3827  smECE=0.0421 +/- 0.0234
```

## Replication: numbers

| appendix | value | final-v2 | verdict |
|---|---|---|---|
| cells evaluated | 414 | **316** | no — see below |
| Param Lens, layer 0 | 1.386 (=ln 4) | **1.3863** | **yes, exactly** |
| Param Lens, layer 1 | 1.022 | 1.2642 | no |
| Tuned Param Lens, layer 1 | 0.682 | 0.9697 | no |
| Tuned 80d-Lens, layer 1 | 0.566 | 0.9459 | no |
| three curves at layer 6 | 0.559 / 0.536 / 0.523 | 1.1170 / 0.8191 / 0.8309 | no |
| Param Lens, layer 7 | 0.533 | 0.9497 | no |
| final model | 0.518 | **0.8272** | no |
| Param Lens layer-1 accuracy | 78.5% | 60.1% | no |
| Param Lens layer-1 smECE | 0.163 | 0.070 | no |
| Param Lens layer-6 smECE | 0.034 | 0.078 | no |
| Param Lens final smECE | 0.027 | 0.042 | no |

**414 is not a population that exists in this run.** The corner is **316 cells at
every K** on the item axis, and the pooled (D22) population at K=500 is 1719. 414
is the count `corner_score.sbatch`'s header records for the *pre-fix* corner
definition at `Item_T_300`. So the appendix's population is neither the corner nor
the pooled set as those are now defined.

**The appendix's 0.518 is better than this run achieves anywhere.** final-v2's
corner log-loss at K=500 is 0.827, and even at K=4000 it is 0.683. So the gap is
not a K mislabel.

**I cannot attribute the difference.** The appendix's model is not any checkpoint
on disk — its figures were committed 2026-06-28, before every rerun. At least four
things differ at once and none can be isolated without a new training run, which
is out of scope: the evaluation population (D22/D23), the two eval bugs (D14's
argmin-on-test selection would bias the quoted loss *down*, D20's max_item), the
F13 search/refit protocol, and the architecture (standard block + Shaw relational
attention + delta_tau 0.7). D14 in particular is the kind of error that produces a
suspiciously low number, but that is a hypothesis, not a finding.

## Replication: claims

**Survives.**
- "The model initialises to maximum uncertainty." The layer-0 Param Lens is
  1.3863 = ln 4 to four decimals. Exact.
- "High-quality predictions are already latent in the full hidden state but have
  not been projected into the output-aligned parameter coordinates." At layer 1
  the tuned lenses read 0.946 / 0.970 while the raw parameter slice reads 1.264,
  and by layer 3 the tuned lenses are at the final model's loss. This is the
  appendix's central claim and it holds.
- "The feature stream carries early-layer information beyond the parameter slice."
  80d beats Tuned Param at layers 1-2 (0.946 vs 0.970), though the margin is small
  and it reverses from layer 3 on.
- "The total predictive information in the hidden state is stable while the raw
  parameter stream catches up." The 80d lens is flat at 0.831 +- 0.001 from layer
  3 to layer 8 while the raw Param Lens climbs from 1.175 to 0.827.

**Contradicted.**
- "At layer 6 all three curves converge." At layer 6 the raw Param Lens is 1.117
  against the final model's 0.827. They converge only at layer 8.
- "By layer 7 the raw Param Lens is within 0.015 of the final model." It is 0.950
  against 0.827 — a gap of 0.123, eight times the quoted figure.
- "The model has committed its beliefs into output-aligned format by depth 6 of 8."
  It commits at depth 8. The last two layers do the bulk of the formatting work
  (1.117 -> 0.950 -> 0.827), which is the opposite of the appendix's reading and
  weakens, rather than supports, the deep-supervision suggestion that closes the
  section.
- "Each layer produces a progressively better-calibrated distribution." Param Lens
  smECE is non-monotone: 0.070, 0.065, 0.078, 0.073, 0.078, 0.078, 0.063, 0.042.

**Weakened.** The aggregation phase is real but less extreme: layer 1 covers 79% of
the uniform-to-final gap here against 94% in the appendix.

## One new observation, offered as suggestive only

A linear probe on the 5-dimensional parameter slice at layers 3-6 scores 0.818-0.819
with smECE 0.024-0.028, against the final model's 0.827 and smECE 0.042 — better
calibrated than the model's own head, while being *trained to imitate it*. That is
the same direction as the overconfidence decomposition (`d_oc` on the dev curve):
the top of the network sharpens predictions past where they should be. On 316 cells
the log-loss difference is well inside noise, so this is a pointer, not a result.

## The figure-clipping defect: cause found

Rendering these two figures reproduced the clipping Tom reported on figure 8, and
it is now diagnosed. `savefig.bbox: "tight"` was never the problem — it is set, and
the figures still clipped. The cause is `DOMAIN3_FONT_SCALE=3`: a 21-character
rotated y label becomes ~420pt of type against a ~360pt axis and runs off the
canvas; a two-line panel title centred on the axes overruns the figure edge in the
right-hand column. Neither `tight_layout()` nor a tight save bbox rescues text that
has already overflowed.

Three fixes, all applied and visually verified here:
1. `layout="constrained"` instead of `tight_layout()`, with the legend placed as
   `loc="outside right center"` so the layout engine reserves room for it rather
   than overlaying it.
2. Short axis labels ("Log loss", not "Test Missing Log Loss") — the title already
   carries the full name.
3. Panel titles at `ts(11)` rather than `ts(14)` in a multi-column grid.

This carries directly to C1/C8 and to figure 8's panels (b) and (d).
