# Removing the duplicated residual stream

Experiment report for Tom's question about the double residual stream.

Runs: `RERUN/runs/resid-01` (baseline + `none`), `resid-02` (identity-folded),
`resid-03` (inner-residual ablation). All arms train on the same freshly
generated splits as the Domain 3 replication (`--data-from replicate-01`), with
the same seeds, so every comparison below can be paired.

---

## 1. What the architecture currently does

In `EntityMarformer._forward_dual`, each of the 8 layers ends like this:

```python
combined = combined + attn_out                       # inner residual (attn)
combined = combined + z_ff                           # inner residual (FFN)

features = features + dropout(proj_out(combined))    # outer residual
params   = params   + dropout(W_param(combined))     # outer residual
combined = torch.cat([features, params], dim=-1)
```

Since `cat([features, params])` *is* the `combined` the layer started with, the
layer as a whole computes

$$h_{\ell+1} \;=\; h_\ell \;+\; M\,c_\ell,
\qquad c_\ell \;=\; h_\ell + \mathrm{attn}(N_1 h_\ell) + \mathrm{ff}(N_2\,\cdot),
\qquad M \;=\; \begin{bmatrix} W_1 \\ W_2 \end{bmatrix}$$

So there are two residual streams stacked: an **inner** one that produces `c`,
and an **outer** one that adds a *linear rotation* `Mc` back onto `h`. The
attention and FFN outputs never reach the next layer directly — only through
`M`.

`M` stacks `proj_out` (75×80) on `W_param` (5×80), so it is a **square 80×80
map**. That matters: it is dimensionally capable of containing an identity.
Whether it does is the empirical question this report answers.

### What the two streams actually hold

With `embedding_dim = 80` and `global_param_dim = 5`, every token's state is

    combined (80) = [ features (75) | params (5) ]

`features` is the entity-identity half: a per-type centroid plus an optional
per-entity deviation. `params` is **not** a generic hidden vector — it is the
model's running logit accumulator. `RatingVariableType.build_param` initialises
it as

| token | `params` at layer 0 |
|---|---|
| masked or missing | `[1, 0, 0, 0, 0]` — mask bit set, logits zero |
| observed, value `v` | `[0, ...]` with `logit_high = 20.0` at slot `1+v` |

and the model's output logits are read straight back out of those same slots,
`params[1 : 1+C]`. So the outer residual makes the output a **sum over layers**:

$$\text{logits} \;=\; \underbrace{p_0}_{\text{input encoding}} \;+\; \sum_{\ell=1}^{8} W_2\,c_\ell$$

This is exactly why the Logit Lens appendix works — `params` is readable as a
distribution at every depth, because it is the same object at every depth.

Two consequences for the ablation:

- For a **masked** token — i.e. every prediction target — `p_0` contributes
  zero to the logits, so nothing is lost by discarding the *input* encoding.
- For an **observed** token, `p_0` carries a ±20 logit of the true value. Under
  `none` that spike is destroyed after layer 1 and must be re-derived by
  attention at every subsequent layer.

---

## 2. The variants

Two independent knobs were added, both exposed as CLI flags on `train.py`,
threaded through `run_marformer_group.sh` as `DOMAIN3_DUAL_OUTER_RESIDUAL` and
`DOMAIN3_INNER_RESIDUAL`, and recorded in `train_config.json` so `test.py`
rebuilds the right architecture.

**`--dual-outer-residual`** — what the layer emits:

| value | layer output | note |
|---|---|---|
| `all` | `h + Mc` | shipped architecture; the double stream |
| `none` | `Mc` | outer add **deleted**; stream carried by `c` alone, rotated each layer |
| `none_id` | `c + Mc = (I + M)c` | outer add **replaced by a structural identity** inside the projection |

**`--inner-residual`** — what happens inside the block:

| value | block body |
|---|---|
| `both` | `c = h + attn(N₁h)`; `c = c + ff(N₂c)` — shipped |
| `ff_only` | `c = attn(N₁h)`;   `c = c + ff(N₂c)` |
| `none` | `c = attn(N₁h)`;   `c = ff(N₂c)` — outer add is the only skip |

Combining `inner_residual="none"` with `dual_outer_residual="none"` is rejected
at construction: it would leave no skip connection at all.

All variants have **identical parameter counts** — `proj_out` and `W_param`
exist in every one, only the forward pass differs — so nothing below is a
capacity difference.

### On `none_id`, the I+W design

`none_id` is the important one. It is the answer to "why not just add an
identity to `W₁;W₂` and simulate the outer residual?" The identity is folded in
**structurally and permanently** — it is not a `W = I` initialisation that
training can drift away from. The layer applies `(I + M)`, and `M` is the only
learned part.

The reason this is not merely cosmetic: under `none`, weight decay pulls `M`
toward the **zero map**, so the layer degrades toward `h' = 0` — catastrophic.
Under `none_id`, decay pulls `M` toward zero too, but the layer then degrades
toward `h' = c` — a clean identity layer. Same regulariser, opposite failure
mode. §4 shows this is the mechanism that matters.

### Dropped: `not_last`

An earlier round also tested `not_last` / `not_last_id` (drop the outer add in
the final layer only). **Tom has ruled this direction out and it is not
reported here.** The runs exist under `resid-01` / `resid-02` if anyone wants
them, but changing only the last layer alters how the output head reads the
final state rather than how signal propagates through the stack, so it does not
speak to the architectural question.

---

## 3. Results

4 variants × 3 sizes (item axis, transductive, K = 100/300/500) × 3 seeds
(42/43/44) = 36 runs, 400 epochs each, everything else at the paper's values.

Compared on the **last** checkpoint. Domain 3 has no validation split (see
[OVERFITTING.md](OVERFITTING.md) §1), so a test-argmin would let each arm pick
its own luckiest checkpoint and the comparison would largely measure selection
noise.

### Absolute test-missing log-loss (chance = ln 4 = 1.3863)

| K | `all` (shipped) | `none` | `none_id` | `all` + `inner=none` |
|---|---|---|---|---|
| 100 | **1.1123** ± 0.0788 | 1.2066 ± 0.0811 | 1.1171 ± 0.0451 | 1.3480 ± 0.2592 |
| 300 | **0.8904** ± 0.0063 | 0.9343 ± 0.0135 | 0.9185 ± 0.0124 | 0.9944 ± 0.0113 |
| 500 | **0.8574** ± 0.0124 | 0.8760 ± 0.0191 | 0.8795 ± 0.0097 | 0.8871 ± 0.0081 |

### Paired against `all` (same seed, same split; positive = worse)

| K | `none` | `none_id` | `all` + `inner=none` |
|---|---|---|---|
| 100 | +0.0942 (se 0.0765, t 1.2) | **+0.0047** (se 0.0700, t 0.1) | +0.2357 (se 0.1112, t 2.1) |
| 300 | +0.0439 (se 0.0106, t 4.1) | +0.0281 (se 0.0105, t 2.7) | +0.1040 (se 0.0098, **t 10.6**) |
| 500 | +0.0186 (se 0.0146, t 1.3) | +0.0221 (se 0.0127, t 1.7) | +0.0297 (se 0.0109, t 2.7) |

Per-seed, since three seeds is thin and the means hide real disagreement:

| K | variant | seed 42 | seed 43 | seed 44 |
|---|---|---|---|---|
| 100 | `none` | +0.143 | +0.196 | −0.056 |
| 100 | `none_id` | −0.013 | +0.134 | −0.107 |
| 100 | `inner=none` | +0.152 | +0.099 | **+0.456** |
| 300 | `none` | +0.056 | +0.023 | +0.053 |
| 300 | `none_id` | +0.049 | +0.016 | +0.019 |
| 300 | `inner=none` | +0.119 | +0.085 | +0.108 |
| 500 | `none` | +0.045 | −0.006 | +0.017 |
| 500 | `none_id` | +0.045 | +0.020 | +0.002 |
| 500 | `inner=none` | +0.052 | +0.020 | +0.018 |

K = 300 is the cell to read. It has by far the tightest spread (baseline sd
0.0063, ~0.7% of the mean), and it is the only size where every variant's sign
is consistent across all three seeds. K = 100 is too noisy to separate anything
except `inner=none`; K = 500 compresses everything into a 0.03-nat band.

---

## 4. Why: the weight geometry

`analysis/residual_geometry.py` loads the final checkpoint of each variant
(seed 42, K = 300) and measures the outer map `M = [W₁; W₂]` per layer.

**Finding 1 — `M` is full rank, so representational capacity is not the issue.**
Rank 80/80 at every layer in every variant. (An earlier version of this report
claimed `M` was rank-limited. That was wrong; corrected in `ce268c2`.)

**Finding 2 — `M` is badly conditioned, and never resembles the identity.**

| variant | cond(`M`) range over 8 layers | ‖M‖_F | cos(`M`, `I`) |
|---|---|---|---|
| `all` | 90 – 367 | 5.04 – 5.17 | −0.041 … +0.008 |
| `none` | 48 – 937 | 5.13 – 5.21 | −0.022 … +0.038 |
| `none_id` | 80 – 682 | 5.02 – 5.17 | −0.038 … +0.013 |
| `all`+`inner=none` | 107 – **5744** | 5.15 – 5.55 | −0.021 … +0.033 |

`cos(M, I) ≈ 0` in **every variant at every layer**, including `none`, where
learning an identity would help most. `‖M − I‖_F ≈ 10.34` throughout, which is
almost exactly $\sqrt{\|M\|_F^2 + 80}$ — i.e. `M` is essentially *orthogonal* to
the identity.

**The model never rediscovers the skip connection on its own.** That is the
central result of this section, and it is why the identity has to be put there
structurally. Plausible reason: AdamW's decay pulls `M` toward 0, and `I` is a
high-norm target (‖I‖_F = 8.94 against ‖M‖_F ≈ 5.1), so nothing pushes toward
it.

**Finding 3 — adding the identity fixes the conditioning, by two orders of
magnitude.**

| variant | cond(`M`) | cond(`I + M`) |
|---|---|---|
| `all` | 90 – 367 | **5.5 – 7.0** |
| `none` | 48 – 937 | **5.3 – 7.5** |
| `none_id` | 80 – 682 | **5.3 – 6.7** |
| `all`+`inner=none` | 107 – 5744 | **5.9 – 12.0** |

This is the whole story in one table. The learned map on its own is ill-conditioned
everywhere; the map *with an identity* is well-conditioned everywhere. Under
`none`, state reaches the output as a product of eight ill-conditioned maps, and
directions aligned with the small singular values (σ_min down to 0.0012) are
crushed geometrically. Under `all` and `none_id`, an additive identity path
exists at every layer and the effective condition number drops to ~6.

That is the ordinary residual-network argument, and it applies to the outer
stream exactly as it does to the inner one.

---

## 5. Reading

**Tom's I+W hypothesis is largely right.** `none_id` recovers most of what
`none` loses: at K = 100 the paired gap goes +0.094 → **+0.005** (95% of the
deficit recovered, statistically indistinguishable from the shipped model), and
at K = 300 +0.044 → +0.028 (~36% recovered). The outer add is, to a good
approximation, redundant **provided the identity is put back structurally**. It
is not redundant if you simply delete it. The mechanism in §4 is exactly the one
Tom proposed: the outer residual's job is to supply an identity path, and a
folded-in `I` supplies the same thing.

It is not a *complete* substitute. At K = 300, where the measurement is
cleanest, `none_id` is still +0.028 worse (t 2.7), and at K = 500 it is
indistinguishable from plain `none`. Something beyond conditioning — most
likely the additive logit accumulation described in §1, which `none_id` also
destroys — accounts for the remainder.

**Removing the inner residual is clearly the worst option, and this inverts the
original framing.** `all + inner=none` is worse at every size, by the largest
margins in the series: +0.236 at K = 100, +0.104 at K = 300 (t 10.6, the
strongest effect anywhere in this report), +0.030 at K = 500. At K = 100 it
lands at 1.3480 against a chance level of 1.3863, with a seed spread of ±0.259
— seed 44 reaches 1.6315, i.e. **worse than chance**; that seed essentially
fails to train.

Compare the two ablations directly at K = 300: removing the *outer* residual
costs +0.044, removing the *inner* one costs +0.104 — **2.4× more**. The inner
residual is the load-bearing one. The geometry agrees: `inner=none` is the only
variant whose conditioning blows up (cond 5744 at layer 6, σ_min = 0.0002),
because with no inner skip the entire signal path runs through `M`.

This is the opposite of the direction the original question anticipated. The
"unified inner residual stream" is the right target, but you get there by
keeping the inner stream and folding the identity into the outer projection —
`none_id` — not by removing the inner one.

---

## 6. Recommendation

1. **Adopt `none_id` if the goal is a single clean residual stream.** At K = 100
   it is free; at K = 300 it costs ~0.03 nats. The architecture becomes
   `h' = (I + M)c` — one stream, one skip, and the output is a genuine
   unembedding of it. Quote the K = 300 cost honestly rather than the K = 100
   result.
2. **Do not delete the outer add outright (`none`).** Worse at every size, for a
   mechanically well-understood reason.
3. **Do not touch the inner residual.** It is more load-bearing than the outer
   one, and removing it makes small-K training unstable.
4. **Still unrun: `use_embed_unembed=True`.** The codebase already contains a
   proper single-residual-stream architecture — standard pre-norm stack, per-type
   embedders and unembedders, no split/concat at all. It is a better-founded
   version of the unified-stream idea than any of the above and has never been
   benchmarked on Domain 3. This remains the obvious next experiment.

---

## 7. Caveats

- 3 seeds per cell. Adequate at K = 300; thin at K = 100, where the baseline sd
  (0.079) exceeds most of the effects being measured.
- Item axis, transductive only. The annotator axis uses different entity dropout
  (`annotator_dropout_rate = 0.8`) and may behave differently.
- Weight geometry is from one seed (42) at one size (K = 300). The conditioning
  gap is large enough (2 orders of magnitude) that seed variation is unlikely to
  reverse it, but it has not been checked across seeds.
- Compared on the last checkpoint, so absolute numbers sit above the paper's
  test-argmin-selected ones (`DISCREPANCIES.md` D14). `all` at K = 500 is 0.857
  here versus 0.830 test-selected in `replicate-01`.
- The `inner=ff_only` middle setting was implemented but never run.

---

## 8. Reproducing

```bash
cd RERUN
./submit_ablation.sh --run-id resid-01 --data-from replicate-01 \
    --variants all,none --sizes 100,300,500 --seeds 42,43,44
./submit_ablation.sh --run-id resid-02 --data-from replicate-01 --variants none_id
./submit_ablation.sh --run-id resid-03 --data-from replicate-01 \
    --variants all --inner none

# once training finishes
sbatch ... jobs/ablation_eval.sbatch          # RUN_ID=resid-0{1,2,3}

python analysis/ablation_combined.py \
    --run-dirs runs/resid-01 runs/resid-02 runs/resid-03 \
    --variants all,none,none_id,all+inner_none --baseline all

python analysis/residual_geometry.py --runs \
    runs/resid-0{1,1,2,3}/RESULTS/.../{all,none,none_id,all+inner_none}/seed42/...
```
