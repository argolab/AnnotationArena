# Removing the duplicated residual stream

Experiment report for Tom's question about the double residual stream.
Run: `RERUN/runs/resid-01`, on the same freshly generated splits as the
Domain 3 replication (`RERUN/runs/replicate-01`).

---

## What the architecture currently does

In `EntityMarformer._forward_dual`, each layer ends like this:

```python
combined = combined + attn_out          # inner residual
combined = combined + z_ff              # inner residual

features = features + dropout(proj_out(combined))   # outer residual
params   = params   + dropout(W_param(combined))    # outer residual
combined = torch.cat([features, params], dim=-1)
```

Since `cat([features, params])` *is* the `combined` the layer started with, the
layer as a whole computes

$$h_{\ell+1} \;=\; h_\ell \;+\; [\,W_1;\,W_2\,]\,c_\ell,
\qquad c_\ell = h_\ell + \mathrm{attn}(h_\ell) + \mathrm{ff}(\cdot)$$

So there really are two residual streams stacked: an inner one that produces
`c`, and an outer one that adds a *linear rotation* of `c` back onto `h`. The
attention and FFN outputs never reach the next layer directly — they only get
there through `[W_1; W_2]`.

### What the two streams actually hold

This matters for reading the result, and it is not obvious from the code.

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

and the model's output logits are read straight back out of the same slots,
`params[1 : 1+C]`. So the outer residual makes the output a **sum over layers**:

$$\text{logits} \;=\; \underbrace{p_0}_{\text{input encoding}} \;+\; \sum_{\ell=1}^{8} W_\text{param}\,c_\ell$$

This is exactly why the Logit Lens appendix works — `params` is readable as a
distribution at every depth, because it is the same object at every depth.

Two consequences for the ablation:

- For a **masked** token — i.e. every prediction target — `p_0` contributes
  zero to the logits, so nothing is lost by discarding the *input* encoding.
- For an **observed** token, `p_0` carries a ±20 logit of the true value. Under
  `none` that spike is destroyed after layer 1 and must be re-derived by
  attention at every subsequent layer.

## The three variants

Added as `dual_outer_residual` (config → `train.py --dual-outer-residual` →
`DOMAIN3_DUAL_OUTER_RESIDUAL` in the group runner):

| variant | last line of every layer |
|---|---|
| `all` | `features += proj_out(c)`, `params += W_param(c)` — shipped architecture |
| `not_last` | same, except the **final** layer emits only `[proj_out(c); W_param(c)]` |
| `none` | **every** layer emits only `[proj_out(c); W_param(c)]` |

`not_last` is Tom's specific proposal: don't add `h` in the last layer, keep
only the part rotated by `W_1;W_2`. `none` is the fully unified inner stream.

The three share identical parameters — `proj_out` and `W_param` exist in all
of them, only the forward pass differs — so any difference is not a capacity
difference.

## Setup

3 variants x 3 sizes (item-axis transductive, K = 100/300/500) x 3 seeds
(42/43/44) = 27 runs, 400 epochs each, everything else at the paper's values.
All arms share byte-identical splits (`--data-from replicate-01`).

Compared on the **last** checkpoint. Domain 3 has no validation split (see
[OVERFITTING.md](OVERFITTING.md) §1), so a test-argmin would let each arm pick
its own luckiest checkpoint and the comparison would largely measure selection
noise. The test-argmin numbers are given too, and they tell the same story.

## Results

Test missing log-loss, last checkpoint, mean ± sd over 3 seeds:

| K_train | `all` | `not_last` | `none` |
|---|---|---|---|
| 100 | **1.1123** ± 0.0788 | 1.1604 ± 0.0852 | 1.2066 ± 0.0811 |
| 300 | **0.8904** ± 0.0063 | 0.8932 ± 0.0295 | 0.9343 ± 0.0135 |
| 500 | **0.8574** ± 0.0124 | 0.8595 ± 0.0191 | 0.8760 ± 0.0191 |

Because every arm shares seeds and splits, the paired differences are much
more informative than those standard deviations (positive = variant is worse):

**`not_last` − `all`**

| K_train | mean | se | t | per-seed |
|---|---|---|---|---|
| 100 | +0.0480 | 0.0068 | **+7.08** | +0.061, +0.039, +0.044 |
| 300 | +0.0027 | 0.0203 | +0.13 | +0.043, −0.018, −0.017 |
| 500 | +0.0020 | 0.0178 | +0.12 | +0.037, −0.012, −0.019 |

**`none` − `all`**

| K_train | mean | se | t | per-seed |
|---|---|---|---|---|
| 100 | +0.0942 | 0.0765 | +1.23 | +0.143, +0.196, −0.056 |
| 300 | +0.0439 | 0.0106 | **+4.14** | +0.056, +0.023, +0.053 |
| 500 | +0.0186 | 0.0146 | +1.28 | +0.045, −0.006, +0.017 |

On the test-argmin metric, `not_last` − `all` is +0.002 / −0.002 / +0.002 —
indistinguishable at every size — while `none` − `all` stays positive
(+0.011, +0.019, +0.010) at every size.

## Reading

**Tom's proposal (`not_last`) is essentially free at K ≥ 300.** The paired
difference is +0.003 and +0.002 nats with standard errors five to ten times
larger, and individual seeds disagree in sign. On the selection-based metric it
is a wash at every size. So dropping the outer add in the final layer — leaving
the output as a pure `[W_1; W_2]` unembedding of the inner stream — costs
nothing measurable, and the architecture gets simpler and easier to describe.

**But it is reliably worse at K = 100**: +0.048 nats, all three seeds positive,
tight spread (t ≈ 7). With little data the direct `h` path into the output
apparently still carries something the rotation alone does not reconstruct.

**Unifying every layer (`none`) is worse.** All six paired comparisons have
positive means, and the effect is clearest where the noise is lowest (K = 300,
t ≈ 4.1). So the outer residual in the *intermediate* layers is doing real
work; this is not redundant machinery that can simply be deleted.

That combination makes sense structurally, though not for the reason I first
wrote. `[proj_out; W_param]` stacks a 75x80 and a 5x80 matrix into a **square
80x80 map, and it is full rank** — checked on trained checkpoints, rank 80/80
at layers 0, 3 and 7 in all three variants. So the problem is not that the map
cannot represent the stream.

It is **conditioning**. The smallest singular value of that map is 0.0013-0.016
depending on layer and variant, against a largest of ~1.1 — condition numbers in
the hundreds. With the outer add removed everywhere, the state reaches the
output only as a product of eight such maps, and directions aligned with the
small singular values are crushed geometrically. With the outer add present,
`params` accumulates *additively*, so the layer-0 encoding and every layer's
contribution reach the output unmultiplied. That is the ordinary residual-network
argument, and it applies to the outer stream exactly as it does to the inner one.

Removing the add in the last layer only costs nothing because no subsequent
layer needs that path — you are changing how the output head reads the final
state, not how signal propagates through the stack.

## Recommendation

1. **Adopt `not_last` if the goal is a cleaner story**, and say so explicitly:
   the output is $[W_1; W_2]\,c_L$, a genuine unembedding, with a single inner
   residual stream feeding it. Note the K = 100 regression, which is real but
   confined to the smallest training size.
2. **Do not adopt `none`.** The intermediate outer residuals earn their place.
3. If the aim is a *single* unified residual stream throughout, the existing
   `use_embed_unembed=True` path already implements exactly that — a standard
   pre-norm stack with per-type embedders and unembedders, no split/concat at
   all. That is a better-founded version of the `none` idea than deleting the
   outer add from the dual-stream model, and it has never been benchmarked
   against the dual-stream default on Domain 3. **That is the comparison I would
   run next**, and it is one command:
   `--variants` does not cover it, but `train.py --use-embed-unembed` does.

## Caveats

- 3 seeds per cell. Enough to establish direction on the `not_last` question,
  thin for the borderline `none` results at K = 100 and 500.
- Item axis, transductive only. The annotator axis has different entity-dropout
  settings (`annotator_dropout_rate = 0.8`) and might behave differently.
- Compared on the last checkpoint, which is the right call here but means the
  absolute numbers sit above the paper's (which are test-argmin-selected —
  `DISCREPANCIES.md` D14). The `all` column at K = 500 is 0.857 on the last
  checkpoint versus 0.830 test-selected in the replication run.

## Reproducing

```bash
cd RERUN
./submit_ablation.sh --run-id resid-01 --data-from replicate-01
# then, once training finishes
sbatch ... jobs/ablation_eval.sbatch          # RUN_ID=resid-01
python analysis/ablation_summary.py --run-dir runs/resid-01
```
