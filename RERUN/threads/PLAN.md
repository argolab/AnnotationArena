# Post-meeting threads — plan and running status

Three independent experiment suites, opened 2026-09-10 after the meeting. Each
is self-contained: its own folder, its own `RUN_ID` prefix, its own results
file, and its own end-of-suite artifact written for people who have not read
this repo (no codenames, no internal shorthand).

| folder | thread | one-line question |
| --- | --- | --- |
| `context-window/` | **1. Context window size** | How many items should one forward pass see, at training and at evaluation time? |
| `value-codec/` | **2. Typed value codec** | Can the hand-built parameter stream be replaced by a learned per-type encoder/decoder without losing accuracy? |
| `block-projection/` | **3. Block output projection** | Is the extra `d -> d` linear at the end of each block doing any work, or can it be folded away? |
| `obs-density/` | **4. Observation density** | Does Marformer still impute, and hold its position against Stan, when 15% or 85% of cells are observed rather than 50%? (opened 2026-10-17) |

Conventions inherited from the September sweep (`../FUTURE_EXPERIMENTS.md`
Part 0) and not renegotiated here:

- Metric is **corner log-loss with `corner_whole=true`**. Never the pooled metric.
- Arms are **paired** on (data draw, model seed, K) and compared at a **fixed
  epoch**, so nothing is selected on the test corner.
- Paired between-arm sd is **0.035**; tail-averaging the last 5 checkpoints cuts
  it by ~1.8x and is used everywhere. Decision threshold **0.03 nats**.
- K=100 is unusably noisy. K=300 is the quietest cheap size.
- `last.ckpt` is not a completion marker; `training_history.json` is.

Cost model (400 epochs, chunk 300): K=300 ~140 s, K=1000 ~446 s, K=2000 ~1003 s,
K=4000 ~2060 s. Larger chunks cost roughly linearly more per epoch.

---

## Status board

**Priority, set 2026-09-10 after review:** threads 2 and 3 first. Thread 1 is
cut back to the single cheapest question that can move the decision and runs in
the background behind them.

| id | task | kind | status | headline |
| --- | --- | --- | --- | --- |
| A1 | Reconcile the contradictory historical chunk evidence | free | **done** | neither direction is established; all three readings are n=1 and their spread IS the noise floor |
| A2 | ~~Train/eval chunk matrix at large K~~ | 36 GPU runs | **cut** | too many cycles to pin down one knob; replaced by A2-lite |
| A2-lite | 300 vs 500 at K=4000 only, 3 seeds x 2 draws | 12 runs | **done** | training chunk 500 is better by **0.016 nats**, consistent in 6/6 cells |
| A3 | Dynamic training chunk size (per-epoch draw) | 12 runs | **reinstated, queued** | 0.016 nats is ~16% of the Stan gap, so more is worth chasing |
| A4 | Memory and runtime ceiling per (chunk, K) | interactive | **done** | 600 fits at K=4000 (70.6 of 80 GiB); 900 does not; time cost of a wider chunk is ~5% |
| B0 | Audit the existing `use_embed_unembed` path | free | **done** | it exists but drops the block projection, decodes C+1, and has no tensorised path -- not reusable as-is |
| B1 | Implement the flagged codec (shared block stack) | code | **done** | `--value-codec typed` on branch `side/value-codec`; same speed as the dual stream (0.31 s/epoch at K=300) |
| B2 | Bit-exactness test: frozen last-6 codec == dual stream | free | **done** | **confirmed, with two corrections** -- forward diff exactly 0.0 |
| B3 | Codec vs dual stream at K=300 and K=2000 | 24 runs | **done** | typed codec is **+0.087 nats WORSE** -- a real regression, well outside noise |
| B4 | Debug ladder: aligned+frozen, aligned+free, random+frozen | 36 runs | **done** | **purely an initialisation problem** -- aligned matches the shipped model within noise |
| B7 | Orthogonal scaled init + 6 seeds to measure the failure rate | 42 runs | **done** | **conditioning is NOT the cause** -- orthogonal blows up at the same 1-in-12 rate |
| B6 | Scale-matched random init (std = logit_high/sqrt(d)) | 12 runs | **done** | **fixes it at K=2000 completely**; systematic gap gone, ~1-in-6 instability remains at K=300 |
| B5 | Does the layer-0 value scale (20 vs 1) cost anything? | 24 runs | **done** | **1.0 is not free**: ~+0.07 at K=300, slightly BETTER at K=2000, and one seed collapsed to uniform |
| B6 | Purge the old `use_embed_unembed` path | code | **done** | gone from model, config, trainer, evaluator and runner; equivalence re-verified after |
| C0 | Weight-folding proof of the projection claim | free | **done** | fold is exact (rel. 1.8e-7) with the inner residual off; impossible with it on (162x output change) |
| C1 | ~~Train with the block output projection removed~~ | 24 runs | **RETRACTED** | the "no projection" arm doubled the residual stream every layer; it measured a broken model |
| C2 | Corrected: projection kept vs removed at matched stream semantics, 4 sizes, dev-set early stopping | 48 runs | **done** | **the standard block wins**: drop the outer add and the linear together, -0.021 at K=2000 |
| C3 | Implementation audit of both arms | free | **done** | found the retraction: initial loss 2067 vs 5.7 |
| C4 | What the learned projection looks like | free | **done** | it keeps its initialisation statistics almost exactly |
| C5 | Freeze the projection at init -- learned transform or random mixer? | 16 runs | **done** | freezing recovers almost nothing (-0.008 dev-sel at K=2000); on the residual path at all, it is in the way |
| C2 | Parameter count and step time with it removed | interactive | not started | |

---

# Thread 1 — Context window size

Folder `context-window/`. Run ids `ctx-*`.

## What is actually known today

The model cannot put a whole large instance in one forward pass, so items are
split into chunks of at most `max_item`. Training has used 300; evaluation has
used 500. Four pieces of evidence exist and **two of them disagree**:

1. **Evaluation chunk sweep (S4, 2026-09-04, 684 re-scored rows, trustworthy).**
   Re-scoring fixed checkpoints at a larger evaluation chunk is free and helps
   at large K: 500-vs-300 is **-0.018** at K=2000 and K=4000, exactly 0.0000
   below K=500, and 1200 is worse than 600. The optimum is interior, near 500.
2. **Training chunk, from the `hp-01` / `hp-mi300` pair (2026-09-04).** Reported
   600-vs-300 as **-0.006 at K=1000 and +0.009 WORSE at K=4000**, and was used
   to retire "context starvation" as an explanation for the large-K plateau.
3. **Training chunk, from the 10k pilot (`chunk_disentangle.py`, 2026-09-10).**
   Same comparison, each arm scored at its own best evaluation chunk:

   | K | best of 300-trained | best of 600-trained | difference |
   | --- | --- | --- | --- |
   | 1000 | 0.7659 | 0.7589 | **-0.007** |
   | 2000 | 0.7459 | 0.7474 | +0.002 |
   | 3000 | 0.7368 | 0.7164 | **-0.021** |
   | 4000 | 0.7565 | 0.7118 | **-0.045** |

   So the 600-trained model is *better* at K>=3000, by well over the decision
   threshold, and it is not an evaluation artifact: it wins at every evaluation
   chunk tested, including 300.
4. **Chunk boundary spike (D24, confirmed).** Wherever an instance first fails
   to fit in one chunk the curve jumps ~+0.15 nats. Raising `max_item` moves the
   spike (600 items -> 900 items), it does not remove it.

Evidence 2 and 3 point in opposite directions at K=4000 (+0.009 vs -0.045).
**Nothing downstream should be decided until that is resolved**, because
evidence 2 is one of the two legs the "the plateau is not context starvation"
conclusion stands on. There is a concrete reason to distrust evidence 2: a
sibling directory named `_hp-mi300-collided` exists, and this project has lost
results five times to the bug where several seeds write to one directory and the
last to finish wins.

## A1 — Reconcile the historical evidence (no GPU)

Audit both trees before spending anything: for `hp-01`, `hp-mi300`,
`_hp-mi300-collided` and the two 10k-pilot arms, record `max_item`, epochs,
seed, data draw, number of completed runs, and whether seeds share a directory.
Re-score whatever survives on a common footing (same checkpoint policy,
tail-averaged, both evaluation chunks). Write `context-window/out/a1_audit.json`
and a short verdict in this file.

Output is one of: (a) evidence 2 is corrupt and is retracted; (b) both are sound
and differ by data draw, in which case the effect is draw-dependent and A2 needs
both draws; (c) they differ by checkpoint policy, which A2 then fixes by design.

### A1 result (2026-09-10) — the historical evidence is underpowered, not contradictory

`context-window/a1_audit.py` and `a1_rescore.py`. Three things came out.

**The tree behind evidence 2 is the one the project already flagged as
damaged.** `runs/legacy/hp-mi300` is empty; every 300-chunk checkpoint in that
comparison lives in `runs/legacy/_hp-mi300-collided`, which
`runs/legacy/README.md` describes as "a known seed collision, kept as evidence".
Both trees do at least share a data draw (`fig6-01`), so the comparison is not
meaningless, but it rests on a tree we have already disowned.

**Evidence 2 also averaged over ten architectures, nine of which we do not
ship.** The capacity screen's arms (`wide160`, `deep16`, `big128x12`, ...) were
pooled into one number. Re-scored on the shipped 80-wide/8-deep architecture
alone, at K=4000, each arm read at its own best evaluation chunk:

| | eval@300 | eval@500 | eval@600 | best |
| --- | --- | --- | --- | --- |
| trained at 300 | 0.8633 | 0.8490 | 0.8569 | 0.8490 |
| trained at 600 | 0.8909 | 0.8468 | **0.8465** | 0.8465 |

**-0.0025** — a tie, not the +0.009 penalty that was reported.

**So there are now three readings of the same comparison at K=4000, and they
span exactly one noise floor:** +0.009 (ten architectures pooled, damaged tree),
-0.0025 (shipped architecture, same trees), -0.045 (shipped architecture, 10k
pilot, a different data draw). Every one of them is a single seed on a single
draw, and the paired between-arm sd is 0.035. The spread of 0.054 is what n=1
looks like.

**Consequence.** The claim that widening the *training* chunk does not help at
large K is **not supported**, and neither is the claim that it does. Context
starvation goes back on the list of live explanations for the large-K plateau
until A2 reports with enough seeds to resolve 0.03. The sentence in
`FUTURE_EXPERIMENTS.md` "Two large-K findings that came free" that retires
context starvation should be marked provisional now, and rewritten when A2
lands.

One incidental finding worth carrying into A2: the 600-trained model is
**0.044 worse when evaluated at 300** than at its own 600. Train/eval chunk
matching matters at least as much as chunk size, which is why A2 scores every
arm at every evaluation chunk rather than at one.

## A2 — The train/eval chunk matrix

The real question the meeting asked: **is a bigger training chunk worth up to
0.045 nats at large K, and is 500/500 enough to get most of it?**

- Training arms: `max_item` in **{300, 500, 600}**.
- Evaluation: every arm re-scored at **{300, 500, 600, 700, 900}** (free).
- Sizes **K in {1000, 2000, 4000}** — the region where chunking binds and where
  the Stan gap lives. Below K=500 chunk size is provably inert.
- 2 data draws x 2 model seeds. Fixed 400 epochs, everything else shipped.
- 3 arms x 3 sizes x 2 draws x 2 seeds = **36 runs**, about 12 GPU-hours.
  Trimmed to 18 (one draw) if the queue is slow, with the second draw added only
  where the first shows an effect.

Decision rule, against the shipped 300-train arm at its own best evaluation
chunk:
- 600 wins by >= 0.03 and 500 recovers most of it -> ship **500/500**, the
  meeting's stated fallback, and re-open context starvation as a live
  explanation for the plateau.
- 600 wins by >= 0.03 and 500 does not -> ship 600 if A4 says it fits.
- Nothing clears 0.03 -> keep 300 train / 500 eval and close the thread.

This also re-measures the D24 boundary spike at each chunk size, since the
matrix contains everything needed.

### A2-lite result (2026-09-11) — 500 is genuinely better than 300, by a small and reliable margin

12 runs at K=4000, two draws x three seeds, each arm scored at evaluation chunks
300/500/600 and read at its own best. Last checkpoint, not tail-averaged: the
K=4000 curve is still descending at epoch 399.

| draw | seed | best of 300-trained | best of 500-trained | difference |
| --- | --- | --- | --- | --- |
| s101 | 42 | 0.7281 | 0.6987 | −0.029 |
| s101 | 43 | 0.7255 | 0.7130 | −0.013 |
| s101 | 44 | 0.7347 | 0.7182 | −0.017 |
| s105 | 42 | 0.6486 | 0.6384 | −0.010 |
| s105 | 43 | 0.6423 | 0.6308 | −0.011 |
| s105 | 44 | 0.6558 | 0.6417 | −0.014 |
| | **mean** | | | **−0.016 ±0.003** |

**A wider training window helps, in every cell, by about 0.013-0.016 nats.**
The effect is below the 0.03 threshold set for arm-vs-arm comparisons but that
threshold does not apply here: the paired sd is **0.0071**, five times tighter
than the 0.035 between-arm figure, because this varies one knob rather than
training a different architecture. Six of six cells agree in sign and the mean is
five standard errors from zero.

**This also settles the contradiction A1 opened.** Neither historical reading was
right: it is not the −0.045 the 10k pilot suggested, nor the +0.009 penalty from
the damaged tree. It is a small, consistent gain, and the earlier numbers were
n=1 readings scattered by a noise floor several times the effect.

**One more thing the scoring makes plain.** Each arm prefers a chunk at or above
the one it trained with: the 500-trained model is best at evaluation chunk 600 in
five of six cells, the 300-trained model at 300 in five of six. So training and
evaluation chunk are not independent knobs, and any future comparison has to read
each arm at its own best rather than at a shared chunk.

**DECIDED (user, 2026-09-11): train at `max_item=500`.** A4 measured the cost — 57.4 GB of
80 at K=4000, 24 GB of headroom, and 5% more wall clock. It buys 0.016 nats at
the size where the gap against Stan is widest, and it moves the chunk-boundary
artefact (D24) out of the range where the paper compares against Stan.

**A3 reinstated (2026-09-11, user decision).** I had dropped it on the grounds
that 0.016 nats is small in absolute terms. That was the wrong frame: the
residual gap against Stan is ~0.10 nats, so this single knob is **16% of the
thing we are trying to close**, and if a dynamic window buys more it is worth
having.

Implemented on `side/dynamic-chunk` as `--item-chunk-range LO,HI`: the chunk size
is drawn once per **epoch**, so every chunk inside a given forward pass has the
same width and nothing zigzags within a pass — the model still sees a range of
widths over training. It reuses the existing fresh-chunks-each-epoch path, so
there is no extra cost beyond rebuilding chunks, which that path already does.

Run at K=4000, where the window matters most, against fixed 300 and fixed 500.

## A3 — Randomised training chunk size

Separate question, separate arm set: **does varying the chunk size during
training teach the model to handle more context than it ever saw?**

New flag `--item-chunk-range LO,HI`: each epoch draws `max_item` uniformly from
[LO, HI] instead of holding it fixed. Arms:

| arm | training chunk |
| --- | --- |
| fixed-300 | 300 (shipped reference) |
| fixed-500 | 500 |
| dynamic-100-500 | uniform in [100, 500] per epoch |

Same sizes, draws and seeds as A2; evaluated at **{300, 500, 700, 900, 1200}**.
12 runs on top of A2. The interesting cell is evaluation at 700-1200, where S4
showed fixed-chunk training degrades: if dynamic training removes that
degradation, chunk size stops being a hyperparameter we have to defend and
becomes something the model is trained to be robust to.

## A4 — The memory ceiling (interactive, no queue)

Before committing to 500 or 600 in the final configuration, measure peak GPU
memory and seconds/epoch for `max_item` in {300, 500, 600, 900} at K in
{2000, 4000} on the reserved H100. Known reference points: `max_item=1200` at
K=4000 needed ~76 GiB and died on an 80 GiB A100, and the 128-wide 12-layer
architecture needed 92.2 of 93.1 GiB. Writes `context-window/out/a4_memory.json`.

### A4 result (2026-09-10) — 500 is comfortable, 600 fits, 900 does not

Peak GPU memory sampled from `nvidia-smi` during 3-epoch runs on one H100
(81,559 MiB usable), `expandable_segments:True`, seconds/epoch from the runner's
own timer:

| K | training chunk | peak GPU | s/epoch |
| --- | --- | --- | --- |
| 2000 | 300 | 18.4 GB | 1.58 |
| 2000 | 500 | 28.3 GB | 1.69 |
| 2000 | 600 | 36.4 GB | 1.71 |
| 2000 | 900 | 48.2 GB | 2.04 |
| 4000 | 300 | 36.2 GB | 3.14 |
| 4000 | 500 | **57.4 GB** | 3.20 |
| 4000 | 600 | **70.6 GB** | 3.30 |
| 4000 | 900 | 80.0 GB | **out of memory** |

Three things this settles for the final configuration:

1. **A wider training chunk is nearly free in time and expensive in memory.**
   Going 300 -> 600 at K=4000 costs **5%** more wall clock and **95%** more
   memory. So the constraint the paper describes -- chunking exists for memory,
   not for speed -- is exactly right, and the cost of relaxing it is paid in one
   currency only.
2. **`max_item=500` is safe at every size we run.** 57.4 of 81.6 GB at K=4000
   leaves 24 GB of headroom, which is comfortable against fragmentation and
   against a bigger data draw.
3. **`max_item=600` fits but has only ~11 GB of headroom**, and 900 is already
   over. If A2 says 600 is worth having, it should run on h200 rather than h100
   for margin; if 500 captures most of the gain, it is the safer ship.

Memory scales close to linearly in the chunk (K=4000: 36 -> 57 -> 71 GB for
300 -> 500 -> 600), so there is no cliff to be surprised by between those points.

---

# Thread 2 — Typed value codec

Folder `value-codec/`. Run ids `codec-*`.

## What the change is

Today the model carries two concatenated streams: a **feature stream** (type
centroid plus a per-entity deviation) and a **parameter stream** whose layout is
hand-built per type, and whose final value *is* the prediction. The proposal is
to delete the second stream and replace it with a learned codec per variable
type `tau`:

- **Encoder** `E_tau: R^(1+C) -> R^d`. For a Likert rating the input is the
  existing binary encoding: a leading wildcard/mask bit, then a one-hot over the
  C=5 rating values. The wildcard bit replaces the masking bit. Random Gaussian
  init.
- **Decoder** `D_tau: R^d -> R^C`, untied from the encoder, no wildcard output.
  Softmax over its C outputs is the predicted distribution; training loss is
  `-log p(observed class)`, with the existing 1 : 1/15 observed-vs-masked
  weighting unchanged.
- Entity types (annotator, item, attribute) have no value; they keep a type
  centroid only, which is exactly the encoder's bias.

Domain 3 has one variable type, so this is a refactor with no new expressive
power *yet*; the point is that it generalises to several value types, which the
hand-built parameter layout cannot.

### B0 result (2026-09-10) — the existing path is close, but is not the proposal

`use_embed_unembed` exists and the evaluation path already reconstructs it, but
it was not reused, for three reasons found by reading it:

1. **It drops the block's trailing projection**, so turning it on would change
   two things at once — the value codec and the block — and the second of those
   is what Thread 3 is measuring. Uninterpretable as a codec experiment.
2. **Its decoder emits C+1 outputs** and the loss then discards the wildcard
   column.
3. **It has no tensorised or batched implementation**: layer 0 is a Python loop
   over every token. The dual stream was tensorised for a 9-17x speedup, so the
   comparison would have been between a fast architecture and a slow one.

## B0 — Audit the path that already exists (no GPU)

**`use_embed_unembed` already exists in the code** (`config.py`, `model.py`
`_build_initial_emb_stream` / `_forward_emb`, a `--use-embed-unembed` flag on
the trainer, and the evaluation path already reconstructs it). It has, as far as
the run trees show, never been trained. It is close to the proposal but not the
same, in three ways that matter:

1. **It drops the block output projection.** The dual-stream block ends with
   `features += proj_out(c)` and `params += W_param(c)`; the existing embedding
   path ends with a plain `x = x + attn; x = x + ff`. That is a second
   architectural change riding along with the codec — and it is exactly the
   thing Thread 3 is about. Two changes at once is how an experiment becomes
   uninterpretable.
2. **The decoder emits C+1 outputs, not C**, and the loss then slices off the
   wildcard column. Harmless but not what was asked for.
3. **It has no tensorised or batched implementation.** Layer 0 is a Python loop
   over every token and the unembedding rebuilds type index lists on every call.
   The dual-stream path was tensorised in the optimisation work for a 9-17x
   speedup; the embedding path would run at the old speed, which would make any
   timing comparison meaningless and the sweep slow.

So B1 does not reuse it as-is.

## B1 — Implement it behind a flag

`--value-codec {dual,typed}` (default `dual`, so nothing existing moves), plus
`--codec-init {random,aligned}` and `--codec-freeze` for the debug ladder.

The design point that makes the whole thread testable: **the typed codec must
reuse the same block stack as the dual stream**, including the block output
projection. Only layer 0 and the output change. Then the debug configuration
below is an exact identity rather than an approximation, and Thread 3's question
stays separable from this one.

Tensorise layer 0 and the decoder the same way the dual stream was tensorised,
and extend the batched-augmentation path, or the sweep cost triples.

## B2 — The equivalence test (this is the "check me here")

The meeting proposed: initialise the encoder as a projection into the **last 6
dimensions** of `R^d`, decode by reading the **last 5**, freeze it, and expect
bit-exactness with the dual stream.

**That is correct, but only under the design in B1, and only with one
correction.** Working through it:

- The dual stream's parameter slice *is* the last `1+C = 6` dimensions of the
  residual vector, and the loss already reads its last C entries as logits. So
  "encode into the last 6, decode the last 5" lines up exactly, and the wildcard
  column is correctly ignored by the decoder.
- **The correction:** layer 0 of the dual stream does not write a one-hot. It
  writes `logit_high = 20.0` in the observed class's slot (and 1.0 in the mask
  slot). So the frozen encoder must be `diag(1, 20, 20, 20, 20, 20)` into the
  last 6 dimensions, not the identity. With a plain identity the test fails by a
  factor of 20 and would look like a bug in the codec.
- **The caveat:** it is bit-exact *only if the block keeps the output
  projection*. Against the embedding path as currently written in the repo it
  would not be, because that path has no `proj_out`/`W_param`, and no choice of
  frozen codec can repair a block-level difference. This is the single most
  important thing to get right before reading any B3 number.

Test, costing nothing: one real graph, assert max absolute difference under
1e-6 against the dual stream forward, then five optimizer steps with the codec
frozen and assert the parameter trajectories still match.

### B1/B2 result (2026-09-10) — the proposal is right, and needed two corrections

Implemented on branch `side/value-codec` (worktree `../Marformer-codec`) as
`--value-codec {dual,typed}` with `--codec-init {random,aligned}` and
`--codec-freeze`. The typed path reuses the dual stream's block stack verbatim,
so only layer 0 and the output differ; layer 0 and the decoder are tensorised
through `GraphTensors` and the batched-augmentation path works, which keeps the
codec at the dual stream's speed (0.31 s/epoch at K=300, against 0.35 measured
for the shipped arm).

One implementation note that matters for how the result reads: the decoder is a
genuine `R^d -> R^C` map with no wildcard output, exactly as proposed, but its C
logits are written into the existing carrier's slots 1..C. Nothing downstream --
loss, evaluation, plotting -- had to change, so the two architectures are scored
by identical code.

**The equivalence check, and what it took to make it pass.**

| check | result |
| --- | --- |
| single forward, aligned + frozen vs dual | **0.0** (exact) |
| batched-augmentation forward | **0.0** (exact) |
| shared parameters after 5 optimizer steps | 4.8e-06 (float32 accumulation) |
| same, with an identity encoder instead of the scaled one | 20.3 — **wrong** |
| same, without pinning the embedding tail | 2.8e-04 — drifts |

So the meeting's claim is confirmed: encode into the last `1+C` dimensions,
decode the last `C`, freeze, and the typed codec *is* the dual-stream
architecture. Two things had to be right first, and both are the kind of detail
that would have read as a bug in the codec:

1. **The encoder is not the identity.** Layer 0 of the dual stream writes `1.0`
   in the wildcard slot but `logit_high = 20.0` in the observed class's slot, so
   the aligned encoder is `diag(1, 20, ..., 20)`. With a plain identity the
   logits are off by 20.3 — wrong by a factor of twenty, not subtly wrong.
2. **The codec's slice has to be cleared at layer 0.** The typed architecture's
   type centroids and entity deviations are `model_dim` wide, so by default the
   *entity* types — which have no encoder of their own — write random values
   into the dimensions the codec owns. Zeroing those tails is what took the
   forward difference to exactly 0.0.

**And one genuine asymmetry the check turned up.** Freezing the codec makes the
forward pass identical but *not* training. The typed architecture is strictly
more permissive at layer 0: gradient can push its centroids and deviations into
the value slice, which the dual stream's narrower centroids cannot reach. Left
alone the two separate at ~1e-4 per step. The aligned+frozen configuration
therefore also pins that gradient, so it is a true control rather than an
approximate one. Worth remembering if the debug ladder is ever read as "frozen
means identical" — it means identical only once the tail is pinned.

**A note for the paper, if the codec ships.** Because the aligned codec is the
dual stream exactly, the two architectures are not rival designs; the dual
stream is the special case of the codec where the encoder is a fixed scaled
injection and the decoder a fixed selection. That is a cleaner thing to write
than a comparison.

### B5 — the value scale, and why it decides how this thread is written up

The equivalence check turned up that the dual stream writes `logit_high = 20.0`
into an observed value's slot, so the aligned encoder is
`diag(1, 20, ..., 20)` rather than the identity. That scale was never chosen
deliberately -- it is a constant that has been in the code since the beginning --
and it is the *only* thing standing between the typed codec and the simplest
possible statement of it.

So the question is not "how do we reproduce 20", it is **"does 20 buy anything
at all?"** Run: the shipped dual architecture at `logit_high` 20 vs 1, K=300 and
K=2000, 3 model seeds x 2 draws, 24 cells in 2 jobs (`codec-b5`).

- **If 1.0 costs nothing** (within 0.03): rewrite layer 0 to use 1.0, and the
  typed codec with a plain one-hot encoder is then bit-exact to the shipped
  model. One architecture, no scale factor to explain, and the encoder is
  literally the injection proposed at the meeting. This is the outcome to hope
  for and the one the code is already set up for.
- **If 20 is load-bearing**: keep it, and the aligned encoder keeps its scale.
  If that also turns out to be where a typed-codec regression comes from, the
  diagnostic is an encoder initialised as `diag(0, 20, ..., 20)` -- the value
  scale without the wildcard channel -- to separate "the scale matters" from
  "the wildcard channel matters".

**Already proven, free** (`b2_equivalence.py`): with the scale set to 1.0 the
aligned encoder *is* a plain 0/1 injection (asserted), and the forward
difference against the dual stream is **0.0**. So the "if" above is the only
open part; the algebra is done.

### B5 result (2026-09-11) — the value scale is NOT free, and the effect flips with K

24 runs, paired, tail-averaged. Per-cell, because the mean hides the story:

| draw | seed | K=300 (20 -> 1) | K=2000 (20 -> 1) |
| --- | --- | --- | --- |
| s101 | 42 | +0.078 | -0.022 |
| s101 | 43 | **+0.582** | **+0.145** |
| s101 | 44 | +0.079 | +0.046 |
| s105 | 42 | +0.066 | -0.018 |
| s105 | 43 | +0.024 | -0.023 |
| s105 | 44 | +0.058 | -0.013 |
| | **median** | **+0.072** | **-0.015** |

Three findings, none of which the pooled mean (+0.086 ±0.047) shows:

1. **At K=300 the scale is worth ~0.06-0.08 nats**, consistently, in every one
   of the six cells. Dropping it to 1.0 is a real loss in the small-data regime.
2. **At K=2000 it reverses**: four of six cells are *better* at 1.0, by 0.013 to
   0.023. The scale behaves like a prior that helps when data is scarce and
   mildly hurts when it is not.
3. **One cell collapsed.** s101/seed43 at K=300 reached **1.3864 = ln(4)**, the
   uniform distribution over the four classes — a dead run, and its K=2000
   sibling is the other outlier. One of six seeds diverging is itself the
   finding: layer-0 scale 1.0 is less stable, not merely slightly worse.

**So the hoped-for simplification does not survive contact with the data.** The
recommendation for thread 2 is to **keep the scale at 20** and have the aligned
encoder keep its `diag(1, 20, ..., 20)` form. It is one extra sentence to
explain, and it buys stability plus 0.07 nats where the data is thin.

**This also supplies the leading hypothesis for B3's regression.** A randomly
initialised encoder draws weights from N(0, 0.02), so against a one-hot input it
writes layer-0 values of order **0.02**, where the dual stream writes **20** — a
thousandfold scale mismatch at the point where the value enters the residual
stream. That is exactly the kind of thing that makes an architecture which is
provably equivalent at initialisation train to a worse place. B4's aligned rungs
test it directly, and if that is the story the fix is an encoder initialisation
scaled to the value it encodes rather than to a generic 0.02.

## B3 — Does it recover the shipped accuracy?

The meeting's framing: pick one small-data and one abundant-data setting where
we already know the answer, and see whether the codec gets back there.

- **K=300** (small data, and the quietest size we have) and **K=2000**
  (abundant). Item axis, transductive, fixed 400 epochs.
- Arms: `dual` (reference) and `typed` with random init.
- 2 draws x 3 model seeds x 2 sizes x 2 arms = **24 runs**, about 4 GPU-hours.

Pass: within 0.03 nats of the dual stream at both sizes. Then the codec is
adopted as the general mechanism and Domain 3 numbers are unchanged.

### B3 result (2026-09-11) — the codec does NOT recover the shipped accuracy

24 runs, 2 draws x 3 model seeds x 2 sizes, fixed 400 epochs, scored at
evaluation chunk 500, tail-averaged, paired against the `codec_dual` arm trained
from the same code.

| arm | K=300 | K=2000 | pooled |
| --- | --- | --- | --- |
| hand-built parameter stream (reference) | 0.0000 | 0.0000 | 0.0000 |
| **learned typed codec** | **+0.0733 ±0.0214** | **+0.1007 ±0.0217** | **+0.0870 ±0.0151** |

Reference absolute corner log-loss: 0.7803 at K=300, 0.7013 at K=2000.

**This is a real regression, not noise.** It is 2.9x the 0.03 decision threshold,
it is consistent in sign across both sizes and all six (draw, seed) pairs, and
the pooled effect is nearly six standard errors from zero. For scale: it is
comparable to the *entire* residual-variant spread from S1 (up to 0.143), and
most of the way to the unexplained Stan gap (0.104) — so it cannot ship as-is.

**And it is an optimisation problem, not an expressiveness one.** That is not a
guess: B2 proves the aligned, frozen codec computes *exactly* the same function
as the dual stream, bit for bit, forward and batched. The typed architecture
therefore contains the shipped one as a special case; a randomly initialised
codec lands somewhere training does not recover from. Which is what B4 is for.

The gap grows with K (+0.073 -> +0.101), which argues against "it just needs more
epochs at small data" and towards the encoder never learning a good code.

## B4 — Debug ladder

Now unconditional, and running. Three rungs in the `codec-b3` tree so they are
paired against that suite's own reference on the same draws and seeds:

| rung | arm | what it separates |
| --- | --- | --- |
| 1 | `codec_align_frozen` | **must** land on the reference exactly. If it does not, the bug is in training, not in the codec's initialisation, and everything below is moot. |
| 2 | `codec_align_free` | same starting point, free to move. If this holds but random init does not, the problem is purely *where the encoder starts*. |
| 3 | `codec_rand_frozen` | a random code that never adapts. Separates "the encoder cannot learn" from "the encoder starts somewhere bad and drags the rest with it". |

Rung 1 is also a live regression test of the whole training path: B2 checked five
optimizer steps, this checks four hundred.

### B4 result (2026-09-11) — the codec is sound; only its starting point is wrong

All 36 runs in, paired against the same `codec_dual` reference on the same draws
and seeds.

| rung | K=300 | K=2000 | pooled |
| --- | --- | --- | --- |
| hand-built parameter stream (reference) | 0.0000 | 0.0000 | 0.0000 |
| **1. aligned encoder, frozen** | +0.0181 ±0.0119 | +0.0004 ±0.0046 | **+0.0093 ±0.0067** |
| **2. aligned encoder, free to learn** | +0.0213 ±0.0113 | −0.0015 ±0.0041 | **+0.0099 ±0.0067** |
| 3. random encoder, free to learn (B3) | +0.0733 ±0.0214 | +0.1007 ±0.0217 | +0.0870 ±0.0151 |
| **4. random encoder, frozen** | +0.1955 ±0.0912 | +0.3529 ±0.1221 | **+0.2742 ±0.0764** |

**Rung 1 lands on the reference.** +0.009 pooled, inside the noise floor and a
third of the decision threshold — so the training path is correct end to end, not
just for the five steps B2 checked. Whatever is wrong with the codec, it is not
the architecture and not the plumbing.

**Rung 2 lands there too.** A codec that *starts* in the right place and is then
free to move stays there (+0.010). So the encoder being learnable is not the
problem either.

**Rung 4 is the tell.** A random code that can never adapt is **+0.274** — three
times worse than a random code that can (+0.087). Learning recovers two-thirds of
the damage and cannot reach the rest.

**Verdict: this is purely an initialisation problem**, exactly as hoped. Ordered
by where the encoder starts rather than by what it is allowed to do, the four
rungs are monotone, and the only arms that fail are the two that start random.

**And the mechanism is magnitude.** The hand-built stream writes `logit_high`
= 20.0 into an observed value's slot, so the value enters the residual stream
with L2 norm 20. A `N(0, 0.02)` encoder delivers about **0.02** — three orders of
magnitude quieter. The value is effectively inaudible at layer 0, and the model
spends its budget learning to amplify it instead of using it. B5 independently
showed this scale is load-bearing: dropping it to 1.0 costs 0.07 nats at K=300
and collapsed one run outright.

### B6 — the fix, now running

Keep the code random, fix its volume: `--codec-init scaled` draws the encoder
from `N(0, logit_high / sqrt(d))`, which gives an output norm of about
`logit_high` — a random code that arrives at the magnitude the stream expects.
12 runs, same cells, paired against the same reference.

If it lands within 0.03 of the reference, the typed codec ships and the value
scale simply becomes part of the encoder's initialisation rather than a constant
baked into a hand-built layout. If it does not, the remaining suspect is the
*decoder* side, and the next rung is a scaled decoder with a random encoder.

### B6 result (2026-09-11) — the scale fix works, and exposes what is left

`--codec-init scaled` draws the encoder from `N(0, logit_high / sqrt(d))`, so the
code is still random but arrives with L2 norm ~20 instead of ~0.02.

| arm | K=300 | K=2000 | pooled |
| --- | --- | --- | --- |
| random encoder, standard init (B3) | +0.0733 ±0.0214 | +0.1007 ±0.0217 | +0.0870 ±0.0151 |
| **random encoder, scale-matched** | +0.0732 ±0.0638 | **+0.0010 ±0.0054** | +0.0371 ±0.0324 |
| aligned encoder (the ceiling) | +0.0181 ±0.0119 | +0.0004 ±0.0046 | +0.0093 ±0.0067 |

**At K=2000 the fix is complete.** +0.001, all six cells inside ±0.017, against
+0.101 for the same architecture with the standard initialisation. One line of
initialisation closes the entire gap, which is about as direct a confirmation of
the magnitude diagnosis as the experiment could give.

**At K=300 the mean is unchanged but the variance tripled**, and the per-cell
table says why:

| draw | seed | shipped | scaled | diff |
| --- | --- | --- | --- | --- |
| s101 | 42 | 0.8140 | 0.8248 | +0.011 |
| s101 | 43 | 0.7926 | 0.8082 | +0.016 |
| s101 | 44 | 0.8220 | 0.8765 | +0.055 |
| s105 | 42 | 0.7379 | 0.7775 | +0.040 |
| s105 | 43 | 0.7855 | 0.7423 | **−0.043** |
| s105 | 44 | 0.7298 | **1.0555** | **+0.326** |

Five of six cells sit between −0.043 and +0.055, median **+0.028**; the sixth
blew up. Drop it and the K=300 mean is +0.016 — the same place as the aligned
ceiling.

**So the systematic bias is fixed and what remains is instability.** One run in
six lands badly at small K, which is the same signature as B5, where scale 1.0
collapsed one run to the uniform distribution. Both point at the same soft spot:
at K=300 the value encoder's initialisation occasionally puts training somewhere
it does not leave.

**Where this leaves thread 2.** The typed codec is viable and the mechanism is
understood end to end: the architecture is provably the shipped one as a special
case (B2), the training path is correct (B4 rung 1), and the regression was
entirely a layer-0 magnitude mismatch (B6 at K=2000). It is not yet shippable at
small K because of the 1-in-6 failures. The obvious next step is to keep the
magnitude but fix the conditioning — an orthogonal encoder scaled to
`logit_high`, which is random and loud but cannot start with a near-degenerate
code — plus a seed-robustness check at K=300. That is a cheap, well-posed
follow-up rather than an open question.

### B7 — measuring and, hopefully, removing the instability (running)

Two candidate causes for the 1-in-6 failures at K=300, and B6 cannot separate
them: a Gaussian draw can start with two rating levels pointing in nearly the
same direction, which is a bad code to be stuck with when there is little data to
correct it; or small data simply cannot correct *any* unlucky start.

`--codec-init orthogonal` addresses the first: orthonormal columns scaled to
`logit_high`, so every value gets its own direction at the magnitude the stream
expects. Verified at construction — an encoded value has norm exactly 20.0 and
the maximum absolute cosine between two value directions is 0.0. It is still not
a hand-written layout: which direction each value receives remains arbitrary.

Run: three arms (hand-built reference, scaled, orthogonal) x **six** model seeds
x two draws at K=300, so the failure rate is measured rather than inferred from a
single bad run, plus the orthogonal arm at K=2000 to confirm it keeps the win the
scaled init already delivered.

### B7 result (2026-09-11) — conditioning is not the cause

Full n, both draws, six model seeds:

| arm | K=300 median | K=300 worst | blowups | K=2000 median | blowups |
| --- | --- | --- | --- | --- | --- |
| random, standard scale | +0.063 | +0.139 | 0/6 | +0.118 | 0/6 |
| random, volume-matched | +0.009 | +0.326 | **1/12** | +0.001 | 0/6 |
| random, orthogonal | +0.007 | +0.360 | **1/12** | −0.005 | 0/6 |

**Orthogonal does not fix the instability.** It matches the volume-matched arm
almost exactly — same median, same blowup rate, a slightly worse worst case. So
the failures are not a badly conditioned draw; they are small-data fragility.

**A correction to what I said an hour ago.** On the first six cells the
orthogonal arm showed 0 failures and I reported that it looked like the fix. The
remaining six contained one blowup. Six paired cells cannot measure a 1-in-12
rate — that was reading a rate off a sample too small to hold one — and the
partial number should not have been passed on, even labelled as partial.

**What both arms do settle:** the volume fix is real and complete at K=2000
(+0.001 and −0.005 against +0.118 for the standard init), and at K=300 both
scaled arms have a median of ~+0.008 against +0.063. The systematic penalty is
gone. What is left is one run in twelve, at the smallest size, landing somewhere
training never leaves.

**Which makes the temperature head the live candidate** (B8, queued): the failed
runs sit at the uniform distribution, and a learned temperature is a direct,
one-scalar escape from it, where a badly scaled encoder has to be undone by every
decoder weight at once.

## B4 — original conditional plan

Runs in order, stopping at the first rung that explains the gap:

1. **Aligned init, frozen.** Must reproduce the dual stream exactly (B2 already
   proves the forward pass; this proves it end to end through training).
2. **Aligned init, unfrozen.** Same starting point, free to move. If this
   matches the dual stream but random init does not, the problem is
   initialisation, not the architecture.
3. **Random init, frozen.** Separates "the encoder cannot learn" from "the
   encoder starts somewhere bad".

If rung 2 recovers and random init does not, that is an optimisation problem and
the thread pauses for the further ideas offered at the meeting.

### B6 — the old embedding path is gone (2026-09-10)

`use_embed_unembed` and everything it reached -- the second `__init__` branch,
`value_embedders`, `type_embeddings_emb`, `deviation_tables_emb`,
`emb_deviation_norm`, `unembedders`, `emb_blocks`, `_build_initial_emb_stream`,
`_forward_emb`, the `_init_bias_as_type_centroid` helper that only it used, the
trainer flag, the saved-config key, the evaluator's reconstruction line and the
runner's flag -- has been deleted on `side/value-codec`. Two architectures that
both claim to be "the embedding one" is exactly the confusion this thread is
meant to remove. `grep -r embed_unembed` returns nothing, the model still
imports, and `b2_equivalence.py` still reports an exact 0.0 afterwards.

Checkpoints written before the purge are unaffected: none of them set the flag,
and the evaluator's reconstruction defaulted it to False.

---

# Thread 3 — Block output projection

Folder `block-projection/`. Run ids `proj-*`.

## The claim, checked

Each block currently ends with a `d -> d` linear applied to the block's output,
written as two halves (`proj_out` to the feature slice, `W_param` to the
parameter slice) which stack into a single `d x d` matrix `M`. The meeting's
question: since the feed-forward network already ends in a `d_ff -> d`
projection with no activation after it, `M` should compose with it into one
projection and be removable for free.

**The composition is real but it does not apply to the shipped architecture.**
The block is

    c = h + attention(norm1(h))
    c = c + feedforward(norm2(c))
    h' = h + M c

`M` is applied to `c`, not to the feed-forward output. `c` also contains the
identity path and the attention output, so `M W_out` is not a refactor of `M` —
dropping `M` changes what the block can compute. Concretely, `h' = h + M c` and
`h' = h + attention + feedforward` are different functions and no re-weighting
turns one into the other.

The composition argument *is* exactly right in one configuration already in the
code: with the inner residual turned off (`inner_residual=none`), `c` is the
feed-forward output alone, so `M W_out` collapses to a single `d_ff -> d`
matrix and `M` is genuinely redundant. Confirmed by reading: the feed-forward is
`Linear -> ReLU -> Dropout -> Linear`, with **no activation after the last
linear**, so nothing blocks the fold.

So the thread splits into a free proof and one cheap training run.

## C0 — Weight folding (no GPU)

Build a model with the inner residual off, fold `M` into the feed-forward output
projection (`W' = M W_out`, `b' = M b_out + b_M`), delete `M`, and assert the two
forward passes agree to floating-point tolerance on a real graph. This turns
"should be able to combine into one projection" into a demonstrated identity for
the case where it holds, and demonstrates by counterexample that it does not
hold for the shipped configuration.

### C0 result (2026-09-10) — the fold is exact where the meeting said, and nowhere else

`block-projection/c0_fold.py`, on a real K=300 graph, folding `M` into the
feed-forward's output linear (`W' = M W_out`, `b' = M b_out + b_M`):

| block form | max abs difference | relative |
| --- | --- | --- |
| inner residual **off** | 3.8e-06 | **1.8e-07** (float32 exact) |
| inner residual **on** (shipped) | 5.1e+03 | 1.6e+02 |

So the composition argument is right, and it is decisive — in the configuration
where the block's output *is* the feed-forward output. In the shipped block the
projection is applied to identity + attention + feed-forward, so there is
nothing adjacent to fold it into, and removing it is a change of function class
rather than a re-parameterisation. That is what C1 measures.

## C1 — Remove it and retrain

Flag `--block-output-projection {full,none}`; `none` gives `h' = h + attn + ff`,
the textbook block. This is the fifth member of the residual-variant family that
was never run — the existing four vary *where the projection's output is added*
(`all`, `not_last`, `none`, `none_id`), never whether the projection exists.

- K=1000 as the meeting suggested, plus K=300 (quietest, nearly free).
- 2 draws x 3 seeds x 2 sizes x 2 arms = **12 runs**, under 2 GPU-hours.

Prior from the September residual sweep: this family of changes moves the number
by 0.05-0.14 nats, all in the wrong direction, so "no change" would be a genuine
surprise. If removal is free it deletes `num_layers x d x d` parameters (8 x 80
x 80 = 51,200, about a fifth of the model) and simplifies the architecture
section to a standard transformer block.

### C1 pre-flight (2026-09-10)

Flag `--block-output-projection {full,none}` added on branch
`side/block-projection` (worktree `../Marformer-blockproj`), plumbed through the
trainer, the saved config, the evaluation path and the job scripts. Verified
before submitting: a checkpoint written before the flag reconstructs as `full`;
`none` is not inert (output changes); and `none` genuinely deletes the modules
rather than pinning them to the identity — **51,840 of 457,546 parameters,
11.3% of the model**. Submitted as `proj-c1`, 24 runs.

**A trap worth writing down, because it has now cost this project four jobs.**
`RERUN/threads` is a symlink shared by every worktree, so `__file__` inside a
thread script resolves into the *main* checkout, and deriving the import root
from it silently imports the unpatched model — the flag then reads as inert.
Thread scripts take their import root from `RANKING_ROOT` (exported by
`config/cluster.sh` from the calling worktree) and assert that the module they
imported actually came from there.

### C1 RETRACTED (2026-09-11) — the ablation was broken, not the architecture

**The +0.348 number is withdrawn.** The arm labelled "projection removed" did not
compute `h' = h + attn + ff`. Removing the projection was implemented as
substituting the identity for `M` while keeping the outer *add*, which gives

    h' = h + c = h + (h + attn + ff) = 2h + attn + ff

— the residual stream doubles at every layer, 2^8 = 256x over the stack. An
implementation audit caught it in one line: at initialisation the arm sits at a
cross-entropy of **2067** against **5.7** for the shipped architecture. It was
not an ablation of the projection; it was a diverging model, and the experiment
measured how badly it diverged.

**How it got past review.** The flag was checked for exactly the things that
usually go wrong — the default is unchanged, the flag is not inert, the modules
are really gone — and it passed all three. None of them asks whether the
resulting model is *sane*. A one-line check of the loss at initialisation would
have caught it before 24 runs, and is now part of the audit
(`c3_implementation_check.py`).

**A second lesson, independent of the bug.** The comparison was made at a fixed
epoch. Reading the same runs at each arm's own best epoch nearly halves the gap
(+0.173 at K=300, +0.125 at K=1000), because the broken arm degrades sharply
after its optimum while the shipped one is flat. Even a correct ablation compared
at a fixed epoch would have overstated the effect. **From now on every arm
comparison uses early stopping on a held-out dev split**, and C2 does.

### C2 — the corrected experiment (running)

The projection and the outer residual add are two different things, and the first
attempt confounded them. Three arms separate them, all with `block` validation
carve and dev-set early stopping, at K = 300 / 1000 / 2000 / 4000, two draws x
two seeds:

| arm | block | what it isolates |
| --- | --- | --- |
| `proj_keep_add` | `h' = h + M c` | the shipped architecture |
| `proj_keep_rot` | `h' = M c` | projection kept, outer add removed |
| `proj_drop_rot` | `h' = c` | **the textbook block** |

`proj_drop_rot` vs `proj_keep_rot` differ in exactly one thing — whether the
projection exists — at matched residual semantics. That is the clean test.
`proj_keep_rot` **is** S1's `none` arm (+0.0525 there), so `proj_keep_add` vs
`proj_keep_rot` reproduces a known number and checks the whole setup.

Trimmed to **K = 300 and 2000 only** (user, 2026-09-11) — the effect does not
need a size sweep to be read — and the freed budget went into a third model seed,
so each cell is n = 6 rather than n = 4.

**Prediction registered before the runs land (user's hypothesis).** The reason
`proj_keep_rot` regressed in S1 is that the projection *gates the residual path*:
with the outer add removed, every layer's output — including the identity part —
passes through `M`, so the stream is multiplied eight times over the stack and
the skip connection it is supposed to provide is no longer a skip at all. If that
is right, then removing the outer add and the projection **together**
(`proj_drop_rot`) should not regress, because it restores a clean additive
residual: `h' = h + attn + ff`. It may even beat the shipped block.

That is a falsifiable ordering, and it is the interesting one:

    proj_drop_rot  <=  proj_keep_add  <  proj_keep_rot

If instead `proj_drop_rot` lands with `proj_keep_rot`, the projection is doing
real work and the gating story is wrong. Either way the answer is informative,
which is what the first version of this experiment failed to be.

### C4 result (2026-09-11) — the projection barely moves from its initialisation

Trained checkpoints, all four training sizes, every layer:

| quantity | value | what the initialisation gives |
| --- | --- | --- |
| Frobenius norm | 5.04 – 5.17 | 5.16 |
| RMS weight | 0.063 | **0.0645** (PyTorch default) |
| effective rank | 57.1 – 57.8 of 80 | ~57 (dense random) |
| identity component | −0.024 to +0.007 | 0 |
| F&larr;F / F&larr;V / V&larr;F / V&larr;V RMS | all ≈0.063 | all equal |

Individual weights do move — up to 0.23, twice the initialisation range — and the
checkpoint load was verified (167 of 170 tensors change on load), so this is not
an untrained model. But the *statistics* are the initialisation's, and they are
the same at K=300 and K=4000. There is no structure: no identity component, no
low-rank structure, and no specialisation between the block that writes the value
slice from the features (`V<-F`, where a prediction would have to be formed) and
any other.

**Which reframes the question.** If the projection is load-bearing but not
learned, it is doing its work as a fixed high-rank random mixing — rotating the
stream so each layer's update is spread across all dimensions rather than
landing back where it started. C5 tests it directly: freeze `M` at its random
initialisation and train everything else. If that costs nothing, "the projection
matters but training it does not" is a far more interesting claim about the
architecture than the one we started with.

### The original C1 text, kept for the record

24 runs, 2 draws x 3 model seeds x 2 sizes, fixed 400 epochs, scored at chunk
500, tail-averaged, paired against the shipped architecture.

| arm | K=300 | K=1000 | pooled |
| --- | --- | --- | --- |
| projection kept (shipped) | 0.0000 | 0.0000 | 0.0000 |
| **projection removed** | **+0.3967 ±0.0435** | **+0.3002 ±0.0412** | **+0.3485 ±0.0320** |

Reference absolute: 0.7839 at K=300, 0.7131 at K=1000.

**Removing it costs 0.35 nats — eleven standard errors, and more than twice the
worst residual variant in the whole September sweep.** It is the largest single
architectural effect measured in this project. So the trailing `d -> d`
projection is not a redundant layer that happens to sit after the feed-forward;
it is doing a large share of the block's work.

**Thread 3's question is answered completely, and the answer has two halves that
have to be stated together:**

1. **The composition argument is correct.** C0 demonstrated the fold exactly
   (relative error 1.8e-07) in the configuration where the block's output *is*
   the feed-forward output.
2. **It does not apply to the shipped block**, where the projection is applied to
   identity + attention + feed-forward. There is nothing adjacent to fold it
   into, and C1 measures what removing it actually costs: 0.35 nats.

So this is not "we kept a redundant layer"; it is "the projection is applied
somewhere that makes it irreducible". Worth one sentence in the architecture
section, because a reader who notices the two adjacent linears will ask exactly
this question.

## C2 — What it costs (interactive)

Parameter count and seconds/epoch for both arms, so "free" can be stated in
parameters and wall clock as well as in nats.

---

# The task pool

Built 2026-09-10, and everything now runs through it. `RERUN/pool/`:

| file | what it does |
| --- | --- |
| `enqueue.sh` | turns an arms file (or a one-off command) into task files |
| `worker.sh` | claims a task, runs it, repeats until out of time |
| `submit_workers.sh` | submits worker jobs across several partitions |
| `run_local_workers.sh` | runs workers inside a held interactive allocation |
| `status.sh` | what is done, running, waiting, failed, and hours remaining |

**Why.** The account holds about ten Slurm jobs at once, and a sweep is a
hundred small trainings. One job per training spends the sweep queueing. Here the
work sits in a directory and a handful of long-lived workers pull from it, so the
number of jobs is decoupled from the amount of work.

**What it buys, concretely.**

- **Any partition.** Workers are submitted to `-p h100,l40s,a100`, so whichever
  GPU class frees up first does the work. This is safe despite the single-partition
  warning in `config/cluster.sh`: the held-job failures that warning describes were
  traced to `--export=ALL,VAR=value`, not to partition lists.
- **Estimates drive packing.** Each task carries a measured estimate (from the
  cost model and `context-window/out/a4_memory.json`) and a GPU-memory figure. A
  worker takes the longest task that still fits in its remaining walltime, and
  refuses tasks bigger than its memory budget -- so the 60 GB K=4000 cells go to a
  worker submitted with a 70 GB budget and never strand a 22 GB one.
- **Early exit.** When nothing left fits in the time remaining, the worker stops
  rather than holding a GPU it cannot use, and says why.
- **Idempotent.** Every task records the file its run produces. Re-enqueueing a
  partly-finished sweep tops it up instead of repeating it -- visible in the log as
  "already has its output; marking done without rerunning".
- **Crash-safe.** Claims are `mkdir` (atomic here, no daemon) with a heartbeat; a
  claim whose owner has gone quiet for 30 minutes is reclaimed.
- **Priorities.** Lower number first, so threads 2 and 3 drain ahead of thread 1.

Six traps, all found the hard way and all now commented in the scripts:

- **A failing task must not kill its worker.** `lib.sh` set `-e`, and worker.sh
  sources it, so the first task that exited non-zero killed the worker *before*
  it could record the failure or release its claim. Three workers died silently
  mid-task, their jobs reported COMPLETED 0:0, and the pool cheerfully said
  "0 failed" while holding stale locks. `lib.sh` no longer sets `-e`, the task
  call is wrapped in `set +e`, and a failure now records the last lines of its
  log alongside the exit code.
- **Do not write a script and immediately execute it on a shared filesystem.**
  The worker wrote each task's command to `cmd.sh` inside its lock directory and
  ran it. One task died with `rc=127` and "No such file or directory" naming a
  script that was on disk, correct and complete, by the time anyone looked --
  a write-then-read coherence window on weka. The command is now executed from a
  shell variable and the file kept only for debugging. `pool/requeue_failed.sh`
  clears diagnosed failures so a worker picks them up again.
- **"Nothing fits" has two causes, and only one of them is terminal.** A worker
  that found every remaining task over its memory budget treated that like
  running out of time and exited. Time is terminal -- it only decreases, so a
  task that does not fit now never will. Memory is not: the task may simply
  belong to a bigger worker, and more work arrives continually. Conflating them
  made the workers on the **reserved GPU quit** because the only tasks left were
  the 60 GB ones, and that node then sat idle through the next batch. A
  memory-blocked worker now waits and re-polls; only a time-blocked one exits.
  The diagnostic message said "shortest remaining task needs ~1376s x 4.07" with
  146,681 seconds left, which should have been the giveaway -- it was reporting
  the time test while the memory test was what actually failed.
- **Snapshot the worker code at launch.** Bash reads a script by byte offset as
  it runs, so editing `worker.sh` while workers are executing it can drop them
  into the middle of a different statement. I did exactly that adding the
  calibration -- the running workers survived, but only by luck. Both launchers
  now copy `worker.sh`, `lib.sh` and `calibrate.py` into a per-launch
  `queue/rev_<timestamp>/` and run the copy, so a later edit cannot reach
  anything already in flight.
- **Estimates must be calibrated, not trusted.** The cost model was measured on
  an idle GPU; workers share a node deliberately, and the first completed task
  took **13 minutes against a 4-minute estimate**. A worker trusting the raw
  number claims work the walltime then kills mid-training. `calibrate.py` takes
  the median of actual/estimated over completed tasks (with headroom, and a
  three-sample floor) and every worker scales its fit test by it.
- **Size the GPU budget from the device, not from the submit command.** Workers
  are submitted across partitions on purpose, so the same script lands on an
  80 GB H100, a 141 GB H200 or a **44 GB L40S**. A fixed 22 GB per worker -- fine
  on an H100 -- put three 18 GB trainings on an L40S and OOMed the third. Each
  worker now reads `nvidia-smi` at startup and takes 85% of the real device
  divided by the workers sharing it, and says which GPU it got. Worker density
  is then a coverage decision: three per job gives 12 GB on an L40S, which is too
  small for a K=2000 training, so the general pool runs two per job (18.8 GB on
  L40S, 34 GB on H100) and a separate one-worker job handles the 60 GB K=4000
  cells.
- **`--runner-root` alone does not select a worktree.** `config/cluster.sh`
  exports `DOMAIN3_REPO_ROOT_OVERRIDE` and `PYTHONPATH` pointing at its own tree,
  and `run_marformer_group.sh` resolves the trainer from that override -- so
  tasks enqueued from the main pool ran a side branch's runner against the
  *main* checkout's `train.py` and died with "unrecognized arguments:
  --value-codec". Each task now pins both variables after sourcing cluster.sh.
  This is the third time in this project that a script has silently imported the
  wrong tree; the other two are in the S1 result and the C1 pre-flight.

- **Do not call a shell variable `GROUPS`.** Bash keeps it as a built-in array of
  the caller's group ids and ignores assignment, so `${GROUPS}` expanded to a gid
  and queued **1,470 empty jobs**. None ran; all were cancelled.
- **One `srun` step for the whole fan-out, not one per training.** A per-command
  `srun -n1` gets `cpus-per-task=1`; this workload is CPU-bound, so each training
  ran at **18 s/epoch against 0.35** with nothing in any log to say so.
- **Cap the thread count.** Torch opens one thread per core by default, so six
  unconstrained trainings on a 30-core node ran at **8.6 s/epoch** -- they fight
  rather than work. Workers set `OMP_NUM_THREADS = cpus / workers`.
- **`bc` is not installed on the compute nodes**, and a missing `bc` made every
  "does it fit" test silently succeed, so a worker with ten minutes left would
  claim an hour-long task and have it killed at the walltime. The arithmetic is
  `awk` now.

# How work is submitted, after two lessons

**Few jobs, many trainings.** The account runs about ten jobs at once, so the
one-array-per-cell shape of `side/submit_arms.sh` leaves most of a sweep PENDING
while nodes idle. `threads/submit_condensed.sh` enumerates the cells, splits them
round-robin into a handful of groups, and gives each group one job that works
through its share several at a time. Same results tree, same variant paths, same
env -- evaluation and analysis cannot tell which submitter produced a run. Size
the workers against `context-window/out/a4_memory.json`: they share one GPU.

**The reserved node runs whatever the queue is blocking.**
`threads/run_local.sh` takes the same arms file and runs the cells inside the
held allocation.

Two traps, both of which cost real time here and are now commented in the
scripts:

- **Do not call a shell variable `GROUPS`.** Bash keeps `GROUPS` as a built-in
  array of the caller's group ids and silently ignores assignment to it, so
  `${GROUPS}` expanded to a gid and the first run of the condensed submitter
  queued **1,470 empty jobs** before it was killed. None of them ran; all were
  cancelled.
- **One `srun` step for the whole fan-out, not one per training.** A per-command
  `srun -n1` gets `cpus-per-task=1`, and this workload is CPU-bound, so each
  training ran at **18 s/epoch instead of 0.35** -- fifty times slower, with
  nothing in any log to say so. The step also has to be launched
  `--cpu-bind=none`: the allocation's CPU mask is not contiguous, so asking for
  `-c N` fails with "CPU binding outside of job step allocation".

# Execution order and use of the reserved node

A single H100 is held interactively until **2026-09-12 21:28** (job 373263,
node gh115, 30 CPUs, 360 GB). Queue time on `h100` is currently short, so the
default remains `sbatch`; the reservation is for work that must not wait.

1. **Now, on the reserved node, no queue:** A1, B0, C0 — all three are free and
   two of them can invalidate the experiments that follow. Then A4 and C2.
2. **Queue immediately:** A2 and C1. They need no new model code (A2 is an
   existing knob; C1 is a one-line flag), so they can be in flight while the
   codec is written.
3. **Write and test B1/B2 on the reserved node** while A2 and C1 run, then queue
   B3.
4. **A3** after A2 reports, since its arms are chosen from what A2 finds.
5. If anything sits queued behind a long wait while the reserved node is idle,
   cancel the queued copy and run it on the reservation instead.

Each thread's suite ends with a standalone interactive artifact written for the
team, covering the question, the evidence, and the recommendation.

---

## Thread E — is the early-stopping protocol sound? (opened 2026-09-11, from the Slack thread)

Jason: *"the early stopping point must be chosen on a separate validation set, not
the test set. My read of the graph is charitable: I'm assuming that the validation
data will be able to identify a stopping point that's close to the minimum of these
test graphs."* This measures that assumption instead of assuming it.

**Construction.** Paired runs, same generative draw and same model seed:
`val-block-01/valblock-search/seed42` (trained on 85 %, validation block scored every
epoch) against `replicate-01` (trained on 100 %, run the full 400 epochs, test-scored
every 25). `e*` comes from the shipped selector (`analysis/select_epochs.py`,
smooth 5, argmin, 25-epoch floor) — imported, not reimplemented. Regret is
`test_full(e*) − min_e test_full(e)`.

**Result (`threads/early-stopping/out/e1_seed42.json`, 12 sizes, K = 200…4000).**

| | nats |
|---|---|
| mean stopping regret | **+0.0162** |
| worst size (K = 500) | +0.0291 |
| best size (K = 3000) | +0.0028 |
| mean cost of *not* stopping (run to 400) | +0.0191 |
| worst cost of not stopping (K = 800) | +0.0472 |

For scale: paired between-arm sd is 0.035, so every size's regret is inside the
noise of a single architecture comparison.

**The mechanism is basin width, not selection accuracy.** The epoch gap between
`e*` and the test-curve argmin is often enormous — 362 vs 200 at K = 700 — and the
regret is +0.0165 anyway. Within 0.01 nats of the floor the curve is flat across
100–250 epochs at most sizes. The selector does not have to be right, only to land
in the basin. That is the honest version of Jason's charitable read, and it is a
better argument than claiming the selection is accurate.

**Observed-token loss** sits at 1e-5 and below at every size and epoch, in both arms
(also closes the thread-3 breakdown deliverable: all curve movement is the
missing-rating term).

**K = 100 is excluded, not failed.** Its training partition has no items to carve a
block from, so there is no validation signal at all; the shipped selector falls back
to the axis median there.

**Artifact:** https://claude.ai/code/artifact/baa75bad-f08b-4b22-a34a-eba42f513a67

**Open:** three-seed replication (seeds 43/44 full-partition 400-epoch runs at
K = 200/300/500/1000/2000) queued in the pool at priority 8 as tasks 00303–00312.

### E-note: the "extra linear is free" claim is false for the shipped block

`threads/block-projection/out/c0_fold.json`, run before this thread opened:

- inner residual **off** — the block output is the feed-forward output alone, the
  two linears are adjacent, the fold is exact (relative error 1.8e-7).
- inner residual **on** — the shipped setting — the projection is applied to
  `h + attn + ff`, not to `ff` alone, so there is nothing to fold into. Same
  substitution moves the output by a **relative 162**.

So the trailing `d→d` projection *does* add expressivity in the shipped
architecture, and dropping it is a real ablation that has to be run, not a free
simplification. This is consistent with — and is the mechanism behind — the gating
hypothesis: `M` carries no identity component (`identity_alpha ≈ −0.003`, effective
rank 57.5/80, `c4_matrix.json`), so under `dual_outer_residual=none` every layer's
identity path is pushed through a dense near-random rotation.

### Pool incident (no work lost)

Four K=2000 `proj-c2` tasks reported `rc=2`: `run_marformer_group.sh: line 404:
unexpected EOF`. Cause: the blockproj worktree's copy of that script gained 8 lines
(the projection-flag logging) **while jobs were executing it**, and bash reads by
byte offset. Training had already finished and written every checkpoint,
`training_history.json` and `launcher_metadata.json`; only the closing `echo` was
lost. Tasks 00181/00185/00193/00197 marked done by hand. The pool snapshots
`worker.sh` per launch but **not the runner tree** — that is the gap to close.

---

### B7 — does an orthogonal, volume-matched encoder remove the small-data failures? **No.**

Scored from the codec worktree, both draws, `corner_at_val_argmin` (honest early
stopping), reference `codec_dual`.

| arm | K=300 | K=2000 | all |
|---|---|---|---|
| `codec_rand_orth` | +0.0342 ±0.0350 | +0.0036 ±0.0088 | +0.0240 |
| `codec_rand_scaled` | +0.0351 ±0.0328 | +0.0010 ±0.0054 | +0.0237 |
| `codec_align_free` | +0.0213 ±0.0113 | −0.0015 ±0.0041 | +0.0099 |
| `codec_typed` (unscaled) | +0.0733 | +0.1007 | +0.0870 |
| `codec_rand_frozen` | +0.1955 | +0.3529 | +0.2742 |

**Read per cell, which is the point.** At K=300, 12 runs per arm:

```
draw  seed   rand_orth   rand_scaled
s101    44     +0.0543       +0.0506
s105    44     +0.2808       +0.2842      <- the catastrophic run, in BOTH arms
```

Failure rate is **2/12 in both arms**, 0/6 at K=2000 in both, and the failures are
**the same seeds** — seed 44 blows up under orthogonal and under plain scaled
initialisation alike, on both draws, with deltas matching to three decimals.
Orthogonality changes nothing about which runs fail.

**Conclusion: this is small-data sensitivity, not conditioning.** The encoder being
well-conditioned is not what was broken. The honest statement for the paper is that
the typed codec matches the hand-built dual stream in the mean once the encoder is
*loud* (scaling alone gets there: +0.024, inside noise at K=300 and at parity by
K=2000), but at K=300 roughly one run in six falls into a bad optimum, and the fix
for that is more data or a warm start — not a better initialisation. At K=2000 the
codec is clean: 0/6 failures, parity in the mean.

Ordering is unchanged by orthogonality, so `codec_rand_scaled` is the simpler
recipe and there is no reason to ship the orthogonal variant.

### B8 — can a learned temperature dimension recover the `logit_high` scale? **No.**

`codec_temp` +0.266, `codec_orth_temp` +0.264 overall — far worse than their own
baseline (`codec_typed`, +0.087), and worse at every size.

Not a plumbing bug; the head is built and wired correctly (`param_dim − 1 + extra`,
decoder shape (5, 80) against (4, 80) without it, C = 4). **It simply never learns
to use it.** Reading the trained head at K=300:

```
temp row norm 0.594   bias −0.040   ->   temperature 0.976
```

i.e. no sharpening at all, against the 0.05 it would need to reproduce the 20x that
`logit_high` gives the hand-built stream for free. So the extra dimension is dead
weight that also destabilises training — dividing every logit by a noisy softplus
early in training is a worse-conditioned objective than just scaling the encoder.

**The fix that works is the loud encoder, not a learned temperature.** B8 is closed
negative; do not ship `codec_temperature`.

### B9 — temperature purged, scaled init made the default, branch committed (2026-09-11)

User decision: remove the temperature head rather than leave it switched off, and
commit the codec branch with the working initialisation as the default.

- `--codec-temperature` and `DOMAIN3_CODEC_TEMPERATURE` **deleted** from
  `config.py`, `model.py`, `train.py`, `test.py` and `run_marformer_group.sh`.
  `arms_b8_temperature.json` deleted too.
- `codec_init` now defaults to **`scaled`** (was `random`) in both `config.py`
  and the runner. `random` is retained only to reproduce the regression.
- Regression test after the edit: `b2_equivalence.py` still reports
  `EQUIVALENT`, forward max abs diff **0.0** including the batched path.
- Committed on `side/value-codec` as `4d6841f`.

**`value_codec` itself still defaults to `dual`.** The commit makes the codec
correct when selected; it does not switch the model over to it. That stays
blocked on B7: 2 runs in 12 fail at K=300 and no initialisation fixes it.

**Arithmetic correction worth recording.** The fix is *not* `N(0, 1)`. The
encoder is `Linear(param_dim, model_dim)` with no bias, so a one-hot value reads
out one column: norm ≈ `std · sqrt(model_dim)`.

| encoder | per-entry | vector norm | vs hand-built |
| --- | --- | --- | --- |
| hand-built slot | 20 in one slot | 20 | — |
| `N(0, 0.02)` (old default) | 0.02 | 0.18 | 112x quieter |
| `N(0, 1)` | 1.0 | 8.94 | 2.2x quieter — **untested** |
| `N(0, logit_high/sqrt(d))` = `N(0, 2.236)` | 2.24 | **20.0** | matched |


### A3 result (2026-09-12) — dynamic chunking buys nothing and costs 4.2x. **Not adopted.**

K=4000, two draws x two seeds, each arm scored at evaluation chunks
{300, 500, 700, 900, 1200} and read at its own best. Tail-averaged.

| arm | 300 | 500 | 700 | 900 | 1200 | best |
| --- | --- | --- | --- | --- | --- | --- |
| fixed-300 | 0.6889 | 0.6907 | 0.6896 | 0.6949 | 0.7104 | 0.6889 |
| **fixed-500** | 0.6846 | 0.6805 | **0.6720** | 0.6720 | 0.6802 | **0.6720** |
| dynamic 100-500 | 0.6920 | 0.6861 | 0.6759 | 0.6748 | 0.6792 | 0.6748 |

Paired, each arm read at its own best against fixed-500:

| arm | delta | n |
| --- | --- | --- |
| fixed-300 | **+0.0155 ±0.0046** | 4 |
| dynamic 100-500 | **+0.0021 ±0.0016** | 4 |

**Three findings.**

1. **Dynamic is not better.** +0.0021 ±0.0016 against fixed-500 — indistinguishable,
   and the sign is the wrong way. It does not beat fixed-500 at *any* evaluation
   width, including 900 and 1200 where neither arm trained. The extrapolation
   rationale is simply absent.
2. **It costs 4.2x in training time**: 10.3-10.8 s/epoch against 2.45 for both
   fixed arms at K=4000. Re-sampling the window each epoch rebuilds the item
   graph and that dominates the step. The user's condition was that it not train
   much more slowly; it does, for no gain.
3. **fixed-500 already extrapolates.** Its best evaluation width is 700-900, not
   the 500 it trained at, and it is still better at 1200 than at 300. Whatever
   generalisation a dynamic schedule was supposed to teach, the fixed-500 model
   already has. fixed-300, by contrast, degrades past its training width
   (0.6889 at 300 -> 0.7104 at 1200), which is the expected shape.

**Incidental confirmation of A2-lite.** fixed-300 vs fixed-500 comes out at
+0.0155 ±0.0046 here, against the −0.016 ±0.003 measured in A2-lite on different
scoring machinery. Two independent readings of the same knob agreeing to 0.001.

**Thread 1 closed. Ship `max_item=500` train and eval; do not ship dynamic chunking.**
The `--item-chunk-range` flag stays on `side/dynamic-chunk`, unmerged.

### B10 (2026-09-17) — is `logit_high` required, or just "loud enough"? **Just loud enough.**

User's objection, and it is the right one: the point of the learned codec is to
remove the hand-built constant 20, but `codec_init="scaled"` sets its std to
`logit_high / sqrt(model_dim)` — so the constant is still in the loop. B10 tests
whether the exact magnitude matters by running `N(0,1)`, which contains no
constant at all.

Six paired runs, K=2000, two draws x three seeds, scored at eval chunk 500
against `codec_dual` on the same draw and seed.

| encoder | output norm | tail-avg | best epoch |
| --- | --- | --- | --- |
| `N(0, 0.02)` (old default) | 0.18 | +0.0965 ±0.0225 | +0.0945 |
| **`N(0, 1)`** | **8.94** | **−0.0088 ±0.0033** | **−0.0067 ±0.0029** |
| `N(0, logit_high/sqrt d)` | 20.0 | +0.0016 ±0.0039 | +0.0030 |
| orthogonal at norm 20 | 20.0 | +0.0027 ±0.0069 | +0.0039 |

Per cell, `unit` is at or below the reference in 5 of 6 runs. The −0.009 is below
the 0.015 decision threshold, so **read it as parity, not as a win.**

**Conclusion: the requirement is a threshold, not a calibration.** Norm 8.9
clears it, norm 0.18 does not, and the crossing is somewhere between. There is no
evidence the encoder has to match `logit_high`, so the constant can leave the
codec path. `codec_init` now defaults to **`unit`** (`04eff0e`).

**Limits.** K=2000 only. `scaled` fails 2 runs in 12 at K=300 and `unit` has not
been tested there, so small-K behaviour is unmeasured for both — this does not
touch the B7 finding. `logit_high` also still governs the dual stream, which
remains the default `value_codec`; B10 removes the constant from the codec path,
not from the codebase.

### Two bugs found on the way, both mine, both worth remembering

1. **The B9 temperature purge broke `train.py`'s CLI.** The regex removed the
   `--codec-temperature` option string but left the enclosing
   `parser.add_argument()`, so argparse raised `TypeError: _get_positional_kwargs()
   missing 'dest'` at import and the trainer could not start. It shipped in
   `4d6841f` because the check I ran — `b2_equivalence.py` — builds models
   directly and never touches the CLI. **A regression test has to exercise the
   path that changed.** Fixed in `0836941`; there is now a cheap AST check that
   asserts every `add_argument` call has an option string, which does not need a
   five-minute torch import.
2. **`s7_curves.py` silently scores nothing when handed a symlinked arm dir.**
   `base.rglob("*_MARFORMER")` does not descend into symlinked directories, so a
   tree built by symlinking the arm produced `[]` and `COMPLETED` in five
   seconds. Symlink the leaf `*_MARFORMER` run dirs instead. Worth hardening:
   a scorer that returns an empty list for an unreadable tree can quietly drop
   arms from a sweep.

---

### C2/C5 result (2026-09-18) — ADOPT THE STANDARD BLOCK. Thread 3 closed.

Written up at the time in `threads/reports/team-update.html` and the artifact
https://claude.ai/artifact/YFdhZZZ2fWZVBqUQbsNRkG, but **never recorded here**,
so the status board above said "running" for two weeks and the decision looked
open when it was not. Recorded now; the scored JSON is being regenerated into
`side/out/arms_proj-c2-s{101,105}.json` because the original was never saved.

Four arms, K=300 and K=2000, three model seeds x two draws, paired. Nats worse
than `proj_keep_add` (the old shipped block); negative is better.

| arm | what it is | K=300 dev-sel | K=2000 dev-sel | K=2000 tail |
| --- | --- | --- | --- | --- |
| `proj_keep_add` | outer add + trailing linear | 0 ref | 0 ref | 0 ref |
| `proj_keep_rot` | linear only, no outer add | +0.150 | −0.014 | −0.013 |
| **`proj_drop_rot`** | **standard block — neither** | +0.052 (+0.007 best ep) | **−0.021** | **−0.023** |
| `proj_frozen_add` | linear frozen at init | +0.035 | −0.008 | +0.001 |

**The two pieces were never independent.** Removing the outer add alone costs
+0.150 at K=300 — that is the regression that stopped this being simplified
before, and it reproduces here. Removing the linear *as well* cuts it to +0.052,
and to +0.007 at the best epoch. The mechanism is C4's: the projection has no
identity component (alignment −0.003, effective rank 57.5/80), so without the
outer add every layer's identity path is pushed through a dense near-random
rotation eight times over. Taken one at a time each piece looks load-bearing;
taken together they cancel, and what is left is the standard block.

**Caveat for the paper, not for the decision.** At K=300 the standard block is
at parity only at its best epoch; under the dev-selected protocol it costs
+0.052, because a ~110-target dev block early-stops these variants less
reliably (the shipped block's own regret at this size is ~+0.018). K=100 has no
dev block at all. Honest phrasing: **equal architecture, worse stoppability**.

**Consequence for the final run:** `block_output_projection` exists only on
`side/block-projection`. That branch has to be merged into `patches` before the
final grid is launched, and `RERUN/FUTURE_EXPERIMENTS.md` row 11 (`residual
variant: all`) is superseded by this.

### B11 (2026-10-05) — `unit` at small K: no gain, and the 1-in-N blow-up is still there. **Codec stays off.**

B10 established parity at K=2000 and said in its own limits that it measured
nothing below that. The final grid's item axis starts at K=100 and the
annotator axis at J=10, so this closes the gap. `codec_unit` against
`codec_dual`, paired on (seed, cell), draw s101, six model seeds, scored on the
corner at eval chunk 500, tail-averaged over 5 checkpoints.

| seed | J=10 | J=30 | J=50 | K=100 | K=300 |
| --- | --- | --- | --- | --- | --- |
| 42 | −0.0098 | +0.0823 | +0.0702 | −0.0045 | **+0.2018** |
| 43 | +0.0253 | +0.0602 | +0.0179 | +0.0247 | +0.0224 |
| 44 | −0.0264 | −0.0259 | +0.0153 | −0.0195 | −0.0006 |
| 45 | −0.0327 | −0.0212 | +0.0132 | −0.0355 | −0.0164 |
| 46 | +0.0121 | +0.0578 | −0.0274 | +0.0119 | +0.0405 |
| 47 | +0.0532 | −0.0479 | −0.0223 | +0.0559 | −0.0351 |

30 cells: `unit` worse in 16, mean **+0.0146**, median +0.0120 — under the
0.015 threshold, and the sign is close to a coin flip. So there is no
systematic penalty and no gain either: **parity at small K on average.**

**But one cell in 30 blew up**: seed 42 at K=300, **+0.202 nats**. Drop it and
the mean falls to +0.0082 over 29 cells, which is the whole story — the central
tendency is fine and the tail is not. This is B7's signature at a lower rate
(B7 saw 2 in 12 for `scaled`); `unit` is better behaved than `scaled` but not
clean.

**Why that decides it against shipping.** The final run is **one model seed**,
so nothing in the grid would reveal a collapsed cell. At p ≈ 1/30 per cell
across the ~36 cells at K ≤ 1000, the chance of at least one visibly broken
point on the headline curve is roughly 70%, with no replicate to catch it.
`value_codec=dual` ships.

The paper claim survives intact and is the one worth making: the learned codec
**matches** the hand-built stream wherever there is enough data, and
`logit_high` leaves the codec path entirely (B10) — the architecture is more
principled without betting the figure on a tail risk. `side/value-codec` stays
unmerged.

**Note on an intermediate read.** A quick proxy over the tail-averaged *training*
objective flagged seed 44 as the failure; the corner metric says seed 42 at
K=300. The proxy got the existence of a failure right and the cell wrong — it is
not a substitute for corner scoring.
