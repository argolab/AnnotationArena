# Paper edits from the pre-final-run sweep — for Tom to edit, then hand back to apply

Baseline: `Marformer-Paper` @ HEAD as of 2026-09-08. Line numbers drift; each
item is anchored by quoted text, which is what will actually be searched for.

**How to edit:** change anything inside the ```latex blocks — that text is
inserted verbatim. Delete an item (or put `SKIP` on its heading line) to drop
it. Don't edit the ANCHOR lines; they locate the insertion point.

Legend: **[rewrite]** = replace the anchor text. **[reply]** = `\response{Tom}`
appended inside an existing note. **[insert]** = new text after the anchor.

Every EVIDENCE line points at the measurement in `RERUN/FUTURE_EXPERIMENTS.md`
or `RERUN/DISCREPANCIES.md`. Nothing here is proposed without a number behind it.

---


## B1 — LLM-Rubric hyperparameters: three unstated or wrong numbers  [rewrite]  L1533

ANCHOR: `The batch size is $1$ due to GPU memory constraints, and training runs for 400 epochs.`

CONTEXT: §LLM-Rubric experimental setup, the paragraph beginning "We train with
AdamW using learning rate $2\times10^{-4}$...". Your existing `\tom{}` note on
this sentence asks which config we actually use; this resolves it.

EVIDENCE: S6 audit (2026-09-04) against three real `train_config.json` files
(sizes 10, 40, 100), all recording `epochs=300, max_item=10,
item_dropout_rate=0.7`. The evaluation clause is from S8 / D30: the published
number is produced by `plot_llmrubric_cpm_sigmoid.py`, which recomputes at
`max_item=10`; re-scoring the same checkpoints at 225 is **+0.50 nats worse**,
because the model never saw a 225-item context in training.

```latex
The batch size is $1$ due to GPU memory constraints, and training runs for 300 epochs.
We set the item dropout rate $\delta_\mathrm{item}=0.7$, and restrict the item graph to
$10$ items per forward pass---both during training and at evaluation, so that the model
is evaluated on the context size it was trained for.
```

---


## B2 — Domain 3 chunking footnote: one size stated, two used, and "the gap is small"  [rewrite]  L1317

ANCHOR: `For training sets beyond 500 items, we restrict the item graph to at most 500 items per forward pass (chunking) to control memory.`

CONTEXT: currently inside the commented-out Domain 3 architecture paragraph. If
that paragraph is reinstated, this footnote needs replacing; if it stays
commented out, the same text belongs wherever chunking is finally described.

EVIDENCE: S4 (job 299893, 684 corner-scored rows, no training). Below K=500 the
two settings are bit-identical (0.0000). Above it, eval chunk 500 beats 300 by
0.018 nats at K=2000 and K=4000. The optimum is interior: chunk 1200 is worse
than 600 by 0.0098. Separately, *training* chunk 600 is worse than 300 by
0.0094 at K=4000 — so 300/500 is not a compromise, both numbers are preferred.

```latex
\footnote{We restrict the item graph to $300$ items per forward pass during training,
to bound the memory of the backward pass.  Inference has no backward pass and can
afford more context, so evaluation uses $500$; Stan is given the whole tensor at once
regardless, so this is the fair comparison rather than a thumb on the scale.  Below
$K=500$ the two settings are identical.  Above it the larger evaluation chunk is worth
$0.018$ nats at $K=4000$.  More context is not uniformly better: the optimum is
interior, and evaluating with chunks of $1200$ is worse than with $600$.}
```

---


## B3 — Domain 3 epoch budget: the reported numbers are undertrained at large $K$  [insert]  L952

ANCHOR: `Domain 3 as reported trains the full 400 epochs with no validation split.  In the new results we have switched domain 3 to the standard setting: carve a validation block out of the training partition, select the epoch budget on it, then merge it back and refit on the full training set for that many epochs.`

CONTEXT: §Optimization / regularization, inside your existing `\response{Tom}`
to Jason's question about standardizing early stopping across domains. This
extends that answer with what the switch actually costs and buys.

EVIDENCE: W3d. Oracle stopping epoch grows with $K$ (616 / 641 / 932 at
$K=1000$ / $2000$ / $4000$, smoothed). Validation-selected epochs on two seeds
reach $-0.022$ nats at $K=4000$ and $+0.003$ (i.e. nothing) at $K=1000$. A flat
1200-epoch budget is $+0.083$ **worse** at $K=1000$. This is the one design
change in the sweep that moves a headline number.

```latex
  The selected budget is not a constant: it grows with the training set, from roughly
  500 epochs at $K=1000$ to roughly 900 at $K=4000$.  Reaching it is worth $0.022$ nats
  at $K=4000$ and nothing measurable at $K=1000$.  A fixed budget is wrong in both
  directions --- the 400 epochs behind our previously reported numbers is too short at
  large $K$, whereas a fixed 1200 is $0.083$ nats \emph{worse} at $K=1000$, because past
  its optimum the curve climbs rather than plateauing.  The large-$K$ points we reported
  previously are therefore undertrained by roughly $0.02$ nats.
```

---


## B4 — LayerNorm on $\vpsi$: your question, now with evidence  [rewrite]  L811

ANCHOR: `\jason{Is LayerNorm really applied to the parameters $\vpsi$?  I thought not.\response{Tom} It is. LayerNorm is applied to the combined stream.}`

CONTEXT: §Constructing the residual stream, on the sentence introducing the
pre-norm Transformer block. The existing exchange answers "is it?" but not
"should it be?", which is the question a reader will actually have.

EVIDENCE: S1 (`side/s1-featnorm`, 14 runs, K=300, 2 seeds). Normalizing only the
feature slice moves the corner by $\pm 0.02$ with no consistent sign across three
residual settings, while the residual setting itself moves it by up to $0.143$.
The hypothesis that the outer residual exists to compensate for normalizing
$\vpsi$ predicts the variants would converge once $\vpsi$ is excluded. They do not.

```latex
LayerNorm is applied to the combined stream, so the parameter component $\vpsi$ is
normalized along with the features.\footnote{We also tried normalizing only the feature
component, leaving $\vpsi$ untouched.  It does not improve the model ($\pm 0.02$ nats,
with no consistent sign), and---more informatively---it does not shrink the gap between
outer-residual variants, which spans $0.14$ nats either way.  The double residual stream
is therefore doing its own work rather than compensating for the normalization of
$\vpsi$.}
```

---


## B5 — the $\delta_\tau$ rule, stated as a convention rather than a tuned gain  [insert]  L958

ANCHOR: `Thus, for each attribute type $\tau$, we define a \defn{dropout rate} $\delta_\tau \in [0,1]$.`

CONTEXT: §Regularization, where $\delta_\tau$ is introduced. The paper currently
never says what values we use or how they were chosen; different domains use
different values for no stated reason.

EVIDENCE: S2, four draws total across two axes. Item axis: $0.7$ vs the shipped
$1.0$ is $+0.0004 \pm 0.0002$ at K=300 and $-0.0008 \pm 0.0004$ at K=4000.
Annotator axis, two independent data draws: $0.7$ vs the shipped $0.8$ is
$+0.0002 \pm 0.0008$ ($n=6$). Turning dropout off entirely is also within noise
once both draws are pooled ($-0.0009 \pm 0.0082$).

```latex
In practice we set $\delta_\tau = 0.7$ for the attribute type being generalized over and
$\delta_\tau = 0$ for the remaining types.  This choice is a convention rather than a
tuned hyperparameter: on domain 3 every value we tried in $[0,1]$ was within measurement
noise of every other (for instance $0.7$ against $0.8$ on the annotator axis differs by
$0.0002 \pm 0.0008$ nats over two data draws), so we state one rule for uniformity across
domains rather than reporting per-domain values.
```

---


## B6 — Figure 6 protocol: draws, error bands, and Stan's range  [insert]  near the domain 3 results figure

ANCHOR: (the paragraph introducing the domain 3 item-generalization figure — please point me at the right sentence, I did not want to guess)

CONTEXT: §Domain 3 results. Currently the figure is a single generative draw with
no error band, and Stan is plotted only to $K=500$.

EVIDENCE: D28 — the data-draw standard deviation is 0.085 nats, against a
model-seed sd of 0.014 and a Marformer/Stan gap of ~0.11. One unlucky draw moves
a curve by most of the effect being argued about; the original seed 42 is such a
draw. On fresh draws Stan keeps descending past $K=500$ and overtakes Marformer.

```latex
Each point is the mean over three independently generated data sets, with the shaded
band showing their spread; the seed-to-seed variation of the model itself is roughly six
times smaller and is not shown separately.  We plot Stan over the full range rather than
stopping at $K=500$, since the two methods cross beyond that point.
```

---


## B7 — loss weights: no paper change, but a note for the record  [reply]  L1035

ANCHOR: `\jason{What is the value of $\lambda_\mathrm{copy}$?}`

CONTEXT: §Domain 3 training. The paper's normalized form ($\lambda_\mathrm{copy}
= 1/15$) was always correct; the *code* used the unnormalized $15 : 1$. The two
were verified equivalent, so nothing in the paper needs changing --- but a reader
comparing paper to released code will notice, and one sentence pre-empts it.

EVIDENCE: S5 (`s5par`, 4 runs). $(15 : 1)$ and $(1 : 1/15)$ at the **same**
learning rate track each other epoch by epoch for 400 epochs, maximum absolute
difference 0.003 nats. AdamW is invariant to a global rescale of the loss, so no
learning-rate compensation is needed.

```latex
\response{Tom} $\lambda_\mathrm{copy}=1/15$ in domain 3.  The released code applies the
equivalent unnormalized weights ($15$ on masked, $1$ on observed); under AdamW a global
rescale of the objective is absorbed by the optimizer's second-moment normalization, and
the two parameterizations produce trajectories that agree to within $0.003$ nats at every
epoch with no change to the learning rate.
```

---


## B8 — LLM-Rubric figure reproducibility  [no paper change — note for us]

Not a paper edit. Recording it here so it is not forgotten: the RERUN tree
regenerates a **two-curve** version of the LLM-Rubric figure, because Stan CPM,
MIWAE and REMASKER were never re-run there (D29). The paper's own figure has all
five series and the result stands --- Marformer is below Stan CPM at every size
but the smallest. Only the end-to-end reproducibility of that figure is missing.
