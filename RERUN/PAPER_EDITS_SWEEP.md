# Paper edits from the pre-final-run sweep — for Tom to edit, then hand back to apply

Baseline: `Marformer-Paper` @ **f990f22**, pulled 2026-09-10 (five commits of
Tom's edits since the 09-08 baseline). Line numbers drift; each item is anchored
by quoted text, which is what will actually be searched for.

**Every item now names its enclosing `\section`/`\subsection`, resolved against
that commit.** This is because B1 was anchored in the wrong section — it would
have written LLM-Rubric's hyperparameters into the *synthetic* Domain 3 setup —
and B2 turned out to be anchored on `%`-commented-out LaTeX. Read the SECTION
line before the ADDS block; it is the check that caught both.

**What Tom's edits already superseded:** B2 entirely (the live chunking sentence
is now correct at L1536), and the epoch half of B1 (Domain 3 now says "early
stopping" rather than "400 epochs").

**How to edit:** change anything inside the ```latex blocks — that text is
inserted verbatim. Delete an item (or put `SKIP` on its heading line) to drop
it. Don't edit the ANCHOR lines; they locate the insertion point.

Legend: **[rewrite]** = replace the anchor text. **[reply]** = `\response{Tom}`
appended inside an existing note. **[insert]** = new text after the anchor.

Each item now shows **REMOVES** (the text as it stands in `main.tex` today, so
you can see exactly what disappears) and then **ADDS** (the replacement, in the
editable block). Where an item removes nothing, REMOVES says so and quotes the
surrounding text for orientation. Only the ADDS block is applied verbatim ---
the REMOVES blocks are there to read, not to edit.

Every EVIDENCE line points at the measurement in `RERUN/FUTURE_EXPERIMENTS.md`
or `RERUN/DISCREPANCIES.md`. Nothing here is proposed without a number behind it.

---



## B1 — LLM-Rubric \oursys configuration: state it, in the LLM-Rubric section  [insert]  after L1899

SECTION (verified 2026-09-10 against `Marformer-Paper` @ f990f22):
`\section{Appling \oursys on Real Annotation Data}` -> `\subsection{LLM-Rubric}`
(heading L1888). Insert after the dataset paragraph ending
"...with item dropout $\delta_\itemidx = 0.7$ for the \oursys."

**This item was wrong and has been rewritten. Tom caught it.** The previous
version anchored on L1533, which is inside
`\subsection{\oursys Configuration for the Annotation Domain}` — the **synthetic**
Domain 3 configuration. Applying it would have written LLM-Rubric's 300 epochs and
`max_item=10` into the synthetic domain's setup, where they are simply false.
The LLM-Rubric section has no \oursys configuration paragraph at all, which is
the actual defect; this adds one.

Nothing needs removing from the Domain 3 section any more: Tom has already
changed "training runs for 400 epochs" to "training runs with early stopping"
there, and marked the `\tom{}` note resolved. The only leftover is that the
resolved note still carries the LLM-Rubric numbers inside it — see B1b.

EVIDENCE: S6 audit (2026-09-04) against three real `train_config.json` files
(sizes 10, 40, 100), all recording `epochs=300, max_item=10,
item_dropout_rate=0.7`. The matched-chunk clause is S8: the published number is
produced by `plot_llmrubric_cpm_sigmoid.py`, which recomputes at `max_item=10`;
re-scoring the same checkpoints at 225 is **+0.5043 nats worse** averaged over
the ten sizes, because the model never saw a 225-item context in training. That
train/eval chunks must match is a real methodological detail currently stated
nowhere in the paper.

REMOVES: nothing. This is new text. For orientation, the sentence it follows:

```latex
We use a 175/25/25 train/validation/test split over conversations and follow the item generalization protocol of \cref{sec:data-annot}, varying $\nitem_{\mathrm{train}} \in \{10, 20, 30, 40, 50, 75, 100, 125, 150, 175\}$ with item dropout $\delta_\itemidx = 0.7$ for the \oursys.
```

ADDS:

```latex
The \oursys configuration follows \cref{sec:marformer-annot}, with two differences
forced by the smaller data. Training runs for $300$ epochs. The item graph is
restricted to $10$ items per forward pass, and---unlike the synthetic setting, where
evaluation can afford more context than training---evaluation uses the same $10$-item
restriction. Matching the two matters: re-scoring the same checkpoints with the full
$225$-item context costs $0.50$ nats, because the model is never exposed to a context
of that size during training.
```

---



## B1b — the resolved `\tom{}` note in the Domain 3 config still carries LLM-Rubric numbers  [rewrite]  L1532

SECTION (verified 2026-09-10): `\subsection{\oursys Configuration for the Annotation Domain}`
— the synthetic Domain 3 setup. This is the note Tom has already marked
`\response{Tom} Done replicating, text updated.`

The prose around it is now correct. The note itself is spent: its first half
(the $15\times$ / $\lambda_\mathrm{copy}=1/15$ identity) is now stated in the
sentence it annotates, and its second half asks about LLM-Rubric, which B1
answers in the right section. Suggest deleting it. If you would rather keep the
$\lambda_\mathrm{copy}$ reasoning visible, keep only that half.

EVIDENCE: S5 — the two forms are the same optimisation under AdamW (max
$\lVert\mathrm{diff}\rVert$ 0.003 over 400 epochs), so $15{:}1$ and
$1{:}1/15$ describe one objective, not two.

REMOVES:

```latex
\tom{Masked loss weight 15 with observed loss weight 1 is a 15$\times$ scaling of the masked loss; normalized as in \cref{eq:trainreg} that is $\lambda_\mathrm{copy}=1/15$.  Also, our local LLM-Rubric config says 300 epochs (not 400), $\delta_\mathrm{item}=0.7$, and \texttt{max\_item=10}, I should run replication to see which one we use, but @prabhav, you might know which config we use to run LLM-rubric?\response{Tom} Done replicating, text updated.}
```

ADDS: (nothing — delete the note)

```latex
```

---



## B2 — SUPERSEDED, do not apply. Its anchor is commented-out LaTeX.

SECTION (verified 2026-09-10): the anchor text
"For training sets beyond 500 items, we restrict the item graph to at most 500
items per forward pass (chunking) to control memory." survives **only** at L1316,
inside a `%`-commented block that ends `% END OF BLUE`. It is not in the rendered
paper. Editing it would have changed dead text and left the live statement alone.

The live statement is at L1536 and Tom has **already corrected it**:

```latex
For larger training sets, each forward pass is restricted to at most 300 items to control memory usage. Inference can afford more context, so evaluation uses $500$ items. We empirically find the raise from $300$ to $500$ at evaluation time improve log loss.
```

That says what S4 measured (identical below K=500, 0.018 in favour of 500 above),
so B2's content has landed. **Nothing to do.** One optional tightening, since
"We empirically find the raise ... improve log loss" is slightly loose and the
effect is size-dependent:

ADDS (optional replacement for the third sentence only):

```latex
The two sizes give identical results below $\nitem_{\mathrm{train}}=500$; above it, the larger evaluation context is worth $0.018$ nats.
```

---



## B3 — Domain 3 epoch budget: the reported numbers are undertrained at large $K$  [insert]  L952

SECTION (verified 2026-09-10 against `Marformer-Paper` @ f990f22):
`\section{A Neural Architecture for Posterior Marginal Prediction}` ->
`\subsection{\oursys Training}` (heading L928). This is the **shared** training
section, not a domain-specific one, which is the right home for a protocol
statement that applies to Domain 3.


ANCHOR: `Domain 3 as reported trains the full 400 epochs with no validation split.  In the new results we have switched domain 3 to the standard setting: carve a validation block out of the training partition, select the epoch budget on it, then merge it back and refit on the full training set for that many epochs.`

CONTEXT: §Optimization / regularization, inside your existing `\response{Tom}`
to Jason's question about standardizing early stopping across domains. This
extends that answer with what the switch actually costs and buys.

EVIDENCE: W3d. Oracle stopping epoch grows with $K$ (616 / 641 / 932 at
$K=1000$ / $2000$ / $4000$, smoothed). Validation-selected epochs on two seeds
reach $-0.022$ nats at $K=4000$ and $+0.003$ (i.e. nothing) at $K=1000$. A flat
1200-epoch budget is $+0.083$ **worse** at $K=1000$. This is the one design
change in the sweep that moves a headline number.

REMOVES: nothing. The text below is appended inside your existing
`\response{Tom}`, immediately after the sentence quoted in ANCHOR, which stays
exactly as it is.

ADDS:

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

SECTION (verified 2026-09-10): `\subsection{\mbox{\oursys} Architecture}`
(heading L766, `\label{sec:architecture}`). Correct home — this is an
architecture claim.


ANCHOR: `\jason{Is LayerNorm really applied to the parameters $\vpsi$?  I thought not.\response{Tom} It is. LayerNorm is applied to the combined stream.}`

CONTEXT: §Constructing the residual stream, on the sentence introducing the
pre-norm Transformer block. The existing exchange answers "is it?" but not
"should it be?", which is the question a reader will actually have.

EVIDENCE: S1 (`side/s1-featnorm`, 14 runs, K=300, 2 seeds). Normalizing only the
feature slice moves the corner by $\pm 0.02$ with no consistent sign across three
residual settings, while the residual setting itself moves it by up to $0.143$.
The hypothesis that the outer residual exists to compensate for normalizing
$\vpsi$ predicts the variants would converge once $\vpsi$ is excluded. They do not.

REMOVES (main.tex today — Jason's question and your inline answer, which the
replacement folds into running prose plus a footnote):

```latex
\jason{Is LayerNorm really applied to the parameters $\vpsi$?  I thought not.\response{Tom} It is. LayerNorm is applied to the combined stream.}
```

ADDS:

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

SECTION (verified 2026-09-10): `\subsection{\oursys Training}` (heading L928),
same shared section as B3. Correct home — $\delta_\tau$ is defined there.


ANCHOR: `Thus, for each attribute type $\tau$, we define a \defn{dropout rate} $\delta_\tau \in [0,1]$.`

CONTEXT: §Regularization, where $\delta_\tau$ is introduced. The paper currently
never says what values we use or how they were chosen; different domains use
different values for no stated reason.

EVIDENCE: S2, four draws total across two axes. Item axis: $0.7$ vs the shipped
$1.0$ is $+0.0004 \pm 0.0002$ at K=300 and $-0.0008 \pm 0.0004$ at K=4000.
Annotator axis, two independent data draws: $0.7$ vs the shipped $0.8$ is
$+0.0002 \pm 0.0008$ ($n=6$). Turning dropout off entirely is also within noise
once both draws are pooled ($-0.0009 \pm 0.0082$).

REMOVES: nothing. The anchor sentence stays as it is:

```latex
Thus, for each attribute type $\tau$, we define a \defn{dropout rate} $\delta_\tau \in [0,1]$.
```

ADDS (immediately after it, or at the end of that paragraph — your call):

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

REMOVES: nothing — this is new text, and I have not identified the anchor
sentence. Please point me at it.

ADDS:

```latex
Each point is the mean over three independently generated data sets, with the shaded
band showing their spread; the seed-to-seed variation of the model itself is roughly six
times smaller and is not shown separately.  We plot Stan over the full range rather than
stopping at $K=500$, since the two methods cross beyond that point.
```

---



## B7 — loss weights: no paper change, but a note for the record  [reply]  L1035

SECTION (verified 2026-09-10): `\subsection{\oursys Configuration}`
(heading L1025, `\label{sec:marformer-bn}`) — this is **Domain 1**, graphical
models, not Domain 3. That is correct for this item: the `\jason{}` question
about $\lambda_\mathrm{copy}$ is asked there. Be aware the reply concerns the
objective in general, so if it reads as Domain-1-specific it may belong in
\cref{sec:marformer-annot} instead.


ANCHOR: `\jason{What is the value of $\lambda_\mathrm{copy}$?}`

CONTEXT: §Domain 3 training. The paper's normalized form ($\lambda_\mathrm{copy}
= 1/15$) was always correct; the *code* used the unnormalized $15 : 1$. The two
were verified equivalent, so nothing in the paper needs changing --- but a reader
comparing paper to released code will notice, and one sentence pre-empts it.

EVIDENCE: S5 (`s5par`, 4 runs). $(15 : 1)$ and $(1 : 1/15)$ at the **same**
learning rate track each other epoch by epoch for 400 epochs, maximum absolute
difference 0.003 nats. AdamW is invariant to a global rescale of the loss, so no
learning-rate compensation is needed.

REMOVES: nothing. The text below is appended inside the existing
`\response{Tom}` in Jason's note, which currently reads:

```latex
\jason{What is the value of $\lambda_\mathrm{copy}$?  Unless we already gave a default value when we first introduced it\response{Tom} \texttt{weight\_decay=0.01} in all domains, so $\lambda_{L_2}=0.005$.  $\lambda_\mathrm{copy}=1/15$ in domain 3.  ...}
```

ADDS (replacing the `$\lambda_\mathrm{copy}=1/15$ in domain 3.` sentence with a
fuller version):

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