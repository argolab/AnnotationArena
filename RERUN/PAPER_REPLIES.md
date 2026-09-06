# Draft replies for main.tex — for Tom to edit, then hand back to apply

Baseline: `Marformer-Paper` @ 4b8168a. Line numbers will drift; each item is
anchored by the quoted text, which is what will actually be searched for.

**How to edit:** change anything inside the ````latex`blocks. That text is inserted verbatim. Delete an entire item (or set`SKIP` on its heading line)
to drop it. Don't edit the ANCHOR lines — they locate the insertion point.

Legend: **[reply]** = `\response{Tom}` appended inside an existing note, just
before its closing brace. **[insert]** = new `\tom{...}` after the anchor
sentence. **[rewrite]** = replace the anchor text.

---



## A1 — LayerNorm on the parameter stream  [reply]  L811

ANCHOR: `\jason{Is LayerNorm really applied to the parameters $\vpsi$?  I thought not.}`

CONTEXT: §Constructing the residual stream. "At each layer $\ell \in [1,L]$, the token representations are updated by a standard pre-norm Transformer block with a ReLU feedforward module:" — then eqs. (attention) and (feedforward).

CODE: `norm_1`/`norm_2` are `NormLayer(model_dim)` applied to
`combined = [features; params]`, so the $\vpsi$ slice is normalized too.

```latex
\response{Tom} It is. LayerNorm is applied to the combined stream.
```

---



## A2 — $W_1$/$W_2$ shared across types  [reply]  L820

ANCHOR: `Do you actually use different $W_1$ and $W_2$ matrices for each type?`

CONTEXT: Jason's inline note asking whether the per-type presentation is just
$I + [\mathbf{W}_1;\mathbf{W}_2]$ in disguise.

CODE: one `proj_out` ($d \to d-d'$) and one `W_param` ($d \to d'$) per layer,
shared across types; $d' = \max_\tau d'_\tau$ is the same for every type.

```latex
\response{Tom} No, they are shared across types.  Could be changed, but there is only one type in domain 3 so it is currently indistinguishable; currently it is shared between entity and rating variables.  We worked out at the last meeting that this is not exactly the same as $I+W$ - experiments on this posted in the channel. Our version actually has slightly better inductive bias on the small data regime.
```

---



## A3 — value of $\lambda_\mathrm{copy}$ and $\lambda_{L_2}$  [reply]  L1033

ANCHOR: `\jason{What is the value of $\lambda_\mathrm{copy}$?  Unless we already gave a default value when we first introduced it}`

CONTEXT: §Marformer Configuration (Domain 1). "For $L_2$ regularization, we set
weight_decay=0.01 within AdamW (the default setting in PyTorch)…"

CODE: `weight_decay=0.01` everywhere; D3 masked/observed loss weights 15/1;
domain 1's loss covers only masked variables (no copy term).

```latex
\response{Tom} \texttt{weight\_decay=0.01} in all domains, so $\lambda_{L_2}=0.005$.  $\lambda_\mathrm{copy}=1/15$ in domain 3.  Domain 1's loss covers only masked variables, so $\lambda_\mathrm{copy}=0$ there --- @prabhav double check me.
```

---



## A4 — standard AdamW hyperparameters  [reply]  L950

ANCHOR: `\jason{can we say anything about \emph{standard} hyperparameters for AdamW that we use in all domains?}`

CONTEXT: §Optimization algorithm, right after footnote fn:weightdecay.

CODE: D1 `training_eval.py:198` + clip 1.0 + ReduceLROnPlateau; D2
`imputer_gaussian.py:596`; D3 `train.py:377` (fused AdamW, constant LR).

```latex
\response{Tom} Yes: AdamW with PyTorch defaults $\beta=(0.9,0.999)$, $\epsilon=10^{-8}$, \texttt{weight\_decay=0.01}, no warmup, in every domain.  Learning rate is $10^{-4}$ in domains 1--2 and $2\times10^{-4}$ in domain 3.  Domain 1 additionally uses ReduceLROnPlateau (factor 0.5) and gradient clipping at norm 1.0; domains 2--3 use a constant rate and no clipping.
```

---



## A5 — minibatches  [reply]  L950

ANCHOR: `Tom also asked about this: "Is this just standardized mini-batching (at least in domain 3?) or is this something special?"`

CONTEXT: Jason's long note asking whether $K$ maskings of one instance beats $K$
maskings of $K$ instances, and whether there is an efficiency reason.

CODE: D2 pools $K{=}3$ maskings of all instances, shuffles, batches at 16
(`imputer_gaussian.py:605-655`).  D1 masks fixed at generation
(`sample_generator.py:157`).  D3 has $N=1$; `train.py:398-402,455-470`.

```latex
\response{Tom} From the code I currently see in Domain 1 and 2 (@prabhav, @haojun you should double check me here), domain 2 already does what you suggest: it redraws $K=3$ maskings of every instance each epoch, pools all $K\!\cdot\!N$ of them, shuffles, and cuts minibatches of 16 --- so the $K$ copies of one instance land in different minibatches.  Domain 1 does not redraw at all; masks are fixed when the sample pool is generated.  For domain 3, $N=1$ (the training set is a single annotation tensor), so a minibatch is necessarily $K=5$ maskings of that one tensor; they share one graph build and go through as one batched forward.
```

---



## A6 — validation set / early stopping across domains  [reply]  L952

ANCHOR: `We should standardize the treatment of convergence and early stopping across all 3 domains, right?`

CONTEXT: Jason's `\Jason{...}` note asking whether we early-stop on a validation
set, what size it is, whether it counts toward the training size, and whether
competitors get it.

CODE: D1 `training_eval.py:218` patience 45 on val loss, val =
`test_dataset[:50]` (`experiment_runner.py:346`).  D2 none.  D3 400 epochs, no
val split; new protocol `imputer/entity_mf/validation.py` (block carve, frac
0.15, seed 7) + `analysis/select_epochs.py` + refit.

```latex
\response{Tom} From the code I currently see in Domain 1 and 2 (@prabhav, @haojun double check me): domain 1 early-stops on validation loss with patience 45, but the validation set is the first 50 \emph{test} instances.  Domain 2 has no early stopping.  Domain 3 as reported trains the full 400 epochs with no validation split.  In the new results we have switched domain 3 to the standard setting: carve a validation block out of the training partition, select the epoch budget on it, then merge it back and refit on the full training set for that many epochs.
```

---



## A7 — does the footnote match the implementation  [reply]  L956

ANCHOR: `\jason{does the footnote match the implementation?}`

CONTEXT: Footnote claiming novel attributes are trained to embedding $\vec{0}$
and that it is convenient to initialize attribute and type embeddings to zero.

CODE: `model.py:353` deviation tables `torch.zeros`; `config.py:29`
`type_embedding_init="normal"` = BERT-style $\mathcal{N}(0, 0.02^2)$.

```latex
\response{Tom} Per-entity deviations are initialized to exactly $\vec{0}$ as described, and with $\delta_\tau=1$ they get no gradient and stay there.  But type embeddings are $\mathcal{N}(0,0.02^2)$.
```

---



## A8 — rule of thumb for $\delta_\tau$  [reply]  L956

ANCHOR: `\jason{is there any rule of thumb about how to set $\delta_\tau$?}`

CONTEXT: End of the dropout paragraph, just before §Domain 1.

CODE: `run_marformer_group.sh:147-165` (item axis 1.0/0.0; annot axis 0.0/0.8);
`RERUN/jobs/llmrubric_train.sbatch:93` (0.7); domains 1--2 use 0.

```latex
\response{Tom} We have been trying a few.  Currently: item axis $\delta_\mathrm{item}=1.0$, $\delta_\mathrm{annot}=0$; annotator axis $\delta_\mathrm{item}=0$, $\delta_\mathrm{annot}=0.8$; real LLM-Rubric $\delta_\mathrm{item}=0.7$; domains 1--2 all 0.  So: nonzero (we did 1/0.8/0.7) for the type you must generalize over, 0 for the rest. I'm planning to do some experiment to unify that nonzero value. Or else we can report them as is and say we use the validation set to pick it.
```

---



## A9 — how was the learning rate chosen  [reply]  L1037

ANCHOR: `And why is it so small?}`

CONTEXT: §Domain 1 config, "We run AdamW with learning rate $0.0001$…"

CODE: no sweep or validation-based selection anywhere in the repo for lr.

```latex
\response{Tom} Chosen by hand and then frozen, not tuned. There was no validation data at the time.  We can tune it now that there is.
```

---



## A10 — what is an epoch / how do you minibatch  [reply]  L1037

ANCHOR: `How do you minibatch?}`

CONTEXT: same sentence, "for up to 100 epochs".

CODE: D3 `train.py:398-402` — `train_dataloader` yields `mask_augmentations`
entries of batch 1, so an epoch is 5 steps.  `--batch-augmentations` (ee53e26)
would collapse them into one batched forward, but every RERUN submit script sets
`DOMAIN3_BATCH_AUGMENTATIONS=false`, and LLM-Rubric defaults to false — so it is
off in all reported runs.  D1 `DataLoader(..., batch_size=32, shuffle=True)`.

```latex
\response{Tom} In domain 3 there is no minibatching: an epoch is $K=5$ gradient steps, each of batch size 1 (one chunk graph).  Domain 1 is a shuffled pass over $\Train$ in minibatches of 32.  (We have since added an option to run the $K$ maskings as one batched forward instead, which would make an epoch a single step and want a correspondingly larger learning rate, but it is switched off in all the runs reported here. We can run with minibatch if we want.)
```

---



## A11 — "early stopping" on training loss  [reply]  L1039  (commented-out line)

ANCHOR: `Did you just mean a stopping criterion?}`

CONTEXT: A `%`-commented sentence: "We use early stopping conditioned on the
training loss with a threshold of 0.001 and a patience of 10 epochs."
NOTE: this line is commented out, so the reply will not render until it is
uncommented.  Kept because the same claim is live two lines above.

CODE: `training_eval.py:215-227` — val loss, patience 45, no threshold.

```latex
\response{Tom} The commented-out text does not match the code: domain 1 stops on \emph{validation} loss with patience 45, no 0.1\% threshold, and the validation set is the first 50 test instances.  @prabhav
```

---



## A12 — the 1010 training size  [reply]  L1119

ANCHOR: `should we replace 1010 with 500?}`

CONTEXT: §Results, "By 1010 training instances (column 2)…"

CODE: `policies.py:get_budget_sequence` starts at `--start-examples 10` and adds
`--increment`; committed default is 150 (would give 1060), so 1010 implies a
run with increment 100.

```latex
\response{Tom} Checked, not mislabeled. The sequence starts at 10 and increments by 100, so this is the midpoint at 1010.  We probably want to redo it with 1000, or 1200, to match up with domain 2. @prabhav
```

---



## A13 — weight decay in domain 2  [reply]  L1237

ANCHOR: `Answer: Haojun thinks it's the default, and Tom thinks we *always* use the default`

CONTEXT: §Domain 2 MAP baseline, "using the AdamW optimizer with learning rate
$0.01$, until convergence".

CODE: `imputer_gaussian.py:596` `AdamW(self.parameters(), lr=lr)` — no
`weight_decay` passed, so PyTorch's 0.01.

```latex
\response{Tom} Checked the code: 0.01, the PyTorch default.  @haojun double check if you have newer code.
```

---



## A14 — masked loss weight $\lambda=15$  [reply]  L1298  (INSIDE A COMMENTED-OUT BLOCK)

ANCHOR: `Also, why is $\lambda$ not mentioned in later sections?}`

CONTEXT: This paragraph sits inside `% {\color{blue} … % }} % END OF BLUE`
(lines ~1290-1301), so it is dead text.  The live equivalent is B5 below, which
states the same weights with no note attached.  Reply here only if the block is
coming back; otherwise take the B5 insert instead.

CODE: `--masked-loss-weight 15.0 --observed-loss-weight 1.0`.

```latex
\response{Tom} Neither --- it just means a 15$\times$ scaling on the masked loss.  This explains why the learning rate is low (2e-4); we could equivalently multiply the learning rate by 15 and call it $1 : 1/15$, which is a bit strange but matches how \cref{eq:trainreg} is written. We should fix this before the final rerun.
```

---



## A15 — "do we really do early stopping?"  [reply]  L1945

ANCHOR: `\jason{do we really do early stopping?}`

CONTEXT: §Discussion, "Perhaps overfitting is adequately prevented by our $L_2$
regularization and early stopping, or by the fact that random masking generates
exponentially many training examples."

CODE: reported D3 runs pass `--epochs 400` with no patience flag; the new
carve/select/refit protocol is `validation.py` + `select_epochs.py`.

```latex
\response{Tom} Not in the reported runs.  I have updated the pipeline: domain 3 now carves a validation set from the training partition, selects the stopping epoch on it, and refits on the full training set. That should be used in the final re-runs. (we should think about what set of hyperparmter do we tune)
```

---



## A16 — feedforward depth  [insert]  L816

ANCHOR: `\jason[color=orange]{Prabhav will check how we tuned the feedforward depth, i.e., ratio of feedforward layers to attention layers)}`

CONTEXT: "Hyperparameters include the number and dimensionality of attention
heads, and the number and dimensionality of intermediate layers in the
feedforward network."

CODE: `config.py:21` `num_ffn_layers = 1`; runner default 1.  In `FeedForward`, `num_layers=1` gives `Linear(d,d_ff) -> ReLU -> Linear(d_ff,d)`, the standard single-hidden-layer FFN.  Nothing in the repo uses 4.

```latex
directly modify text (minimally)to fix 4 to be 1, I can confirm we use 1.
```

---



## B1 — Domain 1 model sizes  [insert]  L1028

ANCHOR: `Tiny ($L=2$ layers, $d=32$ dimensions), Small ($L=4, d=64$), and Large ($L=6, d=128$).`

CODE: `architecture.py:522-539` — Tiny $L{=}2$/32/2 heads, Small $L{=}3$/64/4,
Large $L{=}4$/128/8.

```latex
\tom{Code says Tiny $L{=}2$, Small $L{=}3$, Large $L{=}4$ (not 4/6), with 2/4/8 attention heads.  @prabhav, @haojun can you double check.}
```

---



## B2 — Domain 1 masking rate  [insert]  L1030

ANCHOR: `that masks each observed variable with independent 0.15 probability.`

CODE: `sample_generator.py:97-100` `random.sample` of an exact count;
`obs_ratio = 1 - missing_rate` with `missing_rate=0.4`, so 40% masked, drawn
once at pool generation and reused every epoch.

```latex
\tom{Code says an exact count, not an i.i.d.\ coin flip, and 40\% masked rather than 15\%, fixed once at data generation rather than redrawn per epoch.  @prabhav, can you double check, I might not be viewing the newest code you have locally.}
```

---



## B3 — Domain 1 unstated optimizer settings  [insert]  L1037

ANCHOR: `We run AdamW with learning rate $0.0001$`

CODE: batch 32 (`experiment_runner.py:349`), clip 1.0
(`training_eval.py:120`), ReduceLROnPlateau (`training_eval.py:199`).

```latex
\tom{Code also uses batch size 32, gradient clipping at norm 1.0, and ReduceLROnPlateau, none of which are stated. We should add those for complete-ness @prabhav, can you double check.}
```

---



## B4 — Domain 2 "same hyperparameters as Domain 1"  [insert]  L1228

ANCHOR: `We use the same set of hyperparameters, training procedures, and metrics as in \cref{sec:marformer-bn}.`

CODE: `gaussian_exp.py:313` `training_epochs = 20`; `gaussian_exp.py:314`
`batch_size = 16`.  Domain 1 uses 100 and 32.

```latex
\tom{Code says 20 epochs and batch size 16, not domain 1's 100 and 32.  @haojun, can you double check.}
```

---



## B5 — live Domain 3 config paragraph  [insert]  L1514

ANCHOR: `The batch size is $1$ due to GPU memory constraints, and training runs for 400 epochs.`

CONTEXT: §Marformer Configuration for the Annotation Domain — the live
paragraph.  This is where the $\lambda=15$ clarification belongs if the
commented-out block (A14) does not come back.

CODE: matches, except the LLM-Rubric runs use 300 epochs,
$\delta_\mathrm{item}=0.7$ and `max_item=10`, none of which are stated.

```latex
\tom{Masked loss weight 15 with observed loss weight 1 is a 15$\times$ scaling of the masked loss; normalized as in \cref{eq:trainreg} that is $\lambda_\mathrm{copy}=1/15$.  Also, our local LLM-Rubric config says 300 epochs (not 400), $\delta_\mathrm{item}=0.7$, and \texttt{max\_item=10}, I should run replication to see which one we use, but @prabhav, you might know which config we use to run LLM-rubric?}
```

---



## C1 — chunking footnote  [rewrite]  L1298  (INSIDE THE COMMENTED-OUT BLOCK)

ANCHOR: `For training sets beyond 500 items, we restrict the item graph to at most 500 items per forward pass (chunking) to control memory. Full passes over all items would yield marginally better results but are computationally prohibitive at $K=5000$; empirically the gap is small.`

CONTEXT: dead text (same blue block as A14).  If the block stays commented, this
should instead be added as a new footnote on B5.  See DISCREPANCIES.md D1/D24.

CODE: training `--max-item 300`; `test.py:238` hardcodes eval `max_item=500`.

```latex
We chunk the item graph at 300 items per forward pass during training, which is what fits in training GPU memory.  At evaluation time more GPU is available, and we found empirically that 500 yields slightly better performance, so evaluation chunks at 500.
```

---



## C2 — new paragraph on validation and early stopping  [insert]

ANCHOR: place after the `\Jason{Do we also regularize by using a validation set…}`
note in §Marformer Training (i.e. right after A6's paragraph).

CODE: `validation.py:55-91` block carve, `VAL_FRAC=0.15`, `VAL_SPLIT_SEED=7`;
`analysis/select_epochs.py`; refit driven by `EPOCH_TABLE` in
`RERUN/jobs/marformer_train.sbatch`.

CAVEAT: DISCREPANCIES.md D26 finds this carve under-stops the non-transductive
arm (selects ~208-312 vs a corner optimum of ~124; at K=2000 the refit is worse
than not stopping).  Tracked as suite S7 in FUTURE_EXPERIMENTS.md.  Consider
holding this paragraph until S7 resolves.

```latex
\paragraph{Validation and early stopping.} We carve a validation set from the training partition, holding out 15\% of the training items (or annotators, on the annotator axis) so that the validation set poses the same generalization question as the test set.  We train on the remainder, select the number of epochs at which validation loss is minimized, then merge the held-out block back and refit on the full training set for that many epochs.  Only the number of epochs is selected this way; all other hyperparameters are fixed across sizes and seeds.  The validation block is drawn from --- and counted within --- the stated training-set size, so the competing methods see the same amount of data.
```

---



## Left alone deliberately

- the `20` encoding in domain 2 (`\jason{correct?  Haojun will check the number "20"}`) — Haojun's
- SMAR parents-only, and whether domain 2 regenerates a random graph — no code in repo
- whether the same $P^*$ are reused for the MAR rerun — no code in repo
- everything about rewriting prose, redoing experiments, figures, or citations

---



# Recovering notes lost to commented-out blocks

64 notes currently sit on `%`-commented lines, so they do not render and are
invisible in the PDF. They fall into 33 blocks. **Eight of them already carry
answers** — that is real information that will be lost if the blocks are ever
deleted rather than restored. Recover these first:


| line | note (abbreviated)                                                                               | answer buried with it                                                                                                  |
| ---- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| 414  | `\tom{$a$ might be confused as action --- should we change suffix?}`                             | `\response{prabhav}` I had run out of good symbols at this point lol. Feel free to change it.                          |
| 708  | `\jason{double-check that it's nats not bits}`                                                   | `\response{Tom}` Checked, everything is in nats.                                                                       |
| 765  | `\Tom{should we have an "AnnotationArena API" section?}`                                         | `\response{prabhav}` do it in the appendix; `\response{jason}` that's for the next paper — we only support MCAR so far |
| 848  | `\jason{should we say the observed bit $b_j$ is always 1? how is the mask bit set on entities?}` | `\response{tom}` it's always 0, i.e. entities are treated as observed                                                  |
| 1157 | `\jason{your original text mentioned variable elimination and variational inference...}`         | `\response{prabhav}` (see block)                                                                                       |
| 1298 | `\jason{why validation loss here and training loss before?}`                                     | `\response{...}` Prabhav thinks there's no early stopping actually                                                     |
| 1838 | `\Jason{does this include points for all Likert categories?}`                                    | `\response{Tom}` Yes — calibration flattens all $C=4$ categories into one reliability calculation.                     |
| 1838 | `\Jason{is calibration measured on validation data or a separate held-out test set?}`            | `\response{Tom}` Test set. No separate validation split in these experiments.                                          |


The last one is worth promoting into the live text: it answers the same question
Jason asks live in §Marformer Training (item A6 above).

## All 33 blocks, by size


| lines                                                                                                                                                 | notes    | answered | what it is                                              |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | ------------------------------------------------------- |
| 1699–1846                                                                                                                                             | 5        | 2        | "PLACEHOLDER for old material" — 129 lines, the big one |
| 1420–1453                                                                                                                                             | 2        | 0        | ordinal threshold construction                          |
| 1295–1316                                                                                                                                             | 4        | 1        | Domain 3 config (blue block) — items A14/C1 above       |
| 320–337                                                                                                                                               | 8        | 0        | figure-design notes for the overview diagram            |
| 464–474                                                                                                                                               | 2        | 0        | Bayes-optimal action / loss notation                    |
| 560–565                                                                                                                                               | 4        | 0        | SMAR / DAMAR definitions                                |
| 492–501                                                                                                                                               | 1        | 0        | conditioning on incomplete information                  |
| 2349–2358                                                                                                                                             | 2        | 0        | SummEval + Stan-variant appendix stubs                  |
| 703–710                                                                                                                                               | 1        | 1        | deriving test examples from an observed instance        |
| 761–765                                                                                                                                               | 1        | 2        | "CUT SINCE IT FELT LIKE FILLER"                         |
| 939–943                                                                                                                                               | 1        | 0        | doubly expected log-loss                                |
| 410–416                                                                                                                                               | 2        | 1        | probability space / attribute notation                  |
| 284–289, 371, 375–377, 393, 402, 420–423, 446, 449, 477, 485, 505, 598–600, 682–685, 756, 848, 1039, 1122–1124, 1157, 1180–1182, 1369–1371, 2398–2403 | 1–2 each | 4 total  | one- and two-line asides                                |




## Suggested recovery order

1. **Answered notes** (table above) — pull the eight `\response` pairs out to
  live text or a resolved-notes appendix. Highest information density.
2. **1295–1316** — this is the Domain 3 hyperparameter paragraph. Its content is
  duplicated live at L1489–1521, so the notes should be *moved* onto the live
   paragraph rather than the block uncommented. Items A14, C1 and B5 above
   already do this.
3. **1699–1846** — needs a human pass: "check nothing is left out" is literally
  the block's own title, and it is 129 lines.
4. Everything else — uncomment the note, drop the dead prose around it.

**Note (2026-09-06):** Domain 3 now uses index macros (`\ncrit`/`\critidx`,
`\njudge`/`\judgeidx`, `\nitem`/`\itemidx`, plus `\ratesub`/`\ratetuple`/
`\tensordims`) defined at the top of the annotation section. The conversion
skipped commented-out lines, so any block recovered from the list above still
spells `$I$`/`$i$`/`$K$`/`$k$` literally and must be converted to the macros when
it is uncommented — otherwise it will not follow the eventual I/K swap. The
blue block at 1295–1316 holds most of these.

A one-off way to see them all rendered: temporarily convert `%\jason{` to
`\jason{` in a scratch copy and compile. Don't do this on `main.tex` directly —
some of these blocks contain unbalanced math that will not compile.