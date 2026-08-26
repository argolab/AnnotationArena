# Making Marformer training GPU-efficient

Branch `marformer-optimization`. Every number below was measured on `gh132`
(H100 NVL, 8 CPUs, conda env `torch`) with `RERUN/bench/run_bench.sh`, which
runs the *real* trainer on the `replicate-01` splits with the hyperparameters
`scripts/DOMAIN3/run_marformer_group.sh` uses, and prints the training-history
loss alongside the wall clock so a speedup can never be bought with accuracy.

Wall-clock figures are quoted as **s/epoch**, taken as the slope between two
runs of different length, because ~10 s of every run is Python import and bundle
loading and would otherwise flatter the short runs.

---

## 1. The diagnosis

The starting point was an established fact: the GPU sits at 9–11% utilisation
and an L40S beats an H100. The usual conclusion — "the model is too small to
fill a GPU" — is wrong. The model is small (453K parameters), but that is not
why the GPU is idle.

`RERUN/bench/profile_train.py` wraps each stage of a step in CUDA-synchronised
timers. On the baseline, 10 epochs of `Item_T_500`:

```
section                      total_s    calls    ms/call    % wall
step/total                     10.78       50     215.51     45.2%
fwd/total                      10.10      180      56.10     42.3%
eval/split                      8.08       40     202.00     33.9%
fwd/init_streams                7.85      180      43.63     32.9%
step/backward                   7.48       50     149.54     31.3%
epoch_end/total                 4.07       10     406.79     17.0%
loss/trainable                  2.57      150      17.13     10.8%
fwd/attn                        0.84     1440       0.58      3.5%
WALL                           23.87
```

**The attention — the only part that is actually the model — is 3.5% of wall
clock.** The rest is Python-side marshalling:

1. `_build_initial_streams` (33%) looped over every token, calling
   `build_param` and writing `features[0, idx] = vec` a row at a time. At
   L≈9000 that is ~18k single-row CUDA scatters per forward, plus one
   `torch.rand(1).item()` per token for entity dropout — a device→host sync
   each time.
2. `step/backward` (31%) is the same loop's fault. Each `features[0, idx] =`
   is its own autograd `index_put_` node, so backward replayed all ~18k of them.
3. `loss/trainable` (11%) rebuilt the target vector with a Python loop and
   built each type mask with a list comprehension over all L tokens.
4. Evaluation (17–34%) ran every epoch, through the same code, plus two
   `.item()` calls per rating token.

So the GPU was idle because the CPU could not feed it, and no GPU-side change —
`torch.compile`, CUDA graphs, bigger GPUs, more GPUs — could have helped while
the graph→tensor conversion was interpreted per token.

---

## 2. What was changed

### 2.1 Tensorise the graph → tensor path (`fastgraph.py`)

Everything in those loops except the per-token `status` is fixed by the graph
topology, which never changes after construction. `GraphTensors` computes it
once, caches it on the graph per device, and the hot paths become gathers:

| was | is |
|-----|-----|
| `for tok: features[0, idx] = emb + dev` | `type_embedding[type_ids]`, then one `index_add` per deviation table |
| `for tok: params[0, idx] = build_param(...)` | `where(hidden, mask_row, param_obs)` |
| `for i in idx.tolist(): targets.append(...)` | one `cross_entropy` over cached targets |
| 3 × `.item()` per loss call | six scalars in a single transfer |
| 2 × `.item()` per rating token in eval | one softmax/argmax, four `.tolist()` |

Entity dropout becomes one Bernoulli kernel instead of M `.item()` syncs. That
is a distribution-level equivalence rather than a value-level one, so the RNG
stream differs from the reference — the only intentional numerical difference
in this branch.

`RERUN/bench/test_equiv.py` asserts each new path matches the loop it replaced
on a real split: exact for the streams, ≤2.4e-7 elsewhere.

### 2.2 Launch-overhead trimming

- **einsum for the relational and pointer biases.** The shared-bias path built
  `rel_scores` as broadcast-multiply-then-sum, materialising a `[B, L, L, R]`
  intermediate (and its gradient) for a `[B, L, L]` result — 40 MiB per layer at
  L=1300, R=6.
- **Fold `1/sqrt(head_dim)` into Q.** The scores are `[B, H, L, L]` (135 MiB at
  B=5, L=1298); dividing them was a full pass over the largest tensor in the
  block for a factor that is linear in Q.
- **`attn_mask=None`.** Nothing is ever padded, so the mask was all-true and its
  `masked_fill` was a 135 MiB read-modify-write per layer for no effect.
- **fused AdamW.** ~100 small parameter tensors meant the unfused optimiser was
  14.5% of GPU time.
- **Evaluation graph cache.** `evaluate_entity_marformer_split` rebuilt the item
  chunking, every chunk graph, and the `[L, L, R]` edge mask and `[L, L, 3]`
  K_aug that are cached *on the graph object* — then threw it all away, every
  epoch. The evaluated variable lists never change.
- **`@torch.no_grad()` on `on_train_epoch_end`.** Lightning does not disable
  autograd in that hook, so every epoch built and freed a backward graph for
  forwards that are never differentiated.

### 2.3 `--batch-augmentations`

An epoch was `mask_augmentations` (=5) separate batch-of-one optimizer steps
over the same chunk graphs. The graphs are identical across those augmentations
— only which tokens are hidden differs — so they can be the batch dimension of
one forward. `GraphTensors.sample_status` draws the B masks with three kernels
(argsort of a uniform row is a uniform k-subset); `sample_status_structured`
does the same for the LLM-Rubric always-observed-annotator scheme.

This is the one change that alters the optimisation: one step per epoch instead
of five. The learning rate has to rise with it.

---

## 3. Results

### 3.1 Domain 3, item axis — the production setting (`max_item=300`)

`Item_T_500`, transductive, 400 epochs:

| | s/epoch | 400 epochs | best combined-missing CE |
|---|---|---|---|
| baseline | 2.155 | 873 s | — |
| **optimised, same semantics** | **0.235** | **111 s** | 0.8462 |
| + `--batch-augmentations --tf32 --lr 1e-3` | 0.171 | 72 s | 0.8629 |

Checked against the replication run with the real `test.py`:

| | test-missing log-loss | accuracy |
|---|---|---|
| `replicate-01` reference | 0.8454 | 0.6251 |
| optimised, same semantics | 0.8509 | 0.6216 |

**7.9× faster, same numbers.** The residual difference is the entity-dropout RNG
stream, which no longer matches the reference draw for draw.

At this operating point batching does *not* pay. Given equal wall clock (111 s),
sequential-400 reaches 0.8462 and batched-600 reaches 0.8558. The chunks here
are L≈3900, big enough to keep the GPU busy on their own, so multiplying the
work by five buys nothing.

### 3.2 Domain 3, item axis at `max_item=100`

Same split, smaller chunks (L≈1300) — launch-bound, so batching does pay:

| | s/epoch | 400 epochs | best combined-missing CE |
|---|---|---|---|
| baseline | 2.274 | 920 s | — |
| optimised, same semantics | 0.365 | 156 s | 0.8895 |
| **+ `--batch-augmentations --lr 1e-3`** | **0.120** | **56 s** | **0.8880** |

**16.5×, at equal loss.** The LR sweep at 400 epochs: 2e-4 → 0.9065,
5e-4 → 0.8964, **1e-3 → 0.8890**, 2e-3 → 0.8901.

### 3.3 Domain 3, annotator axis

`Annot_T_30`, `max_item=300`, entity dropout 0.8 and deviation regularisation
actually active:

| | s/epoch | final CE at 90 epochs |
|---|---|---|
| baseline | 1.224 | 0.9279 |
| optimised | 0.134 | 0.9138 |

**9.1×**, curves overlaying throughout.

### 3.4 LLM-Rubric

`LLMRubric_225_25_9_50`, `max_item=10` — a 225-item bundle is ~23 chunks of a
few hundred tokens, the most launch-bound configuration in the repo. 300 epochs:

| | wall | best val-missing CE | final |
|---|---|---|---|
| baseline | LLMRUBRIC_BASE_WALL | — | — |
| optimised, same semantics | 186.8 s | 1.0744 | 1.3568 |
| **+ `--batch-augmentations --tf32 --lr 1e-3`** | **71.2 s** | **1.0201** | **1.0487** |

Batching is *better* as well as faster here, because LLM-Rubric overfits badly
at this size (see `OVERFITTING.md`) and five times fewer optimizer steps means
less of it — the sequential run ends 0.28 nats above its own best.

---

## 4. What did not work

Recorded because these are the obvious next things to reach for.

- **`torch.compile`** (`--compile default`). 0.091 s/epoch against 0.120 eager
  at `max_item=100` — genuinely 25% faster per epoch — but ~12 s of extra fixed
  cost with a warm inductor cache and ~50 s cold. Break-even is around 370
  epochs, so at 400 it is a wash and on a fresh node it is a loss. Left as a
  flag, off by default. `reduce-overhead` (CUDA graphs) was no better.
- **bf16 autocast** (`--amp bf16`). 0.126 s/epoch against 0.120 fp32. The block
  is bandwidth-bound on the score tensor, so halving its width looked obvious,
  but after the other changes the step is CPU-launch-bound again (226 ms CPU vs
  160 ms CUDA per 8 iterations) and autocast only adds cast kernels. Kept as a
  flag, off by default; the param stream carries raw logits (`logit_high = 20`),
  so it is a numerical change, not just a speed knob.
- **TF32** (`--tf32`). No measurable change — the matmuls were never the
  bottleneck. Harmless, so it stays available.
- **Multi-GPU.** Still pointless, for the reason established earlier: there is
  no batch dimension to shard, and sharding the mask augmentations across
  devices would change the optimisation rather than just its speed.
- **`--eval-every 5`.** 0.112 s/epoch against 0.139 — real but small, and it
  thins out the training curve the overfitting analysis reads. Available, off.

---

## 5. How to use it

The semantics-preserving work needs no flags: every existing invocation of
`run_marformer_group.sh` or `jobs/llmrubric_train.sbatch` gets 8–9× for free.

The batching has to be asked for, because it changes the optimisation:

```bash
# Domain 3, small chunks (max_item <= 100)
DOMAIN3_BATCH_AUGMENTATIONS=true DOMAIN3_TF32=true DOMAIN3_LR=1e-3 \
    scripts/DOMAIN3/run_marformer_group.sh ...

# LLM-Rubric
LLMRUBRIC_BATCH_AUGMENTATIONS=true LLMRUBRIC_TF32=true LLMRUBRIC_LR=1e-3 \
    sbatch ... jobs/llmrubric_train.sbatch
```

Do not turn it on at `max_item=300` — measured above, it loses there.

## 6. Reproducing

```bash
RERUN/bench/run_bench.sh --size 500 --max-item 300 --epochs 400 --tag mine
RERUN/bench/run_bench.sh --profile --size 500 --epochs 20 --tag prof   # breakdown
python -m RERUN.bench.test_equiv --data-dir <split>                    # correctness
python -m RERUN.bench.kernel_profile --data-dir <split> --batch 5      # kernels
RERUN/bench/sanity_configs.sh <checkout> 60                            # annot + LLM-Rubric
RERUN/bench/bench_llmrubric.sh <checkout> 50 300 tag [args...]
```

`RERUN/bench/BASELINE.md` holds the pre-optimisation reference so the whole
chain can be re-derived.
