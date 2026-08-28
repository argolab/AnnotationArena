# Marformer training speed — baseline

Node `gh132` (H100 NVL, 8 CPUs), conda env `torch`.
Config: `DOMAIN3-SIGMOID_Item_T_500`, transductive, `max_item=100`,
8 layers, `embedding_dim=80`, `mask_augmentations=5` — i.e. exactly the
hyperparameters `scripts/DOMAIN3/run_marformer_group.sh` uses for the item axis.

## Wall clock (unprofiled)

| tag | epochs | wall | s/epoch | final combined-missing CE |
|-----|--------|------|---------|---------------------------|
| `base_s500_e40` | 40 | 101.6 s | 2.54 | 1.0201 |

A full 400-epoch run is therefore ~17 min, and the 38-run replication grid is
dominated by this number.

## Where the time goes (10 epochs, `--profile`)

Timers synchronise CUDA on entry and exit, so GPU work is charged to the call
that launched it. Sections nest (`step/total` contains `fwd/total`, which
contains `fwd/init_streams`), and the syncs inflate the cheap high-count rows.

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
fwd/k_aug                       0.77      180       4.26      3.2%
mask/mcar                       0.10      150       0.68      0.4%
graph/edge_mask                 0.04      180       0.21      0.2%
graph/build                     0.01        4       3.07      0.1%
step/refresh_tokens             0.01      150       0.07      0.0%
setup/build_chunks              0.01        1       9.37      0.0%
WALL                           23.87
```

## Reading

The actual model math is `fwd/attn` — 3.5% of wall, 0.58 ms per layer call for
a 453K-parameter model. Everything else is Python-side marshalling:

1. **`_build_initial_streams` (33%)** is a Python `for` loop over every token
   that calls `t.build_param(...)` and then writes `features[0, idx] = vec`
   one row at a time. At L≈9k that is ~18k single-row CUDA scatters per
   forward. It also calls `torch.rand(1).item()` per token for entity dropout,
   which is a device→host sync each time.
2. **`step/backward` (31%)** is the same loop's fault: each `features[0, idx] =`
   is a separate autograd `index_put_` node, so backward replays ~18k tiny
   kernels.
3. **`loss/trainable` (11%)** rebuilds the target vector with a Python loop over
   `idx.tolist()` every call, and builds each type mask with a list
   comprehension over all L tokens.
4. **`eval/split` / `epoch_end` (17–34%)** runs a full evaluation pass every
   epoch, which goes through the same `_build_initial_streams`.

So the GPU is idle ~90% of the time (matches the 9–11% utilisation measured
earlier) not because the model is small but because the per-token Python loop
serialises everything. Fixing the marshalling is worth far more than any
GPU-side change, and has to come first — `torch.compile` and CUDA graphs cannot
help while the graph→tensor conversion is interpreted per token.
