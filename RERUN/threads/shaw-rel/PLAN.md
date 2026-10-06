# Thread D — Shaw et al. relational attention

Opened 2026-10-06, from Jason's margin note at `Marformer-Paper/main.tex:839`.

## What he asked

The paper already cites `shaw2018self` for relational attention (main.tex:837).
His note says their version is better and we should consider switching:

> (1) they have two embeddings of ρ, which are added to keys **and the values**
> when i attends to j. We only modify the keys but maybe we should modify the
> values too. (2) they have **learned** embeddings of ρ. We freeze and
> concatenate, but the learning dynamics are probably better their way, and they
> don't have to reserve a fixed number of dimensions, and it's easier their way
> to learn similar representations for similar ρ and near-zero representations
> for unimportant ρ.

Two separable changes, so the arms separate them.

## What ships today

`USE_PER_HEAD_REL=false` in `run_marformer_group.sh`, i.e. the **shared-bias
bit** design: the query gets R extra output dimensions and the "key side" of the
relational term is the frozen indicator bit, so

    scores[h,i,j] += sum_{rho: i -rho-> j} Q_rel[i, rho]

(One paper/code mismatch worth noting while we are here: main.tex says we
*reduce* the key's dimensionality by |Rho| to make room for the bits. The
shipped code does not — it adds R extra **query** outputs and leaves the key at
full width. The per-head mode, `use_per_head_rel=true`, is the one that reserves
dims, and Domain 3 does not use it.)

## What was built

`side/shaw-rel`, worktree `Marformer-shaw`, commit 9d355d8. Two new flags,
both defaulting to the shipped behaviour, so nothing else moves:

| flag | values | what it does |
|---|---|---|
| `--rel-key` / `DOMAIN3_REL_KEY` | `bits` (default), `learned` | `learned` gives each relation a learned per-head embedding `a^K_rho` in key space, added to the key: `scores = q.(k + sum_rho a^K_rho)/sqrt(hd)`. No dims reserved, no extra query outputs — which makes `use_per_head_rel` and `scale_shared_rel` inert, since the two bit designs differ only in where the reserved dims sit. |
| `--rel-value` / `DOMAIN3_REL_VALUE` | `none` (default), `learned` | Shaw's other half, per head and exact: `z_i = sum_j alpha_ij (V(x_j) + a^V_rho)`. Zero-init. |

Both read the **sparse** edge list. The dense `[L, L, R]` mask is 365 MiB per
cached graph at L=3900 and the pre-existing `use_rel_value` needs it; the new
value term contracts j through a gather/scatter keyed by `(i, rho)` instead, so
it does not. The two value terms write the same bias and are refused together.

`d1_sparse_check.py` checks both terms and the whole block against a dense
reference written straight from Shaw's equations: **exact** on the key bias and
the value mass, 6e-8 on the block output, gradients reach both embeddings, and
the shipped `bits` block is byte-for-byte what it was. Run it from any tree with
`Marformer-shaw/imputer/ranking` on `PYTHONPATH`.

## Prediction, on record before the runs

**`shaw_k` alone should be neutral to slightly worse.** The learned key
embedding contributes `q^h_i . a^K_rho = x_i^T W_Q^T a^K_rho`, which is still a
*linear* function of `x_i` — and now has to factor through `W_Q`. The bits give
each head an unconstrained R-dimensional linear readout of `x`; Shaw's is a
rank-constrained version of the same thing. As a score term it is not richer.

What it can buy is exactly what Jason says: `a^K_rho` lives in key space, so
similar relations can take similar embeddings and an unimportant relation can go
to zero in every head at once, and no dimension budget is reserved. Those are
optimisation and parameterisation claims, not capacity claims, so they show up
as a better or worse *trained* number or not at all.

**The value half is the one that widens the function class.** The shipped
architecture has no analogue for it at all. So if anything wins it should be
`bits_v` and `shaw_kv`.

## Arms (`arms_d2.json`), reference `bits`

| tag | rel_key | rel_value |
|---|---|---|
| `bits` | bits | none | ← shipped Domain 3 |
| `shaw_k` | learned | none |
| `shaw_kv` | learned | learned | ← full Shaw |
| `bits_v` | bits | learned |

## Grid and protocol

Jason's sizes: item K = 300, 500, 2000, 4000 and annot J = 10, 30, 50 — which is
exactly the set of cells `final-dual` already has, so the absolute numbers are
comparable to the final run. Three model seeds (42, 43, 44), draw s101, chunk
300, transductive, **700 epochs** with the 15% validation block carved
(`DOMAIN3_VAL_PROTOCOL=block`). 700 covers every stopping epoch the final run
selected on these cells (64–417 transductive, max 564 including NT) with margin.

This is a **dev-selected comparison on the search run**, not a full F13 refit:
each arm trains on 85% of its training partition and is read at the epoch its
own validation block selects. The refit onto 100% is what the headline numbers
need and is worth about +0.005 median; it would double the GPU for a comparison
that is paired anyway, so it is out of scope here.

`d2_analyse.py` reports four reads, all paired within (axis, seed, K):
`dev-sel` (primary), `tail`, `last`, and `best` — the last **diagnostic only**,
because it selects on test.

## Cost

~20 GPU-hours on l40s ≈ 6.6k billing-hours, about 7% of what the project has
left. Everything fits l40s at chunk 300 (K=4000 peaks near 37 GB of 44), and the
new flags add nothing measurable to memory.

## Status

- [x] implemented, `d1_sparse_check.py` all green
- [x] four-arm smoke at K=300 — all train, `train_config.json` records the flags,
      `test.py::_reconstruct` restores them
- [ ] 84 cells queued in the pool as `shaw-d2-s101`
- [ ] scored with `side/s7_curves.py --ckpt-stride 2 --eval-max-item 300`
- [ ] read with `d2_analyse.py --ref bits`
