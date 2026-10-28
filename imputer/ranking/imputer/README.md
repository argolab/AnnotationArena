# imputer

Marformer: a transformer-based imputation model for incomplete annotation data (attributes x annotators x items), built as typed random-variable tokens with no positional embeddings. The actual model lives in `entity_mf/`; the modules at this top level (`data.py`, `transformer.py`, `losses.py`, `utils.py`) are generic building blocks it imports.

## Layout

- `data.py` — `RankingData` (one rating/ranking observation, with explicit
  observed/masked/missing status) and `DataConverter` (turns a JSON data
  bundle into a list of `RankingData`).
- `transformer.py` — `NormLayer`, `FeedForward`, used by
  `entity_mf/model.py`'s `RelationalAttentionBlock`.
- `losses.py` — `PlackettLuceLoss`, the ranking loss used by
  `entity_mf/types.py`'s `PairwiseRankingVariableType`.
- `utils.py` — `sizes_from_configs`, reads item/attribute/annotator/class
  counts out of a `configs.json`.
- `entity_mf/` — the model itself:
  - `config.py` — `EntityMarformerConfig` (architecture/hyperparameter toggles).
  - `types.py` — the type system (see below).
  - `data.py` — `Token`, `Relationship`, `EntityGraph`, and
    `variable_list_to_entity_graph()`, which builds the graph the model consumes.
  - `model.py` — `RelationalAttentionBlock` and `EntityMarformer`.
  - `value_codecs.py` — per-type codecs: how a variable's value enters the
    residual stream at layer 0 and how its logits are read back out (see below).
  - `masking.py` — `MCARMasking` / `StructuredMasking` (the latter keeps an
    LLM annotator's ratings always observed, for real-data training).
  - `eval.py` — cross-type loss aggregation and rating-specific eval metrics.
  - `train.py` — training entry point: `python -m imputer.entity_mf.train`.
  - `test.py` — checkpoint evaluation entry point:
    `python -m imputer.entity_mf.test`.
  - `plot_training_history.py` — standalone plotting tool over saved
    `training_history.json` files; not part of the training/eval pipeline
    itself, not invoked by any script in `scripts/`.

## Defining and using a new type

Every attribute/annotator/item/rating/ranking is an instance of `EntityType`
(`entity_mf/types.py`). There are two kinds:

- **Entity types** (`NullEntityType` subclasses — `AttributeEntityType`,
  `AnnotatorEntityType`, `ItemEntityType`): pure context, no prediction, no
  loss. `build_param` and `build_value` return zeros; `compute_loss_breakdown`
  returns an empty breakdown by default.
- **Variable types** (`EntityType` subclasses — `RatingVariableType`,
  `PairwiseRankingVariableType`): carry a value and a prediction target,
  contribute to the loss via `compute_loss_breakdown` -> `LossBreakdown`.

A variable type with C values has an *extended* domain {⊤, v_1..v_C}: a masked
or missing token takes the placeholder value ⊤ rather than carrying a mask bit.
The type says what the value is — `build_value` returns a distribution over the
extended domain, ⊤ at index 0 — and, optionally, gives a hand-built encoding of
it (`build_param`: ⊤ as 1.0 in slot 0, a value as `logit_high` in its slot).

To add one, subclass `EntityType` (or `NullEntityType` for a pure entity) and
implement `build_value`, `build_param` (only the slice codec reads it), and,
for variable types, `compute_loss_breakdown`. The **core loss
aggregation is generic** — `eval.py`'s `_aggregate_loss_from_breakdowns`
iterates over whatever is in the `types: Dict[str, EntityType]` dict and
calls `compute_loss_breakdown` on each, so a correctly implemented type will
train correctly once registered. Registration is a plain dict, built by
`build_default_domain3_types()` in `types.py` — either extend that factory or
build your own `Dict[str, EntityType]` and pass it through wherever `types`
is threaded (`train.py`, `test.py`, `eval.py`, `data.py`).

**The registry alone is not enough — token/graph construction is hardcoded,
not registry-driven.** `entity_mf/data.py`'s `variable_list_to_entity_graph()`
is domain-3-specific: it explicitly loops over `types["attribute"]`,
`types["annotator"]`, `types["item"]` to create entity tokens, explicitly
picks `"rating"` vs `"ranking_pairwise"` based on `var.is_listwise`, and
explicitly wires the `ATTR`/`ANNOT`/`ITEM` `Relationship` edges. A new entity
type needs its own token-construction block (and edges to whatever it should
connect to) added here; a new variable type needs its own branch for how raw
`RankingData` fields map onto `Token.raw_data`.

**A few features silently ignore types they don't know about, rather than
erroring** — worth checking if you rely on them for a new variable type:
- `model.py`'s `_get_k_aug` (the optional pointer mechanism, `use_pointer=True`)
  only looks at tokens with `type_name in ("rating", "ranking_pairwise")`.
- `train.py` counts variable vs. entity tokens and computes per-chunk loss
  weights against that same `("rating", "ranking_pairwise")` tuple.
- `eval.py`'s rating accuracy/expected-value metrics and `test.py`'s metric
  display are specific to `types["rating"]` — a new variable type trains and
  gets aggregate loss numbers, but won't get this richer per-type metric
  breakdown without adding its own block.

## Codecs: how a value enters and leaves the stream

How a value becomes a vector is not the type's business but its codec's, and
the encoder and decoder are chosen separately, per type
(`--value-codec SPEC`, `--type-codec TYPE=SPEC`):

| spec | encode (layer 0) | decode (layer L) |
|---|---|---|
| `slice` (alias `dual`) | copy `build_param`'s row into the last `global_param_dim` dims, which the centroids give up | read the same slots back |
| `linear` (alias `typed`) | learned projection of `build_value`'s row into the full hidden dim | linear probe on the full hidden dim |
| `mlp` | the same with an MLP | an MLP probe |
| `linear:width=T` | projection into the last T dims only (`,reserve` also narrows the centroids) | probe on the last T dims |
| `slice/mlp`, `linear/slice`, … | the first half | the second half |

`slice` is the shipped dual stream and `linear:init=unit` the typed codec, bit
for bit; `linear:init=aligned,freeze` is the linear codec that *is* the slice
codec. The grammar and every option are documented at the top of
`value_codecs.py`.

**Checklist for a new type:**
1. Subclass `EntityType`/`NullEntityType` in `types.py`, implement `build_value`
   and `build_param` (+ `compute_loss_breakdown` if it's a variable type).
2. Add it to `build_default_domain3_types()` (or your own registry dict).
3. Add its token/edge construction to `variable_list_to_entity_graph()` in
   `entity_mf/data.py`.
4. If it's a variable type and you use `--use-pointer`, extend the tuple in
   `model.py`'s `_get_k_aug`.
5. If you want it counted correctly in training-loop bookkeeping, extend the
   `("rating", "ranking_pairwise")` tuples in `train.py`.
6. If you want per-type accuracy/expected-value metrics, add a block for it
   in `eval.py` (see the `types["rating"]` section) and `test.py`.
