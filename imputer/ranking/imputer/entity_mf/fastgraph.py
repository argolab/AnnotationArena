"""Tensorised view of an EntityGraph.

The original forward/loss path walks `graph.tokens` in Python once per call:
one `build_param` per token, one `features[0, idx] = ...` scatter per token,
one `torch.rand(1).item()` per token for entity dropout, and a fresh
`[tok.type_name == name for tok in tokens]` list comprehension per type per
loss. At L ~ 9000 tokens that is the whole training step -- the model itself is
453K parameters and accounts for a few percent of wall clock.

Everything in that loop except `status` is a property of the graph's topology,
which never changes after construction. `GraphTensors` computes it once, caches
it on the graph per device, and exposes it as flat tensors so the forward pass
and the loss become a handful of gathers.

Only `status` (0=missing, 1=masked, 2=observed) changes between steps, and it
is refreshed with a single list comprehension + one host-to-device copy.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from .data import EntityGraph
from .types import EntityType


class TypeSlice:
    """Per-type static index tensors and prediction targets."""

    __slots__ = ("name", "positions", "mask", "targets", "soft_targets", "vectorised")

    def __init__(
        self,
        name: str,
        positions: torch.Tensor,
        mask: torch.Tensor,
        targets: torch.Tensor | None,
        soft_targets: torch.Tensor | None,
        vectorised: bool,
    ):
        self.name = name
        self.positions = positions          # [M] long, token indices of this type
        self.mask = mask                    # [L] bool
        self.targets = targets              # [M] long class labels, or None
        self.soft_targets = soft_targets    # [M, C] float, or None (means: use hard)
        self.vectorised = vectorised        # False -> caller must use the token-loop path


class GraphTensors:
    """Static tensors for one EntityGraph on one device."""

    def __init__(self, graph: EntityGraph, types: Dict[str, EntityType],
                 global_param_dim: int, device: torch.device):
        tokens = graph.tokens
        L = len(tokens)
        self.device = device
        self.num_tokens = L
        self.global_param_dim = global_param_dim
        self.type_names: List[str] = sorted(types.keys())
        name_to_row = {n: i for i, n in enumerate(self.type_names)}

        type_ids = torch.empty(L, dtype=torch.long)
        entity_ids = torch.full((L,), -1, dtype=torch.long)
        for i, tok in enumerate(tokens):
            type_ids[i] = name_to_row[tok.type_name]
            if tok.entity_id is not None and tok.entity_id >= 0:
                entity_ids[i] = tok.entity_id
        self.type_ids = type_ids.to(device)
        self.entity_ids = entity_ids.to(device)

        # Param stream. build_param's output depends on raw_data and on whether
        # the token is currently masked/missing; the masked/missing answer is
        # always the same "mask bit" vector, so precompute the observed answer
        # and select between them at call time.
        param_obs = torch.zeros(L, global_param_dim)
        mask_row = torch.zeros(L, global_param_dim)
        is_variable = torch.zeros(L, dtype=torch.bool)
        for i, tok in enumerate(tokens):
            t = types[tok.type_name]
            if not getattr(t, "is_variable", False):
                continue
            is_variable[i] = True
            mask_row[i, 0] = 1.0
            raw = tok.raw_data or {}
            if raw.get("is_missing", False):
                # A missing token is never un-masked, so its observed row is
                # never selected; and asking build_param for it would raise.
                continue
            unmasked = dict(raw)
            unmasked["is_masked"] = False
            param_obs[i] = t.build_param(unmasked, device=torch.device("cpu"),
                                         global_param_dim=global_param_dim).cpu()
        self.param_obs = param_obs.to(device)
        self.mask_row = mask_row.to(device)
        self.is_variable = is_variable.to(device)

        # Deviation lookups, one index tensor per type that has a table.
        # Annotator id per token (-1 for entity tokens), for structured masking.
        annot = torch.full((L,), -1, dtype=torch.long)
        for i, tok in enumerate(tokens):
            if tok.raw_data is not None:
                annot[i] = int(tok.raw_data.get("annotator_id", -1))
        self.annot_ids = annot.to(device)

        self.dev_positions: Dict[str, torch.Tensor] = {}
        self.dev_entity_ids: Dict[str, torch.Tensor] = {}
        for name in self.type_names:
            t = types[name]
            if not (t.variation.enabled and t.variation.num_entities > 0):
                continue
            sel = ((self.type_ids == name_to_row[name]) & (self.entity_ids >= 0)
                   & (self.entity_ids < t.variation.num_entities))
            pos = sel.nonzero(as_tuple=False).squeeze(-1)
            if pos.numel():
                self.dev_positions[name] = pos
                self.dev_entity_ids[name] = self.entity_ids[pos]

        # Per-type slices + targets.
        self.slices: Dict[str, TypeSlice] = {}
        for name in self.type_names:
            t = types[name]
            mask = self.type_ids == name_to_row[name]
            pos = mask.nonzero(as_tuple=False).squeeze(-1)
            targets = soft = None
            vectorised = True
            if getattr(t, "is_variable", False) and pos.numel():
                targets, soft, vectorised = self._build_targets(t, tokens, pos)
            self.slices[name] = TypeSlice(name, pos, mask, targets, soft, vectorised)

        self._status = torch.empty(L, dtype=torch.long, device=device)
        self._status_cpu = torch.empty(L, dtype=torch.long)
        self.refresh(graph)
        # Statuses as the graph was constructed, before any training mask was
        # applied. sample_status() rebuilds from this, so it does not matter
        # what the last step left behind.
        self.base_status = self._status.clone()

    @staticmethod
    def _build_targets(t: EntityType, tokens, pos: torch.Tensor):
        """Hard/soft class targets for one variable type, or (None, None, False)."""
        num_classes = getattr(t, "num_classes", None)
        if num_classes is None:
            # Pairwise ranking (or any future type): leave it to the token loop.
            return None, None, False
        idx = pos.tolist()
        hard: List[int] = []
        soft_rows: List[List[float]] = []
        has_soft = False
        for i in idx:
            raw = tokens[i].raw_data or {}
            rv = raw.get("rating_value", None)
            if rv is None:
                return None, None, False
            rv = int(rv)
            hard.append(rv)
            dist = raw.get("rating_dist", None)
            if dist is not None and len(dist) == num_classes:
                soft_rows.append([float(x) for x in dist])
                has_soft = True
            else:
                row = [0.0] * num_classes
                row[rv] = 1.0
                soft_rows.append(row)
        dev = pos.device
        targets = torch.tensor(hard, dtype=torch.long, device=dev)
        soft = torch.tensor(soft_rows, dtype=torch.float32, device=dev) if has_soft else None
        return targets, soft, True

    # ------------------------------------------------------------------ status
    def refresh(self, graph: EntityGraph) -> None:
        """Pull the current per-token status. The only thing that changes per step."""
        self._status_cpu.copy_(torch.tensor([tok.status for tok in graph.tokens],
                                            dtype=torch.long))
        self._status.copy_(self._status_cpu, non_blocking=True)

    @property
    def status(self) -> torch.Tensor:
        return self._status

    def sample_status(self, n_maskable: int, masking_rate: float, batch: int) -> torch.Tensor:
        """
        Draw `batch` independent MCAR masks over the first `n_maskable` tokens.

        Returns [B, L] statuses (0=missing, 1=masked, 2=observed). Equivalent to
        calling MCARMasking.mask() `batch` times -- exactly
        floor(n_maskable * masking_rate) tokens masked per draw, chosen without
        replacement -- but as three kernels instead of B * n_maskable Python
        objects. Sorting a uniform row is the standard way to get a random
        permutation, so taking its first k entries is a uniform k-subset.

        Chunk-graph layout puts the maskable tokens first; everything after
        n_maskable keeps its constructed status.
        """
        L = self.num_tokens
        status = self.base_status.unsqueeze(0).expand(batch, L).clone()
        if n_maskable <= 0:
            return status
        k = int(n_maskable * masking_rate)
        k = max(0, min(n_maskable, k))
        if k == 0:
            return status
        pick = torch.rand(batch, n_maskable, device=self.device).argsort(dim=1)[:, :k]
        rows = torch.arange(batch, device=self.device).unsqueeze(1).expand(batch, k)
        head = status[:, :n_maskable]
        head[rows.reshape(-1), pick.reshape(-1)] = 1
        return status

    def sample_status_structured(self, n_maskable: int, always_observed_ids,
                                 human_observed_rate: float, batch: int) -> torch.Tensor:
        """
        Batched counterpart of StructuredMasking.mask.

        Annotators in `always_observed_ids` stay observed; of the rest, exactly
        floor(n_human * human_observed_rate) are kept observed per draw and the
        remainder are masked. At human_observed_rate=0 -- the LLM-Rubric setting
        -- there is nothing random left, so all `batch` rows are identical and
        only the dropout differs between them.
        """
        L = self.num_tokens
        status = self.base_status.unsqueeze(0).expand(batch, L).clone()
        if n_maskable <= 0:
            return status
        always = torch.tensor(sorted(always_observed_ids), dtype=torch.long,
                              device=self.device)
        head_annot = self.annot_ids[:n_maskable]
        is_human = ~torch.isin(head_annot, always)
        hpos = is_human.nonzero(as_tuple=False).squeeze(-1)          # [n_human]
        n_human = int(hpos.numel())
        if n_human == 0:
            return status
        head = status[:, :n_maskable]
        head[:, hpos] = 1
        k_obs = int(n_human * human_observed_rate)
        k_obs = max(0, min(n_human, k_obs))
        if k_obs:
            pick = torch.rand(batch, n_human, device=self.device).argsort(dim=1)[:, :k_obs]
            rows = torch.arange(batch, device=self.device).unsqueeze(1).expand(batch, k_obs)
            head[rows.reshape(-1), hpos[pick.reshape(-1)]] = 2
        return status

    @property
    def hidden(self) -> torch.Tensor:
        """True where the token's value must not be revealed (masked or missing)."""
        return self.is_variable & (self._status != 2)


def graph_tensors(graph: EntityGraph, types: Dict[str, EntityType],
                  global_param_dim: int, device: torch.device) -> GraphTensors:
    """Get (building once) the cached GraphTensors for this graph and device."""
    cache = getattr(graph, "_tensor_cache", None)
    if cache is None:
        cache = {}
        graph._tensor_cache = cache
    key = (str(device), global_param_dim)
    gt = cache.get(key)
    if gt is None:
        gt = GraphTensors(graph, types, global_param_dim, device)
        cache[key] = gt
    return gt
