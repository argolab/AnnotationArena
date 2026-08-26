from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imputer.transformer import NormLayer, FeedForward

from .config import EntityMarformerConfig
from .data import EntityGraph
from .fastgraph import graph_tensors
from .types import EntityType


class RelationalAttentionBlock(nn.Module):
    """
    Relational self-attention with two selectable designs:

    use_per_head_rel=True  (default, new design):
      Q: D -> D  (each head gets k_content_dim content dims + R relational dims)
      K: D -> H * k_content_dim  (content only)
      scores = (Q_content @ K_content^T + einsum(Q_rel, edge_mask)) / sqrt(head_dim)
      Each head learns independent relational weights.
      Requires: head_dim > R  (i.e. model_dim // num_heads > num_relationships).

    use_per_head_rel=False  (old/shared-bias design):
      Q: D -> D + R  (D content dims shared across heads, R shared relational dims)
      K: D -> D      (content only)
      scores = (Q_content @ K_content^T) / sqrt(head_dim)  +  Q_rel @ edge_mask^T
      Single shared relational bias added identically to all heads.

    Optional extensions (both modes):
      use_pointer:     K_aug obs-obs shared-identity bias (3 extra shared Q dims, like old Marformer).
      use_rel_value:   V_{ij} = V(x_j) + sum_r e_r * edge_mask[i,j,r].
      use_addone_attn: attn = exp(s) / (1 + sum_j exp(s_j)), sum <= 1.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_relationships: int,
        dropout: float,
        use_per_head_rel: bool = True,
        use_pointer: bool = False,
        use_rel_value: bool = False,
        use_addone_attn: bool = False,
        scale_shared_rel: bool = False,
        use_graph_mask: bool = False,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_relationships = num_relationships
        self.use_per_head_rel = use_per_head_rel
        self.use_pointer = use_pointer
        self.use_rel_value = use_rel_value
        self.use_addone_attn = use_addone_attn
        self.scale_shared_rel = scale_shared_rel
        self.use_graph_mask = use_graph_mask
        assert model_dim % num_heads == 0, (
            f"model_dim {model_dim} must be divisible by num_heads {num_heads}"
        )
        self.head_dim = model_dim // num_heads
        R = num_relationships

        if use_per_head_rel:
            # Per-head: each head has (head_dim - R) content dims + R relational dims
            self.k_content_dim = self.head_dim - R
            assert self.k_content_dim > 0, (
                f"head_dim {self.head_dim} must be > num_relationships {R}. "
                f"Increase model_dim or reduce num_heads."
            )
            q_out_dim = model_dim + (3 if use_pointer else 0)
            self.Q = nn.Linear(model_dim, q_out_dim)
            self.K = nn.Linear(model_dim, num_heads * self.k_content_dim)
        else:
            # Shared-bias: full head_dim for content, separate R-dim shared relational vector
            self.k_content_dim = self.head_dim  # full head_dim used for content
            q_out_dim = model_dim + R + (3 if use_pointer else 0)
            self.Q = nn.Linear(model_dim, q_out_dim)
            self.K = nn.Linear(model_dim, model_dim)  # content only

        self.V = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)

        # Relation-specific value embeddings: shared across heads, init to zero (no-op at start)
        if use_rel_value:
            self.value_rel = nn.Parameter(
                torch.zeros(R, model_dim)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_mask: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        K_aug: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         [B, L, D] combined feature+param representation.
            edge_mask: [L, L, R] binary edge indicators.
            attn_mask: [B, L] bool mask for valid tokens.
            K_aug:     [L, L, 3] obs-obs shared-identity indicators (optional, for pointer).
        """
        B, L, D = x.shape
        H = self.num_heads
        R = self.num_relationships
        hd = self.head_dim
        kc = self.k_content_dim
        edge_mask_f = edge_mask.to(x.dtype)                        # [L, L, R]

        V_base = self.V(x).view(B, L, H, hd).transpose(1, 2)     # [B, H, L, hd]

        if self.use_per_head_rel:
            # ── Per-head relational attention ───────────────────────────────────
            # Q: D -> D (+ 3 for pointer); split per head into [content | rel]
            Q_proj = self.Q(x)                                     # [B, L, D] or [B, L, D+3]
            Q_full = Q_proj[..., :D].view(B, L, H, hd).transpose(1, 2)  # [B, H, L, hd]
            if self.use_pointer:
                Q_ptr = Q_proj[..., D:]                            # [B, L, 3]
            K_full = self.K(x).view(B, L, H, kc).transpose(1, 2) # [B, H, L, kc]
            # Scale Q rather than the scores: the scores are [B, H, L, L]
            # (135 MiB at B=5, L=1298), so dividing them is a full extra pass
            # over the largest tensor in the block for a factor that is linear
            # in Q anyway.
            inv_scale = 1.0 / math.sqrt(hd)
            Q_content = Q_full[..., :kc] * inv_scale              # [B, H, L, kc]
            Q_rel     = Q_full[..., kc:] * inv_scale              # [B, H, L, R]
            content_scores = torch.matmul(Q_content, K_full.transpose(-2, -1))       # [B, H, L, L]
            rel_scores     = torch.einsum("bhir,ijr->bhij", Q_rel, edge_mask_f)      # [B, H, L, L]
            scores = content_scores + rel_scores
        else:
            # ── Shared-bias relational attention (old design) ───────────────────
            # Q: D -> D+R (+ 3 for pointer); K: D -> D; rel bias shared across heads
            Q_proj = self.Q(x)                                     # [B, L, D+R] or [B, L, D+R+3]
            Q_content = Q_proj[..., :D]                            # [B, L, D]
            Q_rel_shared = Q_proj[..., D:D+R]                     # [B, L, R]  (shared across heads)
            if self.use_pointer:
                Q_ptr = Q_proj[..., D+R:]                          # [B, L, 3]
            K_content = self.K(x)                                  # [B, L, D]
            # Same reason as the per-head branch: fold the score scale into Q.
            inv_scale = 1.0 / math.sqrt(hd)
            Qh = Q_content.view(B, L, H, hd).transpose(1, 2) * inv_scale   # [B, H, L, hd]
            Kh = K_content.view(B, L, H, hd).transpose(1, 2)               # [B, H, L, hd]
            if self.scale_shared_rel:
                Q_rel_shared = Q_rel_shared * inv_scale
            content_scores = torch.matmul(Qh, Kh.transpose(-2, -1))        # [B, H, L, L]
            # Relational bias: [B, L, L] -> broadcast over heads [B, 1, L, L].
            # einsum rather than broadcast-multiply-then-sum: the latter
            # materialises a [B, L, L, R] intermediate (and its gradient), which
            # at L=1300, R=6 is 40 MiB per layer for a result that is 6.5 MiB.
            rel_scores = torch.einsum("bir,ijr->bij", Q_rel_shared, edge_mask_f).unsqueeze(1)
            scores = content_scores + rel_scores

        # ── Pointer bias (shared across heads) ─────────────────────────────────
        if self.use_pointer and K_aug is not None:
            # Q_ptr: [B, L, 3], K_aug: [L, L, 3]
            # Same [B, L, L, 3] intermediate to avoid as in the relational bias.
            ptr_bias = torch.einsum("bip,ijp->bij", Q_ptr, K_aug.to(Q_ptr.dtype))  # [B, L, L]
            scores = scores + ptr_bias.unsqueeze(1)                        # broadcast over H

        # ── Graph mask (hard): allow only edge_mask + K_aug + self ─────────────
        if self.use_graph_mask:
            graph_mask = edge_mask.any(dim=-1)                                   # [L, L]
            if K_aug is not None:
                graph_mask = graph_mask | K_aug.any(dim=-1)                      # [L, L]
            graph_mask = graph_mask | torch.eye(L, dtype=torch.bool, device=x.device)
            scores = scores.masked_fill(~graph_mask[None, None], torch.finfo(scores.dtype).min)

        # ── Attention mask ──────────────────────────────────────────────────────
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.bool()
            key_mask = attn_mask[:, None, None, :]                # [B, 1, 1, L]
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        # ── Softmax or add-one attention ────────────────────────────────────────
        if self.use_addone_attn:
            # Numerically stable: shift by max before exp
            scores_shifted = scores - scores.max(dim=-1, keepdim=True).values
            exp_s = torch.exp(scores_shifted)
            if attn_mask is not None:
                exp_s = exp_s.masked_fill(~key_mask, 0.0)
            attn = exp_s / (1.0 + exp_s.sum(dim=-1, keepdim=True))
        else:
            attn = F.softmax(scores, dim=-1)

        attn = self.dropout(attn)

        # ── Value aggregation + optional relation value augmentation ───────────
        out = torch.matmul(attn, V_base)   # [B, H, L, hd]

        if self.use_rel_value:
            # Contract j first to avoid materializing [L, L, D] intermediate (OOM for large graphs).
            # attn_r_mass[b, i, r] = sum_j attn_mean[b, i, j] * edge_mask[i, j, r]
            attn_mean = attn.mean(dim=1)                                                  # [B, L, L]
            attn_r_mass = torch.einsum("bij,ijr->bir", attn_mean, edge_mask_f)           # [B, L, R]
            # bias[b, i, d] = sum_r attn_r_mass[b, i, r] * value_rel[r, d]
            bias = attn_r_mass @ self.value_rel                                           # [B, L, D]
            bias_h = bias.view(B, L, H, hd).permute(0, 2, 1, 3).contiguous()            # [B, H, L, hd]
            out = out + bias_h

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(out)


def _init_type_embedding(p: nn.Parameter, mode: str, feature_dim: int) -> None:
    """Initialize a type centroid embedding parameter."""
    if mode == "normal":
        nn.init.normal_(p, mean=0.0, std=0.02)
    elif mode == "scaled_normal":
        nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(feature_dim))
    elif mode == "kaiming":
        nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")
    else:
        raise ValueError(
            f"Unknown type_embedding_init: {mode!r}. "
            f"Choose from 'normal', 'scaled_normal', 'kaiming'."
        )


def _init_bias_as_type_centroid(bias_data: torch.Tensor, mode: str, dim: int) -> None:
    """Initialize a 1-D bias tensor using the same scheme as type centroid embeddings."""
    if mode == "normal":
        nn.init.normal_(bias_data, mean=0.0, std=0.02)
    elif mode == "scaled_normal":
        nn.init.normal_(bias_data, mean=0.0, std=1.0 / math.sqrt(dim))
    elif mode == "kaiming":
        tmp = bias_data.unsqueeze(0)          # kaiming_normal_ requires >= 2D
        nn.init.kaiming_normal_(tmp, mode="fan_out", nonlinearity="relu")
        bias_data.copy_(tmp.squeeze(0))
    else:
        raise ValueError(
            f"Unknown type_embedding_init: {mode!r}. "
            f"Choose from 'normal', 'scaled_normal', 'kaiming'."
        )


class EntityMarformer(nn.Module):
    """
    Entity Marformer with two selectable architectures (controlled by config.use_embed_unembed):

    Dual-stream (default, use_embed_unembed=False):
      Layer-0: [feature_centroid + deviation | param_from_build_param]  (feature_dim + param_dim = model_dim)
      Blocks:  attn+FFN on combined; explicit proj_out and W_param residuals back to each stream.
      Output:  param stream at layer L.

    Embedding/unembedding (use_embed_unembed=True):
      Layer-0: value_embedder(simple_value) + deviation  (single model_dim vector)
               value_embedder is nn.Linear(param_dim, model_dim) whose bias is the type centroid.
               For entity types (param_dim=0) only the bias (type centroid) is used.
               Name embedding = deviation of parent entities (summed), same dropout logic as before.
      Blocks:  standard pre-norm attn+FFN on a single residual stream; no split/concat.
      Output:  per-type unembedder (nn.Linear(model_dim, param_dim)) applied to final residual;
               returns same [1, L, global_param_dim] shape as the dual-stream path.
    """

    def __init__(
        self,
        config: EntityMarformerConfig,
        types: Dict[str, EntityType],
        num_relationships: int,
    ):
        super().__init__()
        self.config = config
        self.types = types
        self.use_per_head_rel = config.use_per_head_rel
        self.use_pointer = config.use_pointer
        self.use_rel_value = config.use_rel_value
        self.use_addone_attn = config.use_addone_attn
        self.use_param_output_head = config.use_param_output_head
        self.use_embed_unembed = config.use_embed_unembed
        # Set by enable_compile(); None means run the eager block stack.
        self._compiled_blocks = None
        self.dual_outer_residual = getattr(config, "dual_outer_residual", "all")
        self.inner_residual = getattr(config, "inner_residual", "both")
        if self.inner_residual not in ("both", "ff_only", "none"):
            raise ValueError(
                f"Unknown inner_residual: {self.inner_residual!r}. "
                "Choose 'both', 'ff_only', or 'none'."
            )
        if self.inner_residual == "none" and self.dual_outer_residual == "none":
            raise ValueError(
                "inner_residual='none' with dual_outer_residual='none' leaves the "
                "block with no skip connection at all."
            )

        # Global param dim = max over types.
        self.global_param_dim = max(t.param_dim for t in types.values())
        self.param_dim = self.global_param_dim

        # Config embedding_dim = total model dim. Must be divisible by heads.
        self.model_dim = config.embedding_dim
        assert self.model_dim % config.attention_heads == 0, (
            f"embedding_dim {self.model_dim} must be divisible by attention_heads {config.attention_heads}"
        )

        # ── Dual-stream path parameters ───────────────────────────────────────
        if not config.use_embed_unembed:
            assert self.model_dim > self.param_dim, (
                f"embedding_dim {self.model_dim} must be > global_param_dim {self.param_dim}"
            )
            self.feature_dim = self.model_dim - self.param_dim

            self.type_embeddings = nn.ParameterDict(
                {name: nn.Parameter(torch.empty(1, self.feature_dim)) for name in types.keys()}
            )
            for p in self.type_embeddings.values():
                _init_type_embedding(p, config.type_embedding_init, self.feature_dim)

            self.deviation_tables = nn.ParameterDict()
            for name, t in types.items():
                if t.variation.enabled and t.variation.num_entities > 0:
                    self.deviation_tables[name] = nn.Parameter(
                        torch.zeros(t.variation.num_entities, self.feature_dim)
                    )

            self.deviation_norm = NormLayer(self.feature_dim)

            if self.use_param_output_head:
                self.param_output_head = nn.Sequential(
                    nn.Linear(self.model_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.param_dim),
                )

            blocks: List[nn.Module] = []
            for _ in range(config.num_layers):
                blocks.append(nn.ModuleDict({
                    "norm_1":    NormLayer(self.model_dim),
                    "attn":      self._make_attn_block(config, num_relationships),
                    "norm_2":    NormLayer(self.model_dim),
                    "ff":        FeedForward(self.model_dim, d_ff=config.d_ff, dropout=config.dropout, num_layers=config.num_ffn_layers),
                    "proj_out":  nn.Linear(self.model_dim, self.feature_dim),
                    "W_param":   nn.Linear(self.model_dim, self.param_dim),
                    "dropout_2": nn.Dropout(config.dropout),
                }))
            self.blocks = nn.ModuleList(blocks)

        # ── Embedding/unembedding path parameters ─────────────────────────────
        else:
            self.feature_dim = self.model_dim  # whole vector is the single residual stream

            # Value embedders: nn.Linear(param_dim, model_dim) for variable types.
            # The bias of each Linear serves as the type centroid.
            # Entity types (param_dim==0) get a plain Parameter as their type centroid.
            self.value_embedders = nn.ModuleDict()
            self.type_embeddings_emb = nn.ParameterDict()
            for name, t in types.items():
                if t.param_dim > 0:
                    lin = nn.Linear(t.param_dim, self.model_dim, bias=True)
                    nn.init.normal_(lin.weight.data, mean=0.0, std=0.02)
                    _init_bias_as_type_centroid(lin.bias.data, config.type_embedding_init, self.model_dim)
                    self.value_embedders[name] = lin
                else:
                    p = nn.Parameter(torch.empty(1, self.model_dim))
                    _init_type_embedding(p, config.type_embedding_init, self.model_dim)
                    self.type_embeddings_emb[name] = p

            # Deviation tables: model_dim-wide (name embeddings for each entity).
            self.deviation_tables_emb = nn.ParameterDict()
            for name, t in types.items():
                if t.variation.enabled and t.variation.num_entities > 0:
                    self.deviation_tables_emb[name] = nn.Parameter(
                        torch.zeros(t.variation.num_entities, self.model_dim)
                    )

            self.emb_deviation_norm = NormLayer(self.model_dim)

            # Per-type unembedders: model_dim → param_dim (only for variable types).
            self.unembedders = nn.ModuleDict()
            for name, t in types.items():
                if t.param_dim > 0:
                    self.unembedders[name] = nn.Linear(self.model_dim, t.param_dim)

            # Standard pre-norm blocks (no proj_out / W_param split).
            emb_blocks: List[nn.Module] = []
            for _ in range(config.num_layers):
                emb_blocks.append(nn.ModuleDict({
                    "norm_1":  NormLayer(self.model_dim),
                    "attn":    self._make_attn_block(config, num_relationships),
                    "norm_2":  NormLayer(self.model_dim),
                    "ff":      FeedForward(self.model_dim, d_ff=config.d_ff, dropout=config.dropout, num_layers=config.num_ffn_layers),
                    "dropout": nn.Dropout(config.dropout),
                }))
            self.emb_blocks = nn.ModuleList(emb_blocks)

    def enable_compile(self, mode: str = "default") -> None:
        """
        Compile the block stack.

        Worth doing because the step is launch-bound, not math-bound: at L~1300
        the GPU does ~11 ms of work per step while the CPU spends ~27 ms queueing
        it. Chunks have different token counts, so the graph is compiled once per
        distinct L (a handful) -- dynamic=False keeps each specialisation fast
        rather than tracing one dynamic kernel for all of them.
        """
        if mode in ("", "none"):
            self._compiled_blocks = None
            return
        if self.use_embed_unembed:
            raise ValueError("enable_compile only covers the dual-stream path")
        self._compiled_blocks = torch.compile(self._dual_blocks, mode=mode, dynamic=False)

    def _make_attn_block(self, config: EntityMarformerConfig, num_relationships: int) -> RelationalAttentionBlock:
        return RelationalAttentionBlock(
            model_dim=self.model_dim,
            num_heads=config.attention_heads,
            num_relationships=num_relationships,
            dropout=config.dropout,
            use_per_head_rel=config.use_per_head_rel,
            use_pointer=config.use_pointer,
            use_rel_value=config.use_rel_value,
            use_addone_attn=config.use_addone_attn,
            scale_shared_rel=config.scale_shared_rel,
            use_graph_mask=config.use_graph_mask,
        )

    def _get_k_aug(self, graph: EntityGraph, device: torch.device) -> "torch.Tensor | None":
        """Build (and cache) the [L, L, 3] obs-obs shared-identity K_aug tensor."""
        if not self.use_pointer:
            return None
        dev_key = str(device)
        if dev_key not in graph._k_aug_cache:
            L = graph.num_tokens
            attr_ids  = torch.full((L,), -1, dtype=torch.long, device=device)
            annot_ids = torch.full((L,), -1, dtype=torch.long, device=device)
            item_ids  = torch.full((L,), -1, dtype=torch.long, device=device)
            for idx, token in enumerate(graph.tokens):
                if token.type_name in ("rating", "ranking_pairwise") and token.raw_data:
                    attr_ids[idx]  = token.raw_data.get("attribute_id", -1)
                    annot_ids[idx] = token.raw_data.get("annotator_id", -1)
                    iids = token.raw_data.get("item_ids", [])
                    item_ids[idx]  = iids[0] if iids else -1

            def _same(ids: torch.Tensor) -> torch.Tensor:
                eq = (ids.unsqueeze(0) == ids.unsqueeze(1)).float()
                valid = (ids >= 0).float()
                return eq * valid.unsqueeze(0) * valid.unsqueeze(1)

            graph._k_aug_cache[dev_key] = torch.stack(
                [_same(attr_ids), _same(annot_ids), _same(item_ids)], dim=-1
            )
        return graph._k_aug_cache[dev_key]

    def _build_initial_streams(
        self,
        graph: EntityGraph,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build initial feature and param streams from the EntityGraph (dual-stream path).

        Everything here except `status` is fixed by the graph topology, so it is
        precomputed once per graph by GraphTensors and this becomes a handful of
        gathers. See _build_initial_streams_ref for the original per-token loop;
        test_equiv.py asserts the two agree.

        Returns:
            features:  [1, L, feature_dim]
            params:    [1, L, param_dim]
            attn_mask: [1, L]
        """
        gt = graph_tensors(graph, self.types, self.param_dim, device)
        gt.refresh(graph)

        type_mat = torch.cat([self.type_embeddings[n] for n in gt.type_names], dim=0)  # [T, F]
        features = type_mat.index_select(0, gt.type_ids)                               # [L, F]

        for name, pos in gt.dev_positions.items():
            table = self.deviation_tables.get(name, None)
            if table is None:
                continue
            variation = self.types[name].variation
            dev = table.index_select(0, gt.dev_entity_ids[name])                       # [M, F]
            if self.training and variation.dropout_rate > 0:
                # One independent Bernoulli draw per token, as in the reference
                # loop -- but as a single kernel instead of M `.item()` syncs.
                keep = (torch.rand(dev.shape[0], 1, device=device) >= variation.dropout_rate)
                dev = dev * keep.to(dev.dtype)
            if self.config.use_deviation_norm:
                dev = self.deviation_norm(dev)
            features = features.index_add(0, pos, dev)

        params = torch.where(gt.hidden.unsqueeze(-1), gt.mask_row, gt.param_obs)       # [L, P]
        # No padding: every token is real. Returning None instead of an all-true
        # mask skips a masked_fill over the [B, H, L, L] score tensor per layer.
        return features.unsqueeze(0), params.unsqueeze(0), None

    def _build_initial_streams_ref(
        self,
        graph: EntityGraph,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Original per-token implementation. Kept as the equivalence reference."""
        L = graph.num_tokens
        features = torch.zeros(1, L, self.feature_dim, device=device)
        params   = torch.zeros(1, L, self.param_dim,   device=device)
        attn_mask = torch.ones(1, L, dtype=torch.bool, device=device)

        for idx, token in enumerate(graph.tokens):
            t = self.types[token.type_name]
            feat_vec = self.type_embeddings[token.type_name].expand(1, -1)
            if token.type_name in self.deviation_tables and token.entity_id >= 0:
                dev_table = self.deviation_tables[token.type_name]
                if 0 <= token.entity_id < dev_table.shape[0]:
                    dev = dev_table[token.entity_id].unsqueeze(0)
                    if self.training and t.variation.dropout_rate > 0:
                        if torch.rand(1).item() < t.variation.dropout_rate:
                            dev = torch.zeros_like(dev)
                    if self.config.use_deviation_norm:
                        dev = self.deviation_norm(dev)
                    feat_vec = feat_vec + dev
            features[0, idx] = feat_vec
            params[0, idx]   = t.build_param(token.raw_data or {}, device=device, global_param_dim=self.param_dim)

        return features, params, attn_mask

    def _build_initial_emb_stream(
        self,
        graph: EntityGraph,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build initial single residual stream (embedding/unembedding path).

        Layer-0 vector for each token = value_embedding(simple_value) + name_embedding(deviation).
        For variable types: value_embedding = Linear(param_dim → model_dim) applied to build_simple_value output.
        For entity types:   value_embedding = type centroid bias only.
        Name embedding = deviation from deviation_tables_emb (same dropout logic as dual-stream).

        Returns:
            x:         [1, L, model_dim]
            attn_mask: [1, L]
        """
        L = graph.num_tokens
        x         = torch.zeros(1, L, self.model_dim, device=device)
        attn_mask = torch.ones(1, L, dtype=torch.bool, device=device)

        for idx, token in enumerate(graph.tokens):
            t = self.types[token.type_name]

            # Value embedding
            if token.type_name in self.value_embedders:
                # Variable type: A * simple_value + b_type  (b_type is the Linear bias)
                simple_val = t.build_simple_value(
                    token.raw_data or {}, device=device, global_param_dim=t.param_dim
                )
                val_emb = self.value_embedders[token.type_name](simple_val)  # [model_dim]
            else:
                # Entity type: type centroid only (bias b)
                val_emb = self.type_embeddings_emb[token.type_name].squeeze(0)  # [model_dim]

            # Name embedding: per-entity deviation (same dropout as dual-stream path)
            name_emb = torch.zeros(self.model_dim, device=device)
            if token.type_name in self.deviation_tables_emb and token.entity_id >= 0:
                dev_table = self.deviation_tables_emb[token.type_name]
                if 0 <= token.entity_id < dev_table.shape[0]:
                    dev = dev_table[token.entity_id]
                    if self.training and t.variation.dropout_rate > 0:
                        if torch.rand(1).item() < t.variation.dropout_rate:
                            dev = torch.zeros_like(dev)
                    if self.config.use_deviation_norm:
                        dev = self.emb_deviation_norm(dev.unsqueeze(0)).squeeze(0)
                    name_emb = dev

            x[0, idx] = val_emb + name_emb

        return x, attn_mask

    def forward(
        self,
        graph: EntityGraph,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Forward pass over a single EntityGraph.

        Returns:
            params: [1, L, global_param_dim]
        """
        if device is None:
            device = next(self.parameters()).device

        if self.use_embed_unembed:
            return self._forward_emb(graph, device)
        return self._forward_dual(graph, device)

    def forward_batched(
        self,
        graph: EntityGraph,
        status: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Forward pass over one EntityGraph under B different masks at once.

        The graph topology -- and therefore the edge mask, K_aug and every
        attention shape but the leading dimension -- is identical across the
        mask augmentations of a training epoch. Running them as a batch turns
        B forward/backward passes into one, which matters because the step is
        launch-bound: at L~1300 the GPU does ~11 ms of work per pass while the
        CPU spends ~27 ms queueing it.

        Args:
            graph:  the chunk graph.
            status: [B, L] long, per-augmentation token statuses
                    (0=missing, 1=masked, 2=observed), from
                    GraphTensors.sample_status.

        Returns:
            params: [B, L, global_param_dim]
        """
        if device is None:
            device = next(self.parameters()).device
        if self.use_embed_unembed:
            raise NotImplementedError("forward_batched only covers the dual-stream path")

        features, params, attn_mask = self._build_initial_streams_batched(graph, status, device)
        edge_mask = graph.build_edge_masks(device=device)
        K_aug = self._get_k_aug(graph, device)
        core = self._compiled_blocks if self._compiled_blocks is not None else self._dual_blocks
        return core(features, params, edge_mask, attn_mask, K_aug)

    def _build_initial_streams_batched(
        self,
        graph: EntityGraph,
        status: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched counterpart of _build_initial_streams; see forward_batched."""
        gt = graph_tensors(graph, self.types, self.param_dim, device)
        B, L = status.shape

        type_mat = torch.cat([self.type_embeddings[n] for n in gt.type_names], dim=0)  # [T, F]
        features = type_mat.index_select(0, gt.type_ids).unsqueeze(0).expand(B, -1, -1)

        for name, pos in gt.dev_positions.items():
            table = self.deviation_tables.get(name, None)
            if table is None:
                continue
            variation = self.types[name].variation
            dev = table.index_select(0, gt.dev_entity_ids[name])                       # [M, F]
            dev = dev.unsqueeze(0).expand(B, -1, -1)
            if self.training and variation.dropout_rate > 0:
                # Independent draws per augmentation as well as per token.
                keep = (torch.rand(B, dev.shape[1], 1, device=device) >= variation.dropout_rate)
                dev = dev * keep.to(dev.dtype)
            if self.config.use_deviation_norm:
                dev = self.deviation_norm(dev)
            features = features.index_add(1, pos, dev)

        hidden = gt.is_variable.unsqueeze(0) & (status != 2)                           # [B, L]
        params = torch.where(hidden.unsqueeze(-1), gt.mask_row, gt.param_obs)          # [B, L, P]
        return features, params, None                                                  # no padding

    def _outer_residual_mode(self, layer_idx: int, num_blocks: int) -> str:
        """How this layer combines its projection with the outer stream.

        Returns "add_h" (baseline outer residual), "replace" (rotated part only),
        or "add_c" (identity folded into the projection).
        """
        mode = self.dual_outer_residual
        is_last = layer_idx == num_blocks - 1
        if mode == "all":
            return "add_h"
        if mode == "none":
            return "replace"
        if mode == "not_last":
            return "replace" if is_last else "add_h"
        if mode == "none_id":
            return "add_c"
        if mode == "not_last_id":
            return "add_c" if is_last else "add_h"
        raise ValueError(
            f"Unknown dual_outer_residual: {mode!r}. Choose 'all', 'not_last', "
            "'none', 'none_id', or 'not_last_id'."
        )

    def _forward_dual(self, graph: EntityGraph, device: torch.device) -> torch.Tensor:
        """Dual-stream forward pass (original architecture)."""
        features, params, attn_mask = self._build_initial_streams(graph, device=device)
        edge_mask = graph.build_edge_masks(device=device)
        K_aug = self._get_k_aug(graph, device)
        core = self._compiled_blocks if self._compiled_blocks is not None else self._dual_blocks
        return core(features, params, edge_mask, attn_mask, K_aug)

    def _dual_blocks(
        self,
        features: torch.Tensor,
        params: torch.Tensor,
        edge_mask: torch.Tensor,
        attn_mask: torch.Tensor | None,
        K_aug: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        The block stack, as a pure tensor -> tensor function.

        Split out of _forward_dual so it can be handed to torch.compile: the
        graph-dependent work (which is Python, and cannot be traced) all happens
        before this point, and everything from here is fixed-shape tensor code.
        """
        combined = torch.cat([features, params], dim=-1)  # [1, L, model_dim]
        num_blocks = len(self.blocks)
        for layer_idx, block in enumerate(self.blocks):
            # Inner residual stream. "both" is the shipped architecture; dropping
            # these adds turns the block into a single residual unit whose body is
            # a serial attention -> FFN composition, leaving the outer add as the
            # only skip connection.
            normed_attn = block["norm_1"](combined)
            attn_out    = block["attn"](normed_attn, edge_mask=edge_mask, attn_mask=attn_mask, K_aug=K_aug)
            combined    = combined + attn_out if self.inner_residual == "both" else attn_out

            normed_ff = block["norm_2"](combined)
            z_ff      = block["ff"](normed_ff)
            combined  = combined + z_ff if self.inner_residual in ("both", "ff_only") else z_ff

            # Outer residual. `combined` already carries an inner residual stream, so
            # adding proj_out/W_param back onto features/params gives a second, parallel
            # stream -- the "double residual stream". dual_outer_residual selects which
            # layers keep that outer add; dropping it leaves only the rotated part.
            feat_update  = block["dropout_2"](block["proj_out"](combined))
            param_update = block["dropout_2"](block["W_param"](combined))
            mode = self._outer_residual_mode(layer_idx, num_blocks)
            if mode == "add_h":
                # Baseline: add onto the stream as it entered the layer.
                features = features + feat_update
                params   = params   + param_update
            elif mode == "replace":
                # Only the rotated part survives; no identity path across layers.
                features, params = feat_update, param_update
            else:  # "add_c" -- identity folded into [W1; W2]
                # h' = c + [W1; W2] c, i.e. the stacked projection is (I + M')
                # rather than M'. Same single-stream shape as "replace", but the
                # identity path is restored -- and unlike "add_h" the attention
                # and FFN output reaches the next layer unmultiplied. Weight
                # decay now pulls M' toward 0, which means toward the identity
                # layer, instead of toward the zero map.
                features = combined[..., : self.feature_dim] + feat_update
                params   = combined[..., self.feature_dim :] + param_update
            combined = torch.cat([features, params], dim=-1)

        if self.use_param_output_head:
            return self.param_output_head(combined)
        return params

    def _forward_emb(self, graph: EntityGraph, device: torch.device) -> torch.Tensor:
        """Embedding/unembedding forward pass (single residual stream)."""
        x, attn_mask = self._build_initial_emb_stream(graph, device=device)
        edge_mask    = graph.build_edge_masks(device=device)
        K_aug        = self._get_k_aug(graph, device)

        for block in self.emb_blocks:
            x = x + block["dropout"](block["attn"](block["norm_1"](x), edge_mask=edge_mask, attn_mask=attn_mask, K_aug=K_aug))
            x = x + block["dropout"](block["ff"](block["norm_2"](x)))

        # Unembed: apply per-type linear to each token's residual to recover param vectors.
        L = graph.num_tokens
        params = torch.zeros(1, L, self.global_param_dim, device=device)
        for type_name, unembedder in self.unembedders.items():
            type_indices = [i for i, tok in enumerate(graph.tokens) if tok.type_name == type_name]
            if not type_indices:
                continue
            idx_t      = torch.tensor(type_indices, dtype=torch.long, device=device)
            tok_params = unembedder(x[0, idx_t])                     # [N, param_dim]
            params[0, idx_t, : self.types[type_name].param_dim] = tok_params

        return params
