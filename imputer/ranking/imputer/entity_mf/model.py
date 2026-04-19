from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imputer.transformer import NormLayer, FeedForward

from .config import EntityMarformerConfig
from .data import EntityGraph
from .types import EntityType

_TRIPLET_EPS = 1e-8


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
            Q_content = Q_full[..., :kc]                          # [B, H, L, kc]
            Q_rel     = Q_full[..., kc:]                          # [B, H, L, R]
            content_scores = torch.matmul(Q_content, K_full.transpose(-2, -1))       # [B, H, L, L]
            rel_scores     = torch.einsum("bhir,ijr->bhij", Q_rel, edge_mask_f)      # [B, H, L, L]
            scores = (content_scores + rel_scores) / math.sqrt(hd)
        else:
            # ── Shared-bias relational attention (old design) ───────────────────
            # Q: D -> D+R (+ 3 for pointer); K: D -> D; rel bias shared across heads
            Q_proj = self.Q(x)                                     # [B, L, D+R] or [B, L, D+R+3]
            Q_content = Q_proj[..., :D]                            # [B, L, D]
            Q_rel_shared = Q_proj[..., D:D+R]                     # [B, L, R]  (shared across heads)
            if self.use_pointer:
                Q_ptr = Q_proj[..., D+R:]                          # [B, L, 3]
            K_content = self.K(x)                                  # [B, L, D]
            Qh = Q_content.view(B, L, H, hd).transpose(1, 2)     # [B, H, L, hd]
            Kh = K_content.view(B, L, H, hd).transpose(1, 2)     # [B, H, L, hd]
            content_scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(hd) # [B, H, L, L]
            # Relational bias: [B, L, L] -> broadcast over heads [B, 1, L, L]
            rel_scores = (Q_rel_shared.unsqueeze(2) * edge_mask_f.unsqueeze(0)).sum(-1).unsqueeze(1)
            if self.scale_shared_rel:
                rel_scores = rel_scores / math.sqrt(hd)
            scores = content_scores + rel_scores

        # ── Pointer bias (shared across heads) ─────────────────────────────────
        if self.use_pointer and K_aug is not None:
            # Q_ptr: [B, L, 3], K_aug: [L, L, 3]
            ptr_bias = (Q_ptr.unsqueeze(2) * K_aug.unsqueeze(0)).sum(-1)  # [B, L, L]
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


class EntityMarformer(nn.Module):
    """
    Minimal viable Entity Marformer with:
      - feature stream updated by one FFN
      - param stream updated by a separate FFN
      - relational attention over a unified token sequence.
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
        self.use_triplet_rating_base = bool(config.use_triplet_rating_base)
        self.triplet_rating_tanh = bool(config.triplet_rating_tanh)
        if self.use_triplet_rating_base:
            self.triplet_rating_logit_scale = nn.Parameter(torch.ones(1))
            # Periodic triplet stats (set by Lightning each epoch); print at most once per epoch.
            self._triplet_diag_epoch: int = -1
            self._triplet_diag_interval: int = 10
            self._triplet_diag_logged_this_epoch: bool = False
        else:
            self.triplet_rating_logit_scale = None  # type: ignore[assignment]

        # Global param dim = max over types (no longer passed in).
        self.global_param_dim = max(t.param_dim for t in types.values())
        self.param_dim = self.global_param_dim

        # Config embedding_dim = total model dim (feature + param). Must be divisible by heads.
        self.model_dim = config.embedding_dim
        assert self.model_dim % config.attention_heads == 0, (
            f"embedding_dim {self.model_dim} must be divisible by attention_heads {config.attention_heads}"
        )
        assert self.model_dim > self.param_dim, (
            f"embedding_dim {self.model_dim} must be > global_param_dim {self.param_dim}"
        )

        self.feature_dim = self.model_dim - self.param_dim

        # Per-type base embeddings (centroids) for both entity and variable tokens.
        self.type_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(1, self.feature_dim))
                for name in types.keys()
            }
        )
        for p in self.type_embeddings.values():
            _init_type_embedding(p, config.type_embedding_init, self.feature_dim)

        # Per-entity deviations/variations (where enabled)
        self.deviation_tables = nn.ParameterDict()
        for name, t in types.items():
            if t.variation.enabled and t.variation.num_entities > 0:
                table = nn.Parameter(torch.zeros(t.variation.num_entities, self.feature_dim))
                self.deviation_tables[name] = table

        self.deviation_norm = NormLayer(self.feature_dim)
        if self.use_param_output_head:
            self.param_output_head = nn.Sequential(
                    nn.Linear(self.model_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.param_dim)
                )

        # Transformer blocks
        blocks: List[nn.Module] = []
        for _ in range(config.num_layers):
            attn = RelationalAttentionBlock(
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
            ff = FeedForward(
                self.model_dim,
                d_ff=config.d_ff,
                dropout=config.dropout,
                num_layers=config.num_ffn_layers,
            )
            block_dict = {
                "norm_1": NormLayer(self.model_dim),
                "attn": attn,
                "norm_2": NormLayer(self.model_dim),
                "ff": ff,
                "proj_out": nn.Linear(self.model_dim, self.feature_dim),
                "W_param": nn.Linear(self.model_dim, self.param_dim),
                "dropout_2": nn.Dropout(config.dropout),
            }
            blocks.append(nn.ModuleDict(block_dict))
        self.blocks = nn.ModuleList(blocks)

    @staticmethod
    def _triplet_scalars(ha: torch.Tensor, hb: torch.Tensor, hc: torch.Tensor, d_feat: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unnormalized cubic sum and a scale-fair version for logits.

        tri_unnorm = sum_d ha[d] hb[d] hc[d]  (can be huge)

        tri_norm: L2-normalize each of ha, hb, hc, then sum_d (no extra divide).
        This keeps magnitude in a usable range while avoiding width over-shrinkage.
        """
        tri_unnorm = (ha * hb * hc).sum()
        ha_n = ha / (torch.norm(ha, p=2) + _TRIPLET_EPS)
        hb_n = hb / (torch.norm(hb, p=2) + _TRIPLET_EPS)
        hc_n = hc / (torch.norm(hc, p=2) + _TRIPLET_EPS)
        tri_norm = (ha_n * hb_n * hc_n).sum()
        return tri_unnorm, tri_norm

    def _apply_triplet_rating_base(
        self,
        graph: EntityGraph,
        features: torch.Tensor,
        out_params: torch.Tensor,
    ) -> None:
        """
        For each rating variable token, add (learnable_scale * tri_norm) to param index 1
        (mu_raw), using the final feature rows of the linked attribute, annotator, and first
        item entity tokens.

        tri_norm uses L2-normalized entity features with no additional /D scaling,
        so magnitude stays informative; optional tanh applies only to this normalized
        scalar (not the raw cubic sum).
        """
        if not self.use_triplet_rating_base or self.triplet_rating_logit_scale is None:
            return
        if graph.attr_token_start is None or graph.annot_token_start is None or graph.item_id_to_token_index is None:
            return
        na = self.types["attribute"].variation.num_entities
        nj = self.types["annotator"].variation.num_entities
        ast = graph.attr_token_start
        jst = graph.annot_token_start
        inv = graph.item_id_to_token_index
        scale = self.triplet_rating_logit_scale.reshape(())
        epoch = int(getattr(self, "_triplet_diag_epoch", -1))
        interval = int(getattr(self, "_triplet_diag_interval", 10))
        want_diag = (
            self.training
            and epoch >= 0
            and interval > 0
            and (epoch % interval == 0)
            and not getattr(self, "_triplet_diag_logged_this_epoch", True)
        )
        tri_unnorm_list: List[torch.Tensor] = []
        tri_norm_list: List[torch.Tensor] = []
        tri_final_list: List[torch.Tensor] = []
        for idx, tok in enumerate(graph.tokens):
            if tok.type_name != "rating":
                continue
            raw = tok.raw_data or {}
            aid = int(raw.get("attribute_id", -1))
            jid = int(raw.get("annotator_id", -1))
            iids = raw.get("item_ids") or []
            if not iids:
                continue
            kid = int(iids[0])
            if not (0 <= aid < na and 0 <= jid < nj):
                continue
            itok = inv.get(kid)
            if itok is None:
                continue
            ia = ast + aid
            ib = jst + jid
            ha = features[0, ia, :]
            hb = features[0, ib, :]
            hc = features[0, itok, :]
            tri_unnorm, tri_norm = self._triplet_scalars(ha, hb, hc, self.feature_dim)
            tri_final = torch.tanh(tri_norm) if self.triplet_rating_tanh else tri_norm
            if want_diag:
                tri_unnorm_list.append(tri_unnorm.detach())
                tri_norm_list.append(tri_norm.detach())
                tri_final_list.append(tri_final.detach())
            out_params[0, idx, 1] = out_params[0, idx, 1] + scale * tri_final

        if want_diag:
            self._triplet_diag_logged_this_epoch = True
            sc = float(scale.detach().float().item())
            if not tri_unnorm_list:
                print(
                    f"[EntityMarformer] triplet_rating diag (epoch {epoch}, first train chunk): "
                    "no valid rating triplets in this graph — triplet base not applied here."
                )
            else:
                u0 = torch.stack(tri_unnorm_list).float()
                u1 = torch.stack(tri_norm_list).float()
                u2 = torch.stack(tri_final_list).float()
                contrib = (scale.detach().float() * u2).abs()
                std0 = float(torch.std(u0, unbiased=False).item())
                std1 = float(torch.std(u1, unbiased=False).item())
                std2 = float(torch.std(u2, unbiased=False).item())
                print(
                    f"[EntityMarformer] triplet_rating diag (epoch {epoch}, first train chunk): "
                    f"n_ratings={u0.numel()}  logit_scale={sc:.6g}  D_feat={self.feature_dim}  every_{interval}_epochs\n"
                    f"  tri_unnorm (raw cubic sum) min/mean/max/std = "
                    f"{float(u0.min().item()):.6g} / {float(u0.mean().item()):.6g} / "
                    f"{float(u0.max().item()):.6g} / {std0:.6g}\n"
                    f"  tri_norm   (L2-entity, no /D) min/mean/max/std = "
                    f"{float(u1.min().item()):.6g} / {float(u1.mean().item()):.6g} / "
                    f"{float(u1.max().item()):.6g} / {std1:.6g}\n"
                    f"  tri_final  (after optional tanh) min/mean/max/std = "
                    f"{float(u2.min().item()):.6g} / {float(u2.mean().item()):.6g} / "
                    f"{float(u2.max().item()):.6g} / {std2:.6g}  tanh={self.triplet_rating_tanh}\n"
                    f"  |scale*tri_final| mean/max = "
                    f"{float(contrib.mean().item()):.6g} / {float(contrib.max().item()):.6g}"
                )

    def _build_initial_streams(
        self,
        graph: EntityGraph,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build initial feature and param streams from the EntityGraph.

        Returns:
            features: [1, L, D_feat]
            params: [1, L, D_param]
            attn_mask: [1, L] (True for valid tokens)
        """
        L = graph.num_tokens
        features = torch.zeros(1, L, self.feature_dim, device=device)
        params = torch.zeros(1, L, self.param_dim, device=device)
        attn_mask = torch.ones(1, L, dtype=torch.bool, device=device)

        for idx, token in enumerate(graph.tokens):
            t = self.types[token.type_name]
            base = self.type_embeddings[token.type_name]  # [1, feature_dim]
            feat_vec = base.expand(1, -1)
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

            p = t.build_param(token.raw_data or {}, device=device, global_param_dim=self.param_dim)
            params[0, idx] = p

            attn_mask[0, idx] = True

        return features, params, attn_mask

    def forward(
        self,
        graph: EntityGraph,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Forward pass over a single EntityGraph.

        Returns:
            params: [1, L, D_param] final parameter stream.
        """
        if device is None:
            device = next(self.parameters()).device

        features, params, attn_mask = self._build_initial_streams(graph, device=device)
        edge_mask = graph.build_edge_masks(device=device)  # [L, L, R]

        # Build K_aug for pointer mechanism: [L, L, 3] obs-obs shared-identity indicators
        K_aug: torch.Tensor | None = None
        if self.use_pointer:
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
                        # Use first item_id (like Marformer pointer)
                        iids = token.raw_data.get("item_ids", [])
                        item_ids[idx]  = iids[0] if iids else -1
                # [L, L] indicator: 1 if same id AND both valid (id >= 0)
                def _same(ids: torch.Tensor) -> torch.Tensor:
                    eq = (ids.unsqueeze(0) == ids.unsqueeze(1)).float()  # [L, L]
                    valid = (ids >= 0).float()
                    return eq * valid.unsqueeze(0) * valid.unsqueeze(1)
                graph._k_aug_cache[dev_key] = torch.stack(
                    [_same(attr_ids), _same(annot_ids), _same(item_ids)], dim=-1
                )  # [L, L, 3]
            K_aug = graph._k_aug_cache[dev_key]

        combined = torch.cat([features, params], dim=-1)  # [1, L, model_dim]
        for block in self.blocks:
            normed_attn = block["norm_1"](combined)
            attn_out = block["attn"](
                normed_attn,
                edge_mask=edge_mask,
                attn_mask=attn_mask,
                K_aug=K_aug,
            )  # [1, L, model_dim]
            combined = combined + attn_out

            normed_ff = block["norm_2"](combined)
            z_ff = block["ff"](normed_ff)
            combined = combined + z_ff

            back_feat = block["proj_out"](combined)   # [1, L, feature_dim or model_dim]
            features = features + block["dropout_2"](back_feat)
            back_param = block["W_param"](combined)   # [1, L, param_dim]
            params = params + block["dropout_2"](back_param)
            combined = torch.cat([features, params], dim=-1)

        if self.use_param_output_head:
            out_params = self.param_output_head(combined)
        else:
            out_params = params
        if self.use_triplet_rating_base:
            self._apply_triplet_rating_base(graph, features, out_params)
        return out_params
