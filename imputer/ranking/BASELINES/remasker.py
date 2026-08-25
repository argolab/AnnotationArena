"""
remasker.py
-----------
CategoricalReMasker: faithful adaptation of ReMasker (Alps-Lab) for ordinal /
categorical annotation imputation.

Architecture is identical to the original (Conv1d feature embedding, sincos
position encoding, ViT encoder, mask-token decoder) except:
  - output head: Linear(decoder_embed_dim, C) instead of scalar + tanh
  - loss: cross-entropy restricted to masked+observed positions instead of MSE

Reference: "ReMasker: Imputing Tabular Data with Masked Autoencoding"
           https://github.com/alps-lab/remasker
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader, TensorDataset

try:
    from timm.models.vision_transformer import Block
except ImportError:
    raise ImportError("timm is required: pip install timm")


# ─────────────────────────────────────────────────────────────────────────────
# Positional embedding (from original ReMasker utils.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_1d_sincos_pos_embed(embed_dim: int, num_pos: int, cls_token: bool = False) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    pos = np.arange(num_pos, dtype=np.float64)
    out = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# Feature embedding (from original MaskEmbed)
# ─────────────────────────────────────────────────────────────────────────────

class MaskEmbed(nn.Module):
    """Project scalar features to embed_dim via Conv1d (kernel=1)."""
    def __init__(self, embed_dim: int, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x: (B, 1, D)
        x = self.proj(x)        # (B, embed_dim, D)
        x = x.transpose(1, 2)  # (B, D, embed_dim)
        x = self.norm(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Masked Autoencoder (categorical version)
# ─────────────────────────────────────────────────────────────────────────────

class CategoricalMAE(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embed_dim: int = 32,
        depth: int = 4,
        num_heads: int = 4,
        decoder_embed_dim: int = 32,
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        eps = 1e-6
        norm_layer = partial(nn.LayerNorm, eps=eps)

        self.num_features = num_features
        self.num_classes  = num_classes

        # ── Encoder ──────────────────────────────────────────────────────────
        self.mask_embed = MaskEmbed(embed_dim, norm_layer)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(
            torch.zeros(1, num_features + 1, embed_dim), requires_grad=False
        )
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_features + 1, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        # Categorical head: one logit per class per feature
        self.decoder_pred = nn.Linear(decoder_embed_dim, num_classes, bias=True)

        self._init_weights()

    def _init_weights(self):
        pos = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_features, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        dec_pos = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_features, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec_pos).float().unsqueeze(0))

        w = self.mask_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_linear_ln)

    def _init_linear_ln(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _random_masking(self, x, m_input):
        """
        x:       (N, L, D_emb)
        m_input: (N, L) — 1 for context cols (always keep), 0 for all others
        Always keeps context tokens, always masks non-context tokens.
        At eval: same behaviour (M_test already has only context cols).
        """
        N, L, D = x.shape

        # Context tokens always small noise → sorted to front → kept.
        # Non-context tokens always noise=1.0 → sorted to back → removed.
        noise = torch.rand(N, L, device=x.device) * 0.5   # [0, 0.5) for context
        noise[m_input < 0.5] = 1.0

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        len_keep = max(int(m_input.sum(dim=1).min().item()), 1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_masked, ids_restore

    def forward_encoder(self, x, m_input):
        x = self.mask_embed(x)          # (N, D, embed_dim)
        x = x + self.pos_embed[:, 1:, :]

        x, ids_restore = self._random_masking(x, m_input)

        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        n_mask = ids_restore.shape[1] + 1 - x.shape[1]
        mask_tokens = self.mask_token.expand(x.shape[0], n_mask, -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x  = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Categorical logits per position
        x = self.decoder_pred(x)   # (N, 1+D, C)
        x = x[:, 1:, :]            # (N, D, C)
        return x

    def forward_loss(self, targets_int, logits, m_label):
        """
        targets_int: (N, D) int64, 0-indexed class labels (-1 for positions with no GT)
        logits:      (N, D, C)
        m_label:     (N, D) float — 1 for target (human/expert) cols with ground truth
        Loss is computed only on target columns that have a ground-truth label.
        """
        N, D, C = logits.shape
        valid = ((targets_int >= 0).float() * (m_label > 0.5).float())

        ce = F.cross_entropy(
            logits.reshape(N * D, C),
            targets_int.clamp(min=0).reshape(N * D),
            reduction="none"
        ).reshape(N, D)

        return (ce * valid).sum() / (valid.sum() + 1e-8)

    def forward(self, x_norm, m_input, m_label, targets_int):
        """
        x_norm:   (N, 1, D) normalized float — context cols filled, targets zeroed
        m_input:  (N, D) float — 1 for context (LLM/turker) cols
        m_label:  (N, D) float — 1 for target (human/expert) cols with ground truth
        targets_int: (N, D) int64 — 0-indexed labels, -1 where no ground truth
        """
        latent, ids_restore = self.forward_encoder(x_norm, m_input)
        logits = self.forward_decoder(latent, ids_restore)
        loss   = self.forward_loss(targets_int, logits, m_label)
        return loss, logits


# ─────────────────────────────────────────────────────────────────────────────
# Public wrapper: CategoricalReMasker
# ─────────────────────────────────────────────────────────────────────────────

class CategoricalReMasker:
    """
    ReMasker adapted for ordinal/categorical data.

    Inputs are 1-indexed integer ratings (1..C); internally normalized to [0,1].
    fit() trains the model; transform() returns (N, D, C) probability arrays.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 32,
        depth: int = 4,
        num_heads: int = 4,
        decoder_embed_dim: int = 32,
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.5,
        max_epochs: int = 300,
        batch_size: int = 64,
        lr: float = 1e-3,
        min_lr: float = 1e-5,
        warmup_epochs: int = 20,
        weight_decay: float = 0.05,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.C            = num_classes
        self.embed_dim    = embed_dim
        self.depth        = depth
        self.num_heads    = num_heads
        self.dec_embed    = decoder_embed_dim
        self.dec_depth    = decoder_depth
        self.dec_heads    = decoder_num_heads
        self.mlp_ratio    = mlp_ratio
        self.mask_ratio   = mask_ratio
        self.max_epochs   = max_epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.min_lr       = min_lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.device       = torch.device(device)
        self.seed         = seed
        self.model: CategoricalMAE | None = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Map 1-indexed values in [1, C] to [0, 1]."""
        return (X - 1.0) / max(self.C - 1, 1)

    def _adjust_lr(self, optimizer, epoch):
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    def fit(
        self,
        X: np.ndarray,
        M: np.ndarray,
        context_cols_mask: np.ndarray | None = None,
        target_cols_mask:  np.ndarray | None = None,
        random_mask_observed: bool = False,
        train_mask_ratio: float | None = None,
    ) -> "CategoricalReMasker":
        """
        X: (N, D) float, 1-indexed values, NaN for missing
        M: (N, D) float, 1 = observed (all annotations, both context and target)
        context_cols_mask: (D,) bool — LLM/turker columns, always in encoder input
        target_cols_mask:  (D,) bool — human/expert columns, always masked from encoder;
                           loss is computed only on these where ground truth exists.
        If masks not provided, falls back to treating all observed cols as context (original behaviour).
        """
        torch.manual_seed(self.seed)
        N, D = X.shape

        if random_mask_observed:
            M_input = M.astype(np.float32)
            M_label = M.astype(np.float32)
        elif context_cols_mask is not None and target_cols_mask is not None:
            # Encoder input: context cols only (target cols zeroed)
            M_input = (M * context_cols_mask[None, :]).astype(np.float32)
            # Loss mask: target cols with observed ground truth
            M_label = (M * target_cols_mask[None, :]).astype(np.float32)
        else:
            M_input = M.astype(np.float32)
            M_label = M.astype(np.float32)

        # Normalize: only context cols filled, targets zeroed
        X_norm = np.where(M_input > 0.5, self._normalize(np.nan_to_num(X, nan=1.0)), 0.0).astype(np.float32)

        # Integer targets: valid only for target cols with observed ground truth
        targets = np.where(M_label > 0.5, np.nan_to_num(X, nan=1.0).astype(int) - 1, -1).astype(np.int64)

        X_t   = torch.tensor(X_norm,  dtype=torch.float32)
        MI_t  = torch.tensor(M_input, dtype=torch.float32)
        ML_t  = torch.tensor(M_label, dtype=torch.float32)
        T_t   = torch.tensor(targets, dtype=torch.int64)

        dataset    = TensorDataset(X_t, MI_t, ML_t, T_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = CategoricalMAE(
            num_features=D,
            num_classes=self.C,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            decoder_embed_dim=self.dec_embed,
            decoder_depth=self.dec_depth,
            decoder_num_heads=self.dec_heads,
            mlp_ratio=self.mlp_ratio,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            betas=(0.9, 0.95), weight_decay=self.weight_decay
        )

        self.model.train()
        for epoch in range(self.max_epochs):
            self._adjust_lr(optimizer, epoch)
            epoch_loss = 0.0
            n_batches  = 0
            for x_b, mi_b, ml_b, t_b in dataloader:
                x_b  = x_b.unsqueeze(1).to(self.device)   # (B, 1, D)
                mi_b = mi_b.to(self.device)
                ml_b = ml_b.to(self.device)
                t_b  = t_b.to(self.device)

                if random_mask_observed:
                    mask_ratio = self.mask_ratio if train_mask_ratio is None else float(train_mask_ratio)
                    keep_prob = max(0.0, min(1.0, 1.0 - mask_ratio))
                    sampled = (torch.rand_like(mi_b) < keep_prob).float() * mi_b
                    row_has_input = sampled.sum(dim=1) > 0
                    if not bool(row_has_input.all()):
                        missing_rows = (~row_has_input).nonzero(as_tuple=False).flatten()
                        for row_idx in missing_rows.tolist():
                            obs_cols = (mi_b[row_idx] > 0.5).nonzero(as_tuple=False).flatten()
                            if obs_cols.numel() > 0:
                                sampled[row_idx, obs_cols[0]] = 1.0
                    mi_b = sampled

                optimizer.zero_grad()
                loss, _ = self.model(x_b, mi_b, ml_b, t_b)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"  [ReMasker] epoch {epoch+1}/{self.max_epochs}  loss={epoch_loss/n_batches:.4f}")

        return self

    @torch.no_grad()
    def transform(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        X: (N, D) float, 1-indexed observed values, NaN for missing
        M: (N, D) float, 1 = observed (context), 0 = missing (to predict)
        Returns: (N, D, C) probability arrays (softmax of logits)
        """
        if self.model is None:
            raise RuntimeError("Call fit() before transform().")

        N, D = X.shape
        X_norm = np.where(M > 0.5, self._normalize(np.nan_to_num(X, nan=1.0)), 0.0).astype(np.float32)
        # Dummy targets (not used in eval path but needed for forward signature)
        targets = np.full((N, D), -1, dtype=np.int64)

        X_t = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(1).to(self.device)  # (N,1,D)
        M_t = torch.tensor(M,      dtype=torch.float32).to(self.device)
        T_t = torch.tensor(targets, dtype=torch.int64).to(self.device)

        self.model.eval()
        # Dummy m_label and targets — not used in transform (no loss needed)
        ML_t = torch.zeros_like(M_t)
        # Process in batches to avoid OOM on large datasets
        all_probs = []
        for start in range(0, N, self.batch_size):
            end  = min(start + self.batch_size, N)
            _, logits = self.model(X_t[start:end], M_t[start:end], ML_t[start:end], T_t[start:end])
            probs = F.softmax(logits, dim=-1)   # (B, D, C)
            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)   # (N, D, C)

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("No model to save.")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "num_features":        self.model.num_features,
                "num_classes":         self.C,
                "embed_dim":           self.embed_dim,
                "depth":               self.depth,
                "num_heads":           self.num_heads,
                "decoder_embed_dim":   self.dec_embed,
                "decoder_depth":       self.dec_depth,
                "decoder_num_heads":   self.dec_heads,
                "mlp_ratio":           self.mlp_ratio,
            },
        }, path)
        print(f"  [ReMasker] model saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CategoricalReMasker":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg  = ckpt["config"]
        obj  = cls(num_classes=cfg["num_classes"], device=device)
        obj.model = CategoricalMAE(**cfg).to(torch.device(device))
        obj.model.load_state_dict(ckpt["model_state"])
        obj.model.eval()
        return obj
