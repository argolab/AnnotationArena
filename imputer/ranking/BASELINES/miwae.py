"""
miwae.py
--------
CategoricalMIWAE: faithful adaptation of MIWAE (Mattei & Frellsen, 2019) for
ordinal / categorical annotation imputation.

Architecture is identical to the original (MLP encoder → Gaussian posterior,
MLP decoder, IWAE bound) except:
  - decoder outputs C*p logits (categorical) instead of 3*p (Student's t params)
  - log-likelihood uses masked cross-entropy instead of Student's t log-prob
  - imputation returns IS-weighted softmax probabilities (N, D, C) instead of
    IS-weighted mean of Student's t samples

Reference: "MIWAE: Deep Generative Modelling and Imputation of Incomplete Data Sets"
           https://github.com/pamattei/miwae
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# CategoricalMIWAE
# ─────────────────────────────────────────────────────────────────────────────

class CategoricalMIWAE:
    """
    MIWAE adapted for ordinal/categorical data.

    Inputs are 1-indexed integer ratings (1..C); values normalized to [0,1]
    before being passed to the encoder (following original MIWAE preprocessing).
    transform() returns (N, D, C) probability arrays via IS-weighted imputation.
    """

    def __init__(
        self,
        num_classes: int,
        latent_dim: int = 4,
        hidden_dim: int = 128,
        K: int = 20,            # IS samples during training (original: 20)
        L: int = 50,            # IS samples during imputation (original: 10)
        max_epochs: int = 500,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.C          = num_classes
        self.d          = latent_dim
        self.h          = hidden_dim
        self.K          = K
        self.L          = L
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.device     = torch.device(device)
        self.seed       = seed

        # Placeholders; built in fit() once p is known
        self.encoder: nn.Sequential | None = None
        self.decoder: nn.Sequential | None = None
        self.p: int | None = None           # number of features

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Map 1-indexed values [1, C] → [0, 1]."""
        return (X - 1.0) / max(self.C - 1, 1)

    def _build_networks(self, p: int):
        d, h, C = self.d, self.h, self.C

        # Encoder: p → h → h → 2d  (mean and log-scale of Gaussian posterior)
        self.encoder = nn.Sequential(
            nn.Linear(p, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 2 * d),
        ).to(self.device)

        # Decoder: d → h → h → C*p  (categorical logits per feature)
        self.decoder = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, C * p),
        ).to(self.device)

        def _ortho_init(layer):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
        self.encoder.apply(_ortho_init)
        self.decoder.apply(_ortho_init)

        self.p = p

    def _miwae_loss(
        self,
        iota_x: torch.Tensor,   # (B, p) zero-filled normalized input
        mask:   torch.Tensor,   # (B, p) float, 1=observed
        targets: torch.Tensor,  # (B, p) int64, 0-indexed class labels (-1 = missing)
    ) -> torch.Tensor:
        """IWAE bound with masked categorical log-likelihood."""
        B, p = iota_x.shape
        K, d, C = self.K, self.d, self.C

        # ── Encode ───────────────────────────────────────────────────────────
        enc = self.encoder(iota_x)                                # (B, 2d)
        mu_q    = enc[:, :d]                                      # (B, d)
        log_s_q = enc[:, d:]                                      # (B, d)
        sigma_q = F.softplus(log_s_q)

        q = td.Independent(td.Normal(mu_q, sigma_q), 1)
        p_z = td.Independent(
            td.Normal(torch.zeros(d, device=self.device),
                      torch.ones(d,  device=self.device)), 1
        )

        # ── Sample z ─────────────────────────────────────────────────────────
        z = q.rsample([K])               # (K, B, d)
        z_flat = z.reshape(K * B, d)     # (K*B, d)

        # ── Decode ───────────────────────────────────────────────────────────
        logits_flat = self.decoder(z_flat)          # (K*B, C*p)
        logits = logits_flat.reshape(K * B, p, C)   # (K*B, p, C)

        # ── Masked categorical log-likelihood ─────────────────────────────────
        # log p(x_j | z) for each observed feature j
        log_probs = F.log_softmax(logits, dim=-1)   # (K*B, p, C)

        tiled_targets = targets.unsqueeze(0).expand(K, -1, -1).reshape(K * B, p)  # (K*B, p)
        tiled_mask    = mask.unsqueeze(0).expand(K, -1, -1).reshape(K * B, p)     # (K*B, p)

        # Gather log-prob of true class; use 0 as dummy for missing positions
        safe_targets = tiled_targets.clamp(min=0)
        true_log_prob = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)  # (K*B, p)

        # Sum over observed features only
        logpx_obs = (true_log_prob * tiled_mask).sum(-1).reshape(K, B)   # (K, B)

        # ── IWAE bound ────────────────────────────────────────────────────────
        logpz = p_z.log_prob(z)              # (K, B)
        logq  = q.log_prob(z)                # (K, B)

        log_w = logpx_obs + logpz - logq     # (K, B)
        neg_bound = -torch.mean(torch.logsumexp(log_w, dim=0) - np.log(K))
        return neg_bound

    def _miwae_impute(
        self,
        iota_x: torch.Tensor,   # (B, p)
        mask:   torch.Tensor,   # (B, p)
    ) -> torch.Tensor:
        """IS-weighted imputation. Returns (B, p, C) probability distributions."""
        B, p = iota_x.shape
        L, d, C = self.L, self.d, self.C

        with torch.no_grad():
            enc = self.encoder(iota_x)
            mu_q    = enc[:, :d]
            sigma_q = F.softplus(enc[:, d:])
            q = td.Independent(td.Normal(mu_q, sigma_q), 1)
            p_z = td.Independent(
                td.Normal(torch.zeros(d, device=self.device),
                          torch.ones(d,  device=self.device)), 1
            )

            z = q.rsample([L])                  # (L, B, d)
            z_flat = z.reshape(L * B, d)

            logits_flat = self.decoder(z_flat)          # (L*B, C*p)
            logits = logits_flat.reshape(L, B, p, C)    # (L, B, p, C)
            probs  = F.softmax(logits, dim=-1)          # (L, B, p, C)

            # IS weights based on observed log-likelihood
            log_probs = F.log_softmax(logits.reshape(L * B, p, C), dim=-1)
            # We need targets for IS weight computation; use argmax of current probs
            # as a proxy. Better: reuse the observed values from iota_x.
            # For IS weights we only need the observed log-likelihood.
            # We'll pass iota_x (zero-filled normalized) and use mask to get targets.
            # Since iota_x encodes observed values as normalized floats, we can
            # reverse-engineer the observed class via rounding:
            obs_class = (iota_x * max(C - 1, 1)).round().long().clamp(0, C - 1)  # (B, p)
            tiled_obs = obs_class.unsqueeze(0).expand(L, -1, -1).reshape(L * B, p)
            tiled_mask = mask.unsqueeze(0).expand(L, -1, -1).reshape(L * B, p)

            true_lp = log_probs.gather(-1, tiled_obs.unsqueeze(-1)).squeeze(-1)   # (L*B, p)
            logpx_obs = (true_lp * tiled_mask).sum(-1).reshape(L, B)

            logpz = p_z.log_prob(z)
            logq  = q.log_prob(z)

            log_w = logpx_obs + logpz - logq            # (L, B)
            imp_weights = F.softmax(log_w, dim=0)       # (L, B) normalized IS weights

            # IS-weighted average of class probability distributions
            # imp_weights: (L, B)  probs: (L, B, p, C)
            probs_weighted = torch.einsum("lb,lbpc->bpc", imp_weights, probs)  # (B, p, C)

        return probs_weighted   # (B, p, C)

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        M: np.ndarray,
        context_cols_mask: np.ndarray | None = None,
        target_cols_mask:  np.ndarray | None = None,
        random_mask_observed: bool = False,
        train_mask_ratio: float | None = None,
    ) -> "CategoricalMIWAE":
        """
        X: (N, D) float, 1-indexed values, NaN for missing
        M: (N, D) float, 1 = observed (all annotations, both context and target)
        context_cols_mask: (D,) bool — LLM/turker cols; encoder sees only these
        target_cols_mask:  (D,) bool — human/expert cols; likelihood restricted to these
        """
        torch.manual_seed(self.seed)
        N, p = X.shape
        self._build_networks(p)

        if random_mask_observed:
            M_input = M.astype(np.float32)
            M_label = M.astype(np.float32)
        elif context_cols_mask is not None and target_cols_mask is not None:
            # Encoder input: context cols only
            M_input = (M * context_cols_mask[None, :]).astype(np.float32)
            # Likelihood mask: target cols with observed ground truth
            M_label = (M * target_cols_mask[None, :]).astype(np.float32)
        else:
            M_input = M.astype(np.float32)
            M_label = M.astype(np.float32)

        # iota_x: only context col values filled (original MIWAE: xhat_0)
        X_norm = np.where(M_input > 0.5, self._normalize(np.nan_to_num(X, nan=1.0)), 0.0).astype(np.float32)
        # 0-indexed targets only for target cols with ground truth
        targets = np.where(M_label > 0.5, np.nan_to_num(X, nan=1.0).astype(int) - 1, -1).astype(np.int64)

        X_t = torch.tensor(X_norm,  dtype=torch.float32)
        M_t = torch.tensor(M_label, dtype=torch.float32)   # likelihood over target cols
        T_t = torch.tensor(targets, dtype=torch.int64)

        dataset    = TensorDataset(X_t, M_t, T_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
        )

        self.encoder.train()
        self.decoder.train()

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            n_batches  = 0
            for x_b, m_b, t_b in dataloader:
                x_b = x_b.to(self.device)
                m_b = m_b.to(self.device)
                t_b = t_b.to(self.device)

                if random_mask_observed:
                    mask_ratio = 0.5 if train_mask_ratio is None else float(train_mask_ratio)
                    keep_prob = max(0.0, min(1.0, 1.0 - mask_ratio))
                    sampled = (torch.rand_like(m_b) < keep_prob).float() * m_b
                    row_has_input = sampled.sum(dim=1) > 0
                    if not bool(row_has_input.all()):
                        missing_rows = (~row_has_input).nonzero(as_tuple=False).flatten()
                        for row_idx in missing_rows.tolist():
                            obs_cols = (m_b[row_idx] > 0.5).nonzero(as_tuple=False).flatten()
                            if obs_cols.numel() > 0:
                                sampled[row_idx, obs_cols[0]] = 1.0
                    x_b = torch.where(sampled > 0.5, x_b, torch.zeros_like(x_b))
                    m_b = sampled

                optimizer.zero_grad()
                loss = self._miwae_loss(x_b, m_b, t_b)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"  [MIWAE] epoch {epoch+1}/{self.max_epochs}  loss={epoch_loss/n_batches:.4f}")

        self.encoder.eval()
        self.decoder.eval()
        return self

    def transform(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        X: (N, D) float, 1-indexed observed values, NaN for missing (target cols)
        M: (N, D) float, 1 = observed (context)
        Returns: (N, D, C) probability arrays
        """
        if self.encoder is None:
            raise RuntimeError("Call fit() before transform().")

        N, p = X.shape
        X_norm = np.where(M > 0.5, self._normalize(np.nan_to_num(X, nan=1.0)), 0.0).astype(np.float32)

        X_t = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        M_t = torch.tensor(M,      dtype=torch.float32).to(self.device)

        self.encoder.eval()
        self.decoder.eval()

        all_probs = []
        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            probs = self._miwae_impute(X_t[start:end], M_t[start:end])
            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)   # (N, D, C)

    def save(self, path: str):
        if self.encoder is None:
            raise RuntimeError("No model to save.")
        torch.save({
            "encoder_state": self.encoder.state_dict(),
            "decoder_state": self.decoder.state_dict(),
            "config": {
                "num_classes": self.C,
                "latent_dim":  self.d,
                "hidden_dim":  self.h,
                "K":           self.K,
                "L":           self.L,
                "p":           self.p,
            },
        }, path)
        print(f"  [MIWAE] model saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CategoricalMIWAE":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg  = ckpt["config"]
        obj  = cls(num_classes=cfg["num_classes"], latent_dim=cfg["latent_dim"],
                   hidden_dim=cfg["hidden_dim"], K=cfg["K"], L=cfg["L"], device=device)
        obj._build_networks(cfg["p"])
        obj.encoder.load_state_dict(ckpt["encoder_state"])
        obj.decoder.load_state_dict(ckpt["decoder_state"])
        obj.encoder.eval()
        obj.decoder.eval()
        return obj
