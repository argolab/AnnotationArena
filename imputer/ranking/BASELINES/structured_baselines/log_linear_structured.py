"""
Multiclass log-linear (softmax) model with the same structured features as the NB baseline.

For target attribute i* and candidate class y:

    score(y) = w_unigram[i*, y] + sum_{sources r} w_bigram[i_src, v_src, i*, y, rel_r]

Indexed implementation (no sparse sklearn pipeline). Trained with PyTorch Adam + CE loss.

Training shows an epoch tqdm bar by default (``pip install tqdm`` if missing). Use
``show_progress=False`` or ``tqdm_batches=True`` (per-epoch batch bar) as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset_adapter import LocalExample
from .feature_utils import NUM_RELATIONS, relation_label


class _LogLinearModule(nn.Module):
    def __init__(self, num_attrs: int, num_classes: int, num_rel: int = NUM_RELATIONS) -> None:
        super().__init__()
        self.I = num_attrs
        self.C = num_classes
        self.R = num_rel
        self.w_uni = nn.Parameter(torch.zeros(num_attrs, num_classes))
        self.w_bi = nn.Parameter(torch.zeros(num_attrs, num_classes, num_attrs, num_classes, num_rel))
        nn.init.normal_(self.w_uni, std=0.01)
        nn.init.normal_(self.w_bi, std=0.01)

    def forward_scores(self, ex: LocalExample, device: torch.device) -> torch.Tensor:
        """Scores over y = 0..C-1, shape (C,) on device."""
        it = ex.target_i
        s = self.w_uni[it].to(device)
        for (i_s, j_s, k_s, v_s) in ex.sources:
            rel = relation_label(i_s, j_s, k_s, it, ex.target_j, ex.target_k)
            s = s + self.w_bi[i_s, v_s, it, :, rel].to(device)
        return s


@dataclass
class StructuredLogLinear:
    """Trained softmax linear model; wraps torch state after fit."""

    num_attrs: int
    num_classes: int
    module: _LogLinearModule
    device: str

    @classmethod
    def fit(
        cls,
        examples: Sequence[LocalExample],
        num_attrs: int,
        num_classes: int,
        *,
        epochs: int = 30,
        lr: float = 0.05,
        batch_size: int = 256,
        device: Optional[str] = None,
        verbose: bool = False,
        show_progress: bool = True,
        tqdm_batches: bool = False,
        tqdm_desc: Optional[str] = None,
    ) -> "StructuredLogLinear":
        """Train with Adam. Epoch tqdm updates postfix ``mean_nll`` (train CE per epoch)."""
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        mod = _LogLinearModule(num_attrs, num_classes).to(dev)
        opt = torch.optim.Adam(mod.parameters(), lr=lr)
        ex_list = list(examples)
        n = len(ex_list)
        use_tqdm = bool(show_progress and tqdm is not None)
        desc = tqdm_desc or "Log-linear train"

        epoch_pbar: Any
        if use_tqdm:
            epoch_pbar = tqdm(
                range(epochs),
                desc=desc,
                unit="epoch",
                leave=False,
                position=0,
            )
        else:
            epoch_pbar = range(epochs)

        for ep in epoch_pbar:
            perm = torch.randperm(n) if n else torch.tensor([], dtype=torch.long)
            total_loss = 0.0
            steps = 0
            n_batches = (n + batch_size - 1) // batch_size if n else 0
            if tqdm_batches and use_tqdm and n_batches > 0:
                batch_iter = tqdm(
                    range(0, n, batch_size),
                    desc=f"{desc} batches",
                    leave=False,
                    unit="batch",
                    total=n_batches,
                    position=1,
                )
            else:
                batch_iter = range(0, n, batch_size)
            for start in batch_iter:
                idxs = perm[start : start + batch_size]
                if int(idxs.numel()) == 0:
                    continue
                loss_acc = torch.zeros((), device=dev)
                for bi in idxs.tolist():
                    ex = ex_list[bi]
                    scores = mod.forward_scores(ex, dev)
                    y = torch.tensor(ex.y, device=dev, dtype=torch.long)
                    loss_acc = loss_acc + F.cross_entropy(scores.unsqueeze(0), y.unsqueeze(0))
                loss = loss_acc / float(idxs.numel())
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.detach().cpu())
                steps += 1
            mean_ep = total_loss / steps if steps else 0.0
            if use_tqdm:
                epoch_pbar.set_postfix(mean_nll=f"{mean_ep:.4f}", refresh=True)
            if verbose and steps:
                print(f"  epoch {ep+1}/{epochs}  mean_nll_batch={mean_ep:.4f}")
        if use_tqdm:
            epoch_pbar.close()
        return cls(num_attrs=num_attrs, num_classes=num_classes, module=mod, device=str(dev))

    def predict_proba(self, examples: Sequence[LocalExample]) -> np.ndarray:
        dev = torch.device(self.device)
        self.module.to(dev)
        self.module.eval()
        out = np.zeros((len(examples), self.num_classes), dtype=np.float64)
        with torch.no_grad():
            for t, ex in enumerate(examples):
                logits = self.module.forward_scores(ex, dev)
                p = F.softmax(logits, dim=-1).cpu().numpy()
                out[t] = p
        return out

    def predict(self, examples: Sequence[LocalExample]) -> np.ndarray:
        return np.argmax(self.predict_proba(examples), axis=1)

    def evaluate(self, examples: Sequence[LocalExample]) -> Dict[str, float]:
        probs = self.predict_proba(examples)
        y = np.array([ex.y for ex in examples], dtype=np.int64)
        pred = probs.argmax(axis=1)
        acc = float((pred == y).mean()) if len(y) else float("nan")
        nll = float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean()) if len(y) else float("nan")
        return {"accuracy": acc, "mean_nll": nll, "n": float(len(y))}
