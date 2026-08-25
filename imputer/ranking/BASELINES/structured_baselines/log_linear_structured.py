"""
Multiclass log-linear (softmax) model matching the new structured factorization.

Score for target (i*, j*, k*) and candidate class y:

    score(y) =  w_y[y]
              + w_i[i*, y]
              + w_j[j*, y]
              + w_k[k*, y]
              + sum_{attr-pair sources (i', v')} w_attr[i', i*, v', y]
              + sum_{CHANGEJ sources} count(v') * w_change_j[v', y]

Parameters:
  w_y         : (C,)           — prior log-odds on class
  w_i         : (I, C)         — attribute unigram
  w_j         : (J, C)         — annotator unigram
  w_k         : (K, C)         — item unigram
  w_attr      : (I, I, C, C)   — per (i', i*) pair; w_attr[i', i*, v', y]
  w_change_j  : (C, C)         — shared; w_change_j[v', y]

Sources are all transductive observed cells except the target (see dataset_adapter).
Training uses PyTorch Adam + CE loss. Optional val early stopping.

**Early stopping:** pass ``val_examples`` from ``build_eval_examples(bundle, "val")`` and set
``early_stopping_patience`` (default 5). Training stops when validation mean NLL does not improve
by ``min_delta`` for that many epochs; the best validation checkpoint is restored.
If ``val_examples`` is empty or ``early_stopping_patience`` is 0 or ``None``, all ``epochs`` are run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset_adapter import LocalExample
from .factor_routing import route_sources


class _LogLinearModule(nn.Module):
    def __init__(self, num_attrs: int, num_classes: int, num_anns: int, num_items: int) -> None:
        super().__init__()
        I, C, J, K = num_attrs, num_classes, num_anns, num_items
        self.I, self.C, self.J, self.K = I, C, J, K

        self.w_y = nn.Parameter(torch.zeros(C))
        self.w_i = nn.Parameter(torch.zeros(I, C))
        self.w_j = nn.Parameter(torch.zeros(J, C))
        self.w_k = nn.Parameter(torch.zeros(K, C))
        self.w_attr = nn.Parameter(torch.zeros(I, I, C, C))      # [i', i*, v', y]
        self.w_change_j = nn.Parameter(torch.zeros(C, C))        # [v', y]

        for p in self.parameters():
            nn.init.normal_(p, std=0.01)

    def forward_scores(self, ex: LocalExample, device: torch.device) -> torch.Tensor:
        """Scores over y = 0..C-1, shape (C,) on device."""
        it, jt, kt = ex.target_i, ex.target_j, ex.target_k

        s = (
            self.w_y.to(device)
            + self.w_i[it].to(device)
            + self.w_j[jt].to(device)
            + self.w_k[kt].to(device)
        )

        routed = route_sources(ex.sources, it, jt, kt)

        # ATTR_PAIR: one w_attr slice per source
        for (i_src, v_src) in routed.attr_pairs:
            s = s + self.w_attr[i_src, it, v_src].to(device)  # shape (C,)

        # CHANGE_J: weighted by multiplicity
        for v_src, cnt in routed.change_j.items():
            s = s + cnt * self.w_change_j[v_src].to(device)

        return s


@torch.no_grad()
def _mean_nll_on_examples(
    mod: _LogLinearModule,
    ex_list: Sequence[LocalExample],
    device: torch.device,
) -> float:
    if not ex_list:
        return float("nan")
    tot = 0.0
    mod.eval()
    for ex in ex_list:
        scores = mod.forward_scores(ex, device)
        y = torch.tensor(ex.y, device=device, dtype=torch.long)
        tot += float(F.cross_entropy(scores.unsqueeze(0), y.unsqueeze(0)).item())
    mod.train()
    return tot / len(ex_list)


@dataclass
class StructuredLogLinear:
    """Trained softmax linear model; wraps torch state after fit."""

    num_attrs: int
    num_classes: int
    num_anns: int
    num_items: int
    module: _LogLinearModule
    device: str
    epochs_ran: int = 0
    best_val_mean_nll: float | None = None
    stopped_early: bool = False

    @classmethod
    def fit(
        cls,
        examples: Sequence[LocalExample],
        num_attrs: int,
        num_classes: int,
        num_anns: int,
        num_items: int,
        *,
        val_examples: Sequence[LocalExample] | None = None,
        epochs: int = 30,
        lr: float = 0.05,
        batch_size: int = 256,
        device: Optional[str] = None,
        verbose: bool = False,
        show_progress: bool = True,
        tqdm_batches: bool = False,
        tqdm_desc: Optional[str] = None,
        early_stopping_patience: int | None = 5,
        min_delta: float = 0.0,
    ) -> "StructuredLogLinear":
        """Train with Adam. Optional validation early stopping restores best weights."""
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        mod = _LogLinearModule(num_attrs, num_classes, num_anns, num_items).to(dev)
        opt = torch.optim.Adam(mod.parameters(), lr=lr)
        ex_list = list(examples)
        n = len(ex_list)
        val_list = list(val_examples) if val_examples is not None else []
        use_es = early_stopping_patience not in (None, 0) and len(val_list) > 0
        use_tqdm = bool(show_progress and tqdm is not None)
        desc = tqdm_desc or "Log-linear train"

        best_val = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_no_improve = 0
        stopped_early = False
        epochs_ran = 0

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
            mod.train()
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

            epochs_ran = ep + 1

            if use_es:
                val_nll = _mean_nll_on_examples(mod, val_list, dev)
                if val_nll < best_val - min_delta:
                    best_val = val_nll
                    best_state = {k: v.detach().clone() for k, v in mod.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= int(early_stopping_patience):
                        stopped_early = True
                        if use_tqdm:
                            epoch_pbar.close()
                        break

        if use_tqdm and not stopped_early:
            epoch_pbar.close()

        if use_es and best_state is not None:
            mod.load_state_dict(best_state)

        best_val_out: float | None = None
        if use_es and best_state is not None:
            best_val_out = float(best_val) if best_val < float("inf") else None

        return cls(
            num_attrs=num_attrs,
            num_classes=num_classes,
            num_anns=num_anns,
            num_items=num_items,
            module=mod,
            device=str(dev),
            epochs_ran=epochs_ran,
            best_val_mean_nll=best_val_out,
            stopped_early=stopped_early,
        )

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

    def evaluate(self, examples: Sequence[LocalExample]) -> dict[str, float]:
        probs = self.predict_proba(examples)
        y = np.array([ex.y for ex in examples], dtype=np.int64)
        pred = probs.argmax(axis=1)
        acc = float((pred == y).mean()) if len(y) else float("nan")
        nll = float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean()) if len(y) else float("nan")
        out: dict[str, float] = {"accuracy": acc, "mean_nll": nll, "n": float(len(y))}
        if self.best_val_mean_nll is not None:
            out["best_val_mean_nll_fit"] = float(self.best_val_mean_nll)
        out["epochs_ran"] = float(self.epochs_ran)
        out["stopped_early"] = self.stopped_early
        return out
