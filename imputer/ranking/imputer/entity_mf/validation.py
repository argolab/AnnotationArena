"""
Carve a validation split out of the training partition.

Domain 3 ships with an empty validation split (`K_val = 0`, `J_val_split = 0`),
so nothing in this repo can stop early or select a checkpoint honestly: the
"best" checkpoint the paper's figures use is an argmin taken on the *test* set
(discrepancy D14). This module makes a real validation split, at load time, by
relabelling part of the training partition -- the bundle files on disk are never
touched, so every arm still trains on byte-identical data.

Two protocols, because they measure different things:

  block  Hold out a block of items (item axis) or annotators (annotator axis).
         The test block is exactly that -- a block of unseen items, or of unseen
         annotators -- so a block validation set poses the same question the
         test set does, and is the right thing to select on.

  mcar   Hide a random subset of observed training cells. This mirrors how the
         missing cells were produced at generation time, but it asks the
         matrix-completion question ("fill in a cell of an item you have already
         seen"), not the generalisation question. Kept as a comparison, because
         the gap between the two is worth knowing.

Both remove training signal, so the caller is expected to select on the carved
run and then refit on the whole training partition -- which is what keeps the
final model's data budget equal to Stan's.

Note for anyone tempted by a third option: the *already-missing* training cells
are scored every epoch and never trained on, so they are free held-out data.
They are a legitimate signal, but on the item axis they sit on training items,
so they answer the completion question rather than the generalisation one.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List


def _entity_key(axis: str) -> str:
    if axis == "item":
        return "item"
    if axis in ("annot", "annotator"):
        return "annotator"
    raise ValueError(f"unknown axis for block validation: {axis!r}")


def infer_axis(bundle: Any) -> str:
    md = getattr(bundle, "domain3_metadata", None) or {}
    axis = md.get("experiment_axis")
    if axis:
        return "annot" if axis.startswith("annot") else "item"
    raise ValueError("bundle has no domain3_metadata.experiment_axis; pass --val-axis")


def carve_validation(bundle: Any, protocol: str, frac: float, seed: int,
                     axis: str | None = None) -> Dict[str, Any]:
    """
    Move part of the training partition into the validation partition, in place.

    Returns a summary dict that the caller should record in train_config.json --
    without it the split is irreproducible, and a validation number nobody can
    reproduce is worse than no validation number.
    """
    if protocol in ("none", "", None):
        return {"protocol": "none"}
    if not 0.0 < frac < 1.0:
        raise ValueError(f"--val-frac must be in (0, 1), got {frac}")

    observed: List[dict] = bundle.observed_ratings
    missing: List[dict] = bundle.missing_ratings
    rng = random.Random(seed)

    if protocol == "block":
        axis = axis or infer_axis(bundle)
        key = _entity_key(axis)
        entities = sorted({r[key] for r in observed + missing if r["instance"] == "train"})
        n_val = int(round(len(entities) * frac))
        if len(entities) and n_val == 0:
            n_val = 1
        chosen = set(rng.sample(entities, n_val)) if n_val else set()
        moved_obs = moved_mis = 0
        for r in observed:
            if r["instance"] == "train" and r[key] in chosen:
                r["instance"] = "val"
                moved_obs += 1
        for r in missing:
            if r["instance"] == "train" and r[key] in chosen:
                r["instance"] = "val"
                moved_mis += 1
        return {
            "protocol": "block", "axis": axis, "frac": frac, "seed": seed,
            "n_train_entities": len(entities), "n_val_entities": len(chosen),
            "val_entities": sorted(chosen),
            "val_observed": moved_obs, "val_missing": moved_mis,
        }

    if protocol == "mcar":
        # Hide observed training cells: they leave the observed list entirely, so
        # they are as absent from training as a test-missing cell is.
        train_obs = [r for r in observed if r["instance"] == "train"]
        n_val = int(round(len(train_obs) * frac))
        if len(train_obs) and n_val == 0:
            n_val = 1
        picked = set(map(id, rng.sample(train_obs, n_val))) if n_val else set()
        keep, hidden = [], []
        for r in observed:
            if id(r) in picked:
                r["instance"] = "val"
                hidden.append(r)
            else:
                keep.append(r)
        observed[:] = keep
        missing.extend(hidden)
        return {
            "protocol": "mcar", "frac": frac, "seed": seed,
            "n_train_observed": len(train_obs),
            "val_observed": 0, "val_missing": len(hidden),
        }

    raise ValueError(f"unknown --val-protocol: {protocol!r}")


def summarise(summary: Dict[str, Any]) -> str:
    """One line for the job log."""
    p = summary.get("protocol")
    if p == "none":
        return "validation: none (training partition used whole)"
    if p == "block":
        return (f"validation: {summary['n_val_entities']}/{summary['n_train_entities']} "
                f"{summary['axis']} entities held out "
                f"({summary['val_observed']} observed + {summary['val_missing']} missing cells)")
    return (f"validation: {summary['val_missing']}/{summary['n_train_observed']} "
            f"observed training cells hidden (MCAR)")
