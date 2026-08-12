"""Config-derived helpers, used by entity_mf/train.py and entity_mf/test.py."""

from typing import Dict, Any


def sizes_from_configs(configs: Dict[str, Any]) -> Dict[str, int]:
    """Extract sizes from configs.json (datagen section).

    Supports two conventions:
    - Item-split bundles: ``K_train``, ``K_test``, optional ``K_val``.
    - Annotator-test bundles: single ``K`` (all items); annotator splits use
      ``J_train`` / ``J_val`` / ``J_test`` and do not replace item counts.
    """
    if "datagen" not in configs:
        raise ValueError("configs.json missing 'datagen' section")
    dg = configs["datagen"]
    if "K_train" in dg and "K_test" in dg:
        num_items = int(dg["K_train"]) + int(dg.get("K_val", 0)) + int(dg["K_test"])
    elif "K" in dg:
        num_items = int(dg["K"])
    else:
        raise ValueError(
            "configs.datagen must specify item counts: either (K_train, K_test) or K"
        )
    required = ["I", "J", "C"]
    missing = [k for k in required if k not in dg]
    if missing:
        raise ValueError(f"configs.datagen missing keys: {missing}")
    return {
        "num_items": num_items,
        "num_attributes": int(dg["I"]),
        "num_annotators": int(dg["J"]),
        "num_likert_classes": int(dg["C"]),
    }
