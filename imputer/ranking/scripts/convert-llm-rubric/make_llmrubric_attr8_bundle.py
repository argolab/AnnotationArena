#!/usr/bin/env python3
"""
Create an 8-attribute LLMRubric bundle by dropping attribute 9 from the
existing largest 175-item split, while keeping every other aspect identical.

Input:
  DATA/LLM_RUBRIC/LLMRubric_225_25_9_175/

Output:
  DATA/LLM_RUBRIC/LLMRubric_225_25_8_175/
"""

from __future__ import annotations

import json
from pathlib import Path


RANKING_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = RANKING_ROOT / "DATA" / "LLM_RUBRIC" / "LLMRubric_225_25_9_175"
OUTPUT_DIR = RANKING_ROOT / "DATA" / "LLM_RUBRIC" / "LLMRubric_225_25_8_175"
DROP_ATTRIBUTE = 9
NEW_ATTRIBUTE_COUNT = 8


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _filter_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if int(row["attribute"]) != DROP_ATTRIBUTE]


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input bundle directory not found: {INPUT_DIR}")

    bundle = _load_json(INPUT_DIR / "data_bundle.json")
    configs = _load_json(INPUT_DIR / "configs.json")

    all_ratings = _filter_rows(bundle["all_ratings"])
    observed_ratings = _filter_rows(bundle["observed_ratings"])
    missing_ratings = _filter_rows(bundle["missing_ratings"])

    for split_name, rows in (
        ("all_ratings", all_ratings),
        ("observed_ratings", observed_ratings),
        ("missing_ratings", missing_ratings),
    ):
        bad = [row for row in rows if not 1 <= int(row["attribute"]) <= NEW_ATTRIBUTE_COUNT]
        if bad:
            raise ValueError(f"{split_name} contains attributes outside 1..{NEW_ATTRIBUTE_COUNT}")

    stats = dict(bundle["stats"])
    stats["I"] = NEW_ATTRIBUTE_COUNT
    stats["total_ratings"] = len(all_ratings)
    stats["observed_ratings"] = len(observed_ratings)
    stats["missing_ratings"] = len(missing_ratings)

    new_bundle = {
        **bundle,
        "all_ratings": all_ratings,
        "observed_ratings": observed_ratings,
        "missing_ratings": missing_ratings,
        "stats": stats,
    }

    datagen = dict(configs.get("datagen", configs))
    datagen["I"] = NEW_ATTRIBUTE_COUNT
    new_configs = {"datagen": datagen}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT_DIR / "data_bundle.json", new_bundle)
    _write_json(OUTPUT_DIR / "configs.json", new_configs)

    print(f"Wrote {OUTPUT_DIR}")
    print(f"  total_ratings   : {len(all_ratings)}")
    print(f"  observed_ratings: {len(observed_ratings)}")
    print(f"  missing_ratings : {len(missing_ratings)}")
    print(f"  attributes      : {NEW_ATTRIBUTE_COUNT}")


if __name__ == "__main__":
    main()
