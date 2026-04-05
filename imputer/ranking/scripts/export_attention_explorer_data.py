#!/usr/bin/env python3
"""
Export attention tensors + token metadata for the static HTML attention explorer.

Builds the same rating-only, max_item–chunked EntityGraph as analyze_pointer_attention.py,
runs a forward with full per-layer attn_mean, and writes:

  meta.json, tokens.json, edge_mask.bin, k_aug.bin, attn_layer{L}.bin (float32 tensor format),
  + copies attention_explorer_static/* (index.html, app.js, style.css, tensor_bin.js).
  For **three-logit** plots and NPZ, run ``analyze_pointer_attention.py`` with the same ``--run-dir``,
  ``--partition``, ``--max-item``, and ``--chunk-index`` (model flags come from ``train_config.json``).
  ``meta.json`` and the HTML “See also” panel reference ``--diagnostics-output-dir`` (default ``<run-dir>/attn_diagnostics``).

Also writes **attention_explorer_standalone.html**: one self-contained file (inline CSS/JS + base64
tensors) that works from ``file://`` after download—no separate ``.bin``/``.json`` files needed.

For the multi-file layout, serve over HTTP (``fetch``), e.g.:

    cd <output-dir> && python -m http.server 8765

Then open http://localhost:8765/index.html
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import struct
from itertools import product
from pathlib import Path

import numpy as np
import torch

import sys

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import analyze_pointer_attention as _apa

build_partition_graphs = _apa.build_partition_graphs
load_model = _apa.load_model
normalize_cli_path = _apa.normalize_cli_path
resolve_run_dir = _apa.resolve_run_dir

STATIC_SRC = Path(__file__).resolve().parent / "attention_explorer_static"

REL_NAMES = ["ATTR", "ATTR_INV", "ANNOT", "ANNOT_INV", "ITEM", "ITEM_INV"]
PTR_NAMES = ["ptr_attr", "ptr_annot", "ptr_item"]


def _rating_overlap_key(si: bool, sj: bool, sk: bool) -> str:
    """Disjoint bucket key for rating–rating overlap (must match app.js ratingOverlapBucket)."""
    c = int(si) + int(sj) + int(sk)
    if c == 0:
        return "r_none"
    if c == 3:
        return "r_ijk"
    if c == 1:
        return "r_only_i" if si else "r_only_j" if sj else "r_only_k"
    if si and sj and not sk:
        return "r_ij"
    if si and not sj and sk:
        return "r_ik"
    if not si and sj and sk:
        return "r_jk"
    return "r_other"


def _assert_disjoint_overlap_keys() -> None:
    seen = {}
    for si, sj, sk in product((False, True), repeat=3):
        k = _rating_overlap_key(si, sj, sk)
        seen.setdefault(k, []).append((si, sj, sk))
    assert len(seen) == 8, f"expected 8 disjoint patterns, got {len(seen)}: {seen.keys()}"
    for k, v in seen.items():
        assert len(v) == 1, f"duplicate mapping for {k}: {v}"


def tensor_to_bin_bytes(arr: np.ndarray) -> bytes:
    """Same payload as write_tensor_bin, returned as bytes."""
    a = np.asarray(arr, dtype=np.float32)
    shape = a.shape
    parts = [b"ATND", struct.pack("<B", len(shape))]
    for s in shape:
        parts.append(struct.pack("<I", int(s)))
    parts.append(a.tobytes())
    return b"".join(parts)


def write_tensor_bin(path: Path, arr: np.ndarray) -> None:
    """Little-endian header ATND + ndim + shape (uint32) + float32 row-major data."""
    with open(path, "wb") as f:
        f.write(tensor_to_bin_bytes(arr))


def _build_k_aug_cpu(graph, L: int) -> np.ndarray:
    """Replicate EntityMarformer K_aug on CPU (float32 0/1)."""
    attr_ids = np.full(L, -1, dtype=np.int64)
    annot_ids = np.full(L, -1, dtype=np.int64)
    item_ids = np.full(L, -1, dtype=np.int64)
    for idx, token in enumerate(graph.tokens):
        if token.type_name in ("rating", "ranking_pairwise") and token.raw_data:
            attr_ids[idx] = int(token.raw_data.get("attribute_id", -1))
            annot_ids[idx] = int(token.raw_data.get("annotator_id", -1))
            iids = token.raw_data.get("item_ids", [])
            item_ids[idx] = int(iids[0]) if iids else -1

    def same(ids: np.ndarray) -> np.ndarray:
        eq = (ids.reshape(1, -1) == ids.reshape(-1, 1)).astype(np.float32)
        valid = (ids >= 0).astype(np.float32)
        return eq * valid.reshape(1, -1) * valid.reshape(-1, 1)

    k0 = same(attr_ids)
    k1 = same(annot_ids)
    k2 = same(item_ids)
    return np.stack([k0, k1, k2], axis=-1).astype(np.float32)


def token_records(graph) -> list[dict]:
    out = []
    for i, t in enumerate(graph.tokens):
        rec: dict = {"i": i, "type": t.type_name, "entityId": int(t.entity_id)}
        if t.type_name == "rating" and t.raw_data:
            item = t.raw_data.get("item_ids", [None])[0]
            aj = int(t.raw_data.get("annotator_id", -1))
            ak = int(t.raw_data.get("attribute_id", -1))
            rec["itemId"] = int(item) if item is not None else -1
            rec["annotatorId"] = aj
            rec["attributeId"] = ak
            rec["label"] = f"r item={rec['itemId']} ann={aj} attr={ak}"
        elif t.type_name == "item":
            rec["label"] = f"item e={t.entity_id}"
        elif t.type_name == "attribute":
            rec["label"] = f"attr e={t.entity_id}"
        elif t.type_name == "annotator":
            rec["label"] = f"ann e={t.entity_id}"
        else:
            rec["label"] = f"{t.type_name}:{i}"
        out.append(rec)
    return out


def write_standalone_html(
    path: Path,
    meta: dict,
    tokens: list[dict],
    edge_mask: np.ndarray,
    k_aug: np.ndarray,
    attn_layers: list[np.ndarray],
) -> None:
    """Single HTML file with inlined CSS, JS, and base64 tensor payloads (works offline / file://)."""
    css = (STATIC_SRC / "style.css").read_text(encoding="utf-8")
    css = css.replace(
        '"IBM Plex Sans", "Segoe UI", system-ui, sans-serif',
        "system-ui, -apple-system, Segoe UI, sans-serif",
    )
    tensor_js = (STATIC_SRC / "tensor_bin.js").read_text(encoding="utf-8")
    app_js = (STATIC_SRC / "app.js").read_text(encoding="utf-8")

    embed = {
        "meta": meta,
        "tokens": tokens,
        "edgeMaskB64": base64.b64encode(tensor_to_bin_bytes(edge_mask)).decode("ascii"),
        "kAugB64": base64.b64encode(tensor_to_bin_bytes(k_aug)).decode("ascii"),
        "attnLayerB64": [
            base64.b64encode(tensor_to_bin_bytes(layer_arr)).decode("ascii") for layer_arr in attn_layers
        ],
    }
    # UTF-8 safe + avoids raw `</script>` in strings breaking HTML parsing.
    payload_utf8 = json.dumps(embed, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.b64encode(payload_utf8).decode("ascii")
    embed_js = (
        "(function(){\n"
        "  var b64 = "
        + json.dumps(payload_b64)
        + ";\n"
        "  var bin = atob(b64);\n"
        "  var u8 = new Uint8Array(bin.length);\n"
        "  for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);\n"
        "  window.ATTENTION_EXPLORER_EMBED = JSON.parse(new TextDecoder('utf-8').decode(u8));\n"
        "})();\n"
    )

    body = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Attention explorer (standalone)</title>
  <style>
""" + css + """
  </style>
</head>
<body>
  <header>
    <h1>Entity Marformer — attention mass explorer</h1>
    <label>
      Layer
      <select id="layer-select"></select>
    </label>
  </header>
  <div id="token-grid"></div>

  <div id="modal" class="modal-backdrop" aria-hidden="true">
    <div class="modal" role="dialog" aria-labelledby="modal-title">
      <header>
        <h2 id="modal-title">Token</h2>
        <button type="button" class="close-btn" id="modal-close" aria-label="Close">&times;</button>
      </header>
      <div class="fan-wrap">
        <svg id="fan-svg" width="420" height="420" viewBox="-210 -210 420 420"></svg>
        <div class="legend" id="fan-legend"></div>
      </div>
      <div class="filters">
        <h3>Edge + pointer filters</h3>
        <div class="mode-toggle">
          <label><input type="radio" name="fmode" value="and" checked /> AND (all selected)</label>
          <label><input type="radio" name="fmode" value="or" /> OR (any selected)</label>
        </div>
        <div id="filter-checkboxes"></div>
        <div class="filter-mass" id="filter-mass"></div>
      </div>
      <div class="see-also" id="see-also"></div>
    </div>
  </div>

  <script>
""" + tensor_js + """
  </script>
  <script>
""" + embed_js + """
  </script>
  <script>
""" + app_js + """
  </script>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export data for interactive attention explorer. "
        "Model flags are taken from <run-dir>/train_config.json (same as training).",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--data-dir",
        default=None,
        type=Path,
        help="Override data bundle path only if train_config data_dir resolution fails.",
    )
    parser.add_argument("--partition", choices=("train", "test"), default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-item", type=int, default=None)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument(
        "--diagnostics-output-dir",
        default=None,
        type=Path,
        help="Path shown in the explorer as where logit PNG/NPZ live (default: <run-dir>/attn_diagnostics).",
    )
    args = parser.parse_args()

    _assert_disjoint_overlap_keys()

    run_dir = resolve_run_dir(Path(args.run_dir))
    out_dir = normalize_cli_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    data_override = normalize_cli_path(args.data_dir) if args.data_dir else None
    model, bundle, converter, sizes, tc = load_model(run_dir, device, data_dir=data_override)

    max_item = args.max_item
    if max_item is None:
        max_item = tc.get("training", {}).get("max_item")
    if max_item is not None and max_item <= 0:
        max_item = None

    graphs = build_partition_graphs(
        bundle,
        converter,
        sizes,
        tc,
        max_graphs=1,
        partition=args.partition,
        max_item=max_item,
        chunk_index=args.chunk_index,
    )
    graph = graphs[0]
    L = graph.num_tokens

    diag_dir = args.diagnostics_output_dir
    if diag_dir is None:
        diag_dir = run_dir / "attn_diagnostics"
    diag_dir = diag_dir.resolve()

    # Topology on CPU
    edge_mask = graph.build_edge_masks(device=torch.device("cpu")).float().numpy()  # [L,L,6]
    k_aug = _build_k_aug_cpu(graph, L)
    write_tensor_bin(out_dir / "edge_mask.bin", edge_mask)
    write_tensor_bin(out_dir / "k_aug.bin", k_aug)

    attention_debug: list = []
    with torch.no_grad():
        model(
            graph,
            device=device,
            attention_debug=attention_debug,
            attention_debug_store_full_attn_mean=True,
        )

    num_layers = len(attention_debug)
    attn_layer_arrays: list[np.ndarray] = []
    for layer_idx, layer_dict in enumerate(attention_debug):
        am = layer_dict.get("attn_mean")
        if am is None:
            raise RuntimeError(f"Layer {layer_idx}: attn_mean missing (store_full_attn_mean path broken)")
        arr = am.numpy().astype(np.float32)  # [1,L,L]
        attn_layer_arrays.append(arr[0])
        write_tensor_bin(out_dir / f"attn_layer{layer_idx}.bin", arr[0])

    use_pointer = bool(tc.get("model", {}).get("use_pointer", False))
    meta = {
        "L": L,
        "numLayers": num_layers,
        "numRelations": 6,
        "relationNames": REL_NAMES,
        "pointerNames": PTR_NAMES,
        "partition": args.partition,
        "chunkIndex": args.chunk_index,
        "usePointer": use_pointer,
        "useGraphMask": bool(model.config.use_graph_mask),
        "runDir": str(run_dir),
        "diagnosticsOutputDir": str(diag_dir),
        "logitPlotsNote": (
            "Three-logit breakdown (content / relational / pointer) is in PNG + NPZ from "
            "scripts/analyze_pointer_attention.py with the same --run-dir, --partition, "
            "--max-item, and --chunk-index (model flags from train_config.json)."
        ),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    tok_list = token_records(graph)
    with open(out_dir / "tokens.json", "w") as f:
        json.dump(tok_list, f, indent=2)

    write_standalone_html(
        out_dir / "attention_explorer_standalone.html",
        meta,
        tok_list,
        edge_mask,
        k_aug,
        attn_layer_arrays,
    )

    if not STATIC_SRC.is_dir():
        raise FileNotFoundError(f"Missing static assets: {STATIC_SRC}")
    for name in ("index.html", "app.js", "style.css", "tensor_bin.js"):
        src = STATIC_SRC / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing {src}")
        shutil.copy2(src, out_dir / name)

    print(f"Wrote explorer data to {out_dir}")
    print(f"  Offline: open attention_explorer_standalone.html (file:// or double-click)")
    print(f"  Server mode: cd {out_dir} && python -m http.server 8765  → index.html")
    print(f"  Logit diagnostics folder (see also): {diag_dir}")


if __name__ == "__main__":
    main()
