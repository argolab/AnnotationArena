#!/usr/bin/env python
"""Inject the payload into the overfitting artifact template.

Kept separate from the template so the page can be re-rendered as the two
in-flight measurements (validation-fitted temperature scaling, the L2 sweep)
land, without hand-editing a 250 KB file.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--template", required=True)
ap.add_argument("--payload", required=True)
ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()

raw = Path(a.payload).read_text()
json.loads(raw)                                   # fail here, not in the browser
# The payload rides in a <script type="application/json"> block, so the only
# sequence that can break out of it is a literal "</script"; escaping the three
# characters keeps the document well-formed without touching parsed values.
safe = raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")

html = Path(a.template).read_text()
assert html.count("__PAYLOAD__") == 1, "template must hold exactly one __PAYLOAD__"
Path(a.out).write_text(html.replace("__PAYLOAD__", safe))
print(f"{Path(a.out)}  {Path(a.out).stat().st_size/1e6:.2f} MB")
