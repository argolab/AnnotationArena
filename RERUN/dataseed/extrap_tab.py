#!/usr/bin/env python
"""
The extrapolation section, appended to the "as published" tab.

Appended AFTER the stashed original body rather than merged into it: the tab is
labelled "as published" and the published panels have to keep meaning that. The
10k pilot is a different generative draw, so dropping a K=10000 point onto a
seed-42 curve would read as an extension of that curve when it is not one.
"""
from __future__ import annotations
import base64, json
from pathlib import Path


def img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def section_html(outdir: Path) -> str:
    tbl = outdir / "extrapolation_table.json"
    png = outdir / "extrapolation.png"
    if not (tbl.is_file() and png.is_file()):
        return ""
    rows = json.loads(tbl.read_text())["rows"]
    body = ""
    for r in rows:
        dp = f'{r["delta_prev"]:+.4f}' if r["delta_prev"] is not None else "&mdash;"
        f1 = f'{r["from_1000"]:+.4f}' if r["from_1000"] is not None else "&mdash;"
        cls = "g" if (r["delta_prev"] or 0) < 0 else ("d" if r["delta_prev"] else "sd")
        body += (f'<tr><td class=k>{r["size"]}</td><td>{r["corner"]:.4f}</td>'
                 f'<td class="{cls}">{dp}</td><td>{f1}</td></tr>')
    return f"""
<section>
<h2>Added September: what happens past K=4000</h2>
<p class=kick>The 10k pilot &middot; a different generative draw &middot; corner-scored</p>
<div class="box warn">
<p><b>This is not an extension of the curves above.</b> The pilot has its own
generative draw and therefore its own corner, and D28 measured 0.085 nats of
draw-to-draw movement in the absolute level &mdash; most of any effect worth
looking for. Its numbers cannot be read against seed 42's. What <em>can</em> be
compared across draws is the <b>shape</b>, which is what panel (b) shows: every
curve shifted to its own K=1000, so the draw offset cancels and only the slope
survives.</p>
<p>The pilot also runs at <code>max_item=300</code> throughout. The
<code>max_item=600</code> arm exists only to K=4000 &mdash; it dies at K=6000 with
a CUDA OOM, 78.4 GiB in use on an 80 GB H100, because activation memory scales
with the <em>number</em> of chunks as well as their width. The shipped chunk size
is what makes a 10k run possible at all.</p>
</div>
<figure><img alt="The 10k pilot, absolute and shape-only" src="{img(png)}">
<figcaption>(a) The pilot on its own axis, transductive and non-transductive.
(b) Every draw shifted to its own K=1000 value; grey are the two main draws,
which stop at 4000, red is the pilot carrying on to 10000.</figcaption></figure>
<div class=tw><table>
<caption>Marformer transductive, 400 epochs, last checkpoint, <code>max_item=300</code></caption>
<thead><tr><th>K</th><th>corner</th><th>&Delta; vs previous size</th><th>&Delta; vs K=1000</th></tr></thead>
<tbody>{body}</tbody></table></div>
</section>"""
