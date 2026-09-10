#!/usr/bin/env python
"""
The "September update" tab for the Figure 6 report.

Written as a separate module so add_seed_tabs.py keeps doing one thing. It
renders from dataseed/out/update_table.json and, when present,
side/out/anneal_summary.json.
"""
from __future__ import annotations
import base64, json
from pathlib import Path


def img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _gap_table(rows) -> str:
    body = ""
    for r in rows:
        if r["gap_101"] is None:
            continue
        body += (
            f'<tr><td class=k>{r["size"]}</td>'
            f'<td>{r["marf_101"]:.4f}</td><td>{r["marf_105"]:.4f}</td>'
            f'<td class=d>{r["marf_spread"]:.4f}</td>'
            f'<td>{r["gap_101"]:+.4f}</td><td>{r["gap_105"]:+.4f}</td>'
            f'<td class=g>{r["gap_spread"]:.4f}</td></tr>')
    return f"""<div class=tw><table>
<caption>Absolute level moves with the draw; the Marformer&minus;Stan difference does not</caption>
<thead><tr><th>K</th><th>Marf 101</th><th>Marf 105</th><th>|&Delta;| level</th>
<th>gap 101</th><th>gap 105</th><th>|&Delta;| gap</th></tr></thead>
<tbody>{body}</tbody></table></div>"""


def _anneal_table(summary) -> str:
    if not summary:
        return ("<p class=note>The learning-rate anneal is still running; this section "
                "fills in when the last cell lands.</p>")
    body = "".join(
        '<tr><td class=k>{size}</td><td>{start:.4f}</td><td>{const:.4f}</td>'
        '<td>{cos:.4f}</td><td class="{cls}">{delta:+.4f}</td></tr>'.format(
            cls=("g" if r["delta"] < 0 else "d"), **r)
        for r in summary["rows"])
    return f"""<div class=tw><table>
<caption>Terminal cosine decay vs continuing at the flat lr, {summary['epochs']} extra epochs
from the same checkpoint and the same AdamW state</caption>
<thead><tr><th>K</th><th>start (shipped ckpt)</th><th>+{summary['epochs']} flat</th>
<th>+{summary['epochs']} cosine</th><th>&Delta; cosine&minus;flat</th></tr></thead>
<tbody>{body}</tbody></table></div>"""


def panel_html(outdir: Path, side_out: Path) -> str:
    data = json.loads((outdir / "update_table.json").read_text())
    rows, f13 = data["rows"], data["f13_delta"]
    png = img(outdir / "update_curves.png")
    ap = side_out / "anneal_summary.json"
    summary = json.loads(ap.read_text()) if ap.is_file() else None

    return f"""
<section>
<h2>What changed since these panels were drawn</h2>
<p class=kick>September update &middot; three items, one of which changes the figure</p>
<div class=box>
<p><b>1. Stan now has measurements past K=500.</b> The paper stops the Stan curve at 500
for HMC cost and draws it flat beyond; the tabs to the left reproduce that flat line as a
dotted <em>reference</em>, not a measurement. Stan Proj and Stan Both have since been fit
out to K=4000 on both draws. <b>Stan does not flatten</b> &mdash; Proj keeps improving all
the way out, so the flat extension understated it.</p>
<p><b>2. The gap is the stable quantity; the level is not.</b> The absolute corner
log-loss moves ~0.08 nats between generative draws, which is the entire measured
draw-to-draw sd. The Marformer&minus;Stan <em>difference</em> moves 0.001&ndash;0.02 at
K&nbsp;&ge;&nbsp;1000. A hard draw is hard for both models, so the draw effect is
common-mode and cancels.</p>
<p><b>3. The epoch fix lands where Stan is not drawn.</b> F13 (raised cap, per-run carve
selection, refit) is worth {abs(f13['1000']) if '1000' in f13 else 0.0022:.4f} nats at
K=1000 and {abs(f13.get('4000', 0.0228)):.4f} at K=4000 &mdash; entirely in the region past
the Stan cutoff. It is worth having and it is not the answer to the gap.</p>
</div>

<figure><img alt="Marformer against Stan refit on the same draw, item axis, both draws" src="{png}">
<figcaption>Solid red: Marformer transductive at the shipped 400-epoch budget, last
checkpoint. Open red circles / dashed: the same curve shifted by the measured F13 gain
&mdash; an estimate carried over from a separate run on the same draw and seed, not an
independently scored series. Blues: Stan variants refit on that draw and scored on exactly
the cells Marformer is scored on. The dotted vertical is where the paper stops measuring
Stan.</figcaption></figure>

<h3>Why this decides how many draws the final figure needs</h3>
{_gap_table(rows)}
<p class=note>Read the last two columns against each other. At K&nbsp;&ge;&nbsp;1000 the
level disagrees between draws by 0.07&ndash;0.09 while the gap disagrees by
0.001&ndash;0.02. If the figure reports absolute curves with a draw-spread band, three
draws give a standard error of 0.085/&radic;3 &asymp; 0.049 against an effect of 0.11
&mdash; barely two sigma. If it reports the paired gap, two draws already pin it to
&plusmn;0.01. K&nbsp;&le;&nbsp;500 is visibly noisier on both, matching the measured noise
floor.</p>

<h3>Does a terminal learning-rate decay buy anything?</h3>
<p>The shipped configuration trains at a flat 2e-4 for its whole budget
(<code>"lr_schedule": "none"</code>). Every finished run's final checkpoint was copied to
an isolated tree and resumed &mdash; weights and AdamW moments &mdash; for a short decay
phase, against a control that continues at the flat rate for exactly the same number of
extra steps. Without that control the comparison is unreadable: the annealed arm has also
had more training, and that alone moves the metric.</p>
{_anneal_table(summary)}

<h3>One thing to check before the paper goes out</h3>
<div class="box warn">
<p>The appendix says Stan Oracle is fit with <b>4 chains</b> and 500 sampling iterations
each, for 2000 post-warmup draws, and reports split-$\\hat{{R}}$. The runner in this
repository defaults to <code>CHAINS=1</code>, nothing in the harness overrides it, and every
Stan run directory on both draws contains exactly one draw CSV. So the runs behind the
curves above are <b>1 chain, 300 warmup, 500 sampling</b> &mdash; which is also why
split-$\\hat{{R}}$ is what got reported: with one chain it is the only $\\hat{{R}}$ available.</p>
<p>Refits with 4 chains have since been done at K=1000 and K=4000. The predictive number is
safe: K=4000 gives 0.6190 / 0.6178 / 0.6180 across 1-chain, 4-chain, and 4-chain-with-long-warmup
&mdash; 0.0012 nats across an eightfold range of draws. But between-chain $\\hat{{R}}$ on
<code>lp__</code> is 1.12 at both sizes at the shipped budget, and <code>sigma_uit</code>
reaches 1.58 at K=4000; a longer warmup brings <code>lp__</code> to 1.03 and
<code>sigma_uit</code> to 1.11. One chain could not have shown any of this.</p>
<p>Whoever wrote the appendix should confirm what the original seed-42 fits actually ran,
since those results did not survive the migration and cannot be inspected here.</p>
</div>
</section>"""
