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
    def cell(v, fmt="{:.4f}"):
        return fmt.format(v) if v is not None else "&mdash;"

    body = ""
    for r in summary["rows"]:
        cd = r.get("corner_delta")
        body += (
            f'<tr><td class=k>{r["size"]}</td>'
            f'<td>{r["start"]:.4f}</td><td>{r["const"]:.4f}</td><td>{r["cos"]:.4f}</td>'
            f'<td class="{"g" if r["delta"] < 0 else "d"}">{r["delta"]:+.4f}</td>'
            f'<td>{cell(r.get("corner_const"))}</td><td>{cell(r.get("corner_cos"))}</td>'
            f'<td class="{"sd" if cd is None else ("g" if cd < 0 else "d")}">'
            f'{cell(cd, "{:+.4f}")}</td></tr>')
    return f"""<div class=tw><table>
<caption>Terminal cosine decay vs continuing at the flat lr, {summary['epochs']} extra epochs
from the same checkpoint and the same AdamW state</caption>
<thead><tr><th rowspan=2>K</th><th colspan=4>whole-graph missing CE (~13.6k cells)</th>
<th colspan=3>test corner (316 cells) &mdash; the reported metric</th></tr>
<tr><th>shipped</th><th>+flat</th><th>+cosine</th><th>&Delta;</th>
<th>+flat</th><th>+cosine</th><th>&Delta;</th></tr></thead>
<tbody>{body}</tbody></table></div>
<p class=note><b>The two metrics disagree, and that is the result.</b> On the quiet
whole-graph metric the schedule is worth 0.013 at K=500 and nothing
(&le;0.003) at K&nbsp;&ge;&nbsp;1000. On the corner &mdash; what the paper reports &mdash;
the sign flips across sizes and every value sits inside the 0.035 paired between-arm
sd measured in Part&nbsp;0. An effect visible only in the metric we do not report,
at a magnitude the metric we do report cannot resolve, is not an effect to ship.
The epoch budget (F13) already collects what there is to collect here.</p>"""


def _runtime_section(outdir: Path) -> str:
    rt = outdir / "runtime_table.json"
    if not rt.is_file() or not (outdir / "runtime_curves.png").is_file():
        return ""
    data = json.loads(rt.read_text())
    by = {}
    for r in data["rows"]:
        by.setdefault(r["method"], {}).setdefault(r["size"], {})[r["draw"]] = r["minutes"]
    order = ["Marformer (T)", "Marformer NT", "Stan Oracle", "Stan Bin",
             "Stan Proj", "Stan Both"]
    body = ""
    for m in order:
        row = by.get(m, {}).get(4000)
        if not row:
            continue
        a, b = row.get("101"), row.get("105")
        cell = lambda v: f"{v:,.0f}" if v else "&mdash;"
        ratio = ""
        base = by.get("Marformer (T)", {}).get(4000, {}).get("101")
        if a and base:
            ratio = f"{a / base:.1f}x"
        body += (f'<tr><td class=k>{m}</td><td>{data["hardware"]["stan" if m.startswith("Stan") else "marformer"]}</td>'
                 f'<td>{cell(a)}</td><td>{cell(b)}</td><td class=sd>{ratio}</td></tr>')
    png = img(outdir / "runtime_curves.png")
    return f"""
<h3>What each method cost to fit <span style="font-weight:400;color:var(--ink-faint)">&mdash; temporary, rough</span></h3>
<div class="box warn">
<p><b>Treat every number here as an order of magnitude, not a measurement.</b> These are
<code>elapsed_seconds</code> and <code>fit_elapsed_seconds</code> scraped from the launcher
metadata of runs submitted for other purposes, on a shared cluster, n=1, with no control
over node contention. Draw 105's Stan Oracle is <em>slower</em> at K=3000 than at K=4000,
which is not a property of the model. The paper's runtime figure needs a controlled pass.</p>
<p>Two asymmetries are deliberate and stated rather than hidden. <b>Different hardware:</b>
Marformer gets a GPU and Stan does not, because HMC here is CPU-only &mdash; that is the
comparison being made ("what does this curve cost to produce"), not a per-core one.
<b>Different work:</b> every Stan fit here is <b>one chain</b>. The final run moves to 4
chains and 2000 sampling iterations, so these are a <em>lower bound</em> on the shipped
configuration.</p>
</div>
<figure><img alt="Wall clock to fit against training size, both draws" src="{png}">
<figcaption>Log&ndash;log. Hardware is in the legend. Marformer's two curves overlap almost
exactly &mdash; the transductive and non-transductive arms cost the same.</figcaption></figure>
<div class=tw><table>
<caption>At K=4000, the largest size both families share</caption>
<thead><tr><th>method</th><th>hardware</th><th>draw 101 (min)</th><th>draw 105 (min)</th>
<th>vs Marformer (T)</th></tr></thead>
<tbody>{body}</tbody></table></div>
<p class=note><b>The headline is real but narrower than it looks.</b> Against <b>Stan
Oracle</b> &mdash; the well-specified model, the one the gap is measured against &mdash;
Marformer is ~20x faster at K=4000 (37 min against 12.6 hours), and the ratio grows with K.
Against the <em>misspecified</em> Stan variants it is not: Proj and Both fit in 26&ndash;27
minutes, comparable to Marformer or better. So the amortisation argument should be made
against the model that is actually competitive on accuracy, not against Stan in general.</p>"""


def panel_html(outdir: Path, side_out: Path) -> str:
    data = json.loads((outdir / "update_table.json").read_text())
    rows, f13 = data["rows"], data["f13_delta"]
    png = img(outdir / "update_curves.png")
    # Computed out of line: nesting a conditional and a format spec inside the
    # page f-string is a parse hazard for no benefit.
    f13_1k = f"{abs(f13.get('1000', 0.0022)):.4f}"
    f13_4k = f"{abs(f13.get('4000', 0.0228)):.4f}"
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
<p><b>3. The epoch fix closes about a sixth of the gap.</b> F13 (raised cap, per-run
carve selection, refit) is worth {f13_1k} nats at K=1000 and {f13_4k} at K=4000,
against a K=4000 gap of 0.126. It is worth having and it is not the answer.
<em>An earlier version of this panel said the fix "lands where Stan is not drawn" &mdash;
true only while Oracle stopped at K=500 in our scoring, which was the bookkeeping
error described above. Oracle runs to K=4000, so F13's gains sit squarely inside the
compared region.</em></p>
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
draws give a standard error of 0.085/&radic;3 &asymp; 0.049 against an effect of 0.126
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

{_runtime_section(outdir)}

<h3>One thing to check before the paper goes out</h3>
<div class="box warn">
<p>The appendix says Stan Oracle is fit with <b>4 chains</b> and 500 sampling iterations
each, for 2000 post-warmup draws, and reports split-R&#770;. The runner in this
repository defaults to <code>CHAINS=1</code>, nothing in the harness overrides it, and every
Stan run directory on both draws contains exactly one draw CSV. So the runs behind the
curves above are <b>1 chain, 300 warmup, 500 sampling</b> &mdash; which is also why
split-R&#770; is what got reported: with one chain it is the only R&#770; available.</p>
<p>Refits with 4 chains have since been done at K=1000 and K=4000. The predictive number is
safe: K=4000 gives 0.6190 / 0.6178 / 0.6180 across 1-chain, 4-chain, and 4-chain-with-long-warmup
&mdash; 0.0012 nats across an eightfold range of draws. But between-chain R&#770; on
<code>lp__</code> is 1.12 at both sizes at the shipped budget, and <code>sigma_uit</code>
reaches 1.58 at K=4000; a longer warmup brings <code>lp__</code> to 1.03 and
<code>sigma_uit</code> to 1.11. One chain could not have shown any of this.</p>
<p>Whoever wrote the appendix should confirm what the original seed-42 fits actually ran,
since those results did not survive the migration and cannot be inspected here.</p>
</div>
</section>"""
