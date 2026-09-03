#!/usr/bin/env python
"""
Build the residual-ablation page from the corner-scored payloads.

Self-contained on purpose: the audience is teammates and an advisor who have
not been following the thread, so the page states the architecture, the
question, the protocol and the numbers without assuming the rest of the
register.
"""
from __future__ import annotations
import argparse, base64, json
from pathlib import Path

VAR_LABEL = {
    "all":            ("all", "shipped — outer add kept in every layer"),
    "none_id":        ("none_id", "outer add replaced by a structural identity, (I+M)c"),
    "none":           ("none", "outer add deleted, Mc"),
    "all+inner_none": ("all + inner=none", "inner adds deleted, outer add kept"),
}
ORDER = ["all", "none_id", "none", "all+inner_none"]


def img(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="reports/build")
    ap.add_argument("--head", default="reports/style/shared_head.html")
    ap.add_argument("-o", "--out", default="reports/build/resid_report.html")
    a = ap.parse_args()
    b = Path(a.build)
    table = json.loads((b / "resid_corner_table.json").read_text())
    sizes = sorted({r["size"] for r in table})
    by = {(r["size"], r["variant"]): r for r in table}

    def cell(size, var, field, fmt="{:.4f}", cls=""):
        r = by.get((size, var))
        if not r or r.get(field) is None:
            return "<td>&mdash;</td>"
        return f'<td class="{cls}">{fmt.format(r[field])}</td>'

    abs_rows = "".join(
        "<tr><td class=k>{K}</td>".format(K=s)
        + "".join(cell(s, v, "final_mean",
                       "{:.4f}", "k" if v == "all" else "") for v in ORDER)
        + "</tr>" for s in sizes)

    def paired_row(s):
        tds = []
        for v in ORDER[1:]:
            r = by.get((s, v))
            if not r or r["paired_vs_all"] is None:
                tds.append("<td>&mdash;</td>"); continue
            d, se = r["paired_vs_all"], r["paired_se"] or 0
            t = abs(d / se) if se else 0
            cls = "d" if d > 0 and t >= 2 else ("sd" if abs(t) < 2 else "g")
            tds.append(f'<td class="{cls}">{d:+.4f}<span class=sd> (t {t:.1f})</span></td>')
        return f"<tr><td class=k>{s}</td>" + "".join(tds) + "</tr>"

    paired_rows = "".join(paired_row(s) for s in sizes)

    infl_rows = "".join(
        "<tr><td class=k>{K}</td>".format(K=s)
        + "".join(cell(s, v, "inflation", "{:+.4f}") for v in ORDER)
        + "</tr>" for s in sizes)

    best_rows = "".join(
        "<tr><td class=k>{K}</td>".format(K=s)
        + "".join(cell(s, v, "min_mean", "{:.4f}") for v in ORDER)
        + "</tr>" for s in sizes)

    head = Path(a.head).read_text()
    curves = img(b / "resid_curves_corner.png")
    hdrs = "".join(f"<th>{VAR_LABEL[v][0]}</th>" for v in ORDER)
    hdrs_p = "".join(f"<th>{VAR_LABEL[v][0]}</th>" for v in ORDER[1:])

    html = f"""<title>The Double Residual Stream</title>
{head}
<div class=wrap>
<header>
<p class=eyebrow>Marformer &middot; Domain 3 &middot; architecture ablation</p>
<h1>Does the double residual stream earn its place?</h1>
<p class=stand>Marformer stacks two residual streams. We deleted each one, and replaced the
outer one with a structural identity, to find out which is redundant. The shipped design wins,
but not for the reason expected: the variants reach a <em>lower</em> best point and then overfit
past it faster.</p>
</header>

<section>
<h2>The architecture, in one block</h2>
<p class=kick>What is being ablated</p>
<p>Every one of the 8 layers ends like this. <code>combined</code> is the token state, split into
an entity-identity half (<code>features</code>, 75 dims) and a running logit accumulator
(<code>params</code>, 5 dims):</p>
<div class=tw><table><tbody>
<tr><td style="text-align:left;font-family:var(--fm)">combined = combined + attn(...)<br>
combined = combined + ff(...)</td><td style="text-align:left">the <b>inner</b> residual stream</td></tr>
<tr><td style="text-align:left;font-family:var(--fm)">features = features + W&#8321;&middot;combined<br>
params&nbsp;&nbsp; = params&nbsp;&nbsp; + W&#8322;&middot;combined</td><td style="text-align:left">the <b>outer</b> residual stream</td></tr>
</tbody></table></div>
<p>Because <code>[features | params]</code> <em>is</em> the <code>combined</code> the layer started
from, the layer computes <b>h &rarr; h + Mc</b>, where <code>M = [W&#8321;; W&#8322;]</code> is a square
80&times;80 map and <code>c</code> is the inner stream's output. So there really are two stacked
streams, and the attention and FFN outputs never reach the next layer except through
<code>M</code>.</p>
<div class=box>
<p><b>The question.</b> Two stacked residual streams look redundant. If <code>M</code> could
simply learn to contain an identity, the outer add would be doing nothing the projection could not
do by itself. Is it redundant &mdash; and if so, can it be folded in?</p>
</div>
</section>

<section>
<h2>The four variants</h2>
<p class=kick>All have identical parameter counts &mdash; only the forward pass differs</p>
<div class=tw><table>
<thead><tr><th>variant</th><th>layer computes</th><th>what it tests</th></tr></thead>
<tbody>
<tr><td class=k>all</td><td>h + Mc</td><td>the shipped model</td></tr>
<tr><td class=k>none_id</td><td>(I + M)c</td><td>outer add folded into the projection, permanently</td></tr>
<tr><td class=k>none</td><td>Mc</td><td>outer add simply deleted</td></tr>
<tr><td class=k>all + inner=none</td><td>h + Mc, no inner adds</td><td>the <em>inner</em> stream deleted instead</td></tr>
</tbody></table></div>
<p class=note>A fifth and sixth variant (<code>not_last</code>, <code>not_last_id</code> &mdash;
drop the outer add in the final layer only) were trained and are excluded: changing one layer
alters how the output head reads the final state rather than how signal propagates, so it does
not speak to the architectural question.</p>
</section>

<section>
<h2>Learning curves on the test corner</h2>
<p class=kick>3 seeds averaged &middot; 400 epochs &middot; item axis, transductive</p>
<figure><img alt="Corner log-loss through training for four residual variants" src="{curves}">
<figcaption>Log-loss on the fixed 414-cell test corner, evaluated at every periodic checkpoint.
Dotted grey line is chance (ln&nbsp;4 = 1.386). Curves are the mean over seeds 42/43/44.</figcaption>
</figure>
<div class="box warn">
<p><b>These numbers are new.</b> The ablation was originally scored on the pooled metric &mdash;
every missing token in the graph, most of which sits on training-partition items. The paper says
it reports the 414-cell corner. Everything on this page is rebuilt on the corner, and the curves
change the reading.</p>
<p><b>Every variant overfits, and the ranking depends on where you stop.</b> All four curves
bottom out well before epoch 400 and rise afterwards. At K&nbsp;=&nbsp;300 the variant with the
<em>lowest</em> minimum (<code>inner=none</code>, 0.881) has the <em>worst</em> final checkpoint
(1.040) &mdash; it fits fastest and then degrades hardest. A last-checkpoint comparison at a fixed
400 epochs is therefore partly a measurement of overfitting rate, not of how well each
architecture can fit.</p>
</div>
</section>

<section>
<h2>The numbers</h2>
<p class=kick>Last checkpoint &middot; corner log-loss &middot; chance = 1.3863</p>
<div class=tw><table>
<caption>Absolute corner log-loss, mean of 3 seeds</caption>
<thead><tr><th>K</th>{hdrs}</tr></thead>
<tbody>{abs_rows}</tbody></table></div>

<div class=tw><table>
<caption>Paired against <code>all</code> &mdash; same seed, same split. Positive = worse than shipped.</caption>
<thead><tr><th>K</th>{hdrs_p}</tr></thead>
<tbody>{paired_rows}</tbody></table></div>

<div class=tw><table>
<caption>Best checkpoint reached &mdash; the floor each architecture can get to</caption>
<thead><tr><th>K</th>{hdrs}</tr></thead>
<tbody>{best_rows}</tbody></table></div>

<div class=tw><table>
<caption>Overfitting: last checkpoint minus each run's own best checkpoint</caption>
<thead><tr><th>K</th>{hdrs}</tr></thead>
<tbody>{infl_rows}</tbody></table></div>
<p class=note>The paired table compares the <b>last</b> checkpoint. Domain 3 ships with an empty
validation split, so a best-checkpoint comparison would let each variant pick its own luckiest
point on test &mdash; the best-checkpoint table is shown as a floor, not as a selectable result.
Read the two together: they disagree, and that disagreement is the finding.</p>
</section>

<section>
<h2>Why: the model never rediscovers the skip connection</h2>
<p class=kick>Weight geometry of the outer map M, seed 42</p>
<p>Independent of the loss numbers, one structural fact came out clean.
If the outer add were redundant, <code>M</code> should drift toward containing an identity.
It never does &mdash; in any variant, at any layer, including <code>none</code>, where an identity
would help most:</p>
<div class=tw><table>
<thead><tr><th>variant</th><th>cos(M, I)</th><th>cond(M)</th><th>cond(I + M)</th></tr></thead>
<tbody>
<tr><td class=k>all</td><td>&minus;0.041 &hellip; +0.008</td><td>90 &ndash; 367</td><td class=g>5.5 &ndash; 7.0</td></tr>
<tr><td class=k>none_id</td><td>&minus;0.038 &hellip; +0.013</td><td>80 &ndash; 682</td><td class=g>5.3 &ndash; 6.7</td></tr>
<tr><td class=k>none</td><td>&minus;0.022 &hellip; +0.038</td><td>48 &ndash; 937</td><td class=g>5.3 &ndash; 7.5</td></tr>
<tr><td class=k>all + inner=none</td><td>&minus;0.021 &hellip; +0.033</td><td>107 &ndash; <b>5744</b></td><td>5.9 &ndash; 12.0</td></tr>
</tbody></table></div>
<p><code>M</code> is full rank (80/80) everywhere, so this is not a capacity limit. It is
<em>orthogonal to the identity</em> and ill-conditioned; the same map <em>with</em> an identity
added is well-conditioned everywhere. Under <code>none</code>, state reaches the output through
eight ill-conditioned maps and directions aligned with the small singular values are crushed
geometrically. Plausible cause: AdamW's weight decay pulls <code>M</code> toward zero, and the
identity is a high-norm target (&#8214;I&#8214; = 8.94 against &#8214;M&#8214; &asymp; 5.1), so
nothing pushes toward it.</p>
</section>

<section>
<h2>What to do</h2>
<p class=kick>Recommendations</p>
<ul>
<li><b>Keep the shipped architecture for now.</b> It is best or tied-best at the last checkpoint
at all three sizes, and it is the most stable: the smallest overfitting inflation and the
tightest seed spread.</li>
<li><b>The inner residual is the one that cannot simply be dropped.</b> At K&nbsp;=&nbsp;100,
<code>inner=none</code> ends at 1.348 against a chance level of 1.386, with a seed spread of
&plusmn;0.26 &mdash; one seed lands worse than chance and essentially fails to train. At larger K
it trains fine and even reaches the lowest floor, so the failure is one of stability at small
data, not of capacity.</li>
<li><b>Re-run this with proper early stopping before deciding anything.</b> Since the ranking
flips between the minimum and epoch 400, and the variants differ mostly in <em>when</em> they turn
over, the honest comparison needs a validation carve and a refit &mdash; the protocol already
built for Figure&nbsp;6. That is the single highest-value follow-up here.</li>
<li><b>The <code>none_id</code> result is size-dependent and should not be quoted as a headline.</b>
It is free at K&nbsp;=&nbsp;100 (+0.005), costs +0.039 at K&nbsp;=&nbsp;300, and at
K&nbsp;=&nbsp;500 is <em>worse</em> than simply deleting the outer add. The mechanism below is
real; the size of the benefit is not yet established.</li>
<li><b>Still unrun:</b> <code>use_embed_unembed=True</code>. The codebase already contains a
proper single-stream architecture (standard pre-norm stack, per-type embedders and unembedders,
no split/concat). It is a better-founded version of the unified-stream idea and has never been
benchmarked here. The obvious next experiment.</li>
</ul>
</section>

<section>
<h2>Caveats</h2>
<ul>
<li>3 seeds per cell, item axis and transductive only. At K&nbsp;=&nbsp;100 the baseline seed sd
(0.079) exceeds most of the effects being measured; K&nbsp;=&nbsp;300 is the cleanest cell
(baseline sd 0.004).</li>
<li>Fixed 400 epochs, no early stopping, last checkpoint. Given that all four variants overfit,
this is the main limitation of the comparison rather than a footnote.</li>
<li>The corner is 414 fixed cells. A companion result (D28) measured that corner log-loss moves
by up to <b>&plusmn;0.09</b> between independent draws of the synthetic data. That does not affect
the <em>paired</em> comparisons here &mdash; every variant sees the identical draw and the identical
corner &mdash; but it does bound how much the absolute numbers mean.</li>
<li>Weight geometry is from one seed at one size. The conditioning gap spans two orders of
magnitude, so seed variation is unlikely to reverse it, but it has not been checked.</li>
</ul>
</section>

<footer>
<p>Runs: <b>RERUN/runs/resid-01</b> (all, none), <b>resid-02</b> (none_id), <b>resid-03</b>
(inner=none) &mdash; 4 variants &times; 3 sizes &times; 3 seeds, 400 epochs each.
Corner scoring: <code>RERUN/fig6/corner_curve.py</code>; figures and tables:
<code>RERUN/reports/build_resid_figs.py</code>. Full write-up:
<code>RERUN/reports/RESIDUAL_ABLATION.md</code>.</p>
</footer>
</div>
"""
    Path(a.out).write_text(html)
    print(f"-> {a.out}  ({len(html)/1e6:.2f} MB)")

if __name__ == "__main__":
    main()
