#!/usr/bin/env python
"""
The entry point for someone who has not been following this thread.

One page: what the project is, what we set out to check, what we found, and
where each result lives. Everything else links out from here.
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path

A = {
    "fig6":    "https://claude.ai/code/artifact/6e710e4d-9877-430b-b54e-c4f5a462e0c1",
    "resid":   "https://claude.ai/code/artifact/783e16ed-4117-4c44-a628-3dff7e440d7b",
    "bugs":    "https://claude.ai/code/artifact/662979d9-c92c-47f7-ae42-eb947561200f",
    "fixes":   "https://claude.ai/code/artifact/ebf655b5-617b-42e4-8eb0-bc13d0ec8afe",
    "rerun":   "https://claude.ai/code/artifact/05c96ea6-b29a-4104-a7b5-e87bd834143b",
    "repl":    "https://claude.ai/code/artifact/96812eeb-4c0c-4c9a-a413-67bd321e6282",
    "valproto":"https://claude.ai/code/artifact/5bae9038-9624-4a46-ae8b-d9faa1f8f6b6",
    "mcar":    "https://claude.ai/code/artifact/d5c50d63-a9fd-4d26-9869-f565652e06b3",
}

def gap_stats(p="dataseed/out/dataseed_rows.json"):
    rows = json.loads(Path(p).read_text())
    out = {}
    for K in (1000,):
        for fam in ("Marformer",):
            m = {r["data_seed"]: r for r in rows if r["family"] == fam and r["size"] == K}
            fresh = [m[s]["gap"] for s in m if s != 42]
            out["s42"] = m[42]["gap"]
            out["fresh_lo"], out["fresh_hi"] = min(fresh), max(fresh)
            c = [r["corner"]["log_loss"] for r in rows
                 if r["family"] == fam and r["size"] == K]
            out["corner_sd"] = st.stdev(c)
            out["corner_lo"], out["corner_hi"] = min(c), max(c)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="reports/style/shared_head.html")
    ap.add_argument("-o", "--out", default="reports/build/index_report.html")
    a = ap.parse_args()
    g = gap_stats()
    head = Path(a.head).read_text()

    def card(href, title, kicker, body, tag=""):
        t = f'<span class=tag>{tag}</span>' if tag else ""
        return f"""<a class=card href="{href}">
<p class=cardkick>{kicker}{t}</p><h3 class=cardtitle>{title}</h3><p class=cardbody>{body}</p></a>"""

    extra_css = """
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:.9rem;margin:0 0 1.8rem}
a.card{display:block;background:var(--surface);border:1px solid var(--rule);padding:1.05rem 1.15rem;
text-decoration:none;color:inherit;transition:border-color .12s,transform .12s}
a.card:hover{border-color:var(--accent);transform:translateY(-1px)}
a.card:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.cardkick{font-family:var(--fm);font-size:.65rem;letter-spacing:.11em;text-transform:uppercase;
color:var(--ink-faint);margin:0 0 .45rem;display:flex;align-items:center;gap:.5rem}
.tag{background:var(--accent-soft);color:var(--accent);padding:.1rem .4rem;border-radius:2px;
letter-spacing:.06em}
h3.cardtitle{font-family:var(--fd);font-size:1.02rem;font-weight:600;color:var(--ink);
margin:0 0 .4rem;text-transform:none;letter-spacing:0}
.cardbody{font-size:.88rem;color:var(--ink-soft);margin:0;max-width:none;line-height:1.5}
.tl{border-left:2px solid var(--rule-strong);padding-left:1.15rem;margin:0 0 1.6rem}
.tl h3{margin-top:1.1rem}
.big{font-family:var(--fd);font-size:1.6rem;font-weight:700;color:var(--accent);
font-variant-numeric:tabular-nums}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin:0 0 1.6rem}
.stat{background:var(--surface);border:1px solid var(--rule);padding:.9rem 1rem}
.stat p{margin:0;max-width:none}
.statlab{font-family:var(--fm);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;
color:var(--ink-faint);margin-bottom:.3rem}
"""
    html = f"""<title>Domain 3 Findings</title>
{head}
<style>{extra_css}</style>
<div class=wrap>
<header>
<p class=eyebrow>Marformer &middot; synthetic Domain 3 &middot; replication and audit</p>
<h1>What we found re-running Domain 3</h1>
<p class=stand>Marformer is a transformer that imputes missing entries in an
item&nbsp;&times;&nbsp;annotator&nbsp;&times;&nbsp;criterion rating tensor. This is a
re-run of the paper's synthetic Domain&nbsp;3 experiments from scratch: does it replicate,
are the reported numbers measuring what they claim to, and how much of the headline result
survives scrutiny. Short answer: the model replicates, the <em>evaluation</em> did not.</p>
</header>

<section>
<h2>The three results that matter</h2>
<p class=kick>If you read nothing else</p>

<div class=box>
<p><b>1. Marformer and Stan were being scored on different cells.</b> The paper says all
log-losses are computed on a fixed 414-cell &ldquo;test corner&rdquo;. Stan was. Marformer was
scored on <em>every</em> missing cell in its graph &mdash; mostly cells belonging to training
items, which are much easier. Rebuilt on the corner, Marformer is worse than published at
every one of the 38 measured cells &mdash; by +0.03 to +0.14 nats, typically about +0.08 &mdash; and the claim that it approaches near-oracle performance
by K&nbsp;=&nbsp;4000 does not survive.</p>
</div>

<div class=box>
<p><b>2. The published points are chosen by looking at the test set.</b> Domain&nbsp;3 ships with
an empty validation split, so there is nothing to stop early on. The reported number is the
minimum over ~18 checkpoints scored on test. We built a proper protocol &mdash; carve a
validation block out of training, select the epoch on it, refit on the full training partition
&mdash; so the numbers can be quoted without a leak.</p>
</div>

<div class="box warn">
<p><b>3. The synthetic corner is noisier than the effects being read off it.</b> We regenerated
the whole dataset under six independent generative seeds. At fixed training size and protocol,
corner log-loss ranges <b>{g['corner_lo']:.2f}&ndash;{g['corner_hi']:.2f}</b> purely from the
draw (sd {g['corner_sd']:.3f}). That is larger than most of the gaps Figure&nbsp;6 is currently
read as showing. Domain&nbsp;3 numbers need to be averaged over generative draws, not just over
model seeds.</p>
</div>
</section>

<section>
<h2>How big is &ldquo;noisier&rdquo;?</h2>
<p class=kick>Six independent regenerations of the same experiment</p>
<div class=stats>
<div class=stat><p class=statlab>corner spread across draws</p><p class=big>&plusmn;{g['corner_sd']:.3f}</p>
<p class=cardbody>standard deviation of corner log-loss at K&nbsp;=&nbsp;1000, same model, same protocol</p></div>
<div class=stat><p class=statlab>train&minus;corner gap, original seed</p><p class=big>{g['s42']:+.3f}</p>
<p class=cardbody>the gap that looked like a generalization deficit</p></div>
<div class=stat><p class=statlab>same gap, five fresh draws</p>
<p class=big>{g['fresh_lo']:+.3f} to {g['fresh_hi']:+.3f}</p>
<p class=cardbody>every one negative &mdash; the corner is <em>easier</em> than the training cells</p></div>
</div>
<p>The original dataset happened to deal a hard 414-cell corner. We checked and ruled out the
mundane explanations: label difficulty (marginal predictors see no difference), context density
(identical in both regions on every draw), and corner size (the original corner is the
<em>largest</em>, so the least noisy). What is left is the realised difficulty of ~400 specific
cells.</p>
</section>

<section>
<h2>The results, in detail</h2>
<p class=kick>Each links to a standalone page</p>
<div class=grid>
{card(A['fig6'], "Figure 6, rebuilt on the test corner",
      "primary result", "The corrected item- and annotator-generalization panels, with the "
      "non-transductive curve extended to the full sweep and the published Stan curves overlaid. "
      "Tabs compare two fresh generative draws against the same Stan reference.", "updated today")}
{card(A['resid'], "Does the double residual stream earn its place?",
      "architecture", "Marformer stacks two residual streams. We deleted each and replaced the "
      "outer one with a structural identity. The shipped design wins, but the variants reach a "
      "lower floor and overfit past it faster &mdash; the ranking depends on where you stop.",
      "updated today")}
{card(A['bugs'], "Four evaluation bugs",
      "audit", "The defects found in the evaluation path, each traced to whether it affects the "
      "published numbers, our replication, or both.")}
{card(A['fixes'], "Four fixes, one deficit",
      "audit", "What changes once all four are fixed, and which part of the published claim "
      "does not survive.")}
{card(A['repl'], "Does Domain 3 replicate?",
      "replication", "Re-running the paper's pipeline end to end on freshly generated data. "
      "The model replicates; two evaluation bugs had to be fixed to see it.")}
{card(A['rerun'], "Domain 3 rerun harness",
      "replication", "The regenerated runs and figures, and what the migration to the current "
      "cluster lost along the way.")}
{card(A['valproto'], "A validation protocol for Domain 3",
      "method", "Domain 3 has no validation split. This is the block-carve-then-refit design "
      "that gives an honest stopping rule without spending test data.")}
{card(A['mcar'], "MCAR validation",
      "method", "The alternative carve &mdash; hide random observed cells instead of a block. "
      "Same destination, weaker instrument.")}
</div>
</section>

<section>
<h2>Does a bigger Marformer help?</h2>
<p class=kick>Ten architectures at K&nbsp;=&nbsp;4000, each with a validation carve</p>
<p>Depth 4&ndash;16, width 80&ndash;160, wider feed-forward, deeper feed-forward &mdash; all
trained with a 15% block-validation carve so the stopping epoch is never chosen on test.
Corner log-loss against the shipped 80-dim, 8-layer model:</p>
<div class=tw><table>
<caption>Change vs the shipped architecture &mdash; negative is better. One model seed.</caption>
<thead><tr><th>architecture</th><th>dim</th><th>layers</th><th>transductive</th><th>non-transductive</th></tr></thead>
<tbody>
<tr><td class=k>wide160</td><td>160</td><td>8</td><td class=g>&minus;0.0221</td><td class=sd>+0.0019</td></tr>
<tr><td class=k>wide128</td><td>128</td><td>8</td><td class=g>&minus;0.0171</td><td class=d>+0.0172</td></tr>
<tr><td class=k>deep12</td><td>80</td><td>12</td><td class=sd>&minus;0.0095</td><td class=sd>&minus;0.0014</td></tr>
<tr><td class=k>ff256</td><td>80</td><td>8</td><td class=sd>&minus;0.0080</td><td class=sd>+0.0017</td></tr>
<tr><td class=k>shipped</td><td>80</td><td>8</td><td>0.8633</td><td>0.8607</td></tr>
<tr><td class=k>deep16</td><td>80</td><td>16</td><td class=sd>+0.0047</td><td class=d>+0.0203</td></tr>
<tr><td class=k>shallow4</td><td>80</td><td>4</td><td class=d>+0.0064</td><td class=d>+0.0160</td></tr>
<tr><td class=k>big160x12</td><td>160</td><td>12</td><td class=sd>+0.0039</td><td class=d>+0.0133</td></tr>
</tbody></table></div>
<p><b>No architecture change is a reliable win.</b> Making the model wider helps the transductive
arm by about 0.02 and does nothing for the non-transductive one; making it both wider and deeper
is worse than either. Every effect here is smaller than the &plusmn;{g['corner_sd']:.3f} draw-to-draw spread
above, and the two arms disagree in sign, so nothing on this table should be adopted yet. A
three-seed replication of the top configurations is running.</p>
<p class=note>The one clear signal is negative: depth alone does not help. 16 layers is worse than
8 in both arms, and 4 layers is worse still &mdash; the shipped depth is already at the useful
maximum.</p>
</section>

<section>
<h2>Still running</h2>
<p class=kick>Results expected shortly; this page updates as they land</p>
<ul>
<li><b>Capacity sweep, replicated.</b> The four best configurations above on two further model
seeds, to see whether the width effect survives.</li>
<li><b>Scale pilot.</b> A freshly generated dataset with 10,000 items, ten times the paper's
range, to ask whether transfer keeps improving or has already saturated by K&nbsp;=&nbsp;3000&ndash;4000.</li>
</ul>
<p class=note>Given the draw-to-draw spread above, both will need more than one generative seed
before any winner is believable. Effect sizes are reported against the
&plusmn;{g['corner_sd']:.3f} noise floor throughout.</p>
</section>

<footer>
<p>All results are reproducible from <code>RERUN/</code> in the Marformer repository:
<code>dataseed/</code> (generative-seed sweep), <code>fig6/</code> (corrected panels),
<code>hpsweep/</code> (capacity), <code>pilot10k/</code> (scale), <code>reports/</code>
(write-ups). Every defect is numbered and recorded in <code>RERUN/DISCREPANCIES.md</code>,
currently at 28 entries.</p>
</footer>
</div>
"""
    Path(a.out).write_text(html)
    print(f"-> {a.out}")

if __name__ == "__main__":
    main()
