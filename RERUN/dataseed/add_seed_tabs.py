#!/usr/bin/env python
"""
Turn the Figure 6 report into a tabbed page and add one tab per fresh
generative draw.

The existing single-scroll document becomes the first tab unchanged. Each new
tab carries that draw's Marformer curves against the published Stan curves, plus
the numbers behind them.

Idempotent: re-running rebuilds the tab shell from the ORIGINAL body, which is
stashed in an HTML comment the first time round, so the page can be regenerated
as new seeds land without nesting shells.

    python add_seed_tabs.py --report fig6/out/fig6_report.html --seeds 101,105
"""
from __future__ import annotations
import argparse, base64, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from update_tab import panel_html as update_panel_html

ORIG_START, ORIG_END = "<!--ORIG-START-->", "<!--ORIG-END-->"

TAB_CSS = """
.tabs{display:flex;gap:.3rem;flex-wrap:wrap;border-bottom:2px solid var(--ink);
margin:0 0 2.2rem;padding:0}
.tabs button{font-family:var(--fd);font-size:.83rem;font-weight:600;letter-spacing:.01em;
background:transparent;color:var(--ink-faint);border:1px solid transparent;border-bottom:none;
padding:.62rem 1.05rem;cursor:pointer;border-radius:3px 3px 0 0;margin-bottom:-2px}
.tabs button:hover{color:var(--ink)}
.tabs button[aria-selected=true]{background:var(--surface);color:var(--ink);
border-color:var(--rule-strong);border-bottom:2px solid var(--surface)}
.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.tabpanel[hidden]{display:none}
"""

TAB_JS = """
<script>
(function(){
  var bs=[].slice.call(document.querySelectorAll('.tabs button'));
  function sel(id){
    bs.forEach(function(b){
      var on=b.dataset.target===id;
      b.setAttribute('aria-selected',on?'true':'false');
      b.tabIndex=on?0:-1;
      var p=document.getElementById(b.dataset.target);
      if(p)p.hidden=!on;
    });
    try{localStorage.setItem('fig6tab',id)}catch(e){}
    window.scrollTo({top:0,behavior:'smooth'});
  }
  bs.forEach(function(b){b.addEventListener('click',function(){sel(b.dataset.target)})});
  document.querySelector('.tabs').addEventListener('keydown',function(e){
    var i=bs.indexOf(document.activeElement); if(i<0)return;
    var j=e.key==='ArrowRight'?i+1:e.key==='ArrowLeft'?i-1:-1;
    if(j<0)return; e.preventDefault();
    var t=bs[(j+bs.length)%bs.length]; t.focus(); sel(t.dataset.target);
  });
  var saved=null; try{saved=localStorage.getItem('fig6tab')}catch(e){}
  sel(saved&&document.getElementById(saved)?saved:bs[0].dataset.target);
})();
</script>
"""


def img(path: Path) -> str:
    return ("data:image/png;base64,"
            + base64.b64encode(path.read_bytes()).decode("ascii"))


def seed_panel_html(seed: int, outdir: Path) -> str:
    data = json.loads((outdir / f"seed{seed}_table.json").read_text())
    band = data["band"]
    rows = data["rows"]
    png = img(outdir / f"seed{seed}_vs_stan.png")

    def tbl(fam, label):
        rs = [r for r in rows if r["family"] == fam]
        if not rs:
            return ""
        body = "".join(
            "<tr><td class=k>{size}</td><td>{corner:.4f}</td><td class=sd>{n}</td>"
            "<td class=sd>{train}</td><td>{so}</td><td class=\"{cls}\">{delta}</td></tr>".format(
                size=r["size"], corner=r["corner"], n=r["n"],
                train=(f'{r["train"]:.4f}' if r["train"] is not None else "&mdash;"),
                so=(f'{r["stan_oracle"]:.3f}' if r["stan_oracle"] else "&mdash;"),
                cls=("g" if r["stan_oracle"] and r["corner"] < r["stan_oracle"] else
                     "d" if r["stan_oracle"] else "sd"),
                delta=(f'{r["corner"] - r["stan_oracle"]:+.3f}' if r["stan_oracle"] else "&mdash;"))
            for r in rs)
        return f"""<div class=tw><table>
<caption>{label} &mdash; data seed {seed}, corner log-loss vs Stan Oracle refit on the same draw</caption>
<thead><tr><th>K</th><th>corner</th><th>cells</th><th>train cells</th>
<th>Stan Oracle</th><th>&Delta; vs Stan</th></tr></thead>
<tbody>{body}</tbody></table></div>"""

    return f"""
<section>
<h2>Data seed {seed} against the published Stan curves</h2>
<p class=kick>Item axis &middot; fresh generative draw &middot; corner-scored, corner held whole</p>
<div class=box>
<p><b>Both series come from the same data.</b> Stan has been refit on this draw &mdash; 20 HMC
fits, sampler settings and model matrix taken verbatim from the paper's &mdash; and scored on
exactly the cells Marformer is scored on. Earlier versions of this panel could only overlay the
published seed-42 Stan curves, which meant reading a Marformer&ndash;Stan gap across two
different draws of the world.</p>
<p>The published curves are still drawn, faintly dotted, as a reference. The distance between
each dotted line and its solid counterpart is how far <em>Stan alone</em> moves with the draw
&mdash; which turns out to be substantial, and is the reason the same-draw refit was necessary.</p>
</div>
<figure><img alt="Marformer on data seed {seed} against the published Stan curves" src="{png}">
<figcaption>Red: Marformer transductive and non-transductive, trained and corner-scored entirely
within the seed-{seed} tree. Blue solid: the four Stan variants refit on the same draw. Blue
dotted: the same four variants as published, on seed 42.</figcaption></figure>
{tbl("Marformer", "Transductive")}
{tbl("MarformerNT", "Non-transductive")}
<p class=note>Marformer protocol matches the corrected panels: training chunk 300, no shuffling,
400 epochs, model seed 42, last checkpoint, corner held whole in a chunk that carries training
context. Stan: <b>1 chain</b>, 300 warmup, 500 sampling &mdash; which is what the runner defaults
to and what every run directory on both draws contains, though the paper's appendix describes 4
chains; see the September tab. An earlier version of this note said Stan stopped at K=500 because
the paper stops there. That was wrong: the fits exist at every size out to K=4000 and had simply
landed after the scoring pass ran, so they were missing from the table rather than from the
cluster.</p>
</section>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="fig6/out/fig6_report.html")
    ap.add_argument("--seeds", default="101,105")
    ap.add_argument("--outdir", default="dataseed/out")
    ap.add_argument("--side-out", default="side/out",
                    help="where anneal_summary.json lives, for the September tab")
    ap.add_argument("--no-update-tab", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s]
    src = Path(a.report).read_text()
    outdir = Path(a.outdir)

    # Already tabbed? Take the first tab's content back out and rebuild around
    # it, so this can be re-run as new seeds land without nesting shells. The
    # sentinels bound it exactly -- counting <div> would have to track nesting.
    i = src.index("<div class=wrap>")
    head = src[:i]
    if ORIG_START in src:
        body_rest = src[src.index(ORIG_START) + len(ORIG_START):src.index(ORIG_END)]
        hm = re.search(r"(<header>.*?</header>)", src, re.S)
        header = hm.group(1) if hm else ""
    else:
        orig_body = src[i + len("<div class=wrap>"):]
        orig_body = orig_body[:orig_body.rindex("</div>")]
        hm = re.search(r"(<header>.*?</header>)", orig_body, re.S)
        header = hm.group(1) if hm else ""
        body_rest = orig_body.replace(header, "", 1) if header else orig_body

    if ".tabs{" not in head:
        head = head.replace("</style>", TAB_CSS + "</style>")

    btns, panels = [], []
    # The newest material goes first so the page opens on it. Returning readers
    # land on whatever tab they last chose (localStorage), so this only changes
    # the default for someone opening the link fresh -- which, for a page that is
    # circulated before a meeting, is most of them.
    if not a.no_update_tab and (outdir / "update_table.json").is_file():
        btns.append('<button data-target="tab-update" role=tab>September update</button>')
        panels.append('<div class=tabpanel id=tab-update role=tabpanel>'
                      + update_panel_html(outdir, Path(a.side_out)) + '</div>')
    hidden = " hidden" if panels else ""
    btns.append('<button data-target="tab-orig" role=tab>Seed 42 &middot; as published</button>')
    panels.append(f'<div class=tabpanel id=tab-orig role=tabpanel{hidden}>'
                  f'{ORIG_START}{body_rest}{ORIG_END}</div>')
    for s in seeds:
        if not (outdir / f"seed{s}_table.json").is_file():
            print(f"  skip seed {s}: no payload yet"); continue
        btns.append(f'<button data-target="tab-s{s}" role=tab>Data seed {s} vs Stan</button>')
        panels.append(f'<div class=tabpanel id=tab-s{s} role=tabpanel hidden>'
                      f'{seed_panel_html(s, outdir)}</div>')

    out = (head + "<div class=wrap>\n" + header
           + '\n<div class=tabs role=tablist>' + "\n".join(btns) + '</div>\n'
           + "\n".join(panels)
           + TAB_JS + "</div>\n")
    Path(a.report).write_text(out)
    print(f"{len(btns)} tabs -> {a.report}  ({len(out)/1e6:.2f} MB)")

if __name__ == "__main__":
    main()
