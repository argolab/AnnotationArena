#!/usr/bin/env python
"""
Build the interactive Domain 3 report by injecting collected results into the
HTML template.

The template is kept separate and committed; the built page is not, because it
carries a few MB of embedded JSON and is reproducible from the run tree in one
command.

    python collect_domain3.py --run-dir RERUN/runs/seeds-01 -o /tmp/d3.json
    python build_report.py --data /tmp/d3.json -o RERUN/reports/build/domain3.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSON from collect_domain3.py")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--template", default=str(HERE / "report_template.html"))
    ap.add_argument("--title", default="", help="override the <title>")
    ap.add_argument("--marker", default="", help="short badge shown on every figure title")
    ap.add_argument("--baseline", default="",
                    help="a collect_domain3.py JSON to overlay as a dashed reference "
                         "series (only its per-run metrics are embedded, not curves)")
    ap.add_argument("--baseline-label", default="baseline")
    args = ap.parse_args()

    template = Path(args.template).read_text()
    raw = Path(args.data).read_text()

    # The payload rides in a <script type="application/json"> block, so the only
    # sequence that can break out of it is a literal "</script"; JSON escaping
    # of "<" keeps the document well-formed without touching the parsed values.
    payload = raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")

    baseline = {"label": args.baseline_label, "runs": {}}
    if args.baseline:
        src = json.loads(Path(args.baseline).read_text())
        for r in src["runs"]:
            key = f"{r['family']}|{r['axis']}|{r['size']}|{r['seed']}"
            baseline["runs"][key] = {"last": r["test"]["last"], "best": r["test"]["best"],
                                     "runtime_min": r["runtime_min"]}
        print(f"baseline: {len(baseline['runs'])} runs from {args.baseline}")

    html = template.replace("__DATA__", payload)
    html = html.replace("__BASELINE__", json.dumps(baseline).replace("<", "\\u003c"))
    if args.title:
        html = html.replace("<title>Domain 3 Replication</title>", f"<title>{args.title}</title>")
    if args.marker:
        html = html.replace('const FIGURE_MARKER = "";', f'const FIGURE_MARKER = {json.dumps(args.marker)};')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"{out}  {out.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
