#!/usr/bin/env python
"""
Render the built report headlessly and fail on anything a reader would see.

A self-contained page can be structurally valid and still be broken: a throwing
handler leaves half the charts empty, a token defined only inside a media query
paints one theme's text on the other's ground, and neither shows up in a diff.
This loads the page in Chromium, drives the controls the reader will actually
press, and reports console errors, empty chart hosts, and contrast in both
themes. Screenshots go next to the page for a visual look.

    python check_report.py --page RERUN/reports/build/domain3.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

# Controls the reader will press, as (rail container id, button label).
CLICKS = [
    ("c-ckpt", "Best on test"), ("c-ckpt", "Last epoch"),
    ("c-seed", "Per seed"), ("c-seed", "Aggregate"),
    ("c-arms", "Non-transductive"), ("c-arms", "Non-transductive"),
    ("c-axis", "Annotator axis"), ("c-axis", "Item axis"),
    ("c-metric", "Accuracy"), ("c-metric", "Training time"), ("c-metric", "Test log-loss"),
]


def luminance(rgb):
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def parse_rgb(s):
    nums = [float(x) for x in s.replace("rgba(", "").replace("rgb(", "").rstrip(")").split(",")[:3]]
    return nums


def contrast(fg, bg):
    l1, l2 = luminance(parse_rgb(fg)), luminance(parse_rgb(bg))
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True)
    ap.add_argument("--shots", default="", help="directory for screenshots")
    args = ap.parse_args()
    page_path = Path(args.page).resolve()
    shots = Path(args.shots) if args.shots else page_path.parent / "shots"
    shots.mkdir(parents=True, exist_ok=True)

    problems: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for theme in ("light", "dark"):
            ctx = browser.new_context(viewport={"width": 1440, "height": 1000},
                                      color_scheme=theme)
            page = ctx.new_page()
            errors: list[str] = []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.goto(page_path.as_uri())
            page.wait_for_timeout(700)

            # every chart host must contain an <svg>, not the empty-state text
            hosts = page.eval_on_selector_all(
                "#small-multiples .body, #focus-chart",
                "els => els.map(e => ({id: e.id || e.parentElement.querySelector('.name')?.textContent,"
                " svg: !!e.querySelector('svg'), txt: e.textContent.trim().slice(0,60)}))")
            for h in hosts:
                if not h["svg"]:
                    problems.append(f"[{theme}] empty chart: {h['id']} -> {h['txt']!r}")
            if len(hosts) != 7:
                problems.append(f"[{theme}] expected 7 chart hosts, found {len(hosts)}")

            # readable text on the painted ground
            body_bg = page.eval_on_selector("body", "e => getComputedStyle(e).backgroundColor")
            if "rgba(0, 0, 0, 0)" in body_bg:
                problems.append(f"[{theme}] body background is transparent")
            for sel in ("h1", ".sub", ".stamp .v", "td", "th", ".note"):
                el = page.query_selector(sel)
                if not el:
                    continue
                fg = page.eval_on_selector(sel, "e => getComputedStyle(e).color")
                ratio = contrast(fg, body_bg)
                if ratio < 4.5:
                    problems.append(f"[{theme}] low contrast {ratio:.1f}:1 on {sel}")

            # drive the controls
            for container, label in CLICKS:
                btn = page.query_selector(f"#{container} button:text-is('{label}')")
                if not btn:
                    problems.append(f"[{theme}] control not found: {container}/{label}")
                    continue
                btn.click()
                page.wait_for_timeout(120)

            # hover must produce a tooltip. Checked before the click, because
            # selecting a run smooth-scrolls, and scrolling hides the tip by
            # design -- which would otherwise look like a broken tooltip.
            pt = page.query_selector("#focus-chart .pt")
            if pt:
                # Playwright scrolls before hovering, and a scroll hides the tip
                # by design; settle the scroll first so this tests hovering a
                # point that is already on screen, which is the real case.
                pt.scroll_into_view_if_needed()
                page.wait_for_timeout(350)
                # force=True skips the actionability wait. Two series can put a
                # point at the same coordinates (they do at K=100), so the one
                # drawn second intercepts the pointer and a normal hover blocks
                # forever. What is under test is the handler, not hit-testing.
                pt.hover(force=True)
                page.wait_for_timeout(250)
                if page.eval_on_selector("#tip", "e => getComputedStyle(e).opacity") == "0":
                    problems.append(f"[{theme}] hovering a point produced no tooltip")
                elif not page.inner_text("#tip").strip():
                    problems.append(f"[{theme}] tooltip is empty")

            # clicking a data point must populate the detail section
            if pt:
                pt.click(force=True)
                page.wait_for_timeout(400)
                head = page.inner_text("#detail-heading")
                if "No run selected" in head:
                    problems.append(f"[{theme}] clicking a point did not select a run")
                if not page.query_selector("#ckpt-chart svg"):
                    problems.append(f"[{theme}] checkpoint chart empty after selecting a run")
                if not page.query_selector("#hist-chart svg"):
                    problems.append(f"[{theme}] training-curve chart empty after selecting a run")
            else:
                problems.append(f"[{theme}] no data points rendered in the focus chart")

            rows = page.eval_on_selector_all("#table tbody tr", "els => els.length")
            if rows == 0:
                problems.append(f"[{theme}] run table is empty")

            page.screenshot(path=str(shots / f"{theme}.png"), full_page=True)
            if errors:
                problems += [f"[{theme}] console: {e[:200]}" for e in errors[:6]]
            ctx.close()
        browser.close()

    print(f"screenshots -> {shots}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("report renders cleanly in both themes; controls, hover and drill-down all work")


if __name__ == "__main__":
    main()
