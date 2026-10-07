#!/usr/bin/env python
"""Does any installed figure have ink touching its own border?

Every Domain 3 figure is saved with `savefig.bbox: "tight"`, and matplotlib's
tight box leaves `savefig.pad_inches` (0.1in by default) of white around the
content. So ink in the outermost band is not a tight crop -- it is a label, a
legend or a tick that was cut off. That makes clipping detectable instead of a
thing we notice by eye, one panel at a time.

Reports, per figure, which edges have ink in the margin band and how much, so a
one-pixel antialiasing speck does not read the same as a sliced axis title.

    python check_figure_clipping.py PLOTS_DIR [PLOTS_DIR ...] [--dpi 300]
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def ink_mask(path: Path) -> np.ndarray:
    """True where the image differs from its own background.

    The background is taken as the modal colour of the one-pixel border ring,
    not assumed to be white: these PNGs come from several tools and are variously
    RGB on white, RGBA with a transparent ground, and palette images. An earlier
    version thresholded on "darker than near-white" and flagged 42 of 45 figures,
    because a transparent or tinted ground reads as ink everywhere. Deriving the
    ground from the image itself is what makes the test mean anything.
    """
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        arr = np.asarray(im)
        if (arr[..., 3] < 255).any():                  # genuine transparency
            return arr[..., 3] > 24
        im = im.convert("RGB")
    arr = np.asarray(im.convert("RGB")).astype(np.int16)
    ring = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    cols, counts = np.unique(ring.reshape(-1, 3), axis=0, return_counts=True)
    bg = cols[counts.argmax()]
    return np.abs(arr - bg).max(axis=-1) > 24


def scan(path: Path, dpi: int, pad_in: float, frac: float):
    m = ink_mask(path)
    h, w = m.shape
    band = max(1, int(round(pad_in * dpi * frac)))
    edges = {
        "top":    m[:band, :],
        "bottom": m[-band:, :],
        "left":   m[:, :band],
        "right":  m[:, -band:],
    }
    hits = {}
    for name, blk in edges.items():
        n = int(blk.sum())
        if not n:
            continue
        # a run of ink along the edge is a cut label; a few specks are not
        line = blk.any(axis=0) if name in ("top", "bottom") else blk.any(axis=1)
        runs, cur = 0, 0
        for v in line:
            cur = cur + 1 if v else 0
            runs = max(runs, cur)
        hits[name] = (n, runs, len(line))
    return hits, (h, w), band


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--pad-inches", type=float, default=0.1)
    ap.add_argument("--band-frac", type=float, default=0.45,
                    help="fraction of the pad to treat as 'must be blank'")
    ap.add_argument("--min-run", type=int, default=12,
                    help="longest edge-parallel ink run, in px, to call it clipped")
    a = ap.parse_args()

    pngs = sorted(p for d in a.dirs for p in Path(d).rglob("*.png"))
    if not pngs:
        print("no PNGs found"); return 1
    bad = []
    for p in pngs:
        hits, (h, w), band = scan(p, a.dpi, a.pad_inches, a.band_frac)
        real = {k: v for k, v in hits.items() if v[1] >= a.min_run}
        flag = "CLIPPED" if real else ("specks " if hits else "clean  ")
        where = " ".join(f"{k}(run={v[1]}px/{v[2]})" for k, v in sorted(real.items()))
        print(f"{flag} {str(p):<78} {w}x{h} band={band}px  {where}")
        if real:
            bad.append(p)
    print(f"\n{len(bad)} of {len(pngs)} figures have ink cut at an edge")
    for p in bad:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
