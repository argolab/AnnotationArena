"""One multiplier for every font size in the Domain 3 / LLM-Rubric figures.

Every one of these figures is included in the paper inside a 0.48\\linewidth
subfigure, so a 17-inch-wide plot is reproduced at roughly three inches and its
11pt tick labels land near 3pt on the page. The fix is to grow the type
RELATIVE to the plot -- growing the figure instead changes nothing, because
LaTeX scales the whole thing back down to the column width either way.

    DOMAIN3_FONT_SCALE        ticks, axis labels, legends           (default 3)
    DOMAIN3_PANEL_FONT_SCALE  extra factor inside multi-panel grids (default 1.4)
    DOMAIN3_TITLE_SCALE       figure/axis titles                    (default 1.8)

Titles get their own, gentler factor. They start out as the largest text on the
figure, so the same 3x that makes 11pt ticks legible makes a 16pt suptitle
overflow the canvas and eat the plot area -- measured, not assumed, on the
first render. The paper's caption carries the title anyway.

The calibration figures are a 2x2 grid inside one of those half-width
subfigures, so each panel is about a quarter of the text width and needs the
extra factor on top.

Set DOMAIN3_FONT_SCALE=1 to get the pre-2026-10-06 figures back exactly.
"""
from __future__ import annotations
import os

SCALE = float(os.environ.get("DOMAIN3_FONT_SCALE", "3.0"))
PANEL_EXTRA = float(os.environ.get("DOMAIN3_PANEL_FONT_SCALE", "1.4"))
TITLE_SCALE = float(os.environ.get("DOMAIN3_TITLE_SCALE", "1.8"))

# rcParams whose values are point sizes. Anything not listed is left alone --
# scaling "lines.linewidth" by 3 would be a different change with a different
# justification, and marker/line weights are handled separately where needed.
_FONT_KEYS = (
    "font.size", "axes.labelsize", "axes.titlesize",
    "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
    "figure.titlesize",
)


def fs(points: float) -> float:
    """A font size in points, scaled."""
    return points * SCALE


def ts(points: float) -> float:
    """A title size in points, scaled gently -- see the module docstring."""
    return points * TITLE_SCALE


def pfs(points: float) -> float:
    """A font size inside a multi-panel grid, scaled harder."""
    return points * SCALE * PANEL_EXTRA


def scale_rcparams(rc) -> None:
    """Scale every font-size entry of an rcParams-like mapping, in place."""
    for k in _FONT_KEYS:
        v = rc.get(k)
        if isinstance(v, (int, float)):
            rc[k] = v * (TITLE_SCALE if "titlesize" in k else SCALE)
