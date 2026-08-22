"""BilimAI — ink-aware vertical crop (E4.2, 2026-08-22). NOT wired into the pipeline; candidate only.

WHY THIS EXISTS
---------------
Measured on the sealed exam set (plans/CROP-GEOMETRY-2026-08-22.md): the production detector box cuts
2.3 % of each matched line's ink on average, two thirds of it BELOW the box — descender tails (у, р, з,
щ, д). Half of lines lose nothing; a seventh lose more than 5 %.

The obvious fix — pad every box more — does not work, because growth is already a fitted fraction of the
box's own height, and a box that is too small gets a pad that is too small. Page-relative growth was
tried and measured null (-0.0023 CER, p = 0.15).

The obvious other fix — mask everything outside the line's word polygons — was tried and measured
HARMFUL (+0.0717 CER on bad boxes), because the mask was built from the detector's own word boxes, so
any word the detector missed was painted out. `«2 закончить изложение»` became `«2 2-»`.

So this module obeys one rule that makes both failures impossible:

    THE CROP MAY ONLY EVER GROW. It never masks, never erases, never shrinks.

The output is always a superset of today's crop, so the worst case is that it changes nothing.

HOW IT DECIDES
--------------
1. Binarise a window around the box (Otsu; these are photographed notebook pages, blue ink on ruled paper).
2. Connected components of ink.
3. A component belongs to THIS line if it touches the box's core band (the middle `core` of its height)
   AND its centroid sits inside the box's vertical span. The centroid test is what keeps a descender
   dropping in from the line ABOVE from being claimed — that stroke's centroid is above our box.
4. Ruled paper lines are rejected: wide, flat components spanning most of the box width.
5. Grow top and bottom to contain the surviving components, capped by `max_grow` and, when the caller
   supplies neighbouring line boxes, by the midpoint to the neighbour's own ink.

Measured headroom (same plan doc): in 19 of 20 lines there is room to stop cutting the writing without
reaching the neighbouring line at all — the gap is 5-6x the growth needed.

MEASURED, 732 matched lines over 20 exam pages (2026-08-22), ink cut before = 2.30 %.
These are the numbers AFTER the neighbour-cap fix described in grow_box_to_ink — the first version
found no neighbours at all (production boxes overlap, so an edge test never fires) and happily ate a
whole neighbouring line to score 61 % recovery. That number was a bug, not a result.

    max_grow  core   ink cut   recovered   p90 growth   boxes grown >30 %
       0.15    0.4     1.44 %      38 %        18.5 %           0 %
       0.25    0.4     1.26 %      45 %        24.9 %           6 %
       0.35    0.4     1.21 %      47 %        30.6 %          10 %      <- default
       0.45    0.4     1.18 %      49 %        29.6 %          10 %
       0.35    0.6     1.53 %      34 %        24.4 %           7 %

Median growth is 0 % at every setting: 58 % of boxes are already clean and are returned untouched.
Recovery saturates by 0.35. Cost is 1.0 ms per line, against ~700 ms per page for detection itself.
Lines made worse: 0, as the monotone rule guarantees.

CEILING OF THE RECTANGLE. The remaining ~50 % cannot be recovered by any rectangle. On densely
written pages the descender we want and the neighbour's body occupy the SAME rows, just different
columns (2718.jpg: 21 % cut, and the cap allows only 4 % growth). Taking it needs a per-column crop
— the founder's interlocking-polygon idea. That is only safe if the polygon is built from INK; built
from the detector's own word boxes it deletes any word the detector missed (measured +0.0717 CER).

WHAT IS STILL UNKNOWN: whether recovering that ink improves CER. Growing the crop also drags in more
of the neighbouring lines' ink, and only a paired reader A/B can settle which way that trades. The
whole crop-geometry ceiling is +0.0074 CER (crop_cost.py), so the prize here is a fraction of one
point. See plans/CROP-GEOMETRY-2026-08-22.md §4. Geometry tells you the direction; only CER decides.
"""
from __future__ import annotations
import numpy as np

__all__ = ["ink_extent", "grow_box_to_ink"]


def _binarise(win: np.ndarray) -> np.ndarray:
    """Ink mask for one window. Otsu, with a guard against windows that are all paper or all ink."""
    import cv2
    if win.size < 64:
        return np.zeros_like(win, dtype=bool)
    t, _ = cv2.threshold(win, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = win < t
    f = float(ink.mean())
    if f < 0.002 or f > 0.6:            # blank paper, or a shadow/dark photo where Otsu split noise
        return np.zeros_like(win, dtype=bool)
    return ink


def ink_extent(gray: np.ndarray, box, core: float = 0.4, max_grow: float = 0.35,
               neigh_up: float | None = None, neigh_dn: float | None = None,
               rule_w: float = 0.85, rule_h: float = 0.12, min_px: int = 12):
    """Grow `box`'s vertical extent so this line's own ink is not cut. Returns (y0, y1), never shrunk.

    gray      : full-page 8-bit grayscale
    box       : [x0, y0, x1, y1] in page pixels
    core      : fraction of the box height, centred, that defines "certainly this line"
    max_grow  : hard cap on growth each way, as a fraction of the box height
    neigh_up  : hard floor for growing UP   — the bottom of the core of the line above
    neigh_dn  : hard ceiling for growing DOWN — the top of the core of the line below
    rule_w/h  : a component wider than rule_w of the box AND flatter than rule_h is ruled paper, not ink
    min_px    : components smaller than this are speckle
    """
    import cv2
    H, W = gray.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    h = y1 - y0
    if h <= 2 or x1 - x0 <= 2:
        return y0, y1

    pad = max_grow * h
    wy0, wy1 = int(max(0, y0 - pad)), int(min(H, y1 + pad))
    wx0, wx1 = int(max(0, x0)), int(min(W, x1))
    if wy1 - wy0 < 4 or wx1 - wx0 < 4:
        return y0, y1

    ink = _binarise(gray[wy0:wy1, wx0:wx1])
    if not ink.any():
        return y0, y1

    n, lab, stats, cent = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    bw = wx1 - wx0
    core_top = (y0 + 0.5 * h * (1 - core)) - wy0
    core_bot = (y1 - 0.5 * h * (1 - core)) - wy0
    box_top, box_bot = y0 - wy0, y1 - wy0

    top, bot = box_top, box_bot
    for i in range(1, n):
        cx0, cy0, cw, ch, area = stats[i]
        if area < min_px:
            continue
        if cw > rule_w * bw and ch < rule_h * h:          # ruled paper line
            continue
        cy1 = cy0 + ch
        if cy1 <= core_top or cy0 >= core_bot:            # never enters this line's core -> not ours
            continue
        cyc = cent[i][1]
        if cyc < box_top or cyc > box_bot:                # centre lives in a neighbouring line -> not ours
            continue
        top = min(top, cy0)
        bot = max(bot, cy1)

    ny0, ny1 = wy0 + top, wy0 + bot
    ny0 = max(ny0, y0 - pad); ny1 = min(ny1, y1 + pad)    # cap
    ny0 = min(ny0, y0); ny1 = max(ny1, y1)                # MONOTONE: never shrink
    if neigh_up is not None:
        ny0 = max(ny0, min(neigh_up, y0))     # never past the neighbour's core, but never shrink either
    if neigh_dn is not None:
        ny1 = min(ny1, max(neigh_dn, y1))
    return float(max(0.0, ny0)), float(min(float(H), ny1))


def grow_box_to_ink(gray: np.ndarray, box, all_boxes=None, core: float = 0.4, **kw):
    """Convenience wrapper: [x0,y0,x1,y1] -> [x0,y0',x1,y1'], finding the vertical neighbours itself.

    Neighbours are found by BOX CENTRE, not by box edge. Production line boxes are grown 0.25 of their
    height at the top and 0.365 at the bottom, so consecutive boxes routinely OVERLAP — an edge test
    ("is the other box entirely above mine?") therefore finds no neighbour at all and the growth cap
    silently never fires. That bug let the crop swallow a whole neighbouring line on tightly written
    pages (observed on 2015.jpg and 2718.jpg, out/inkcrop_examples.png).

    The cap is the neighbour's CORE — the middle `core` fraction of its box. Growing into a
    neighbour's ascenders or descenders is unavoidable and already happens today; growing into the
    body of its letters is what confuses the reader.

    A neighbour counts only if it overlaps this box by at least a quarter of its width — that keeps a
    margin scribble or the OTHER PAGE of a two-page spread from capping growth (these scans are
    4000x3000 spreads; anything y-based must respect columns)."""
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    cy = 0.5 * (y0 + y1)
    nu = nd = None
    if all_boxes:
        w = max(1.0, x1 - x0)
        for o in all_boxes:
            ox0, oy0, ox1, oy1 = [float(v) for v in o[:4]]
            if min(x1, ox1) - max(x0, ox0) < 0.25 * w:
                continue
            ocy, oh = 0.5 * (oy0 + oy1), oy1 - oy0
            if abs(ocy - cy) < 1e-6:
                continue
            if ocy < cy:                                   # line above -> its core BOTTOM is our floor
                v = ocy + 0.5 * core * oh
                nu = v if nu is None else max(nu, v)
            else:                                          # line below -> its core TOP is our ceiling
                v = ocy - 0.5 * core * oh
                nd = v if nd is None else min(nd, v)
    ny0, ny1 = ink_extent(gray, box, core=core, neigh_up=nu, neigh_dn=nd, **kw)
    return [x0, ny0, x1, ny1]
