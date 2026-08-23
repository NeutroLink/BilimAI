"""BilimAI — who wrote this stroke? (2026-08-22)

FOUNDER RULE, 2026-08-22: **red and its shades belong to the teacher. Every other colour belongs to
the pupil — including green.** No exceptions, no second guessing.

WHY THIS EXISTS. Guardrails that measure "how much writing did we miss?" need an ink mask, and the
obvious ink mask — threshold on brightness — is wrong here: on lined and squared paper the RULING is
dark too, so it counts as ink and every invariant behaves differently per paper type. Real assignments
arrive on lined, squared, narrow-lined and plain paper, so a paper-dependent guardrail is no guardrail.

Colour fixes it. Ruling is pale and washed out; pen ink is dark and saturated. Classify ink by colour
and the same measurement works on every paper. The founder's rule then falls out for free.

    from bilimai.ink import ink_masks
    pupil, teacher = ink_masks(img_bgr)      # two boolean masks, same HxW as the page

Deliberately NOT handled, per instruction: green-pen teachers (green is pupil), and any attempt to
infer authorship from handwriting style. Red is the whole rule.
"""
from __future__ import annotations
import numpy as np

__all__ = ["ink_masks", "ink_stats"]

# OpenCV hue is 0-179, so red straddles the wrap-around: ~0-12 and ~168-179.
RED_LO, RED_HI = 12, 168
SAT_MIN = 60          # a pixel this saturated is coloured ink (blue, red, purple) whatever its darkness
REL_DROP = 0.12       # ...or a pixel at least this much darker than ITS OWN local paper is ink too

# WHY RELATIVE AND NOT ABSOLUTE (2026-08-22). The first version used fixed cuts — saturation >= 60 or
# value <= 130. On 2718 the two pages of the spread are lit identically (paper value 203 vs 199) but
# the LEFT page is written in a much fainter hand: ink value 151, contrast 52, saturation 16. It fell
# between both cuts and almost none of it registered — 1.9 % of the half detected as ink against 9.0 %
# on the right. Faint pen, pencil and old ink all live in that gap, and so does any washed-out photo.
# Measuring darkness against the LOCAL paper instead makes the rule indifferent to how pale the writing
# is, how bright the photo is, and how the light falls across the page.


def ink_masks(img_bgr: np.ndarray, sat_min: int = SAT_MIN, rel_drop: float = REL_DROP,
              min_area: int = 6, drop_rules: bool = True, limit_to_page: bool = True):
    """Return (pupil_mask, teacher_mask) as boolean arrays.

    Ink = saturated-and-not-bright, OR simply dark (so black and pencil survive a washed-out photo).
    Among ink pixels, red hue -> teacher, everything else -> pupil.

    Photographed pages carry a colour cast, so hue is measured AFTER a grey-world white balance;
    without it a warm classroom photo pushes blue ink toward purple and, at the margins, red.

    Two artefacts, both observed on real pages 2026-08-22 (out/ink_rule.png) and both corrected here:

    `limit_to_page` — the dark table or shadow beyond the page edge is dark, so it was being counted as
    pupil ink (a solid band along the bottom of 2718 and 2047). Ink is now restricted to the paper
    itself, found as the largest bright region.

    `drop_rules` — the PRINTED red margin line is red and saturated, so it was being counted as teacher
    ink on every page including ones with no marking at all. Printed rules are rejected by shape: a
    component spanning more than 60 % of the page in one direction while staying thinner than 1.5 % in
    the other. A teacher's underline spans a word or a line, never the whole page, so it survives.
    """
    import cv2
    img = _white_balance(img_bgr)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    ink = (s >= sat_min) | _darker_than_paper(img, rel_drop)
    if limit_to_page:
        ink &= _page_mask(img)
    if min_area > 1:
        ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN,
                               np.ones((2, 2), np.uint8)).astype(bool)
    if drop_rules:
        ink &= ~_printed_rules(ink)
        ink &= ~_ruling(ink)

    is_red = ((h <= RED_LO) | (h >= RED_HI)) & (s >= sat_min)
    teacher = ink & is_red
    pupil = ink & ~is_red
    return pupil, teacher


def _darker_than_paper(img_bgr: np.ndarray, rel_drop: float = REL_DROP, ds: int = 4) -> np.ndarray:
    """True where a pixel is at least `rel_drop` darker than the paper immediately around it.

    The paper level is estimated by a morphological CLOSE with a kernel wider than any pen stroke:
    that erases the writing and leaves the page. Dividing by it flattens illumination, shadow across
    the gutter, and the photo's exposure all at once, so one threshold works on a faint hand and a bold
    one alike. Computed on a downscaled copy for speed, then resized back."""
    import cv2
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (g.shape[1] // ds, g.shape[0] // ds), interpolation=cv2.INTER_AREA)
    k = max(9, (min(small.shape) // 40) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.resize(bg, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return (g.astype(np.float32) / np.maximum(bg, 1.0)) < (1.0 - rel_drop)


def _ruling(ink: np.ndarray, min_run: float = 0.05) -> np.ndarray:
    """Printed ruling and grids, found by SHAPE rather than darkness.

    MEASURED 2026-08-22, and this is why intensity cannot do it: on squared paper the grid darkens the
    page by a median of 0.146 relative to its surroundings, while the faint handwriting on 2718 darkens
    it by only 0.073. The ruling is DARKER than the writing. Any threshold that keeps that writing keeps
    the grid, so the two are not separable by level, colour or contrast — only by form.

    Opening with a long flat kernel keeps only horizontal runs; a long tall kernel only vertical ones.
    `min_run` is 5 % of the page: at 18 % almost nothing was caught, because writing crossing a rule
    breaks it into shorter runs (measured: 66k px caught at 18 %, 529k at 5 %).

    KNOWN COST: at 5 % a long pen stroke or a teacher's underline can be caught too. That is the price
    of separating a grid that is darker than the handwriting on top of it."""
    import cv2
    H, W = ink.shape
    u = ink.astype(np.uint8)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(min_run * W)), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, int(min_run * H))))
    lines = cv2.morphologyEx(u, cv2.MORPH_OPEN, hk) | cv2.morphologyEx(u, cv2.MORPH_OPEN, vk)
    return cv2.dilate(lines, np.ones((3, 3), np.uint8)).astype(bool)


def _page_mask(img_bgr: np.ndarray) -> np.ndarray:
    """The paper itself: the largest bright region, interior holes filled.

    The hole-filling floods from OUTSIDE a 1-pixel border added around the frame. Flooding from (0,0)
    of the original image is wrong whenever the paper reaches the corner — the seed lands inside the
    page, the fill inverts, and the mask comes back covering 100 % of the frame (observed on 2718 and
    2047, which is why the first version of this fix did nothing)."""
    import cv2
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thr = max(90, int(np.percentile(g, 55) * 0.75))
    bright = (g >= thr).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return np.ones(g.shape, bool)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    page = (lab == i).astype(np.uint8)

    padded = cv2.copyMakeBorder(page, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = padded.copy()
    cv2.floodFill(ff, np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8), (0, 0), 1)
    holes = (1 - ff)[1:-1, 1:-1]                 # everything the outside flood could not reach
    page = (page | holes).astype(bool)
    if page.mean() < 0.20:                       # implausible -> do not restrict at all
        return np.ones(g.shape, bool)
    return page


def _printed_rules(ink: np.ndarray, span: float = 0.60, thin: float = 0.015) -> np.ndarray:
    """Mask of printed ruling / margin lines: components crossing most of the page in one direction
    while staying very thin in the other. A teacher's underline spans a word or a line, never the
    whole page, so it survives this test and stays classified as teacher ink."""
    import cv2
    H, W = ink.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    out = np.zeros(ink.shape, bool)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]; ht = stats[i, cv2.CC_STAT_HEIGHT]
        if (w > span * W and ht < thin * H) or (ht > span * H and w < thin * W):
            out |= (lab == i)
    return out


# _blobs_too_big() REMOVED 2026-08-22. It rejected components spanning >50 % of the page in both
# directions, to kill the dark surround of a tilted photograph. Two things made it wrong:
#   1. Once ink is detected by LOCAL contrast, that surround is never flagged in the first place — a
#      uniformly dark region has no local contrast, so gray/background ~ 1 there.
#   2. With faint strokes now detected, a page of handwriting connects into ONE page-spanning
#      component, and the filter deleted it: 2047 lost 92 % of its writing (1,405,225 px -> 110,419).
# The page mask alone handles the surround. Do not reintroduce a size-based reject.


def _white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """Grey-world balance on the PAPER only (the bright 60 % of pixels), so a page dominated by dark
    ink does not drag the correction. Cheap, and it stops a warm photo from reddening blue ink."""
    import cv2
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    paper = g >= np.percentile(g, 40)
    if paper.sum() < 100:
        return img_bgr
    means = np.array([img_bgr[..., c][paper].mean() for c in range(3)], dtype=np.float32)
    if (means <= 1).any():
        return img_bgr
    gain = means.mean() / means
    return np.clip(img_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def ink_stats(img_bgr: np.ndarray, boxes=None) -> dict:
    """Paper-agnostic guardrail numbers.

    `covered` is the one that matters in production: the fraction of the PUPIL's ink that falls inside
    some detector box. It needs no ground truth, so it works on a live page in a school where nobody
    will ever tell us the right answer. A page where 40 % of the writing sits outside every box is
    broken whatever the language, the paper or the assignment type."""
    pupil, teacher = ink_masks(img_bgr)
    tot = int(pupil.sum())
    out = {"pupil_px": tot, "teacher_px": int(teacher.sum()),
           "teacher_share": float(teacher.sum() / max(1, teacher.sum() + tot))}
    if boxes is not None and tot:
        m = np.zeros(pupil.shape, bool)
        H, W = pupil.shape
        for b in boxes:
            x0, y0 = max(0, int(b[0])), max(0, int(b[1]))
            x1, y1 = min(W, int(b[2])), min(H, int(b[3]))
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = True
        out["covered"] = float((pupil & m).sum() / tot)
    return out
