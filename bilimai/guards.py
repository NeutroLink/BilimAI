"""BilimAI — production guardrails. 2026-08-22/23.

Three checks that run on every page in production, none of which needs a correct answer to compare
against. That last point is the whole design constraint: in a school there is no ground truth, ever.

  1. PROMPT FINGERPRINT   the prompt is part of the model, so version it like one
  2. OUTPUT SHAPE         reject malformed or implausible model output before anything consumes it
  3. PAGE INVARIANTS      measure the page against itself: was any of the pupil's writing left outside
                          every box, are the boxes a sane size, do they overlap absurdly

WHY EACH ONE EXISTS — every trigger below is a failure this project actually had:

  * R6 read CER 0.78266 instead of 0.02188 because TEMPLATE=qwen3_vl is a *thinking* template and
    prefixed an empty <think></think> to 100 % of predictions. A prompt/template changed underneath a
    model and nothing noticed. -> check_prompt()
  * 2026-08-22, a VLM asked for line boxes returned 84 items whose raw text ended mid-token («"tex»),
    having burned its whole token budget on the top half of the page; on the other half it returned
    6 boxes for a 64-line page and closed the JSON cleanly, as if finished. -> check_output()
  * The production detector leaves 13 % of lines unboxed, and on one page a third of the pupil's ink
    falls outside every box. Nobody downstream can tell. -> check_page()

Deliberately NOT here: a canary set. Thresholds for that must come from a model we are happy with, and
we do not have one yet (founder, 2026-08-23). Calibrate against real runs, then freeze.
"""
from __future__ import annotations
import hashlib
import json
import re

__all__ = ["prompt_fingerprint", "check_prompt", "check_output", "check_page", "Verdict"]


class Verdict:
    """OK = use it. REVIEW = use but flag for a human. REJECT = do not use."""
    OK, REVIEW, REJECT = "OK", "REVIEW", "REJECT"

    def __init__(self, level=OK, reasons=None, stats=None):
        self.level = level
        self.reasons = reasons or []
        self.stats = stats or {}

    def merge(self, other: "Verdict") -> "Verdict":
        order = {Verdict.OK: 0, Verdict.REVIEW: 1, Verdict.REJECT: 2}
        lvl = self.level if order[self.level] >= order[other.level] else other.level
        v = Verdict(lvl, self.reasons + other.reasons, {**self.stats, **other.stats})
        return v

    def __repr__(self):
        return f"<{self.level}{': ' + '; '.join(self.reasons) if self.reasons else ''}>"


# ---------------------------------------------------------------- 1. prompt fingerprint
def prompt_fingerprint(prompt: str) -> str:
    """Stable short hash of a prompt. Whitespace-normalised, so reformatting is not a version change
    but a wording change is."""
    norm = re.sub(r"\s+", " ", prompt).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]


def check_prompt(prompt: str, expected: str | None) -> Verdict:
    """A prompt that drifts from the one the model was validated with is an unversioned dependency.
    Ship `expected` next to the adapter and fail loudly, not quietly."""
    got = prompt_fingerprint(prompt)
    if expected is None:
        return Verdict(Verdict.REVIEW, [f"prompt not pinned (fingerprint {got})"], {"prompt_fp": got})
    if got != expected:
        return Verdict(Verdict.REJECT,
                       [f"prompt fingerprint {got} != pinned {expected} — model was not validated with this prompt"],
                       {"prompt_fp": got})
    return Verdict(Verdict.OK, [], {"prompt_fp": got})


# ---------------------------------------------------------------- 2. output shape
def check_output(raw: str, items: list, page_wh: tuple[int, int],
                 expected_lines: int | None = None, max_ratio: float = 3.0,
                 coord_max: float = 1000.0) -> Verdict:
    """Reject malformed or implausible detector output. No model needed; this is arithmetic."""
    W, H = page_wh
    reasons, level = [], Verdict.OK
    s = (raw or "").strip()

    if not items:
        return Verdict(Verdict.REJECT, ["no items parsed from model output"], {"n_items": 0})

    # truncation: a complete JSON array ends with ']'. A reply cut off by the token cap does not.
    if s and not s.endswith("]"):
        reasons.append("output does not end with ']' — reply truncated by the token limit")
        level = Verdict.REJECT

    bad_geom = 0
    for b, *_ in ((it.get("bbox") or it.get("bbox_2d"),) for it in items):
        if not b or len(b) != 4:
            bad_geom += 1; continue
        x0, y0, x1, y1 = b
        if x1 <= x0 or y1 <= y0:
            bad_geom += 1
        elif max(x1, y1) > coord_max * 1.02 and max(x1, y1) > max(W, H) * 1.02:
            bad_geom += 1
    if bad_geom:
        reasons.append(f"{bad_geom} of {len(items)} boxes have impossible geometry")
        level = Verdict.REJECT if bad_geom > 0.1 * len(items) else max(level, Verdict.REVIEW, key=len)

    if expected_lines:
        if len(items) > max_ratio * expected_lines:
            reasons.append(f"{len(items)} boxes for ~{expected_lines} expected lines (over-segmenting)")
            level = Verdict.REJECT
        elif len(items) < expected_lines / max_ratio:
            reasons.append(f"only {len(items)} boxes for ~{expected_lines} expected lines (stopped early?)")
            level = Verdict.REJECT
    return Verdict(level, reasons, {"n_items": len(items), "bad_geom": bad_geom})


# ---------------------------------------------------------------- 3. page invariants
def check_page(img_bgr, boxes, min_covered: float = 0.62, review_covered: float = 0.74,
               min_box_h_frac: float = 0.015, max_box_h_frac: float = 0.16) -> Verdict:
    """Measure the page against itself. Needs no ground truth, so it works in a live classroom.

    `covered` — the fraction of the PUPIL's ink (red excluded, ruling and shadow excluded; see
    bilimai/ink.py) that falls inside some box. Writing outside every box is writing nobody will read.

    THRESHOLDS ARE CALIBRATED ON PRODUCTION, NOT INVENTED (2026-08-23, all 60 sealed exam pages,
    RP + val-fitted growth — the detector we actually ship):

        fraction of pupil ink inside some box   p1 0.625   p10 0.743   p50 0.886   p90 0.959
        median box height / page height          p1 0.032   p50 0.047   p99 0.081

    So `min_covered` 0.62 sits at production's WORST observed page (2832.jpg, 62 %) and `review_covered`
    0.74 at its 10th percentile: today's detector would raise a review flag on about one page in ten and
    reject none. That is the correct calibration for a guard — it must not cry wolf on the system it is
    guarding, only on something worse than anything we have seen.

    The box-height window 0.015-0.16 is roughly 2x either side of the p1-p99 range, so it catches a
    detector emitting word-sized or paragraph-sized boxes without touching normal variation.

    Re-calibrate whenever the detector changes. A guard tuned to a retired model is worse than none,
    because it reads as passing.
    """
    from bilimai.ink import ink_stats
    H, W = img_bgr.shape[:2]
    st = ink_stats(img_bgr, boxes)
    reasons, level = [], Verdict.OK
    cov = st.get("covered")
    if cov is not None:
        if cov < min_covered:
            reasons.append(f"only {100*cov:.0f} % of the pupil's ink is inside a box")
            level = Verdict.REJECT
        elif cov < review_covered:
            reasons.append(f"{100*cov:.0f} % of the pupil's ink is inside a box")
            level = Verdict.REVIEW
    if boxes:
        hs = sorted(b[3] - b[1] for b in boxes)
        med = hs[len(hs) // 2] / max(1, H)
        st["median_box_h_frac"] = round(med, 4)
        if med < min_box_h_frac or med > max_box_h_frac:
            reasons.append(f"median box height is {100*med:.1f} % of the page — implausible for a text line")
            level = Verdict.REJECT
    return Verdict(level, reasons, st)
