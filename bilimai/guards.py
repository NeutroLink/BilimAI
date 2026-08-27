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

__all__ = ["prompt_fingerprint", "check_prompt", "check_output", "check_runaway", "check_page",
           "check_line_length", "check_ink_density", "check_script", "Verdict"]


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


def check_runaway(items: list, raw: str | None = None, min_unique: float = 0.5,
                  review_unique: float = 0.8, max_run: int = 5) -> Verdict:
    """Catch a generation that fell into a repetition loop. Needs no ground truth.

    `check_output` can only spot over-segmenting when you already know how many lines to expect, which
    at inference time you do not. This spots the failure from the output's own shape.

    MEASURED (markup probe, 2026-08-23). One page in six derailed: 136 items in 78 s against 6 real
    ones, of which **135 were the identical word** — unique ratio **0.01**, where all five healthy
    pages scored exactly **1.00**. The separation is not marginal, so the threshold does not need to
    be delicate. Same failure family as R6's adapter emitting one line and stopping: a whole-page
    generation fails as a whole, and nothing downstream would have flagged it.

    That run also ended mid-object, so `check_output`'s truncation test would have caught this one
    too — but only because it happened to hit the token cap. A loop that repeats and then closes the
    array cleanly slips past every other check here. Hence a check on repetition itself.
    """
    if not items:
        return Verdict(Verdict.REJECT, ["no items to check for runaway"], {"n_items": 0})

    def key(it):
        t = it.get("text") if it.get("text") is not None else it.get("word")
        return re.sub(r"\s+", " ", str(t or "")).strip().lower()

    keys = [key(it) for it in items]
    nonempty = [k for k in keys if k]
    uniq = len(set(nonempty)) / len(nonempty) if nonempty else 0.0

    run = best = 1
    for a, b in zip(keys, keys[1:]):
        run = run + 1 if a and a == b else 1
        best = max(best, run)

    stats = {"n_items": len(items), "unique_ratio": round(uniq, 3), "longest_repeat_run": best}
    reasons, level = [], Verdict.OK
    if nonempty and uniq < min_unique:
        reasons.append(f"repetition loop: only {uniq:.0%} of {len(nonempty)} items are distinct")
        level = Verdict.REJECT
    elif nonempty and uniq < review_unique:
        reasons.append(f"suspicious repetition: {uniq:.0%} of items are distinct")
        level = Verdict.REVIEW
    if best > max_run:
        reasons.append(f"{best} identical items in a row")
        level = Verdict.REJECT
    if raw is not None and raw.strip() and not raw.strip().endswith("]"):
        reasons.append("reply did not close its JSON array — cut off by the token cap")
        level = Verdict.REJECT
        stats["hit_token_cap"] = True
    return Verdict(level, reasons, stats)


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


# ---------------------------------------------------------------- 4. per-line reader-output guards (0f, 2026-08-27)
# Three pure checks on ONE line read: (text, ink measurements) in, Verdict out. No model, no image
# here — the caller measures the crop (bilimai/pipeline.py page_ink_mask/line_ink_measures) and the
# constants are FIT by eval/dictation/read_guard_probe.py on the sealed pages at PRODUCTION
# geometry (rp-segm+grow-v1 det line boxes + the READ stage's pad=12). Re-fit whenever the detector
# or the crop geometry changes — the check_page precedent: a guard tuned to a retired model is
# worse than none, because it reads as passing.
#
# FLAG-ONLY, by standing rule (bilimai/dictation.py: "the transcript is never 'corrected'; a low
# reader confidence on the word -> needs_review"). These verdicts clamp a line's confidence so the
# EXISTING review flow fires; the text is NEVER mutated. Repetition-stripping in particular is
# actively dangerous here: every triple-word repeat in the sealed line reads is GENUINE pupil
# drill text, GT-confirmed («сирень сирень сирень» 2239, «бегать бегать бегать бегать» 2235,
# «рябинки рябинки рябинки» 2238) — a stripper would delete pupil work and destroy real errors.


def check_line_length(text: str, ink_cols: int, crop_h: int, k: float, ratio: float,
                      min_chars: int = 8) -> Verdict:
    """More text than the ink can warrant -> the tail of the read is invention, not reading.

    warranted_chars = k * (ink_cols / crop_h); trips when len(text) > ratio * max(min_chars,
    warranted_chars). `ink_cols` is the count of ink-bearing columns of the LOCAL-CONTRAST mask
    (bilimai/ink.py _darker_than_paper) inside the production crop — box width lies, because
    production boxes are padded and detectors over-grow.

    k AND ratio ARE MEASURED, NOT INVENTED (read_guard_probe.py, 2026-08-27, 3,707 sealed line
    reads at det geometry): the GENUINE ratio distribution runs p95 1.89 / p99 2.58 / max 3.79 —
    school notebooks hold drill lines, dense essays and faint hands. Boros et al.'s fixed 1.5x
    would false-trip 6.4 % of genuine lines here; `ratio` ships as the fitted p99 x 1.10 instead,
    so only reads beyond anything genuine ever flag (~0.5 % of sealed reads, all of them heavy
    misreads). REVIEW, never REJECT: at this bound a trip means "a human should look", and the
    read may still be partly right."""
    n = len((text or "").strip())
    warranted = k * (ink_cols / max(1, crop_h))
    limit = ratio * max(min_chars, warranted)
    stats = {"chars": n, "ink_cols": int(ink_cols), "crop_h": int(crop_h),
             "warranted_chars": round(warranted, 1), "length_limit": round(limit, 1)}
    if n > limit:
        return Verdict(Verdict.REVIEW,
                       [f"{n} chars where the ink warrants ~{max(min_chars, warranted):.0f} — "
                        f"beyond the calibrated x{ratio:g} bound, the tail is likely invented"], stats)
    return Verdict(Verdict.OK, [], stats)


def check_ink_density(density: float, tau: float) -> Verdict:
    """A confident read of a (near-)blank crop is a hallucination by construction — flag the crop
    from its own pixels, before or regardless of what the reader said (Careless Whisper's
    silence-predictor, transposed to ink).

    `density` is the mean of bilimai/ink.py `_darker_than_paper` over the production crop. That
    mask is the point: raw Otsu is MEASURED-INVALID as a blankness signal (on 2013.jpg text-line
    crops it reads 0.07-0.13 while a BLANK region reads 0.21 — Otsu splits paper-texture noise).
    The local-contrast mask separates cleanly: sealed text lines 0.076+ (p1), blank regions 0.000.
    `tau` ships at half the sealed p1 (read_guard_probe.py) so it cannot cry wolf on the faintest
    genuine line we have seen. REJECT: there is nothing on the paper to read — though the first
    ship treats REJECT as REVIEW-strength (flag-only, no skip)."""
    stats = {"ink_density": round(float(density), 4)}
    if density < tau:
        return Verdict(Verdict.REJECT,
                       [f"ink density {density:.4f} below blank threshold {tau:g} — "
                        f"the crop is (near-)blank, any read of it is invented"], stats)
    return Verdict(Verdict.OK, [], stats)


_CJK_RE = re.compile("[⺀-⻿　-〿぀-ヿ㐀-䶿一-鿿"
                     "가-힯豈-﫿\U00020000-\U0002ebef]")


def check_script(text: str) -> Verdict:
    """CJK inside a Russian/Uzbek read — the observed leak class, and ONLY that class.

    Exactly one incident in 3,707 sealed line reads (2026-08-27): «ВесёX线» on 2817.jpg — a Han
    ideograph inside a Russian line. Latin lookalikes are deliberately OUT of scope this ship
    (the «X» above stays legal): pupils write Latin letters in maths and language drills, so a
    Latin trigger would flag genuine work. Extend only from an observed failure, never ahead of
    one."""
    hits = _CJK_RE.findall(text or "")
    if hits:
        return Verdict(Verdict.REVIEW,
                       [f"non-Cyrillic script leak: {''.join(hits[:5])!r} — "
                        f"{len(hits)} CJK character(s) in the read"], {"cjk_chars": len(hits)})
    return Verdict(Verdict.OK, [], {"cjk_chars": 0})
