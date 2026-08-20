"""E2.6a — letter-level splicing of real ink: manufacture misspelled words with EXACT labels from children's handwriting.
2026-08-19, after R5b showed the objective alone does not move verbatim retention (44 → 45 %): the lever is the data kind.

How it works (one spliced line):
  1. ErrorModel — fitted on the 4,267 real pupil misspellings mined from the training transcripts: which letter becomes which
     (о→а, е→и …), which letters get dropped, which get doubled / inserted, where in the word. Sampling a mistake for a word
     draws from those frequencies, so the manufactured errors follow the real children's distribution.
  2. LetterAligner — the CTC word reader (ai-forever ReadingPipeline CRNN, the same model as the spelling judge) read on the
     word crop; Viterbi alignment gives each letter's pixel span (a letter's visual span runs from its own CTC firing to the
     next letter's firing — calibrated on «слушат», 2026-08-19). Only words the CTC reads exactly right are trusted.
  3. splice — on the line strip at native page resolution: cut the target letter's column band out and paste a donor letter's
     band from ANOTHER word on the SAME page (same pen, same hand): baseline-aligned on the word boxes, scaled to the target
     word height, background-matched, seams feathered. delete = remove the band; double/insert = add a band; CONTROL = replace
     the letter with the same letter from another word (label unchanged) at the same rate, so a seam carries no label.
Everything is CPU; ~30 ms/word for the alignment, ~2 ms per splice.
"""
from __future__ import annotations
import collections, difflib, random
from dataclasses import dataclass, field
import numpy as np

VOWELS = set("аоеиуыэюяё")
LAST_REJECT = ""          # why splice_strip last returned None (diagnostics)
CYR = "абвгдежзийклмнопрстуфхцчшщъыьэюяё"


# ------------------------------------------------------------------------------------------------------------ error model
class ErrorModel:
    """Letter-edit statistics of real pupil misspellings (key = correct, ink = what was written)."""

    def __init__(self):
        self.sub = collections.Counter(); self.dele = collections.Counter(); self.ins = collections.Counter(); self.dbl = collections.Counter()
        self.ops = collections.Counter(); self.pos = collections.Counter(); self.n_pairs = 0

    @staticmethod
    def _core(w): return w.strip(".,;:!?—–-«»\"'()…").lower().replace("ё", "е")

    def fit(self, pairs):
        for p in pairs:
            key, ink = self._core(p["key"]), self._core(p["ink"])
            if not key or not ink or key == ink: continue
            self.n_pairs += 1
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, key, ink, autojunk=False).get_opcodes():
                if tag == "equal": continue
                a, b = key[i1:i2], ink[j1:j2]; posf = round(i1 / max(1, len(key)), 1)
                if tag == "replace" and len(a) == 1 and len(b) == 1: self.sub[(a, b)] += 1; self.ops["sub"] += 1
                elif tag == "delete" and len(a) == 1: self.dele[a] += 1; self.ops["del"] += 1
                elif tag == "insert" and len(b) == 1:
                    prev = key[i1 - 1] if i1 > 0 else "^"
                    if b == prev: self.dbl[b] += 1; self.ops["dbl"] += 1
                    else: self.ins[(prev, b)] += 1; self.ops["ins"] += 1
                else: self.ops["complex"] += 1; continue
                self.pos[posf] += 1
        return self

    def to_json(self):
        return {"n_pairs": self.n_pairs, "ops": dict(self.ops), "sub": {f"{a}>{b}": c for (a, b), c in self.sub.most_common()},
                "del": dict(self.dele.most_common()), "dbl": dict(self.dbl.most_common()), "ins": {f"{a}>{b}": c for (a, b), c in self.ins.most_common()},
                "pos": {str(k): v for k, v in sorted(self.pos.items())}}

    @classmethod
    def from_json(cls, d):
        m = cls(); m.n_pairs = d["n_pairs"]; m.ops = collections.Counter(d["ops"])
        m.sub = collections.Counter({(k[0], k[2]): v for k, v in d["sub"].items()}); m.dele = collections.Counter(d["del"]); m.dbl = collections.Counter(d["dbl"])
        m.ins = collections.Counter({(k[0], k[2]): v for k, v in d["ins"].items()}); m.pos = collections.Counter({float(k): v for k, v in d["pos"].items()})
        return m

    def options(self, word: str):
        """All single-letter edits of `word` the model has evidence for, with weights. Each: (kind, index, new_letter, weight).
        kind: sub (replace letter at index), del (drop letter at index), dbl (double letter at index), ins (insert new letter
        after index)."""
        w = word.lower(); out = []
        for i, a in enumerate(w):
            if a not in CYR: continue
            for (x, b), c in self.sub.items():
                if x == a: out.append(("sub", i, b, c))
            if self.dele[a] and len(w) >= 4: out.append(("del", i, "", self.dele[a]))
            if self.dbl[a]: out.append(("dbl", i, a, self.dbl[a]))
            for (x, b), c in self.ins.items():
                if x == a: out.append(("ins", i, b, c))
        return out

    def sample(self, word: str, rng: random.Random):
        opts = self.options(word)
        if not opts: return None
        tot = sum(o[3] for o in opts); r = rng.random() * tot; acc = 0.0
        for o in opts:
            acc += o[3]
            if acc >= r: return o
        return opts[-1]


def apply_edit(word: str, edit) -> str:
    kind, i, b, _ = edit; w = word
    if kind == "sub": return w[:i] + (b.upper() if w[i].isupper() else b) + w[i + 1:]
    if kind == "del": return w[:i] + w[i + 1:]
    if kind == "dbl": return w[:i + 1] + w[i] + w[i + 1:]
    if kind == "ins": return w[:i + 1] + b + w[i + 1:]
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------------------ aligner
class LetterAligner:
    """CTC forced alignment → per-letter page-pixel spans for a word box. Trusts only words the CTC reads exactly."""

    def __init__(self, threads: int = 1, batch: int = 64, coreml: bool = False):
        from .verifier import CTCWordVerifier
        self.v = CTCWordVerifier(threads=threads, batch=batch, coreml=coreml)

    def _bestpath(self, lp):
        ids = lp.argmax(-1); out = []; prev = -1
        for i in ids:
            if i != prev and i not in (self.v.BLANK, self.v.OOV): out.append(self.v.alpha[i - 2])
            prev = i
        return "".join(out)

    def align_words(self, page_bgr, words: list[dict], min_conf: float = 0.6, min_trusted_frac: float = 0.5, strict: bool = False) -> list[dict | None]:
        """words: [{"text", "bbox"}] (page px). Returns per word None (untrusted) or
        {"spans": [(x0, x1), …] per letter in page px, "conf": [p, …] per letter, "trusted": [bool, …], "crop_box", "text"}.
        Gating is PER LETTER (review 2026-08-19): forced alignment is told the answer, so it works on hands the reader cannot
        guess; a letter is trusted when the CTC posterior of that letter at its firing frame is ≥ min_conf. A word is kept when
        ≥ min_trusted_frac of its letters are trusted (guards against annotator typos, where nothing aligns). strict=True
        restores the old whole-word rule (best path == label)."""
        items = [(k, w) for k, w in enumerate(words) if w["text"] and len(w["text"]) >= 2 and all(c.lower() in CYR for c in w["text"])]
        if not items: return [None] * len(words)
        crops = []; boxes = []
        for _, w in items:
            cb = self.v._crop_box(page_bgr, w["bbox"]); boxes.append(cb); crops.append(page_bgr[cb[1]:cb[3], cb[0]:cb[2]])
        lps = self.v.logprobs(crops)
        out = [None] * len(words)
        for (k, w), cb, lp in zip(items, boxes, lps):
            txt = w["text"].lower().replace("ё", "е")
            if strict and self._bestpath(lp) != txt: continue
            fr = self.v.char_frames(lp, txt); T = lp.shape[0]; x0, y0, x1, y1 = cb; cw, h = x1 - x0, y1 - y0
            nw = min(int(cw * (self.v.H / max(1, h))), self.v.W); ppf = (self.v.W / T) * (cw / max(1, nw))
            ink_frames = max(1, min(T, int(round(nw / (self.v.W / T)))))
            spans = []; conf = []; ok = True; ids = self.v._encode(txt)
            for i in range(len(fr)):
                f0, f1 = fr[i]
                if f0 is None: ok = False; break
                nxt = next((fr[j][0] for j in range(i + 1, len(fr)) if fr[j][0] is not None), None)
                a = max(0.0, f0 - 0.5); b = (nxt - 0.5) if nxt is not None else min(ink_frames, f1 + 2.5)
                spans.append((x0 + a * ppf, x0 + max(b, a + 1) * ppf))
                conf.append(float(np.exp(lp[f0:f1 + 1, ids[i]].max())))          # posterior of THIS letter where it fires
            if not ok: continue
            trusted = [c >= min_conf for c in conf]
            if sum(trusted) < max(1, min_trusted_frac * len(trusted)): continue
            spans = _snap_spans(spans, page_bgr[y0:y1, x0:x1], x0)
            if spans is None: continue
            widths = [b - a for a, b in spans]; med = float(np.median(widths))
            if med < 4 or any(wd > 3.0 * med or wd < 0.15 * med for wd in widths): continue   # implausible letter widths
            out[k] = {"spans": spans, "conf": conf, "trusted": trusted, "crop_box": cb, "text": w["text"],
                      "nll": float(-self.v.scores(lp, [txt])[0] / max(1, len(txt)))}                 # readability of the ORIGINAL word (−log P per letter)
        return out


def _snap_spans(spans, crop, x0):
    """Refine CTC letter spans with the ink itself. The CTC gives letter ORDER and rough positions (its frame is ~8–12 px and
    it fires late on wide cursive letters — «делая» gave «л» 61 px and put the «а» cut inside the «я», 2026-08-19); the ink
    column profile gives the real cut points (valleys = thin joining strokes). Dynamic programming picks n−1 valleys, monotonic,
    each letter ≥ 0.4 median width, minimising |valley − CTC boundary| + ink at the cut. First/last boundaries = ink extent.
    Returns None if a span ends up without ink."""
    bg = _bg(crop); lo = _ink_threshold(crop, bg); a = _ink_alpha(crop, bg, lo)
    lr = _line_rows(a); a[lr] = 0                                          # ignore the printed ruled line
    col = (a > 0.5).mean(0); W = len(col)
    ink_cols = np.where(col > 0.02)[0]
    if len(ink_cols) < 4: return None
    cx0, cx1 = int(ink_cols[0]), int(ink_cols[-1]) + 1
    sm = np.convolve(col, np.ones(5) / 5, mode="same"); peak = max(1e-6, float(sm.max()))
    rel = [(a_ - x0, b_ - x0) for a_, b_ in spans]; n = len(rel); med = float(np.median([b_ - a_ for a_, b_ in rel]))
    if n == 1: return [(x0 + cx0, x0 + cx1)] if col[cx0:cx1].sum() > 0 else None
    ctc = [rel[i][0] for i in range(1, n)]                               # CTC interior boundaries
    # candidate cut positions: local minima of the smoothed profile strictly inside the ink extent (+ the CTC positions as fallback)
    interior = range(cx0 + 2, cx1 - 2)
    cands = sorted({c for c in interior if sm[c] <= sm[c - 1] and sm[c] <= sm[c + 1]} | {int(round(min(max(c, cx0 + 2), cx1 - 3))) for c in ctc})
    if not cands: return None
    min_w = max(2.0, 0.4 * med); max_shift = 2.0 * med
    NEG = 1e18; m = len(cands); D = np.full((n - 1, m), NEG); B = np.zeros((n - 1, m), dtype=np.int64)
    def cost(i, c): return 0.7 * abs(c - ctc[i]) / med + 1.5 * sm[c] / peak
    def wpen(w): return 0.8 * abs(np.log(max(w, 1.0) / med))              # letters of very uneven width are unlikely
    for j, c in enumerate(cands):
        if c - cx0 >= min_w and abs(c - ctc[0]) <= max_shift: D[0, j] = cost(0, c) + wpen(c - cx0)
    for i in range(1, n - 1):
        for j, c in enumerate(cands):
            if abs(c - ctc[i]) > max_shift: continue
            best = NEG; arg = -1
            for k in range(j):
                if c - cands[k] >= min_w and D[i - 1, k] + wpen(c - cands[k]) < best: best, arg = D[i - 1, k] + wpen(c - cands[k]), k
            if arg >= 0: D[i, j] = best + cost(i, c); B[i, j] = arg
    last = [(D[n - 2, j] + wpen(cx1 - c), j) for j, c in enumerate(cands) if D[n - 2, j] < NEG and cx1 - c >= min_w]
    if not last: return None
    _, j = min(last); cuts = [0] * (n - 1)
    for i in range(n - 2, -1, -1):
        cuts[i] = cands[j]; j = B[i, j]
    bounds = [cx0] + cuts + [cx1]
    out = []
    for i in range(n):
        a_, b_ = bounds[i], bounds[i + 1]
        if b_ - a_ < 2 or col[int(a_):int(b_)].sum() <= 0: return None
        out.append((x0 + a_, x0 + b_))
    return out


# ------------------------------------------------------------------------------------------------------------ splicing
def _bg(img):  # paper colour: median of the brightest 30 % pixels, per channel
    flat = img.reshape(-1, 3).astype(np.float32); lum = flat.mean(1); thr = np.percentile(lum, 70)
    sel = flat[lum >= thr]; return np.median(sel, axis=0) if len(sel) else np.array([235, 235, 235], np.float32)


def _feather(dst, x, width=3):
    """Soft vertical seam at column x of `dst` (in place): average columns across the seam."""
    h, w = dst.shape[:2]
    for k in range(1, width + 1):
        a, b = x - k, x + k - 1
        if 0 <= a < w and 0 <= b < w:
            t = k / (width + 1); m = dst[:, a].astype(np.float32) * (1 - t * 0.5) + dst[:, b].astype(np.float32) * (t * 0.5)
            dst[:, a] = m.astype(np.uint8)


def _darkness(img, bg):
    lum = img.astype(np.float32).mean(2); bgl = float(bg.mean()); return (bgl - lum) / max(1.0, bgl)


def _ink_threshold(region, bg):
    """Adaptive pen-ink threshold for one strip region: ~0.65 × the median darkness of its darker pixels, clipped to
    [0.15, 0.4]. Measured 2026-08-19: printed ruled lines sit at darkness 0.12–0.2 (0.35 on one dark page), pen ink median
    0.2–0.55 — a fixed cut-off either erases the line or leaves ghost letters."""
    d = _darkness(region, bg); inkpx = d[d > 0.1]
    if len(inkpx) < 20: return 0.2
    return float(np.clip(np.percentile(inkpx, 50) * 0.65, 0.15, 0.4))


def _ink_alpha(img, bg, lo=0.2, hi=None):
    """Darkness of each pixel relative to the paper → soft alpha in [0, 1] (ink = opaque, paper = transparent)."""
    hi = hi if hi is not None else lo + 0.3
    return np.clip((_darkness(img, bg) - lo) / max(1e-6, hi - lo), 0.0, 1.0)


def _line_rows(alpha, frac=0.6):
    """Rows of a band that are a printed ruled line: a majority of columns are 'ink' at once."""
    return (alpha > 0.5).mean(1) > frac


def _erase_ink(band, bg, lo=0.2):
    """Remove the pen ink inside a band with inpainting — leaves paper texture and the printed ruled lines (rows that are
    'ink' across most columns are not erased unless the pixel is much darker than the line). The mask takes the stroke's
    light halo too (alpha > 0.15, dilated 2 px) and inpaints with radius 5, so the fill is sampled from true paper — with a
    core-only mask the fill came from the halo and looked lighter than the paper (2026-08-19)."""
    import cv2
    a = _ink_alpha(band, bg, lo); lr = _line_rows(a)
    strong = _ink_alpha(band, bg, lo + 0.2) > 0.5
    mask = ((a > 0.15) & (~lr[:, None] | strong)).astype(np.uint8) * 255
    if mask.sum() == 0: return band
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return cv2.inpaint(band, mask, 5, cv2.INPAINT_TELEA)


def _body_rows(region, bg, lo):
    """x-height body of a word region: rows whose ink coverage is ≥ 35 % of the densest row — the longest such run.
    Returns (top, bottom) row indices (bottom ≈ baseline). Ascenders/descenders are thin in the row profile and drop out."""
    a = _ink_alpha(region, bg, lo); cov = (a > 0.5).mean(1)
    if cov.max() <= 0: return 0, region.shape[0]
    on = cov >= 0.35 * cov.max(); best = (0, 0); i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]: j += 1
            if j - i > best[1] - best[0]: best = (i, j)
            i = j + 1
        else: i += 1
    return best[0], best[1] + 1


@dataclass
class Donor:
    band: np.ndarray            # page pixels: rows = donor word rows (± margin), cols = letter span
    body: tuple                 # (top, bottom) rows of the donor WORD's x-height body, in band row coordinates
    lo: float                   # the donor's ink threshold


def take_band(page_bgr, span, word_bbox, margin=0.35) -> Donor | None:
    x0, x1 = int(round(span[0])), int(round(span[1])); wx0, wy0, wx1, wy1 = [int(round(v)) for v in word_bbox]; h = max(1, wy1 - wy0)
    y0 = int(max(0, wy0 - margin * h)); y1 = int(min(page_bgr.shape[0], wy1 + margin * h))
    x0 = max(0, x0); x1 = min(page_bgr.shape[1], max(x0 + 2, x1))
    word_region = page_bgr[y0:y1, max(0, wx0):min(page_bgr.shape[1], wx1)]
    bg = _bg(word_region); lo = _ink_threshold(word_region, bg); band = page_bgr[y0:y1, x0:x1].copy()
    a = _ink_alpha(band, bg, lo); a[_line_rows(a)] = 0
    if (a > 0.5).mean() < 0.01: return None                                 # empty band (alignment drift) — no donor
    return Donor(band, _body_rows(word_region, bg, lo), lo)


def splice_strip(strip: np.ndarray, origin: tuple[float, float], target_span, target_word_bbox, op: str, donor: Donor | None, margin=0.35):
    """strip: line crop (native px) whose top-left page coordinate is `origin`. Returns (new_strip, seam_xs, dw).
    Ink-only compositing (2026-08-19 v2): the target letter is erased by inpainting its ink (paper and ruled lines survive),
    the column band is widened/narrowed on that clean paper, and the donor's INK is painted on top with its own darkness as
    alpha — no rectangles, no paper-tone seams."""
    import cv2
    ox, oy = origin; H, W = strip.shape[:2]
    tx0, tx1 = int(round(target_span[0] - ox)), int(round(target_span[1] - ox)); tx0 = max(0, min(W - 1, tx0)); tx1 = max(tx0 + 1, min(W, tx1))
    tw_y0, tw_y1 = target_word_bbox[1] - oy, target_word_bbox[3] - oy; th = max(1.0, tw_y1 - tw_y0)
    ry0 = int(max(0, tw_y0 - margin * th)); ry1 = int(min(H, tw_y1 + margin * th))          # rows of the word (± margin)
    bg = _bg(strip[ry0:ry1]); lo = _ink_threshold(strip[ry0:ry1], bg)
    if op == "del":
        out = np.concatenate([strip[:, :tx0], strip[:, tx1:]], axis=1); _feather(out, tx0); return out, [tx0], -(tx1 - tx0)
    if op == "pad":                                                          # decoy: insert a band of clean paper (2 seams, +width)
        w_ = tx1 - tx0; clean = strip[:, tx0:tx1].copy(); clean[ry0:ry1] = _erase_ink(clean[ry0:ry1], bg, lo)
        out = np.concatenate([strip[:, :tx1], clean, strip[:, tx1:]], axis=1)
        for x_ in (tx1, tx1 + w_): _feather(out, x_, width=2)
        return out, [tx1, tx1 + w_], w_
    assert donor is not None
    # target word body (x-height rows) and its pen colour
    twx0, twx1 = int(max(0, target_word_bbox[0] - ox)), int(min(W, target_word_bbox[2] - ox))
    treg = strip[ry0:ry1, twx0:max(twx0 + 2, twx1)]; tb0, tb1 = _body_rows(treg, bg, lo); tb0 += ry0; tb1 += ry0
    ta = _ink_alpha(treg, bg, lo); inkpx = treg.reshape(-1, 3)[(ta > 0.8).reshape(-1)]
    ink_colour = np.median(inkpx, axis=0).astype(np.float32) if len(inkpx) >= 10 else np.array([60, 50, 90], np.float32)
    # scale the donor so its x-height body matches the target's; align baselines (body bottoms)
    db0, db1 = donor.body; ratio = (tb1 - tb0) / max(2.0, db1 - db0)
    global LAST_REJECT
    if not (0.7 <= ratio <= 1.4): LAST_REJECT = f"body_ratio {ratio:.2f}"; return None                # outside this the letter would come out the wrong size («этем», 2026-08-19)
    s = float(ratio)
    band = donor.band; bh, bw = band.shape[:2]
    nb = cv2.resize(band, (max(2, int(round(bw * s))), max(2, int(round(bh * s)))), interpolation=cv2.INTER_AREA); dw = nb.shape[1]
    dbg = _bg(nb); alpha = _ink_alpha(nb, dbg, donor.lo)
    lr = _line_rows(alpha); alpha[lr] *= (_ink_alpha(nb, dbg, donor.lo + 0.2) > 0.5)[lr]          # the donor's ruled line is not ink
    b0s, b1s = int(db0 * s), int(max(db0 * s + 2, db1 * s))
    if (alpha[b0s:b1s] > 0.5).mean() < 0.02: LAST_REJECT = "donor_body_empty"; return None       # no ink where the letter body should be
    # keep only ink near the letter body: rows farther than 1.2 body heights above/below are other lines' strokes
    bh_ = max(2.0, b1s - b0s); keep0, keep1 = int(max(0, b0s - 1.2 * bh_)), int(min(nb.shape[0], b1s + 1.2 * bh_))
    alpha[:keep0] = 0; alpha[keep1:] = 0
    if op == "sub":
        clean = strip[:, tx0:tx1].copy(); clean[ry0:ry1] = _erase_ink(clean[ry0:ry1], bg, lo)        # old letter gone, paper kept
        block = cv2.resize(clean, (dw, H), interpolation=cv2.INTER_LINEAR)
        out = np.concatenate([strip[:, :tx0], block, strip[:, tx1:]], axis=1); x_ins = tx0; seams = [tx0, tx0 + dw]; delta = dw - (tx1 - tx0)
    else:                                                                                         # dbl / ins: new paper after the letter
        w0 = max(2, min(tx1 - tx0, 12)); clean = strip[:, tx1 - w0:tx1].copy(); clean[ry0:ry1] = _erase_ink(clean[ry0:ry1], bg, lo)
        block = cv2.resize(clean, (dw, H), interpolation=cv2.INTER_LINEAR)
        out = np.concatenate([strip[:, :tx1], block, strip[:, tx1:]], axis=1); x_ins = tx1; seams = [tx1, tx1 + dw]; delta = dw
    # paint the donor ink with the target's pen colour, baseline-aligned: donor body bottom × s → target body bottom
    y_top = int(round(tb1 - db1 * s)); by0 = max(0, y_top); by1 = min(H, y_top + nb.shape[0])
    if by1 > by0:
        a = np.clip(alpha[by0 - y_top:by1 - y_top] ** 0.6 * 1.15, 0, 1)[..., None]; region = out[by0:by1, x_ins:x_ins + dw].astype(np.float32)   # boost thin strokes (downscaled donors fade)
        out[by0:by1, x_ins:x_ins + dw] = (region * (1 - a) + ink_colour * a).astype(np.uint8)
    for x in seams: _feather(out, x, width=2)
    return out, seams, delta


@dataclass
class Bank:
    """Letter → donors on one page: (word_index, letter_index) with the letter's left/right neighbours (review fix 4: a donor
    whose neighbours match the target's joins more naturally)."""
    by_letter: dict = field(default_factory=lambda: collections.defaultdict(list))

    def add(self, wi, text, spans, trusted=None):
        t = text.lower().replace("ё", "е")
        for li, ch in enumerate(t):
            if ch not in CYR or (trusted is not None and not trusted[li]): continue
            self.by_letter[ch].append((wi, li, t[li - 1] if li > 0 else "^", t[li + 1] if li + 1 < len(t) else "$"))

    def pick(self, letter, rng, exclude_wi, want_w, widths, left="", right=""):
        """A donor (wi, li) for `letter`: width within 0.6–1.6× of the target letter, not from the target word; prefers the same
        left AND right neighbour letters, then one matching neighbour, then any."""
        cands = [(wi, li, l, r) for wi, li, l, r in self.by_letter.get(letter, []) if wi != exclude_wi and 0.6 * want_w <= widths[(wi, li)] <= 1.6 * want_w]
        if not cands: return None
        for tier in ((lambda c: c[2] == left and c[3] == right), (lambda c: c[2] == left or c[3] == right), (lambda c: True)):
            t = [c for c in cands if tier(c)]
            if t: wi, li, _, _ = rng.choice(t); return (wi, li)
        return None


__all__ = ["ErrorModel", "apply_edit", "LetterAligner", "Donor", "take_band", "splice_strip", "Bank", "CYR", "VOWELS"]
