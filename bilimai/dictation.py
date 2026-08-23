"""E5.8 — Dictation engine (T2): compare what the student wrote with what was dictated.

Input : key text (the dictated words/sentences), transcript lines (contract TranscriptLine:
        text + bbox, optionally word bboxes + confidences), options.
Output: contract-shaped dict with `marks`, `score_card` (errors, counts, coverage, score,
        school_grade, summary) — see contracts/dictation.schema.json.

How it works (plain words):
  1. Split both texts into words and punctuation.
  2. Line the student's words up against the dictated words (edit-distance alignment where a
     look-alike word counts as a *substitution*, so «нечего»↔«ничего» pairs up instead of being
     one deletion + one insertion).
  3. Every mismatch becomes an error: spelling (look-alike substitution), missing word,
     extra word, capitalization; then punctuation after each word is compared → missing /
     extra / wrong punctuation.
  4. Errors → marks (underline / insert caret / strike / circle) at word boxes. If the reader
     did not give word boxes, they are estimated from the line box by character position.
  5. Count errors → school grade via a configurable scale.
Rules: the transcript is never "corrected"; a low reader confidence on the word → needs_review.
"""
from __future__ import annotations
import re, unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

PUNCT = ".,;:!?—–-«»\"'()…"
_TOKEN_RE = re.compile(r"\w+(?:[-'’ʼ`]\w+)*|[^\w\s]", re.UNICODE)

# --- grading scales: (max spelling, max punctuation) per grade, checked top-down --------------
SCALES = {
    # classic RU/UZ 5-point dictation scale (school practice; replace with Markaz criteria — E5.9)
    "classic-5point": [("5", 0, 1), ("4", 2, 2), ("3", 4, 4), ("2", 7, 7)],   # else "1"
    "uz-5point-2024": [("5", 0, 1), ("4", 2, 2), ("3", 4, 4), ("2", 7, 7)],
}


def _norm(w: str, ignore_case: bool) -> str:
    w = unicodedata.normalize("NFC", w).replace("ё", "е").replace("Ё", "Е")
    return w.lower() if ignore_case else w


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


@dataclass
class Tok:
    text: str
    is_punct: bool
    bbox: list | None = None
    line_id: str | None = None
    confidence: float | None = None
    punct_after: str = ""            # punctuation glued after this word (filled in tokenize)
    line_pos: int | None = None      # index of this word among the line's tokens (verifier context)


def tokenize(text: str) -> list[Tok]:
    """Words with their trailing punctuation attached as `punct_after`."""
    raw = _TOKEN_RE.findall(text)
    out: list[Tok] = []
    for t in raw:
        if t and all(c in PUNCT for c in t):
            if out and not out[-1].is_punct:
                out[-1].punct_after += t
            else:
                out.append(Tok(t, True))
        else:
            out.append(Tok(t, False))
    return [t for t in out if not t.is_punct]      # leading/orphan punctuation ignored for grading


def map_words_to_boxes(text: str, boxes: list[list[float]], line_bbox: list[float] | None = None) -> list[dict]:
    """Assign each word of `text` (tokenize order) to one of the ink-tight word `boxes` of that line by best horizontal
    overlap with the word's proportional span; returns [{"text", "bbox"}] in reading order (E4.3, 2026-08-19).
    Boxes without any word keep no text; words without an overlapping box get a proportional box at ink height."""
    words = tokenize(text); out = []
    if not words: return out
    spans, pos = [], 0
    for w in words:
        i = text.find(w.text, pos); i = pos if i < 0 else i
        spans.append((i, i + len(w.text) + len(w.punct_after))); pos = i + len(w.text)
    bx = sorted(boxes, key=lambda b: b[0]); n = max(len(text), 1)
    if line_bbox: x0, y0, x1, y1 = line_bbox
    elif bx: x0, y0, x1, y1 = min(b[0] for b in bx), min(b[1] for b in bx), max(b[2] for b in bx), max(b[3] for b in bx)
    else: return [{"text": w.text + w.punct_after, "bbox": None} for w in words]
    W = x1 - x0
    for w, (a, b) in zip(words, spans):
        px0, px1 = x0 + W * a / n, x0 + W * b / n; best = None
        if bx:
            o, bb = max(((min(px1, c[2]) - max(px0, c[0]), c) for c in bx), key=lambda t: t[0])
            if o > 0.3 * (px1 - px0): best = list(bb)
        if best is None:
            if bx:
                ys0 = sorted(c[1] for c in bx); ys1 = sorted(c[3] for c in bx); best = [px0, ys0[len(ys0) // 2], px1, ys1[len(ys1) // 2]]
            else: best = [px0, y0, px1, y1]
        out.append({"text": w.text + w.punct_after, "bbox": [float(v) for v in best]})
    return out


def transcript_tokens(lines: list[dict]) -> list[Tok]:
    """Tokenize transcript lines; attach word boxes (given, or estimated from the line box)."""
    toks: list[Tok] = []
    for ln in lines:
        text = ln.get("text", "")
        words = tokenize(text)
        given = [g for g in (ln.get("words") or []) if g.get("bbox")]
        lb = ln.get("bbox")
        lconf = ln.get("confidence")
        if given and len(given) == len(words):
            for w, g in zip(words, given):
                w.bbox = g.get("bbox"); w.confidence = g.get("confidence", lconf)
        elif lb or given:
            mapped = map_words_to_boxes(text, [g["bbox"] for g in given], lb)
            for w, mw in zip(words, mapped): w.bbox = mw["bbox"]; w.confidence = lconf
        for pi, w in enumerate(words):
            w.line_id = ln.get("id"); w.line_pos = pi; w.confidence = w.confidence if w.confidence is not None else lconf
        toks += words
    return toks


def align(key: list[Tok], hyp: list[Tok], ignore_case: bool) -> list[tuple[int | None, int | None]]:
    """Edit-distance alignment on words. Returns pairs (key_idx | None, hyp_idx | None)."""
    K, H = len(key), len(hyp)
    kn = [_norm(t.text, ignore_case) for t in key]; hn = [_norm(t.text, ignore_case) for t in hyp]
    INF = float("inf")
    D = [[INF] * (H + 1) for _ in range(K + 1)]; B = [[None] * (H + 1) for _ in range(K + 1)]
    D[0][0] = 0
    for i in range(1, K + 1): D[i][0] = i; B[i][0] = "del"
    for j in range(1, H + 1): D[0][j] = j; B[0][j] = "ins"
    for i in range(1, K + 1):
        for j in range(1, H + 1):
            s = _sim(kn[i - 1], hn[j - 1])
            sub = 0.0 if kn[i - 1] == hn[j - 1] else (1.0 - s * 0.6 if s >= 0.5 else 2.05)  # look-alike → cheap substitution (spelling); unrelated → dearer than delete+insert (missing + extra)
            best, arg = D[i - 1][j - 1] + sub, "sub"
            if D[i - 1][j] + 1 < best: best, arg = D[i - 1][j] + 1, "del"
            if D[i][j - 1] + 1 < best: best, arg = D[i][j - 1] + 1, "ins"
            D[i][j], B[i][j] = best, arg
    pairs = []; i, j = K, H
    while i > 0 or j > 0:
        op = B[i][j]
        if op == "sub": pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif op == "del": pairs.append((i - 1, None)); i -= 1
        else: pairs.append((None, j - 1)); j -= 1
    return pairs[::-1]


def _spans(counts: list[int]) -> list[tuple[int, int]]:
    """[(start, end)] into a flat token list, from per-line token counts."""
    out, s = [], 0
    for c in counts:
        out.append((s, s + c)); s += c
    return out

def _badness(key, hyp, pairs, ignore_case) -> int:
    """How many marks this alignment would produce — the objective we minimise when choosing between two."""
    n = 0
    for ki, hj in pairs:
        if ki is not None and hj is not None:
            n += _norm(key[ki].text, True) != _norm(hyp[hj].text, True)
        elif ki is not None:
            n += 1
    return n


def align_order_robust(key: list[Tok], hyp: list[Tok], hyp_spans: list[tuple[int, int]],
                       ignore_case: bool) -> list[tuple[int | None, int | None]]:
    """Alignment that does not punish the pupil for lines arriving in the wrong order.

    WHY THIS EXISTS (measured 2026-08-23). `align` flattens the page into one word stream and aligns it in a
    single pass. One line delivered out of place slips the two streams out of step, and the words around the
    slip are compared against the wrong key words and reported as the pupil's spelling mistakes. On real
    page-level model output: **27.2 invented errors per page**, mean grade 4.75 -> 3.2, half the pages
    mis-graded, for text that was transcribed perfectly. eval/score.py cannot see any of it — it reassembles
    lines in ground-truth order before scoring, so it reports the cost of bad ordering as exactly zero.

    HOW. Align once as before; ask each transcript LINE where its words actually landed in the key; re-sort the
    lines by that; align again. The key is the authority on order, so this needs no correspondence between the
    key's line breaks and the pupil's — which matters, because there is none: a dictation key is a paragraph
    and the child wraps it wherever the page ends. (An earlier version matched key lines to transcript lines
    one-to-one. It scored perfectly when the two happened to share a line structure and collapsed completely
    when they did not — every word came back `extra_word`. Do not reintroduce it.)

    The result is never worse than the flat alignment: both are scored with `_badness` and the better wins.

    Rejected, measured, do not revisit without a new argument: sorting the model's boxes by position on the
    page instead of trusting its emission order reaches only ~8.0 invented errors/page (still worth doing —
    it is free and independent — but not a fix); quantising y into row bands before that sort was WORSE at
    every width tried (0.5/0.8/1.0/1.3 median line-heights).
    """
    flat = align(key, hyp, ignore_case)
    if len(hyp_spans) < 2:
        return flat

    # Locate each line in the key INDEPENDENTLY of the flat alignment. Using the flat alignment's own matches
    # was the obvious idea and it does not work: the lines that are out of place are exactly the ones whose
    # words the flat pass failed to match, so they are the ones with no anchor — it recovered nothing.
    # Instead, vote on the offset. Every word of the line that occurs at key position p, at line position i,
    # is one vote for "this line starts at p - i"; the winning offset is where the line belongs.
    kn = [_norm(t.text, True) for t in key]
    at: dict[str, list[int]] = {}
    for p, w in enumerate(kn):
        at.setdefault(w, []).append(p)
    ranked: list[list[tuple[int, int]]] = []            # per line: [(votes, offset), …] best first
    for s, e in hyp_spans:
        votes: dict[int, int] = {}
        for i, j in enumerate(range(s, e)):
            for p in at.get(_norm(hyp[j].text, True), ()):
                votes[p - i] = votes.get(p - i, 0) + 1
        ranked.append(sorted(((v, o) for o, v in votes.items()), key=lambda t: (-t[0], t[1]))[:4])

    # A page repeats itself — «Изложение» heads both halves of a spread, and short lines recur. Both copies
    # then vote for the same offset, both land on the same anchor, and the sort leaves them where they were.
    # So offsets are CLAIMED: the most confident line assigns first, and a line whose best slot is taken falls
    # through to its next-best. Without this, page 2047 (a spread Qwen read one column at a time, with two
    # duplicate titles) kept 20 invented marks.
    anchors: list[float | None] = [None] * len(hyp_spans)
    taken: set[int] = set()
    for li in sorted(range(len(ranked)), key=lambda i: -(ranked[i][0][0] if ranked[i] else 0)):
        for v, o in ranked[li]:
            if o not in taken:
                anchors[li] = float(o); taken.add(o); break
        else:
            if ranked[li]:
                anchors[li] = float(ranked[li][0][1])

    # A line whose words matched nothing has no opinion about where it belongs — keep it between the
    # neighbours it was already between, rather than inventing a position for it.
    last = -1.0
    for i, v in enumerate(anchors):
        if v is None:
            nxt = next((anchors[j] for j in range(i + 1, len(anchors)) if anchors[j] is not None), None)
            anchors[i] = last if nxt is None else (last + nxt) / 2.0
        last = anchors[i]

    perm = sorted(range(len(hyp_spans)), key=lambda i: (anchors[i], i))     # stable: ties keep page order
    if perm == list(range(len(hyp_spans))):
        return flat                                                        # already in key order — nothing to do

    new2old: list[int] = []
    for i in perm:
        new2old.extend(range(*hyp_spans[i]))
    reordered = [hyp[j] for j in new2old]
    pairs = [(ki, new2old[hj] if hj is not None else None)
             for ki, hj in align(key, reordered, ignore_case)]
    return pairs if _badness(key, hyp, pairs, ignore_case) <= _badness(key, hyp, flat, ignore_case) else flat


def _gap_bbox(prev: Tok | None, nxt: Tok | None) -> list | None:
    if prev and prev.bbox:
        x = prev.bbox[2]; return [x, prev.bbox[1], x + max(8, (prev.bbox[3] - prev.bbox[1]) * 0.3), prev.bbox[3]]
    if nxt and nxt.bbox:
        x = nxt.bbox[0]; return [x - max(8, (nxt.bbox[3] - nxt.bbox[1]) * 0.3), nxt.bbox[1], x, nxt.bbox[3]]
    return None


def grade_dictation(key_text: str, transcript: list[dict], *, ignore_case: bool = False,
                    count_punctuation: bool = True, scale: str = "classic-5point",
                    review_threshold: float = 0.6, language: str = "ru",
                    verifier=None, image=None, line_align: bool = True) -> dict[str, Any]:
    """`verifier` (bilimai.verifier.CTCWordVerifier) + `image`: every SPELLING mismatch is judged on the ink — verdict
    `error` (kept, counted), `review` (kept, drawn, flagged, not counted), `ok` (our misread → mark removed). E5.8 2026-08-19.

    `line_align` (default ON since 2026-08-23): match transcript lines to key lines by content before aligning
    words, so a line delivered out of order cannot invent spelling mistakes. See `align_line_first` for the
    measurements — the old flat behaviour invented 27.2 errors per page on real page-level output and cost half
    the pages more than a whole grade. Pass `line_align=False` for the pre-2026-08-23 behaviour."""
    key = tokenize(key_text)
    hyp = transcript_tokens(transcript)
    if line_align:
        # `transcript_tokens` concatenates tokenize() per line, so these spans are exact by construction.
        hspans = _spans([len(tokenize(ln.get("text", ""))) for ln in transcript])
        pairs = (align_order_robust(key, hyp, hspans, ignore_case)
                 if hspans and hspans[-1][1] == len(hyp)          # else fall back rather than mis-index
                 else align(key, hyp, ignore_case))
    else:
        pairs = align(key, hyp, ignore_case)

    errors: list[dict] = []; marks: list[dict] = []
    matched = 0
    def add(kind, expected, written, bbox, tok: Tok | None, mark_kind, mark_text=None, extra_reason=None):
        conf = tok.confidence if (tok and tok.confidence is not None) else 0.9
        review = conf < review_threshold
        err = {"kind": kind, "expected": expected, "written": written, "bbox": bbox or [0, 0, 0, 0],
               "confidence": round(conf, 3), "needs_review": review}
        if tok and tok.line_id: err["line_id"] = tok.line_id          # contract: string or absent, never null
        if tok and tok.line_pos is not None: err["_line_pos"] = tok.line_pos
        errors.append(err)
        if bbox:
            m = {"kind": mark_kind, "bbox": bbox, "reason": kind, "confidence": round(conf, 3),
                 "needs_review": review, "explanation": _explain(kind, expected, written, language)}
            if tok and tok.line_id: m["line_id"] = tok.line_id
            if mark_text: m["text"] = mark_text
            marks.append(m)

    for idx, (ki, hj) in enumerate(pairs):
        k = key[ki] if ki is not None else None; h = hyp[hj] if hj is not None else None
        if k and h:
            kn, hn = _norm(k.text, True), _norm(h.text, True)
            if kn == hn:
                matched += 1
                if not ignore_case and k.text != h.text and k.text.lower() == h.text.lower():
                    add("capitalization", k.text, h.text, h.bbox, h, "circle")
            else:
                add("spelling", k.text, h.text, h.bbox, h, "underline")
            if count_punctuation:
                kp, hp = k.punct_after, h.punct_after
                if kp and not hp:
                    gb = [h.bbox[2], h.bbox[1], h.bbox[2] + max(8, (h.bbox[3]-h.bbox[1])*0.3), h.bbox[3]] if h.bbox else None
                    add("punctuation_missing", kp, "", gb, h, "insert", mark_text=kp)
                elif hp and not kp:
                    gb = [h.bbox[2] - max(8, (h.bbox[3]-h.bbox[1])*0.3), h.bbox[1], h.bbox[2], h.bbox[3]] if h.bbox else None
                    add("punctuation_extra", "", hp, gb, h, "circle")
                elif kp and hp and kp != hp:
                    gb = [h.bbox[2] - max(8, (h.bbox[3]-h.bbox[1])*0.3), h.bbox[1], h.bbox[2], h.bbox[3]] if h.bbox else None
                    add("punctuation_wrong", kp, hp, gb, h, "circle")
        elif k and not h:
            prev = hyp[pairs[idx - 1][1]] if idx > 0 and pairs[idx - 1][1] is not None else None
            nxt = next((hyp[p[1]] for p in pairs[idx + 1:] if p[1] is not None), None)
            add("missing_word", k.text, "", _gap_bbox(prev, nxt), prev or nxt, "insert", mark_text=k.text)
        elif h and not k:
            add("extra_word", "", h.text, h.bbox, h, "strike")

    # ---- E5.8: key-conditioned verification of spelling marks on the ink (CTC judge; PMI later) ----------------------
    verified = 0; removed = 0; reviewed = 0
    if verifier is not None and image is not None:
        idx = [i for i, e in enumerate(errors) if e["kind"] == "spelling" and e.get("bbox") and e["bbox"] != [0, 0, 0, 0]]
        lines_by_id = {ln.get("id"): ln for ln in transcript}
        items = []
        for i in idx:
            e = errors[i]; ln = lines_by_id.get(e.get("line_id")) or {}
            lw = [t.text + t.punct_after for t in tokenize(ln.get("text", ""))]
            items.append({"bbox": e["bbox"], "key": e["expected"], "read": e["written"], "line_bbox": ln.get("bbox"),
                          "line_words": lw, "wi": e.get("_line_pos")})
        judged = verifier.judge(image, items) if items else []
        keep_e = set(range(len(errors))); mark_of = {}
        for m in marks: mark_of.setdefault((tuple(m["bbox"]), m["reason"]), m)
        for i, j in zip(idx, judged):
            e = errors[i]; v = j["verdict"]; verified += 1
            e["verdict"] = v; e["verifier_margin"] = j["margin"]
            m = mark_of.get((tuple(e["bbox"]), "spelling"))
            if v == "ok":
                keep_e.discard(i); removed += 1
                if m: m["verdict"] = "ok"; m["_drop"] = True
            elif v == "review":
                e["needs_review"] = True; reviewed += 1
                if m: m["verdict"] = "review"; m["needs_review"] = True; m["explanation"] = _explain(e["kind"], e["expected"], e["written"], language) + (" — на проверку" if language == "ru" else " — review")
            else:
                if m:
                    m["verdict"] = "error"
                    if j.get("letters"): m["letters"] = j["letters"]
                    if j.get("best"): e["ink_says"] = j["best"]; m["explanation"] = _explain(e["kind"], e["expected"], j["best"], language)
        errors = [e for i, e in enumerate(errors) if i in keep_e]
        marks = [m for m in marks if not m.get("_drop")]
    for e in errors: e.pop("_line_pos", None)
    n_sp = sum(1 for e in errors if e["kind"] in ("spelling", "missing_word", "extra_word", "capitalization") and e.get("verdict") != "review")
    n_pu = sum(1 for e in errors if e["kind"].startswith("punctuation"))
    grade = "1"
    for g, ms, mp in SCALES.get(scale, SCALES["classic-5point"]):
        if n_sp <= ms and n_pu <= mp: grade = g; break
    coverage = matched / max(len(key), 1)
    conf_vals = [e["confidence"] for e in errors] or [1.0]
    overall_conf = round(min(1.0, sum(conf_vals) / len(conf_vals)), 3)
    needs_review = any(e["needs_review"] for e in errors) or coverage < 0.7

    summary = _summary(n_sp, n_pu, grade, coverage, language)
    return {
        "marks": marks,
        "score_card": {
            "score": int(grade), "max_score": 5, "school_grade": grade, "summary": summary,
            "confidence": overall_conf, "needs_review": needs_review,
            "errors": errors, "n_spelling": n_sp, "n_punctuation": n_pu,
            "n_words_expected": len(key), "n_words_read": len(hyp), "coverage": round(coverage, 3),
            "verifier": ({"name": getattr(verifier, "name", "?"), "judged": verified, "removed_as_misread": removed, "review": reviewed}
                         if verifier is not None and image is not None else None),
        },
    }


def _explain(kind, expected, written, lang):
    ru = {"spelling": f"«{written}» → «{expected}»", "missing_word": f"пропущено слово «{expected}»",
          "extra_word": f"лишнее слово «{written}»", "capitalization": f"«{written}» → «{expected}» (регистр)",
          "punctuation_missing": f"пропущен знак «{expected}»", "punctuation_extra": f"лишний знак «{written}»",
          "punctuation_wrong": f"«{written}» вместо «{expected}»"}
    uz = {"spelling": f"«{written}» → «{expected}»", "missing_word": f"«{expected}» so'zi tushib qolgan",
          "extra_word": f"ortiqcha so'z «{written}»", "capitalization": f"«{written}» → «{expected}» (bosh harf)",
          "punctuation_missing": f"«{expected}» belgisi tushib qolgan", "punctuation_extra": f"ortiqcha belgi «{written}»",
          "punctuation_wrong": f"«{expected}» o'rniga «{written}»"}
    return (uz if lang == "uz" else ru).get(kind, kind)


def _summary(n_sp, n_pu, grade, coverage, lang):
    if lang == "uz":
        s = f"{n_sp} ta imlo va {n_pu} ta tinish belgisi xatosi topildi. Baho: {grade}."
        if coverage < 0.7: s += f" Diqqat: matnning faqat {coverage:.0%} qismi topildi — sahifa to'liq emasmi?"
        return s
    s = f"Найдено ошибок: орфографических — {n_sp}, пунктуационных — {n_pu}. Оценка: {grade}."
    if coverage < 0.7: s += f" Внимание: найдено только {coverage:.0%} текста — возможно, страница неполная."
    return s


# ---------------------------------------------------------------------------------------------------------------------
# Rule-generated plausible pupil misspellings (used by the key-conditioned verifiers: eval/dictation/keyed_verify_pmi.py,
# eval/dictation/ctc_verify.py). Moved here 2026-08-18 so both scorers share one generator.
_VOW = "аоеиуыэюя"
def _nfc(s: str) -> str: return unicodedata.normalize("NFC", s).replace("ё", "е").replace("Ё", "Е")
def word_core(w: str) -> str: return _nfc(w).lower().strip(".,;:!?—–-«»\"'()…")
_ALPHA = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
def candidates(word: str, read_word: str | None = None, max_n: int = 10, wide: bool = False) -> list[str]:
    """Plausible Russian misspellings of `word`; punctuation and capitalisation preserved; `read_word` (the reader's free
    read), if it differs, is appended as one more candidate.
    Default (rules): о/а, е/и, voiced pairs, doubled/dropped consonant, ь/ъ, тся/ться, жи-ши, dropped letter — deterministic
    order seeded by the word, capped at `max_n`. Measured 2026-08-19 on the 86 real pupil misspellings: rules contain the
    actual spelling in 42 % of cases (24 % with max_n=10).
    wide=True: the rule set first, then the whole 1-edit neighbourhood (insert / delete / substitute / adjacent swap over
    the Cyrillic alphabet, ~520 per word) — covers 97 % of the 86; `max_n<=0` means no cap. Use with a scorer that is cheap
    per candidate (CTC) or as a cascade (CTC top-K → PMI)."""
    import random as _random
    w = _nfc(word); pre = ""; suf = ""
    while w and not w[0].isalpha(): pre += w[0]; w = w[1:]
    while w and not w[-1].isalpha(): suf = w[-1] + suf; w = w[:-1]
    lw = w.lower(); out = set()
    swaps = {"о": "а", "а": "о", "е": "и", "и": "е", "я": "е", "б": "п", "п": "б", "в": "ф", "ф": "в", "г": "к", "к": "г", "д": "т", "т": "д", "ж": "ш", "ш": "ж", "з": "с", "с": "з"}
    for i, c in enumerate(lw):
        if c in swaps: out.add(lw[:i] + swaps[c] + lw[i + 1:])
        if i > 0 and c not in _VOW and lw[i - 1] == c: out.add(lw[:i] + lw[i + 1:])          # doubled → single
        if i > 0 and c not in _VOW and lw[i - 1] != c and c not in "ьъй": out.add(lw[:i] + c + lw[i:])   # single → doubled
        if c in "ьъ": out.add(lw[:i] + lw[i + 1:])                                             # drop soft/hard sign
    if "ться" in lw: out.add(lw.replace("ться", "тся"))
    if "тся" in lw and "ться" not in lw: out.add(lw.replace("тся", "ться"))
    for pat, rep in (("жи", "жы"), ("ши", "шы"), ("ча", "чя"), ("ща", "щя"), ("чу", "чю"), ("щу", "щю")):
        if pat in lw: out.add(lw.replace(pat, rep, 1))
    if len(lw) > 4:
        for i in range(1, len(lw) - 1): out.add(lw[:i] + lw[i + 1:])                            # dropped letter
    out.discard(lw); out = sorted(out)
    rng = _random.Random(hash(lw) & 0xffff); rng.shuffle(out)
    if wide:
        e1 = set()
        for i in range(len(lw) + 1):
            for c in _ALPHA: e1.add(lw[:i] + c + lw[i:])                       # insert
        for i in range(len(lw)):
            e1.add(lw[:i] + lw[i + 1:])                                        # delete
            for c in _ALPHA: e1.add(lw[:i] + c + lw[i + 1:])                   # substitute
            if i + 1 < len(lw): e1.add(lw[:i] + lw[i + 1] + lw[i] + lw[i + 2:])  # adjacent swap
        e1.discard(lw); out = out + sorted(e1 - set(out))
    if max_n and max_n > 0: out = out[:max_n]
    def recase(c): return c.capitalize() if w[:1].isupper() else c
    cands = [pre + recase(c) + suf for c in out]
    if read_word and word_core(read_word) != lw and word_core(read_word).isalpha(): cands.append(pre + read_word.strip(".,;:!?—–-«»\"'()…") + suf)
    return cands
