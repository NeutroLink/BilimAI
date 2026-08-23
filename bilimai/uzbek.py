"""E2.6b — Uzbek Latin text normalisation and line chunking for the font-rendered bridge corpus (TRAINING-PLAN §3).

The uzbek_news corpus writes the oʻ/gʻ mark and the tutuq with five look-alike characters (measured on 2 100 files,
2026-08-20): ‘ U+2018 ×46 559, ` U+0060 ×9 289, ’ U+2019 ×6 074, ʻ U+02BB ×2 577, ' U+0027 ×1 513, ʼ U+02BC ×19.
Labels are normalised to the official letters: after o/O/g/G → ʻ U+02BB, between other letters → ʼ U+02BC (tutuq,
incl. foreign-name suffixes: Huracan’ning). A variant next to a non-letter is a quotation mark — those lines are
dropped rather than guessed (rare). score.py must apply the same fold before any UZ number is quoted (plan §5).
"""
from __future__ import annotations
import re
import unicodedata

APOS_VARIANTS = "‘’`´'ʻʼ"
OKINA, TUTUQ = "ʻ", "ʼ"          # oʻ/gʻ mark, tutuq belgisi (official)
_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
               " .,!?:;()-%\"«»“”–—№/+=") | {OKINA, TUTUQ}


def normalize(text: str) -> str | None:
    """Canonical label text, or None if the line should be dropped (ambiguous quote, foreign script, leftovers)."""
    t = unicodedata.normalize("NFC", text).replace(" ", " ")
    out = []
    for i, ch in enumerate(t):
        if ch in APOS_VARIANTS:
            prev = t[i - 1] if i else ""
            nxt = t[i + 1] if i + 1 < len(t) else ""
            if prev in "oOgG":
                out.append(OKINA)
            elif prev.isalpha() and nxt.isalpha():
                out.append(TUTUQ)
            else:
                return None                      # quotation-mark use of a single quote — ambiguous, drop the line
        else:
            out.append(ch)
    s = re.sub(r"\s+", " ", "".join(out)).strip()
    if not s or any(c not in _ALLOWED for c in s):
        return None
    if not re.search(r"[a-zA-Z]", s):
        return None
    return s


def fold_marks(text: str) -> str:
    """SCORING ONLY — collapse every look-alike of the oʻ/gʻ mark and the tutuq to ONE character.

    Why this must exist (measured 2026-08-23). 58 of the 59 fonts in `data/fonts/uz_bridge` cannot draw
    the okina U+02BB and 50 cannot draw the tutuq U+02BC, so `make_uz_font_strips.py` substitutes ‘/’ at
    draw time while the LABEL keeps the official letter. The image therefore shows one character and the
    label asks for another, and no model can be graded on a distinction the ink does not carry.
    On the 48-strip zero-shot run this alone was **65 % of the whole Uzbek error**: CER 0.0216 strict
    against **0.0075** folded. The module docstring above has demanded this fold since it was written;
    it had never been implemented.

    ⚠ NEVER fold when MARKING a pupil. Dropping the oʻ/gʻ mark is a real Uzbek spelling mistake — it is
    the single heaviest op in `corrupt_word` (weight 3.0). Folding there would delete the mistake class
    the dictation engine exists to catch. Fold in the OCR metric; keep `normalize()` for labels and for
    anything that grades a child.
    """
    t = unicodedata.normalize("NFC", text)
    return "".join(OKINA if c in APOS_VARIANTS else c for c in t)


def mark_agreement(ref: str, hyp: str) -> tuple[int, int]:
    """(matched, total) over the mark characters, counted only where the folded strings agree.

    Folding hides whether the reader carries the okina/tutuq distinction at all — which the marker needs.
    This reports it separately instead of losing it: when the model read the line correctly apart from the
    marks, did it also pick the right mark? Returns (0, 0) when the line was misread, so it cannot be
    inflated by lines the model got wrong.
    """
    r, h = unicodedata.normalize("NFC", ref), unicodedata.normalize("NFC", hyp)
    if fold_marks(r) != fold_marks(h):
        return 0, 0
    rm = [c for c in r if c in APOS_VARIANTS]
    hm = [c for c in h if c in APOS_VARIANTS]
    return sum(1 for a, b in zip(rm, hm) if a == b), len(rm)


def corrupt_word(word: str, rng) -> str | None:
    """One child-like Uzbek misspelling of `word` (normalised text), or None if the word is unsuitable.
    Ops weighted by what Uzbek learners actually do: dropping the oʻ/gʻ mark or the tutuq, h↔x confusion,
    vowel substitutions, digraph simplification (sh→s, ng→n), letter drop/double/swap."""
    core = [c for c in word if c.isalpha() or c in (OKINA, TUTUQ)]
    if len(core) < 3 or not any(c.isalpha() for c in word):
        return None
    ops = []
    if OKINA in word or TUTUQ in word: ops += [("apo", 3.0)]
    if any(c in "hxHX" for c in word): ops += [("hx", 2.0)]
    if any(c in "ouiea" for c in word.lower()): ops += [("vowel", 2.0)]
    if "sh" in word.lower() or "ng" in word.lower(): ops += [("digraph", 1.5)]
    ops += [("drop", 1.2), ("double", 0.8), ("swap", 1.0)]
    for _ in range(8):
        op = rng.choices([o for o, _ in ops], weights=[w for _, w in ops])[0]
        w = word
        if op == "apo":
            k = [i for i, c in enumerate(w) if c in (OKINA, TUTUQ)]
            if not k: continue
            i = rng.choice(k); w = w[:i] + w[i + 1:]
        elif op == "hx":
            k = [i for i, c in enumerate(w) if c in "hxHX"]
            i = rng.choice(k); c = w[i]; w = w[:i] + {"h": "x", "x": "h", "H": "X", "X": "H"}[c] + w[i + 1:]
        elif op == "vowel":
            pairs = {"o": "u", "u": "o", "i": "e", "e": "i", "a": "o"}
            k = [i for i, c in enumerate(w) if c.lower() in pairs]
            if not k: continue
            i = rng.choice(k); c = w[i]; r = pairs[c.lower()]; w = w[:i] + (r.upper() if c.isupper() else r) + w[i + 1:]
        elif op == "digraph":
            lw = w.lower()
            if "sh" in lw: i = lw.index("sh"); w = w[:i + 1] + w[i + 2:]
            elif "ng" in lw: i = lw.index("ng"); w = w[:i + 1] + w[i + 2:]
            else: continue
        elif op == "drop":
            k = [i for i, c in enumerate(w) if c.isalpha() and i > 0]
            if not k: continue
            i = rng.choice(k); w = w[:i] + w[i + 1:]
        elif op == "double":
            k = [i for i, c in enumerate(w) if c.isalpha()]
            i = rng.choice(k); w = w[:i] + w[i] + w[i:]
        elif op == "swap":
            k = [i for i in range(len(w) - 1) if w[i].isalpha() and w[i + 1].isalpha() and w[i].lower() != w[i + 1].lower()]
            if not k: continue
            i = rng.choice(k); w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        if w != word and len(w) >= 2:
            return w
    return None


def corrupt_line(text: str, rng, max_words: int = 2) -> tuple[str, int]:
    """Corrupt 1–max_words words of a normalised line; returns (new_text, n_edits) — (text, 0) if nothing was suitable."""
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if sum(c.isalpha() for c in w) >= 3]
    rng.shuffle(idx)
    n = 0
    for i in idx[:rng.randint(1, max_words)]:
        c = corrupt_word(words[i], rng)
        if c: words[i] = c; n += 1
    return " ".join(words), n


def chunk_lines(text: str, rng, lo: int = 20, hi: int = 68) -> list[str]:
    """Greedy word-wrap of an article into dictation-line-sized chunks; each chunk gets its own random target length."""
    chunks = []
    for para in text.split("\n"):
        words = para.split()
        cur: list[str] = []
        target = rng.randint(lo + 8, hi)
        for w in words:
            if cur and len(" ".join(cur + [w])) > target:
                chunks.append(" ".join(cur))
                cur = [w]
                target = rng.randint(lo + 8, hi)
            else:
                cur.append(w)
        if cur:
            chunks.append(" ".join(cur))
    return [c for c in chunks if lo <= len(c) <= hi]
