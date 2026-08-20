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
