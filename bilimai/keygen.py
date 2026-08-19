"""R5b «verbatim» (2026-08-19) — prompts and dictation keys for the ink-beats-prior training objective.

Idea (plans/NOTE-dictation-verification-2026-08-18.md §A, decision 2026-08-19): show the reader the dictation key in the
prompt, but make the key UNRELIABLE at training time (20–30 % of its words damaged with our own school-misspelling model,
words dropped / swapped / added, edges trimmed) while the target is always the ink verbatim. A key that is usually right but
never trustworthy cannot be copied — the model has to read the pixels and learns that "what is written" beats "what should
be written". Sources: Gao et al. 2506.11079 (clean key in prompt made miscue detection worse, error-dense prompt better),
Apple Whisper-miscue 2505.23627 (loss masked on the prompt), nyra labs 2607.18934 (mode tags).

Prompt grammar (same "<image>" + text as R5; LLaMA-Factory masks the whole prompt out of the loss):
    [school] Text Recognition:                                 plain (as v5)
    [school] [verbatim] Text Recognition:                      verbatim mode, no key
    [school] [verbatim] Key: <key text>\nText Recognition:     verbatim mode with a (possibly damaged) dictation key
HWR200 lines keep "[essay] Text Recognition:" — their labels are the canonical essay text, not the ink, so they must not
carry the verbatim tag (they would teach the opposite).
"""
from __future__ import annotations
import random, re
from .dictation import candidates

PROMPT = "Text Recognition:"
TAG_VERBATIM = "[verbatim]"
_WORD = re.compile(r"[^\W\d_]+(?:['’ʻʼ-][^\W\d_]+)*", re.UNICODE)


def prompt(domain: str = "school", verbatim: bool = False, key: str | None = None) -> str:
    tags = f"[{domain}]" + (f" {TAG_VERBATIM}" if verbatim else "")
    return f"{tags} {PROMPT}" if key is None else f"{tags} Key: {key}\n{PROMPT}"


def misspell(word: str, rng: random.Random, wide_p: float = 0.2) -> str:
    """One plausible pupil misspelling of `word` (punctuation kept). Rule set first (о/а, е/и, voiced pairs, doubled/dropped
    consonant, ь/ъ …); with probability wide_p any 1-edit neighbour."""
    if len(word.strip(".,;:!?—–-«»\"'()…")) < 3: return word
    cs = candidates(word, None, 0, wide=True) if rng.random() < wide_p else candidates(word, None, 12)
    return rng.choice(cs) if cs else word


def corrupt(text: str, rng: random.Random, word_rate: float = 0.25, edge_p: float = 0.3, neighbour: str | None = None,
            min_changes: int = 1) -> tuple[str, int]:
    """Damage a clean line of text so it can serve as an untrustworthy key. Returns (key, n_changes).
    Per word (rate `word_rate`): 70 % misspell, 12 % drop, 8 % swap with the next word, 5 % duplicate/insert a neighbour word,
    5 % flip the first letter's case. With probability `edge_p` the key's extent is wrong too: first/last 1–2 words cut, or 1–2
    words of the neighbouring line (`neighbour`) glued on — dictation keys at line level are approximate in extent.
    `min_changes`: if the random pass changed fewer words than this, one extra word is damaged (short lines would otherwise
    come out clean most of the time and teach copying)."""
    words = text.split()
    if not words: return text, 0
    n = 0
    i = 0
    out = []
    while i < len(words):
        w = words[i]
        if _WORD.search(w) and rng.random() < word_rate:
            r = rng.random()
            if r < 0.70: out.append(misspell(w, rng)); n += 1
            elif r < 0.82: n += 1                                           # drop
            elif r < 0.90 and i + 1 < len(words): out += [words[i + 1], w]; i += 1; n += 1   # swap
            elif r < 0.95: out += [w, rng.choice(words)]; n += 1            # stray extra word
            else: out.append((w[0].swapcase() + w[1:]) if w[0].isalpha() else w); n += 1
        else: out.append(w)
        i += 1
    if rng.random() < edge_p and len(out) > 3:
        r = rng.random(); k = rng.randint(1, 2)
        if r < 0.4: out = out[k:]
        elif r < 0.8: out = out[:-k]
        elif neighbour:
            nw = neighbour.split()
            if nw: out = (nw[-k:] + out) if rng.random() < 0.5 else (out + nw[:k])
        n += 1
    if n < min_changes:
        idx = [j for j, w in enumerate(out) if _WORD.search(w)]
        if idx:
            j = rng.choice(idx); w = out[j]; m = misspell(w, rng)
            if m != w: out[j] = m
            elif len(out) > 1: out.pop(j)                                   # too short to misspell → drop it
            else: out[j] = w[0].swapcase() + w[1:] if w[0].isalpha() else w + "."
            n += 1
    return " ".join(out), n


__all__ = ["PROMPT", "TAG_VERBATIM", "prompt", "misspell", "corrupt"]
