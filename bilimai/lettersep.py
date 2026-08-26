"""Letter-separated reader targets — the ONE encode/decode pair (Arm 1, 2026-08-25).

    encode("Привет, мир")            -> "П р и в е т , @ м и р"
    decode("П р и в е т , @ м и р")  -> "Привет, мир"

WHY THIS EXISTS. `plans/exec/2026-08-25-arm1-letter-targets.md` trains the reader to emit one letter
at a time so it cannot lean on its wordpiece vocabulary to tidy a child's misspelling into the word
it expected (Fadeeva et al., arXiv:2402.15307 §3.3: *"'hello' and 'hallo' may correspond to two
different sequences of tokens … especially true in case of non-vocabulary words"*). Everything
downstream expects ordinary words, so the spaced form must be undone at exactly one place.

⚠ WHY NOT `" ".join(text)`, THE OBVIOUS ENCODING. Because it is **not invertible**. Joining every
character with a space turns the gap BETWEEN two words into THREE spaces:

    'Человечество сделало'  ->  'Ч е л о в е ч е с т в о   с д е л а л о'

and `s.replace(" ", "")` then welds the line into one word. `bilimai/dictation.py`'s `_TOKEN_RE`
splits on whitespace, so a welded line does not raise — it produces a plausible-looking flood of
`missing_word` / `extra_word` marks on a page that was read perfectly. Recovering the boundary by
counting spaces is possible in principle but only where the original is at hand; **at inference the
original is exactly what we do not have.** The model emits a string and nothing else. So the
encoding has to be SELF-DESCRIBING: the output alone must say where the words end.

⚠ AND RUN-LENGTH (1 space = inside a word, 3 = between words) IS NOT ENOUGH EITHER. Generation does
not guarantee exact run lengths, and `eval/score.py:39-45` `norm()` collapses runs of spaces to one —
so any normalisation anywhere destroys the boundary irrecoverably. An explicit delimiter survives
whitespace normalisation; a run length does not.

Hence `WORD_SEP`, a single character that cannot occur in the source text. `encode` REFUSES text
containing it rather than producing something it cannot invert.

⚠ ONE PAIR, ONE MODULE. `r5b_data.py` (which builds the targets) and `bilimai/reader.py` (which undoes
them) must import from here. A second implementation is how the §6 figures were published wrong three
times — see `eval/dictation/sync_score_block.py`'s docstring.
"""
from __future__ import annotations

# ⚠ MEASURED, NOT ASSUMED — AND THE MEASUREMENT IS REPRODUCIBLE. The first candidate was "|" and it
# was WRONG: scanning real label lines found **18 occurrences of "|"** and 3 of "\\".
# ⚠ An earlier version of this note claimed 8,943 lines while `tests/test_lettersep.py` scanned only
# 2,569, because its glob matched zero of the derived corpora — an unreproducible measurement, which
# is exactly what this repo's evidence rules exist to stop. The gate now scans **8,437** lines across
# both roots and `test_corpus_is_actually_large` fails if that ever silently collapses again.
# `encode` refuses text containing the separator, so the corpus guard caught it on first contact
# rather than the run producing welded lines. "@" occurs ZERO times, is a single token in every BPE
# vocabulary we use, and carries no meaning in Russian orthography or in maths notation (unlike ^, _,
# ~, # and $, any of which a Tier-2 maths corpus could legitimately contain).
# Re-run `assert_absent()` against any NEW corpus before trusting this choice — that is what the
# helper is for.
WORD_SEP = "@"
_SEP_SPACED = f" {WORD_SEP} "


class NotInvertible(ValueError):
    """`encode` was handed text it cannot round-trip. Raised instead of degrading quietly."""


def encode(text: str) -> str:
    """Plain line -> letter-separated line. Raises `NotInvertible` rather than lose information."""
    if WORD_SEP in text:
        raise NotInvertible(
            f"text contains the word separator {WORD_SEP!r} and could not be decoded back: {text!r}. "
            "Pick a different WORD_SEP, or drop this line from the corpus — do not strip the "
            "character, because then the label no longer matches the ink.")
    words = text.split()
    if not words:
        return ""
    return _SEP_SPACED.join(" ".join(w) for w in words)


def decode(text: str) -> str:
    """Letter-separated line -> plain line. Tolerant of the malformed spacing a model can emit.

    ⚠ A line with NO separator decodes to a single word — which is CORRECT for a genuinely
    single-word line (12.9 % of sealed-exam lines) and WRONG for a line whose separator collapsed.
    Those two cases are identical as strings and cannot be told apart here; judge a run with
    `separator_rate` instead. See its docstring for why no per-line detector exists.
    """
    if not text.strip():
        return ""
    words = [w.replace(" ", "") for w in text.split(WORD_SEP)]
    return " ".join(w for w in words if w)


def separator_rate(texts) -> float:
    """Fraction of lines carrying the word separator. A RUN-level health signal.

    ⚠ WHY THERE IS NO PER-LINE DETECTOR. A first version of this module shipped `is_malformed()`,
    which flagged any separator-less line whose tokens were mostly single characters. It was wrong,
    measurably: 824 of 6,374 sealed-exam lines (12.9 %) and 276 of 5,868 splice labels (4.7 %) are a
    SINGLE WORD, so their correct encoding legitimately contains no separator — 'алфавиту' encodes to
    'а л ф а в и т у'. Uzbek was hit too ('oʻqituvchi', 'gʻalaba', 'sanʼat'). A detector that calls
    one line in eight broken on a healthy run cannot distinguish a real separator collapse from its
    own noise floor.

    It is not a tuning problem, it is impossible in principle: a lost separator and a genuinely long
    single word produce the SAME string. 'Ч е л о в е к и д ё т' (separator lost) and a real word of
    eleven letters are indistinguishable without knowing how many words to expect.

    So the signal moved to where the information exists — the RUN. Compare this rate against the
    same rate over the labels the arm was built from; a large drop means separators are collapsing.
    Dictation additionally knows the expected word count from the teacher's key, which is a stronger
    per-line check available to the caller, not to this module.
    """
    xs = list(texts)
    if not xs:
        return 0.0
    return sum(WORD_SEP in t for t in xs) / len(xs)


def looks_encoded(text: str) -> bool:
    """Cheap POSITIVE check, so a caller can assert an arm is producing what it thinks.

    ⚠ Positive only. A correctly encoded single word carries no separator, so False does NOT mean
    "broken" — 12.9 % of sealed-exam lines are single words. Judge a run with `separator_rate`.
    """
    return WORD_SEP in text


def target_is_encoded(prompt: str) -> bool:
    """Does the target for THIS prompt carry letter separation? Mirror of `r5b_data.tgt()`.

    ⚠ THIS IS WHY THE DECODE CANNOT BE A PER-READER FLAG. Arm 1 letter-separates the INK-bearing
    targets (school, splice) and deliberately leaves HWR200 ESSAY targets plain — their labels are
    canonical text, not ink, so they carry no misspelling to protect. One adapter therefore emits
    BOTH formats, chosen by the prompt.

    Decoding unconditionally destroys the essay half: a plain line has no separator, so `decode`
    strips its spaces and WELDS the whole line into one word. That is not a subtle regression — the
    1500-line HWR200 hold-out score becomes meaningless, silently. And it cannot be detected from the
    string: a correctly-encoded SINGLE-WORD school line also has no separator, and welding it is the
    right answer there. Same string, opposite correct action — so the decision must come from the
    prompt, which is the only place that knows.

    Kept beside `encode`/`decode` so the encode rule and the decode rule cannot drift apart.
    """
    return "[essay]" not in prompt


def assert_absent(texts, corpus_name: str = "corpus") -> None:
    """Gate a corpus before using it: the separator must not occur in any label.

    ⚠ RUN THIS ON EVERY NEW CORPUS. It is the only thing standing between a delimiter collision and
    a training run whose targets silently cannot be decoded. It is how "|" was rejected.
    """
    hits = [t for t in texts if WORD_SEP in t]
    if hits:
        raise NotInvertible(
            f"{corpus_name}: {len(hits)} of {len(list(texts))} lines contain the word separator "
            f"{WORD_SEP!r} and cannot be round-tripped. Examples: {hits[:3]!r}. "
            "Choose a different WORD_SEP and re-run this gate — do NOT strip the character, because "
            "then the label no longer matches the ink.")


def roundtrips(text: str) -> bool:
    """`decode(encode(x)) == x` after whitespace normalisation. The property the corpus is gated on."""
    norm = " ".join(text.split())
    try:
        return decode(encode(text)) == norm
    except NotInvertible:
        return False
