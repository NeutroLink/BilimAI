"""Shared text helpers for the dictation tooling."""
from __future__ import annotations

# Edge punctuation removed before a token is tested against the lexicon: the child's own
# adjacent comma or full stop is not part of the word.
_EDGE_PUNCT = ".,;:!?—–-«»\"'()…"


def strip_p(w: str) -> str:
    """Return `w` with surrounding punctuation removed."""
    return w.strip(_EDGE_PUNCT)
