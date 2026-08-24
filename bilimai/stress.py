#!/usr/bin/env python3
"""Russian lexical stress as FROZEN DATA. Build once, audit, never call at run time.

    eval/.venv/bin/python -m bilimai.stress build      # writes eval/data/stress_table.json
    eval/.venv/bin/python -m bilimai.stress audit      # sample for hand-checking

WHY THIS EXISTS. In Russian an UNSTRESSED «а» and «о» are the same sound, and so are unstressed «и»
and «е». A child writing by ear cannot hear which letter to use — which is why «безударная гласная
в корне слова» is the most-drilled orthography topic in Russian primary school. Under stress both
pairs are perfectly distinct and children rarely err.

That predicts the two largest pupil error rates in `edit_prior`'s §4 table:

    а->о   P(pupil) 0.1174   19.8x the reader      <- аканье, unstressed /o/ and /a/ merge
    и->е   P(pupil) 0.1048   59.8x the reader      <- иканье, unstressed /e/ and /i/ merge

`edit_prior` currently conditions on CHARACTER POSITION, which a paired McNemar says is not
established (p = 0.344). Position is a weak proxy; stress is the variable the phonology actually
turns on. This module supplies it.

WHY A TABLE AND NOT A LIBRARY CALL. Only ~3.5 k distinct key words exist across the mined training
pairs. A frozen table is auditable, reproducible, diffable, and adds no runtime dependency to
`bilimai/`. The generator (RUAccent) is a BUILD-TIME tool only; nothing in the serving path imports
it. ⚠ Never call a web API here, at build time or run time.

PROVENANCE IS RECORDED PER ENTRY, because the entries are not equally trustworthy:

    mono      the word has one vowel; it is stressed by definition. No model involved.
    dict      found in RUAccent's 3.19 M-wordform accent dictionary. Deterministic lookup.
    omograph  found in RUAccent's 19.7 k homograph list (за́мок / замо́к). Genuinely ambiguous
              WITHOUT sentence context, so the index is None and the feature must stay neutral.
              We do NOT guess: a contextless guess here would be a silently wrong feature, the
              exact failure class that produced a published retraction in this project.
    model     not in either dictionary; RUAccent's neural accentuator was used. Lowest confidence.
    fail      nothing produced a parseable mark. Index is None.

A None index is NEUTRAL, never a different feature — `edit_prior` must score it 0.0, identically to
a multi-edit.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from .dictation import word_core as _core

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "eval/data/stress_table.json"
VOWELS = set("аоеиыуэюяё")
core = lambda s: _core(s) if s else ""


def _index_from_marked(marked: str) -> int | None:
    """RUAccent writes «молок+о»: the '+' sits immediately BEFORE the stressed vowel, so the vowel's
    index in the UNMARKED word equals the index of '+' in the marked one."""
    if "+" not in marked:
        return None
    i = marked.index("+")
    plain = marked.replace("+", "")
    if not (0 <= i < len(plain)) or plain[i] not in VOWELS:
        return None
    return i


def build(words, accentizer=None) -> dict:
    """word -> {"i": index of the stressed vowel in core(word) or None, "src": provenance}."""
    if accentizer is None:
        from ruaccent import RUAccent
        accentizer = RUAccent()
        accentizer.load(omograph_model_size="turbo", use_dictionary=True)

    accents = accentizer.accents
    omographs = accentizer.omographs
    out: dict[str, dict] = {}

    for raw in words:
        w = core(raw)
        if not w or w in out:
            continue
        vpos = [i for i, ch in enumerate(w) if ch in VOWELS]
        if not vpos:
            out[w] = {"i": None, "src": "fail"}
        elif "ё" in w:                                   # ё is always stressed, by orthography
            out[w] = {"i": w.index("ё"), "src": "mono" if len(vpos) == 1 else "dict"}
        elif len(vpos) == 1:
            out[w] = {"i": vpos[0], "src": "mono"}
        elif w in omographs:
            out[w] = {"i": None, "src": "omograph"}      # ambiguous without context — do NOT guess
        elif w in accents:
            out[w] = {"i": _index_from_marked(accents[w]), "src": "dict"}
        else:
            try:
                out[w] = {"i": _index_from_marked(accentizer.process_all(w)), "src": "model"}
            except Exception:
                out[w] = {"i": None, "src": "fail"}
        if out[w]["i"] is None and out[w]["src"] not in ("omograph", "fail"):
            out[w] = {"i": None, "src": "fail"}
    return out


def validate(table: dict) -> dict:
    """Every non-None index must be in range of core(word) AND point at a vowel. Raises otherwise."""
    counts: dict[str, int] = {}
    for w, e in table.items():
        counts[e["src"]] = counts.get(e["src"], 0) + 1
        i = e["i"]
        if i is None:
            continue
        assert w == core(w), f"table key {w!r} is not in core() form"
        assert 0 <= i < len(w), f"stress index {i} out of range for {w!r} (len {len(w)})"
        assert w[i] in VOWELS, f"{w!r}[{i}] = {w[i]!r} is not a vowel"
    return counts


def load(path: Path | str | None = None) -> dict:
    p = Path(path or TABLE)
    if not p.exists():
        raise FileNotFoundError(f"{p} — run: python -m bilimai.stress build")
    t = json.load(p.open(encoding="utf-8"))
    validate(t)
    return t


def is_stressed(table: dict, word: str, i: int) -> bool | None:
    """True/False, or None when stress is unknown or ambiguous. None MUST be treated as neutral."""
    e = table.get(core(word))
    if not e or e["i"] is None:
        return None
    return e["i"] == i


# ------------------------------------------------------------------------------------ build / audit
def _corpus_words():
    """Every key word the pupil model and the sealed set are built from."""
    words = []
    p = ROOT / "eval/runs/dictation/train_misspellings_v1.json"
    words += [x.get("key") for x in json.load(p.open(encoding="utf-8"))["pairs"]]
    s = ROOT / "eval/runs/dictation/sealed_errors_v4.json"
    words += [x.get("key") for x in json.load(s.open(encoding="utf-8"))["items"]]
    return [w for w in words if w]


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        words = _corpus_words()
        table = build(words)
        counts = validate(table)
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(table, TABLE.open("w", encoding="utf-8"), ensure_ascii=False,
                  indent=0, sort_keys=True)
        known = sum(1 for e in table.values() if e["i"] is not None)
        print(f"{len(table)} distinct key words from {len(words)} corpus mentions")
        for k in sorted(counts):
            print(f"  {k:9s} {counts[k]:6d}")
        print(f"  KNOWN stress: {known}/{len(table)} = {100*known/len(table):.1f} % coverage")
        print(f"wrote {TABLE.relative_to(ROOT)}")
    elif cmd == "audit":
        import random
        table = load()
        rng = random.Random(0)
        pool = [(w, e) for w, e in table.items() if e["i"] is not None and e["src"] != "mono"]
        print("Hand-check these. The marked vowel should be the one you would stress.\n")
        for w, e in rng.sample(pool, min(40, len(pool))):
            i = e["i"]
            print(f"  {w[:i]}[{w[i]}]{w[i+1:]:<20} src={e['src']}")
    else:
        sys.exit("usage: python -m bilimai.stress [build|audit]")


if __name__ == "__main__":
    main()
