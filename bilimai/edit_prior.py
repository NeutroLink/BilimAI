#!/usr/bin/env python3
"""Scoring improvements measured 2026-08-24. Catch 29.9 % -> 44.1 % at the same false-flag budget.

Both are ZERO-TRAINING: no GPU, no new data, computed from scores already on disk. Held out both
directions (fit on one sealed page set, scored on the other) at matched cost.

    NUMBERS LIVE IN eval/dictation/score_ladder.py -> eval/runs/dictation/score_ladder.json,
    and the prose block it emits is spliced into HANDOFF-2026-08-24.md and ENGINEERING-LOG.md by
    eval/dictation/sync_score_block.py. Re-run those; do NOT retype figures here. They were published
    wrong three times, every time by hand-copying between this docstring and those two documents.

    Headline: min(z_ctc, z_pmi) 29.9 % -> 44.1 % with page normalisation and the edit prior.

    !! SHIP THE UN-POSITIONED EDIT PRIOR. It scores higher than the position-conditioned form and
       has fewer parameters, and a paired McNemar says position conditioning is NOT established at
       this sample size. An earlier draft of this docstring claimed the un-positioned form was
       "actively harmful, never ship it" -- that was a BUG, see _check_arity below, and its sign was
       inverted.
    !! The control rung isolates the standardiser swap, so the per-writer gain is the page term.

WHERE THE REST OF THE LOSS GOES — see section 4. That section is the most important thing in this
file: it says which errors this method can separate IN PRINCIPLE, and which it cannot.

--------------------------------------------------------------------------------------------------
1. ISOLATION — is the winner a SPIKE or a PLATEAU?
--------------------------------------------------------------------------------------------------
The shipping score is `max over ~518 candidates minus the key`. That is an extreme value, and its size
depends on how legible the ink is, not only on whether the word is wrong: on a scrawl, one of 518
candidates beats the key by luck. Measured consequence — in the band just below the flag threshold sat
55 real errors AND 129 correctly-spelled words, «довать» (+2.20, wrong) next to «достала» (+2.17,
fine).

The separating structure: **a real error is LOCAL** — it lifts one candidate clear of the pack.
**Illegibility is GLOBAL** — it lifts the whole pack. So measure the winner's isolation:

    iso = score(top-1) - score(top-20)

⚠ Rank 20, not rank 2. `top1 - top2` was tested and scored **AUC 0.454, worse than chance**: the
second-best candidate is nearly always a one-letter variant of the first, so the gap collapses exactly
when a real error exists. You must reach past the winner's own neighbour cluster.

⚠ Also tested and NOT better: the trimmed-bulk standardisation (median/MAD of ranks 25-400, T and T').
Principled, and it does not beat the crude difference above. Kept simple deliberately.

--------------------------------------------------------------------------------------------------
2. EDIT PRIOR — would a PUPIL make this error, or would the READER?
--------------------------------------------------------------------------------------------------
The verifier is not an error detector; it is a FILTER. A plain spellchecker over the reader's output
already catches 59 % of errors at 12.29 false flags per 100 words — and 12.29 is roughly the reader's
own word error rate, because every misread word becomes a non-word. So the real question is:

    a word looks wrong. Did the PUPIL write it wrong, or did the READER misread it?

That is a likelihood ratio over the single edit between the key and the top candidate:

    LR = log [ P_pupil(edit | letter, position) / P_reader(edit | letter, position) ]

⚠ MUST be conditioned per letter. Raw edit COUNTS correlate r = 0.668 between pupil and reader, which
looks like "they make the same mistakes" — that is a BASE-RATE ARTIFACT, since «а», «о», «е» dominate
both lists simply by being common. Per letter written the correlation is **r = -0.040**: pupils make
these errors **20-60x more often** than the reader (а->о: pupil 0.119 vs reader 0.006). Using raw
counts made the feature actively harmful (catch 88 -> 61). That is about conditioning per LETTER,
which is essential; conditioning additionally on POSITION is a separate question and is NOT
established (see the McNemar in score_ladder.py) -- prefer the simpler un-positioned form.

⚠ Estimate the PUPIL model from `train_misspellings_v1.json` (6,436 mined training pairs), never from
the sealed eval set — 261 errors is far too few, and using them is circular.

Position conditioning uses FOUR coarse buckets -- `round(3*i/(len-1))` spans 0..3 inclusive, not the
3 an earlier draft claimed. It is NOT established as helping (paired McNemar, score_ladder.py), and
the un-positioned form scores higher, so the position argument is kept for study, not for shipping.

--------------------------------------------------------------------------------------------------
3. FUSION — keep min(), do NOT use logistic regression
--------------------------------------------------------------------------------------------------
Tested cross-page: L2 logistic regression over the same four features scored **87/261 vs min()'s
101/261**. The features are not additive. `min()` is a CONJUNCTION — every judge must be suspicious —
and that is what protects precision; a linear combiner lets one strong feature outvote the others.
The `iso` coefficient even flipped sign between folds. Keep the conjunction. `2-of-3` voting, the
monotone middle ground, also lost (99/261) — the veto is doing the work, not the averaging.

--------------------------------------------------------------------------------------------------
4. THE CEILING IS PER-EDIT-TYPE, NOT GLOBAL — read this before adding another feature
--------------------------------------------------------------------------------------------------
Stratifying the 261 sealed errors by which edit the pupil actually made:

    Group sizes are stable (n = 99 vowel, 68 insert/delete, 73 other-substitution, 21 multi-edit,
    summing to 261); the CATCH rates move by several points between rungs, so read them from
    score_ladder.json, never from here. Shape-confusable consonant substitutions are caught at
    roughly half the vowel rate, and score_ladder.py reports the Fisher exact p for that gap.
    The script asserts group n sums to 261 and caught sums to the shipped total.

The misses are NOT in the "hard to see" vowels — those beat the average. They are in consonant
pairs that are near-identical STROKES in Russian cursive. The mechanism, measured:

    edit    P(pupil)  P(reader)   ratio
    а->о     0.1174    0.0059     19.8x     pupil errs by SOUND, reader by SHAPE
    и->е     0.1048    0.0018     59.8x     different causes, ratio separates
    п->т     0.0082    0.0019      4.4x     both err by SHAPE, same cause
    ш->м     0.0050    0.0118      0.42x    reader errs MORE than the pupil - INVERTED

    !! DO NOT QUOTE A PER-LETTER CATCH RATE. At the final scorer п->т is 1 of 4 and ш->м is 0 of 3.
    A 0/3 observation has a 95 % upper bound near 63 %, so those cells cannot tell "unrecoverable in
    principle" from sampling noise. The rates above ARE well powered (6,436 pupil pairs, 11,932
    words); the GROUP gap and these RATES are what the data supports.

The method rests on pupil and reader failing for DIFFERENT REASONS. A dictation exercise mostly
tests sound-driven spelling, and there the decoupling is 20-60x and the method works. Where the
child's error is itself a stroke confusion, the reader shares it - for ш->м the evidence points the
WRONG WAY (0.42x).

The leading HYPOTHESIS is that fixing that group needs a reader which beats the child at shapes
(higher crop resolution or a better encoder) rather than another statistic over the scores we
already have. It is not established: the per-pair evidence is n=3-4, and the group still holds 36 %
of the loss, so it is worth attacking either way.

Practical consequence: report per-edit-type recall, never one global number, AND say which scorer
rung it was measured at. score_ladder.py prints the per-group catch rate at EVERY rung; the spread
    across rungs reaches ~22 points for one group, so a stratum quoted without its rung is meaningless.

    Note what that cross-rung table shows: at the BASE rung vowels and shape-confusable consonants are
    caught at almost the same rate. The gap only opens as the features improve — the features help
    vowels a great deal and the consonant group barely. That is the mechanism showing up in the
    ablation, not just in the rate table.
"""
from __future__ import annotations
import collections, json, math
from pathlib import Path

from .dictation import word_core as _c

core = lambda s: _c(s) if s else ""
ROOT = Path(__file__).resolve().parents[1]


def edit_of(key: str, cand: str, with_position: bool = True):
    """The single character edit turning `key` into `cand`, optionally with a coarse position bucket.

    Returns None for anything that is not a clean single edit — multi-edit candidates carry no
    reliable prior at the data volumes available.
    """
    k, c = core(key), core(cand)
    if not k or not c:
        return None
    if len(k) == len(c):
        d = [(i, x, y) for i, (x, y) in enumerate(zip(k, c)) if x != y]
        if len(d) != 1:
            return None
        i, x, y = d[0]
        return ("sub", x, y, round(3 * i / max(len(k) - 1, 1))) if with_position else ("sub", x, y)
    if abs(len(k) - len(c)) == 1:
        short, long_ = (k, c) if len(k) < len(c) else (c, k)
        for i in range(len(long_)):
            if long_[:i] + long_[i + 1:] == short:
                kind = "del" if long_ is k else "ins"
                return (kind, long_[i], "", round(3 * i / max(len(long_) - 1, 1))) if with_position \
                    else (kind, long_[i], "")
    return None


MAX_MULTI_EDITS = 3


def multi_edits(key: str, cand: str, max_edits: int = MAX_MULTI_EDITS):
    """The sequence of single character edits turning `key` into `cand`, or None.

    ⚠ NOT IN THE SHIPPED SCORE. Measured 2026-08-24 against a pre-committed gate and it LOST:
    multi-edit 106/261 vs 115/261 (McNemar p = 0.035). See
    `eval/dictation/gt_multi_probe.py` for the numbers and the mechanism. Kept because it is tested,
    reusable, and the negative result must stay reproducible.

    WHY. `edit_of` returns None for anything that is not ONE clean edit, so the 21 multi-edit real
    errors of 261 score exactly 0.0 on the edit prior — the feature is blind to them.

    Brill & Moore (2000) model generic string-to-string edits, which is the principled answer. It is
    the WRONG answer at this data volume: 6,436 pupil pairs cannot populate a substring-to-substring
    table, and the cells that matter would all be singletons. Decomposing into single edits instead
    reuses cells that ARE well estimated (543 pupil / 240 reader), at the cost of assuming the edits
    are independent — which is false for a child who mis-hears a whole syllable, but is a far better
    bias/variance trade here than a sparse table.

    ⚠ BOUNDED ON PURPOSE. Beyond `max_edits` the pair is almost certainly an ALIGNMENT SLIP, not a
    misspelling — §4 records two sealed labels where key and ink shared no letters. Summing a long
    edit script would turn that slip into a large, confident-looking log-ratio. Returns None.
    """
    import difflib
    k, c = core(key), core(cand)
    if not k or not c or k == c:
        return None
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=k, b=c, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        if op == "replace":
            if (i2 - i1) != (j2 - j1):            # a ragged replace is not a clean edit sequence
                return None
            out += [("sub", k[i1 + n], c[j1 + n]) for n in range(i2 - i1)]
        elif op == "delete":
            out += [("del", k[n], "") for n in range(i1, i2)]
        elif op == "insert":
            out += [("ins", c[n], "") for n in range(j1, j2)]
        if len(out) > max_edits:
            return None
    return out or None


def good_turing_probs(counter, min_distinct_r: int = 3):
    """Simple Good-Turing. Returns (p_unseen, {event: p}), with sum(p) + p_unseen == 1.

    ⚠ NOT IN THE SHIPPED SCORE. Measured 2026-08-24 against a pre-committed gate and it LOST:
    Good-Turing 111/261 vs 115/261 (McNemar p = 0.388 — no evidence either way). See
    `eval/dictation/gt_multi_probe.py` for the numbers and the mechanism. Kept because it is tested,
    reusable, and the negative result must stay reproducible.

    WHY, over the add-0.5 / floor-50 smoothing it replaces. Add-k assigns unseen mass by fiat; the
    data says what it should be. Measured on this corpus:

        pupil   6,287 events over 543 cells,  N_1 = 164  ->  unseen mass  2.6 %
        reader    731 events over 240 cells,  N_1 = 110  ->  unseen mass 15.1 %

    The reader model is six times sparser, and it is the DENOMINATOR of the pupil/reader ratio, so
    getting its unseen mass wrong biases every edit the reader has never made — which is most of
    them. `p_unseen = N_1 / N` is Good-Turing's central result.

    Discounting uses a log-linear fit to the frequency-of-frequencies (the "LGT" half of
    Gale & Sampson). ⚠ If the fit is degenerate (slope >= -1, i.e. the counts are too flat to
    extrapolate) or there are too few distinct frequencies, it falls back to raw proportions with
    the unseen mass carved out — a bad fit must not silently produce confident nonsense.
    """
    import collections as _c
    import math as _m
    N = sum(counter.values())
    if not N:
        return 1.0, {}
    Nr = _c.Counter(counter.values())
    p_unseen = Nr.get(1, 0) / N
    if p_unseen <= 0.0 or p_unseen >= 1.0:                       # no singletons: nothing to reserve
        p_unseen = 0.0

    rs = sorted(Nr)
    fitted = None
    if len(rs) >= min_distinct_r:
        # Z_r = N_r / (half the gap to the neighbouring observed frequencies) — Gale & Sampson
        xs, ys = [], []
        for idx, r in enumerate(rs):
            lo = rs[idx - 1] if idx else 0
            hi = rs[idx + 1] if idx + 1 < len(rs) else 2 * r - lo
            xs.append(_m.log(r)); ys.append(_m.log(2.0 * Nr[r] / (hi - lo)))
        n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
            if b < -1.0:                                          # slope >= -1 => not extrapolable
                a = my - b * mx
                fitted = lambda r: _m.exp(a + b * _m.log(r))       # noqa: E731

    if fitted is None:
        star = {r: float(r) for r in rs}                           # honest fallback: no discount
    else:
        star = {r: (r + 1) * fitted(r + 1) / fitted(r) for r in rs}

    total = sum(Nr[r] * star[r] for r in rs)
    scale = (1.0 - p_unseen) / total if total > 0 else 0.0
    probs = {e: star[c] * scale for e, c in counter.items()}
    return p_unseen, probs


def build_rate_model(pairs, key_of, other_of, with_position: bool = True):
    """P(edit | source letter) from (intended, produced) pairs. Conditional, never raw counts."""
    edits, chars = collections.Counter(), collections.Counter()
    for x in pairs:
        k = core(key_of(x))
        if not k:
            continue
        chars.update(k)
        e = edit_of(k, other_of(x), with_position)
        if e:
            edits[e] += 1
    return edits, chars


def pupil_model(path=None, with_position: bool = True):
    """The pupil's error habits, from the MINED TRAINING pairs — disjoint from every sealed set."""
    p = Path(path or ROOT / "eval/runs/dictation/train_misspellings_v1.json")
    pairs = json.load(p.open(encoding="utf-8"))["pairs"]
    return build_rate_model(pairs, lambda x: x.get("key"), lambda x: x.get("ink"), with_position)


def reader_model(rows, with_position: bool = True):
    """The reader's confusion habits: every CORRECTLY-spelled word where its output != the key."""
    good = [r for r in rows if r.get("truth") == "correct"]
    return build_rate_model(good, lambda r: r.get("key"), lambda r: r.get("read"), with_position)


def _check_arity(model, with_position: bool, what: str) -> None:
    """Guard against the silent-degeneracy bug of 2026-08-24.

    `pupil_model()`/`reader_model()` key their Counters by 4-tuples when `with_position=True` and by
    3-tuples when False. Calling `log_ratio(..., with_position=False)` against a POSITIONED model
    therefore misses on EVERY lookup — key-set overlap is exactly 0 of 543 — and the feature silently
    collapses to log[(rc[letter]+floor)/(pc[letter]+floor)], i.e. the source letter's corpus
    frequency, carrying no edit information at all. It does not raise, it does not return NaN, it
    just quietly becomes a different and much weaker feature.

    That bug produced a published rung of "65/261, un-positioned prior is actively harmful" which was
    not a measurement of the un-positioned prior at all. Fail loudly instead.
    """
    edits, _ = model
    for k in edits:
        want = 4 if with_position else 3
        if len(k) != want:
            raise ValueError(
                f"{what} model is keyed by {len(k)}-tuples but log_ratio was called with "
                f"with_position={with_position} (expects {want}-tuples). Build the model with the "
                f"SAME with_position flag you pass here — otherwise every lookup misses silently.")
        return


def log_ratio(key, top1, pup, rdr, with_position: bool = True, prior: float = 0.5, floor: int = 50):
    """log[ P_pupil(edit) / P_reader(edit) ]. Positive = far more typical of a pupil than of the reader.

    ⚠ `pup` and `rdr` MUST have been built with the same `with_position` flag passed here. Guarded.
    """
    _check_arity(pup, with_position, "pupil")
    _check_arity(rdr, with_position, "reader")
    e = edit_of(key, top1, with_position)
    if e is None:
        return 0.0
    (pe, pc), (re_, rc) = pup, rdr
    p_pupil = (pe.get(e, 0) + prior) / (pc.get(e[1], 0) + floor)
    p_reader = (re_.get(e, 0) + prior) / (rc.get(e[1], 0) + floor)
    return math.log(p_pupil / p_reader)


MODELS = ROOT / "bilimai/data/edit_prior_models.json"
_SEP = "|"


def save_models(pup, rdr, path=None) -> Path:
    """Freeze the pupil and reader Counters as shipped data.

    ⚠ The reader model is built from SCORED ROWS WITH TRUTH LABELS, which exist only under `eval/`.
    Production cannot rebuild it, so it must be frozen here or the edit prior cannot ship at all.
    Both models are UN-POSITIONED (3-tuples) — the shipped form; see the module docstring.
    """
    p = Path(path or MODELS)
    p.parent.mkdir(parents=True, exist_ok=True)
    # ⚠ The file records `with_position: False`. VERIFY that before writing it, or the flag is a
    # lie and `load_models` will happily hand production a positioned model against an
    # un-positioned call — the silent-collapse bug of §10.6, this time on the shipping path.
    # Without this guard the failure surfaced only as `TypeError: sequence item 3: expected str`
    # from the join below, which names neither the cause nor the file.
    _check_arity(pup, False, "pupil")
    _check_arity(rdr, False, "reader")
    def enc(model):
        edits, chars = model
        return {"edits": {_SEP.join(k): v for k, v in edits.items()}, "chars": dict(chars)}
    json.dump({"pupil": enc(pup), "reader": enc(rdr), "with_position": False},
              p.open("w", encoding="utf-8"), ensure_ascii=False)
    return p


def load_models(path=None):
    """(pupil, reader), both un-positioned. Raises if the file was written positioned."""
    p = Path(path or MODELS)
    if not p.exists():
        raise FileNotFoundError(f"{p} — run eval/dictation/refit_verifier_v6.py")
    blob = json.load(p.open(encoding="utf-8"))
    if blob.get("with_position"):
        raise ValueError(f"{p} holds POSITIONED models; the shipped feature is un-positioned")
    def dec(d):
        return (collections.Counter({tuple(k.split(_SEP)): v for k, v in d["edits"].items()}),
                collections.Counter(d["chars"]))
    return dec(blob["pupil"]), dec(blob["reader"])


def feature_row(ctc_margin, pmi_margin, cand_scores_desc, key, best_cand, pup, rdr):
    """The four shipped features, in the ONE order everything else assumes: ctc, pmi, iso, lr.

    `cand_scores_desc` MUST be sorted descending — `isolation` indexes rank 20 positionally.
    """
    return [float(ctc_margin), float(pmi_margin), isolation(cand_scores_desc),
            log_ratio(key, best_cand, pup, rdr, with_position=False)]


def fused_score(X, files, gmed, gmad):
    """Features -> the shipped score. THE single implementation; eval and production both call it.

    Kept here rather than duplicated into `verifier.py` because a second copy is how the §6 numbers
    were published wrong three times. `min()` is a CONJUNCTION — do not replace it with a weighted
    sum; logistic fusion was measured and lost (87/261 vs 101/261).
    """
    import numpy as _np
    X = _np.asarray(X, dtype=float)
    assert X.ndim == 2 and X.shape[1] == 4, f"expected (n, 4) features, got {X.shape}"
    assert len(files) == len(X), "one page id per feature row"
    P = page_normalise(X, files, _np.asarray(gmed, dtype=float), _np.asarray(gmad, dtype=float))
    return _np.minimum(_np.minimum(P[:, 0], P[:, 1]), P[:, 2]) + 0.5 * P[:, 3]


def isolation(cand_scores, rank: int = 19) -> float:
    """Spike-vs-plateau. Rank ~20, NOT rank 2 — see module docstring."""
    if not cand_scores:
        return 0.0
    return float(cand_scores[0] - cand_scores[min(rank, len(cand_scores) - 1)])


def page_normalise(X, files, gmed=None, gmad=None, min_words: int = 25):
    """Judge each child against their OWN hand. Robust, unsupervised, no labels needed.

    WHY. Measured across 121 pages, the median candidate-margin of CORRECT words ranges from -11.85
    to +1.87 — a spread of 13.72, larger than the median margin of a REAL error (+10.80). One global
    threshold is therefore badly miscalibrated per writer: on a sloppy child's page, correctly spelled
    words routinely out-score a neat child's genuine mistakes.

    Errors are ~2 % of a page, so the page's own median/MAD is dominated by correct words — which is
    what makes this a legitimate calibration rather than label leakage.

    Blending half global / half per-page beats either alone: the pure per-page form forces roughly
    equal flags on every page, which is wrong when pupils genuinely differ in how many mistakes they
    make. The blend keeps some of that real between-child signal.

    ⚠ PASS `gmed`/`gmad` FROM TRAINING DATA. If omitted they are computed from `X` itself, which
    makes the GLOBAL half transductive — it would peek at the very set being scored. The PER-PAGE
    half is legitimately transductive (a page is graded as a unit, so its own words are available),
    but the global half must not be. Measured cost of getting this wrong: about one error in 261,
    so it does not move a conclusion — but it silently invalidates any claim that the global
    standardiser is held out.
    """
    import collections as _c
    import numpy as _np
    Z = X.copy()
    idx = _c.defaultdict(list)
    for i, f in enumerate(files):
        idx[f].append(i)
    if gmed is None:
        gmed = _np.median(X, 0)
    if gmad is None:
        gmad = _np.median(_np.abs(X - gmed), 0) + 1e-9
    for _f, ii in idx.items():
        if len(ii) < min_words:                      # too few words to estimate a hand from
            Z[ii] = (X[ii] - gmed) / gmad
            continue
        med = _np.median(X[ii], 0)
        mad = _np.median(_np.abs(X[ii] - med), 0) + 1e-9
        Z[ii] = (X[ii] - med) / mad
    return 0.5 * ((X - gmed) / gmad) + 0.5 * Z       # half global, half per-writer
