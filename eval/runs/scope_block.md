## The product in one paragraph

BilimAI finds **mistakes** in handwritten schoolwork and shows them to a **teacher**. It does not
grade, score, rank, or put a number on a pupil's work, and no pupil ever sees its output. A false
flag costs a teacher a few seconds; that is the whole cost model.

## The governing rule

> **NEVER INFER A TEACHER'S KEY.**
> The model checks only what is verifiable *without* assignment-level context. Anything that needs
> the original text, the original question, the textbook, or the maths problem is checked **only**
> when the teacher supplies it — and then it is a *comparison*, not a prediction.

The reason is simple. Shown «того» alone, nothing in the ink says the pupil meant «Много». Guessing
would mean inventing a mistake. So we do not guess: without the source text that word is not checked
at all, and with the source text checking it is trivial.

## Two modes, and they are different problems

| | **No key supplied** | **Key supplied by the teacher** |
|---|---|---|
| what we check | only what is self-evident from the ink alone | every deviation from the supplied source |
| spelling | **non-words only** — «актава», «пачему», «рощя» are not Russian words, so the error is self-evident | any difference from the key, including valid-word→valid-word («того» vs «Много») |
| punctuation | **nothing** — a missing comma cannot be judged without the sentence that was dictated | any difference from the key |
| maths | computations that stand alone: «2+2=5» is wrong whatever problem produced it | the pupil's working against the supplied problem |
| the task is | detection | comparison |

**Comparison is not a model problem.** Once the teacher supplies the key, most of the difficulty
disappears — the work is alignment and diffing. The model's hard job is the no-key column.

## The five assignment types

**1. Диктанты — dictation and vocabulary dictation.** The teacher supplies the text. Every
divergence from it is marked. *This is the flagship type and the one nearest to shipping.*

**2. Математика — two tiers, and they are not the same product.**

- **Tier 1 (MVP, in scope now).** Immediate, self-evident computation errors. «2+2=5» is wrong with
  no knowledge of the problem it came from. Needs: read the symbols, verify the arithmetic.
  `bilimai/mathcheck.py` already does the checking.
- **Tier 2 (the real product, later).** The teacher supplies the maths problem and **may not supply
  the answer** — the model may be expected to solve it itself, then judge the pupil's working and
  say where it first went wrong. This needs materially more: reliable reading of maths *in layout*
  (squared paper, plain paper, column arithmetic, multi-line derivations) **and** genuine
  mathematical competence. ⚠ Tier 2 is deferred, **not deleted**. It is the largest capability jump
  in the project, because solving is a different faculty from reading, and no key-derived candidate
  set exists to constrain it.

**3. Тесты — multiple choice.** The teacher supplies the key. The model reports which option the
pupil marked; a tick or a cross per question follows from the key. Reading the option *text* is not
required when the key is positional.

**4. Изложение / bayon — retelling.** ⚠ **Narrowed.** Completeness of retelling, order of ideas and
rubric judgement are **out of scope** — they need the source text and a rubric we do not hold.
What remains: orthography, on the same terms as everywhere else — non-words without a key, full
comparison with one.

**5. Открытые вопросы — open questions.** **Out of scope unless the teacher supplies both the
question and a reference answer.** A pupil's answer alone, without the question and without the
textbook it came from, cannot be assessed. With both supplied it becomes comparison.

## Uzbek is LATIN script — always

**Founder, 2026-08-25.** Uzbek pupils write the **Latin** alphabet. Not Cyrillic, not mixed.

Consequences, so nobody re-derives them:
- `bilimai/uzbek.py` is already Latin-only (a–z plus okina `ʻ` U+02BB and tutuq `ʼ` U+02BC, with the
  five look-alike apostrophes folded). Correct as built.
- **Cyrillic handwriting corpora do NOT bootstrap Uzbek.** HKR, School Notebooks RU and Digital Peter
  are Cyrillic — they are Russian-side assets only. Any plan that lists them as Uzbek data is wrong.
- An Uzbek misspelling generator must cover Latin orthography and the okina/tutuq marks, not Cyrillic
  letter confusions.
- ⚠ Still true and unchanged: every Uzbek number to date is on **font-rendered synthetic** pages.
  Zero real Uzbek pupil pages exist. "Works on fonts we rendered" is not "works on children".

## THE VISION ARCHITECTURE — 2 of 3 move to Qwen. Founder, 2026-08-25.

Three components look at the page. **Two migrate to Qwen. One does not, and the reason is
structural, not a training gap.**

| component | job | decision |
|---|---|---|
| **1. Detector** | find the line and word boxes | **→ Qwen.** Already proven: R9's region finder scored line recall **0.951** against ReadingPipeline's **0.8916**. Trained, gated, and *not yet wired in* — free work on the shelf. |
| **2. Reader** | turn a line of ink into text, and pair each word to the teacher's key | **→ Qwen.** Needs one clean training run. **Gated on verbatim retention, not CER** — the two can move in opposite directions, and optimising CER actively harms the product. |
| **3. Ink checker** | given a word crop and ~520 candidate spellings, say which the ink matches | **STAYS a CRNN-CTC.** |

**Why the ink checker does not move.** Its entire value is having **no vocabulary**. Qwen is a
language model that can see; you cannot remove its vocabulary, because that is what it is. Measured
on our own 261 real pupil misspellings: the LM-backed reader corrected the child toward the key
**58** times, the vocabulary-free CTC only **24**. Published work agrees — VLMs rewrite 29–65 % of
corrupted words, traditional OCR **0 %** (Lee et al. 2026, arXiv:2607.21617), and the only
error-retaining pupil-HTR system in the literature deliberately chose CNN-BLSTM-CTC (Gold et al.,
BEA 2023). A Qwen checker would be *worse at the one job the checker exists for*.

⚠ **But it is not either/or.** The checker does two separable things — LOOK at the ink, and DECIDE
without vocabulary. Only the second must be vocabulary-free. The destination remains **one Qwen
vision encoder with two heads**: a reading head, and a vocabulary-free "does this ink match this
string?" head (Totev & Ward 2023, arXiv:2309.10158, ~135 k parameters). One set of eyes, two mouths.
**Not before the Qwen reader clears its retention bar** — otherwise it is built on an unproven encoder.

⚠ Today the checker is **free**: off-the-shelf, MIT, never trained by us. Replacing it means paying
to train something that currently costs nothing.

⚠ **A concrete gap this exposes: `candidates()` is hardcoded Cyrillic** (32 letters). It does not
work for Uzbek Latin at all. An Uzbek candidate generator needs its own alphabet plus the okina and
tutuq marks — and no amount of vision work substitutes for it.

## Two readers, two silos — founder, 2026-08-25

**GLM and Qwen are developed in parallel and must not affect one another.** Tuning one may not move
the other's numbers.

Why this needed enforcing: two files in `bilimai/data/` describe **one reader's mistakes** —
`verifier_v6.json` (thresholds and scaling) and `edit_prior_models.json` (`reader_model()` is
literally "the reader's confusion habits"). Stored globally, refitting for Qwen would have
overwritten GLM's live production constants, and a GLM run would then have scored against Qwen's
numbers **silently** — nothing errors, the numbers just quietly become wrong.

Now scoped per reader family. **A family with no constants of its own never inherits another's** —
it falls back to v5 and says so. GLM keeps the original filenames, so the shipped reader is untouched.

**The destination** (agreed 2026-08-25): one Qwen vision encoder with **two heads** — a reading head
and a vocabulary-free "does this ink match this string?" head (the Totev & Ward design,
arXiv:2309.10158, ~135k parameters). One set of vision weights; reading and doubting need different
outputs on top. The silos come first so that work cannot disturb the shipping reader.

## What we never do

- Grade, score, rank, or assign marks.
- Infer a teacher's key, question, rubric, or maths problem from what the pupil wrote.
- Judge style, handwriting quality, effort, or logic without a supplied source.
- Show anything to a pupil. **The product is teacher-facing, full stop.**

## How this scope is measured

Not by agreement with a teacher's own marks — teachers mark selectively and mark things outside this
scope, so agreement answers a question we are not asking. The metric is:

> **Of the in-scope deviations, how many do we flag, at what false-flag rate?**

**THE PRODUCT BAR — founder, 2026-08-25: find 7 in 10 of the mistakes a child made, at
≤ 0.65 false flags per 100 scored words** (the RED band, unchanged from 2026-08-24).

⚠ **CORRECTED 2026-08-25.** An earlier version of this section said the bar was unreachable because
catch can never exceed verbatim retention. **That was wrong.** The CTC verifier scores the CROP
against the key's 1-edit neighbourhood, so it never needed the reader's transcription: 24 errors were
caught that the reader had already normalised away (§0e).

**Where the bar actually stands.** ⚠ **The gate was also over-reporting.** Corrected 2026-08-25: production calibrates
`page_normalise` on ~11 reader-mismatch words per page, not the ~86 the eval dumps hold, so the
honest figure is **77/275 = 28.0 %**, not 30.5 %. On words the reader transcribes faithfully the
system catches **70.0 %** — the bar, met. On words it normalises away, **18.3 %**. Overall **44.1 %**.
So retention is not a wall but it is the dominant lever, worth roughly a 4× difference in catch.
⚠ The split is confounded — faithfully-read words are likely clearer-ink words, easier on both axes —
so 70 % is an OPTIMISTIC bound on what fixing retention alone would buy.
Retention is therefore the programme, not a sub-task.