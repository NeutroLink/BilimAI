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

## What we never do

- Grade, score, rank, or assign marks.
- Infer a teacher's key, question, rubric, or maths problem from what the pupil wrote.
- Judge style, handwriting quality, effort, or logic without a supplied source.
- Show anything to a pupil. **The product is teacher-facing, full stop.**

## How this scope is measured

Not by agreement with a teacher's own marks — teachers mark selectively and mark things outside this
scope, so agreement answers a question we are not asking. The metric is:

> **Of the in-scope deviations, how many do we flag, at what false-flag rate?**

Operating point: **≤ 0.65 false flags per 100 scored words** for the RED band.