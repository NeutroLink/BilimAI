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

## VERIFY-ALL — the ink checker reads every aligned word. Founder, 2026-08-27.

**The checker becomes a PARALLEL READER of every aligned word on the page — not an appeals court
for the words the reader already flagged.** Today `bilimai/dictation.py` builds the judging
work-list only from reader–key mismatches, so a word the reader silently tidied to the key looks
"correct" and is never judged. Under verify-all, every aligned word is scored on the ink against
the key spelling; "reader agrees with the key, ink disagrees" becomes a caught autocorrection
instead of an invisible one.

**Recorded as architecture; switch-on is GATED.** Three prior measurements argue against flipping
today, and they stay on the record: 2026-08-19 — judging only mismatch words beat judge-all on
precision at every threshold (`ENGINEERING-LOG.md`, E5.8 entry); 2026-08-23 — judge-all, no gain,
McNemar p = 1.00 (`plans/DICTATION-AUTOPSY-2026-08-23.md`); 2026-08-25 — the audit's 2×2: opening
the gate alone LOST four errors at the shipped budget, the entire gain was calibration
(`plans/AUDIT-2026-08-25-pipeline.md` §2a). What survives those results is the reason for this
decision: only verify-all lifts the catch ceiling (the audit's exchange-rate table), the checker
fine-tune now training (HiGAN-RU) is the lever the 2×2 did not have, and the 2026-08-27
literature review converged on the same architecture independently (verification-not-transcription,
Totev & Ward arXiv:2309.10158; the rewrite mechanism, FaithC4 arXiv:2607.21617; full report:
https://claude.ai/code/artifact/fa8ab823-dc9d-4e77-83ec-2eebb313951e). So: implement behind a
flag, OFF; refit the verifier thresholds on the all-words population (the shipped τ are fitted on
mismatch-only rows); production flips only when the improved checker clears a re-measured catch
bar at the shipped false-flag budget. Touchpoints, pre-flight and gates:
`plans/exec/2026-08-27-verify-all-words.md`.

**When there is no key — design of record, NOT scheduled work.** Without a key the trust chain
changes, the scope does not: the checker verifies the READER'S OWN transcript word by word
("does the ink really spell what the reader wrote?"), a disagreement replaces the reader's word
with the checker's reading, and the existing non-word check judges the verified transcript.
"Non-words only without a key" and NEVER INFER A TEACHER'S KEY stand unchanged.

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

### ⚠ THE BAR IS NOT REACHABLE AS WRITTEN — audit, 2026-08-25

**Both halves of the bar cannot hold at once, and it is not close.** Full derivation, every script and
every retraction: [`plans/AUDIT-2026-08-25-pipeline.md`](AUDIT-2026-08-25-pipeline.md).

**Where the product actually is.** `77/275 = 28.0 %` at 0.64 false flags per 100
(`eval/runs/dictation/gate.json`, `production` block). **95 % CI [18.5, 36.4]** — a page-level
bootstrap of the whole threshold-selection procedure, ~17 points wide. Quote the interval; the point
estimate alone is not a fact about the world.

**Why the bar cannot be met.** `bilimai/dictation.py` hands the verifier ONLY words where the reader
disagrees with the key. Of the 275 in-scope labels:

| outcome | n | meaning |
|---|---|---|
| inspected | **174** | the reader disagreed, so the judges look at it |
| **corrected away** | **59** | the reader wrote the key — nothing ever looks. **Unreachable, not missed** |
| unscoreable | **42** | orphan labels with no word index; the harness cannot evaluate them at all |

Flag every one of the 174 and accept unlimited false alarms, and catch stops at **174/275 = 63.3 %**.
**So 7-in-10 is unreachable at ANY false-flag budget under the shipped gate.** Judging every aligned
word instead lifts the ceiling to 84.7 %, and 70 % then costs **≈ 20 false flags per 100 words — 30×
the stated budget.** This is a product decision about how much teacher time a page is worth, not an
engineering gap; the measured exchange rate is in the audit.

⚠ Note 2026-08-27: verify-all is now the SETTLED DESTINATION (see § VERIFY-ALL above) — gated
switch-on, flag off until the refit checker clears the re-measured bar. The numbers in this
section describe the shipped design and remain true until that flip.

⚠ Note 2026-08-27 (later): two of the three biases below are addressed and the re-key is DONE.
The gate headline is now measured on REAL detector boxes (76/275 = 27.6 %; the GT-box row stays
as continuity — `eval/runs/dictation/gate.json`); the strict rule is restated at its four real
conjuncts (`gate.py` "LABEL FILTER"); and the founder adjudicated all 20 label conflicts —
teacher right — so those labels are re-keyed (position only, no key inferred). Ceiling
63.3 % → 66.2 % (GT boxes) / 62.9 % (detector boxes); reachable-by-judging-all 84.7 % → 92.0 %
(253/275 — the remaining 22 are 21 text-unmatchable orphans plus 1 unjoined keyed label).
Threshold-chosen-on-the-scored-fold remains open. The table above is the 2026-08-25 record.

**Retention is a bounded lever, not the programme.** ⚠ **RETRACTED 2026-08-25.** This section
previously claimed 24 errors were caught that the reader had normalised away, and concluded
"retention is therefore the programme". Both are withdrawn. Those figures (70.0 % / 18.3 % / 44.1 %)
come from `eval/dictation/catch_by_retention.py`, which **never calls `production_subset`** and so
measures a configuration the product does not run. In production that group catches **0 %** by
construction — which `tests/test_pipeline_invariants.py` already asserted, in this repository, while
this file said the opposite. Measured properly: perfect retention reaches **40.1 %**, and even
granting that every one of the 59 recovered words is then caught, the **hard cap is 136/275 = 49.5 %**
— twenty points short of the bar, with a perfect reader. Retention is worth ≈ +12 points.

**The binding constraint is measurement, not the model.** 275 labels over 120 pages cannot resolve the
~8-error effects this programme chases. Every catch figure in the repository carries three
undisclosed upward biases: computed on **ground-truth annotator word boxes**
(`eval/dictation/ctc_verify.py:11`), at a **threshold chosen on the fold it is scored on**, over labels
filtered by `strict:True` — **edit distance exactly 1 AND word length ≥ 6** — which excludes
multi-edit errors entirely and every word shorter than six letters (32.7 % of all scored words).
Re-keying the 42 orphans and adjudicating the 20 teacher/dump label conflicts is the cheapest
available gain in the project and needs no GPU.