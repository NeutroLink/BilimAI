# contracts/ — what goes in, what comes out (E5.1)

One agreed form per assignment type, so the reader, the engines, the renderer, the API and the
teacher UI can be built independently and still fit. Machine-checkable JSON Schema (2020-12).

| File | Type | Tier | Key (input) | Score card (output) |
|---|---|---|---|---|
| `dictation.schema.json` | diktant, lug'at diktant | T2 | the dictated text | error list (spelling / punctuation…), counts, coverage |
| `math.schema.json` | misollar, masala, equations, physics Berilgan/Javob | T1/T3 | per-problem answer (+ kind, tolerance, points) | per-problem verdict, read answer/steps, first wrong step, checker used |
| `mcq.schema.json` | tests with circled letters / grids | T1 | question → accepted letters, policies | per-question detected/expected/verdict |
| `retelling.schema.json` | bayon / изложение | T4 (+T2 spelling) | source text, key points, rubric | criterion scores, key-point coverage, language errors, feedback |
| `open_question.schema.json` | curriculum short/long answers | T3/T4 | per-question model answer / required points | per-answer points, required points hit, factual errors, feedback |

Shared pieces live in `common.schema.json`: **Mark** (one red-pen mark: kind, bbox in original
image pixels, reason, explanation, confidence, `needs_review`), **TranscriptLine** (raw reading —
never auto-corrected), **ScoreCardBase**, **Provenance** (which models/engine produced this),
request/response envelopes.

Rules baked into the shapes:
- Every mark has a **bbox** — nothing is claimed that cannot be pointed at on the page.
- **`needs_review`** must be set whenever a mark rests on a low-confidence reading or a
  dictionary/LLM correction — the teacher glances there first.
- All output is addressed to the **teacher** (`feedback_for_teacher`, explanations in
  `teacher_language`); the teacher decides what reaches the student.
- `provenance` (which models/engine produced the result) is **internal only** — stripped at the
  external API boundary; teachers and partner apps never see tool names.
- `language / script / grade_level / subject` are optional hints normally supplied automatically
  by the partner app; the teacher never types them.
- Transcript text is what the student wrote, mistakes included; corrections live only in marks.

`examples/` holds one request+response per type (the dictation one is our real exam page
2877.jpg). `validate.py` checks them all — run it in CI:

```bash
eval/.venv/bin/python contracts/validate.py
```
