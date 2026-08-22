# BilimAI

AI-powered assignment grading assistant. Reads students' written work, checks it,
drafts grades and feedback — a teacher reviews and approves. Runs on a custom,
fully local AI (no cloud LLM APIs). Languages: **Russian and Uzbek**.

- [Landing page](https://neutrolink.github.io/BilimAI/) — what BilimAI does, in Russian.
- [contracts/](contracts/) — JSON contracts per assignment type.
- [bilimai/](bilimai/) — grading engines, renderer, pipeline; [tests/](tests/).
- [eval/](eval/) — sealed exam + scorer + public bake-off results.
- [mvp/](mvp/) — single-file demo of the grading stage (typed text only).

## Where the reader stands (2026-08-19)

Sealed exam of real Russian school-notebook pages (60 pages, 2,569 handwritten lines, never trained on), scored with
[`eval/score.py`](eval/score.py) — **character error rate** (of 100 letters, how many wrong; lower is better) and **word accuracy**:

| reader | line CER (char-weighted) | word accuracy | lines read perfectly | end-to-end page CER (auto line detection) |
|---|---|---|---|---|
| GLM-OCR stock (no training) | 0.82 | 0.17 | 1 % | 0.68 |
| + LoRA v3 (language side only, 68 k lines) | 0.226 | 0.54 | 19 % | 0.293 |
| + LoRA v4 (+ HWR200 essays) | 0.222 | 0.57 | 20 % | 0.299 |
| **+ LoRA v5 (vision tower trained too, 290 k lines)** | **0.038** | **0.87** | **58 %** | **0.066** (was 0.113 before the detector fix of 2026-08-19; word accuracy end-to-end 0.83) |

The jump to v5 came from one change: training the half of the model that *looks* at the page, not only the half that
writes text. Numbers on text that never appears in training data are the same (0.038), so it is not memorisation.
The end-to-end column uses our own line detector (ReadingPipeline segmenter + measured box growth, [`bilimai/detector.py`](bilimai/detector.py)); the
remaining gap to human boxes is mostly 1–2-word fragments the detector does not fire on. Full leaderboard and method notes: [`eval/runs/README.md`](eval/runs/README.md).

**Dictation judges (verifier numbers, 2026-08-19, all 7,783 words of the sealed exam):** for each word where the reader
disagrees with the key, two judges score the ink — a CTC word reader (no language prior; ranks ~500 one-letter variants)
and the GLM reader with its no-picture score subtracted (PMI). "Both must agree" gives ranking quality (AUC) **0.92**;
at the shipped thresholds the red band costs ≈ 0.65 false red marks per 100 words and catches ≈ 26 % of real misspellings,
the yellow «на проверку» band raises catch to ≈ 71 % at ≈ 6.5 marks per 100 words. Words the reader auto-corrected to the
key (26 % of real misspellings) never reach the judges — the next training round (R5b «verbatim») targets that.
Code: [`bilimai/verifier.py`](bilimai/verifier.py); measurement: `eval/runs/dictation/ctc_r5all_fused*.json`.

### Fresh demo grading (v5 + own detector + word judges)

Dictation-style check of a real pupil page against the dictated text, **the whole product path**: our line/word detector
finds the lines, the reader transcribes them, the dictation engine compares with the key, and every spelling disagreement
is judged on the ink by two independent readers (a letter-by-letter CTC reader and the GLM reader with its language habit
subtracted) — red only when both agree the ink deviates, yellow «на проверку» when they half-agree, nothing when it was
our misread. A teacher reviews.

![v5 demo grading — page 2921, full product](docs/demo-v5.marked.jpg)

6 marks on this page: **1 red — the pupil's genuine slip** (*слушат → служат*; both judges agree; the wrong letter is struck and
the correction written above it — the letter's position comes from the CTC judge's alignment, nothing is hand-placed),
3 yellow highlights («на проверку»: one real deviation, one reader slip, one ambiguous), a caret for a dropped «а» and a
circle for an extra full stop. The page's other genuine slip (*проевить*) is **not** marked: both judges read that ambiguous
letter as «я» — an honest limit (see the verifier numbers above). The pupil's own corrections (a crossed-out word, «он»
written above it, «считать войну» in the margin) are not understood yet — see the plan (E4.6). Request: [`contracts/examples/demo_v5/2921.json`](contracts/examples/demo_v5/2921.json) (drop `options.oracle_lines`
to use the own detector). Reproduce:
`python -m bilimai.pipeline --request contracts/examples/demo_v5/2921.json --out out/v5_demo --adapter models/adapters/glm-ocr-lora-ru-v5`.

## Run the MVP

```bash
# 1. Ollama serving + model
ollama serve &
ollama pull qwen3:8b

# 2. App
cd mvp
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --port 8000
```

Open http://localhost:8000 — "Load sample" cycles a Russian and an Uzbek essay.

Model is configurable: `BILIMAI_MODEL=qwen3:4b .venv/bin/uvicorn app:app --port 8000`.


Training data pipeline, model recipes and the engineering log live in a private repository.
