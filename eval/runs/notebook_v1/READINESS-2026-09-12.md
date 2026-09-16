# my_notebook_v1 — readiness judgment

**Date** 2026-09-12; updated after full error adjudication 2026-09-14. **Judge** Astra. **Scope**
the corpus only—the 57-page photographed notebook set, not the product and not the reader.

Written against `auditing-data-and-ground-truth` and `eval-harness-first`, whose rule is
explicit: a verdict is **ready**, **ready with conditions**, **not ready** or **unknown**, and
**"unknown" never becomes "ready" just because no defect turned up.**

---

## The verdict

| capability | verdict | why |
|---|---|---|
| **Reading faithfulness** (does the child's own spelling survive a model's transcription?) | **ready with conditions** | 2,053 lines of verbatim text over 57 pages, one schema, unique line ids, blackouts named not hidden, geometry from the production detector. The condition is single-annotator risk — see below. |
| **Marking** (does the system flag the right mistakes?) | **ready with conditions** | 251 adjudicated records, 245 counted, clear the ≥100 quantity bar. The conditions are single-annotator truth, 113 deferred records outside the current no-key score, and no pipeline result on this corpus yet. |
| **Key-based marking** (compare the model's read against a source text) | **ready to measure — and this corpus is the instrument** | **Founder ruling, restated 2026-09-12: the verified transcript IS the key.** All 1,992 scored lines therefore carry a source to compare against, not four. What the comparison measures is *marking* — does the model's read diverge from the key where the key says it should, and nowhere else. The caveat is not availability, it is independence: the key is one annotator's reading (see condition 1), so a divergence is charged against a human transcript, never against a certified truth. |

## What is genuinely in hand

- **57 pages, 2,053 lines, 1,992 scored.** Ten validator checks pass (`tools/validate_notebook_v1.py`, exit 0). 61 lines are excluded with a recorded reason: 31 drawings, 15 printed and 15 blank.
- **Transcriber notation is out of the text.** 411 lines carry `text_raw`; `text` is pupil ink only. Every consumer now reads one field and gets the same answer.
- **Hyphen carries recorded once.** 313 carries, 237 verified joins, 76 candidates, 11 side-final. One line in seven ends mid-word here, so a consumer deriving this itself was the largest silent error source in the corpus.
- **Every in-place edit has a ledger.** `CORRECTIONS.jsonl` has 11 rows; the two 2026-09-14 additions record uncertainty-pointer repairs, not transcript-text changes.
- **Geometry is predictions, never hand-drawn.** 2,369 line boxes and 9,384 word boxes from the shipped detector, CPU and CoreML byte-identical on 57/57 pages, zero geometry violations, stamped `PREDICTIONS, NOT GROUND TRUTH`.
- **Sealed and manifested.** `SPLIT.json` records dev 9 pages / 292 scored and sealed 48 / 1,700; `MANIFEST.json` hashes every declared file; `verify_manifest.py` is the integrity gate.

## The three conditions on "reading faithfulness"

1. **One annotator.** Every line was transcribed by Astra alone. Page 34 proved the cost within minutes of being checked: two of my own lines had silently tidied the child (`Взаимствование` → `Заимствование`, `дедикация` → `одинкация`) — the exact failure this product exists to prevent, sitting in what we were calling truth. A stated transcriber error rate needs a second pass.
2. **No second reader is available on this machine.** Measured, not assumed: 17 vision models across 8 vendors, 51 crop-calls, best mean CER **0.583**, and of those 51 answers **44 were hallucinations and 3 exact** — all three of the exact ones on the same easy crop. Not one model corrected the child's spelling toward the dictionary, which is the one failure mode we feared; they invented different words instead, which is worse. `glm-5.3` has no image input at all. Double-keying here must be human, or Astra blind-retest plus founder adjudication.
3. **The error labels share the transcript's annotator.** All 50 previously unlabelled pages were
   reviewed, and pages 1, 7 and 49 are now reviewed negatives rather than missing labels. This fixes
   coverage, not independence.

## Conditions before a product claim

- **Run the locked pipelines.** The corpus can now measure marking, but no reader or ink-checker
  result exists on it yet; dataset readiness is not product performance.
- **Respect deferred scope.** 113 of 251 records need a teacher key or a broader category scorer and
  must not enter the current no-key spelling score.
- **Thresholds may only be fitted on the dev fold** (pages 6, 42–49). The 2026-09-14 inspection
  created and froze sealed ground truth; future rounds must not inspect sealed labels while tuning.

## NOTHING HAS BEEN RUN ON THESE PAGES YET

**Read this before quoting any product number against this corpus.** The only model that has
touched these 57 pages is the **detector** — `models/readingpipeline/segm/segm_model.onnx`, the
shipped ReadingPipeline segmenter, which produced 2,369 line and 9,384 word boxes as predictions.
The **reader has never run on them**, the **ink checker has never run on them**, and no marking
pass has ever been scored against the transcript. Every catch rate, false-flag figure and CER in
this repository comes from the sealed `school_notebooks_RU` exam — a different corpus, printed-line
paper, one hand per page.

So the one number this corpus could produce and has not is the important one: **run the pipeline
over these pages with the transcript as the key, and measure how much of the child's own spelling
survives the read.** That is a single run, it needs no training, and until it exists the words
"ready" and "not ready" above describe the *dataset*, never the product.

## What this corpus is, and is not

It is **not a dictation set** — the pages are class notes, so the source text is the transcript we
made, not a text a teacher read aloud. Under the founder's ruling that the verified transcript
serves as the key, that is enough to measure marking. It is not enough to claim the dictation
product works, because a real dictation key is a teacher's text the pupil never saw.

Sources: `data/derived/my_notebook_v1/{DATASET-CARD.md,MANIFEST.json,SPLIT.json,CORRECTIONS.jsonl,MIGRATIONS.md,TRANSCRIPTION-STATE.md}`,
`eval/runs/notebook_v1/{adjudication_2026-09-14/result.json,error_analysis_v1.json,labels_provisional.json,sample3_adjudication.json,vision_bakeoff.json,detector_v1_boxes.json}`.
