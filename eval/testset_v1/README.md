# BilimAI sealed test set v1

**Never train on these pages.** They exist only to score models (CER/WER, box accuracy).

## ru_pages/ — 20 Russian school-notebook spreads
Source: ai-forever `school_notebooks_RU` **test** split (MIT). Images are git-ignored (24 MB);
copy them from `data/raw/school_notebooks_RU/exam/images/<file>` — filenames listed in
`ru_pages/ground_truth.json`.

Selection: 6 clean pages (no teacher ink), 3 short, 6 medium, 5 long/dense; 2718.jpg is a
deliberately hard case (curled page, heavy red marks); 15+ different students.
Totals: 804 lines · 3,285 words · 20,278 characters.

`ground_truth.json` — per page: width/height, counts, teacher-mark count, and `lines[]`
(text, bbox, column 0=left page/1=right, `words[]` with text + bbox). Built from the
dataset's word polygons grouped by line (`group_id`, geometric fallback for 96 words).

Scoring rules: CER/WER computed per line (matched by box overlap) and per page
(bag-of-lines, order-insensitive) — reported per page and aggregated.
