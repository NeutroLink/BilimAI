# Sealed exam v2 — RU school notebooks (2026-08-18)

60 pages = the 20 pages of v1 (verbatim) + 40 pages added from the unused, labelled school_notebooks_RU **test** split
(never in any training set): 20 with the lowest training-text overlap (mean 5 % of lines have a text twin in the training
labels) + 20 random (mean 57 %) — see `eval/testset_v2_selection.json`. Built by `eval/util/build_exam_gt.py` (reproduces
v1 on 19/20 pages exactly; 2238.jpg differs by 5 chars of orphan margin-note grouping — v1 kept verbatim).
2569 lines / 11867 words / 74206 chars / 259 teacher marks.
Rules: report-only, never select on it, never train on it. Score with `eval/score.py --gt eval/testset_v2/ru_pages/ground_truth.json`.
