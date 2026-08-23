#!/usr/bin/env python
"""Qwen baseline on two product roadblocks that were only ever measured against GLM. 2026-08-23.

Both are computed from ALREADY-SAVED zero-shot page runs — no GPU, no money, no new inference.

1. READING ORDER. bilimai/dictation.py::align walks the pupil's words against the key in sequence, so
   lines arriving out of order derail the marking and invent errors. HANDOFF-2026-08-23 §3 flags this as
   real in the product and INVISIBLE to our metric: eval/score.py matches lines by IoU and assembles in GT
   order, so order literally cannot cost anything there. It has never been measured. A page-level model
   emits an ORDERED list, so for Qwen this is directly observable.

2. TEACHER INK. Ground truth is pupil-only, and the page prompt says "IGNORE anything written in red".
   The ink rule (bilimai/ink.py) is wired into nothing in production. So: of the boxes Qwen produces that
   match no pupil line, how many are actually the teacher's marking? School pages arrive already marked,
   and a marker that grades the teacher's own corrections as the pupil's work is worse than useless.

Teacher ground truth is built from school_notebooks' category 2 (`teacher_comment`) — 1,055 annotations on
the 60 exam pages, every one carrying its transcription.

  eval/.venv/bin/python eval/detectors/qwen_baseline_gaps.py eval/runs/qwen_hard20.json
"""
import argparse, ast, json, sys, unicodedata
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", help="saved qwen_grounding_probe.py output(s)")
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--ann", default=str(ROOT / "data/raw/school_notebooks_RU/exam/annotations_test.json"))
ap.add_argument("--iou", type=float, default=0.5)
ap.add_argument("--out", default=str(ROOT / "eval/runs/qwen_baseline_gaps.json"))
a = ap.parse_args()


def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    i = max(0., x1 - x0) * max(0., y1 - y0)
    return i / max(1e-9, (p[2]-p[0])*(p[3]-p[1]) + (g[2]-g[0])*(g[3]-g[1]) - i)


def contains(outer, inner, frac=0.6):
    """Fraction of `inner` covered by `outer` — teacher marks are single letters, far too small for IoU."""
    x0, y0, x1, y1 = max(outer[0], inner[0]), max(outer[1], inner[1]), min(outer[2], inner[2]), min(outer[3], inner[3])
    i = max(0., x1 - x0) * max(0., y1 - y0)
    return i / max(1e-9, (inner[2]-inner[0]) * (inner[3]-inner[1])) >= frac


def attrs(x):
    t = x.get("attributes")
    if isinstance(t, str):
        try: t = ast.literal_eval(t)
        except Exception: t = {}
    return t or {}


def poly_bbox(seg):
    xs, ys = seg[0][0::2], seg[0][1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


gt = json.load(open(a.gt, encoding="utf-8"))
ann = json.load(open(a.ann, encoding="utf-8"))
cats = {c["id"]: c["name"] for c in ann["categories"]}
byname = {im["file_name"]: im["id"] for im in ann["images"]}
teacher = defaultdict(list)
for x in ann["annotations"]:
    if cats[x["category_id"]] == "teacher_comment":
        teacher[x["image_id"]].append({"bbox": poly_bbox(x["segmentation"]),
                                       "text": (attrs(x).get("translation") or "").strip()})

records = []
for f in a.runs:
    records += json.load(open(f, encoding="utf-8"))

order_ok = order_tot = 0
inversions = 0
teach_hit = extra = matched_pupil = pred_tot = 0
per_page = []
teacher_examples = []

for r in records:
    page = gt.get(r["file"])
    if not page:
        continue
    sw, sh = r["sub_size"]; ox, oy = r["sub_origin"]
    lines = page["lines"]                                    # already sorted by (column, y, x) = reading order
    P = []
    for it in r["items"]:
        b = it["bbox"]
        q = [b[0]/1000*sw + ox, b[1]/1000*sh + oy, b[2]/1000*sw + ox, b[3]/1000*sh + oy]
        if q[2] > q[0] and q[3] > q[1]:
            P.append((q, it.get("text", "")))
    pred_tot += len(P)
    tmarks = teacher.get(byname.get(r["file"]), [])

    seq = []            # GT index for each prediction, in EMISSION order
    for q, t in P:
        best_j, best = None, 0.0
        for j, l in enumerate(lines):
            v = iou(q, l["bbox"])
            if v > best: best, best_j = v, j
        if best >= a.iou:
            seq.append(best_j); matched_pupil += 1
        else:
            extra += 1
            hit = [m for m in tmarks if m["text"] and contains(q, m["bbox"])]
            if hit:
                teach_hit += 1
                if len(teacher_examples) < 8:
                    teacher_examples.append({"file": r["file"], "pred": t[:40],
                                             "teacher_marks_inside": [m["text"] for m in hit][:6]})
    inv = sum(1 for i in range(len(seq)) for j in range(i+1, len(seq)) if seq[i] > seq[j])
    ok = sum(1 for i in range(len(seq)-1) if seq[i] < seq[i+1])
    order_ok += ok; order_tot += max(0, len(seq)-1); inversions += inv
    per_page.append({"file": r["file"], "region": r.get("region"), "pred": len(P),
                     "matched": len(seq), "adjacent_ok": ok, "adjacent": max(0, len(seq)-1),
                     "inversions": inv})

res = {
 "records": len(records), "pred_boxes": pred_tot, "matched_pupil_lines": matched_pupil,
 "reading_order": {
   "adjacent_pairs": order_tot, "in_correct_order": order_ok,
   "pct_correct": round(100*order_ok/max(1, order_tot), 1),
   "total_inversions": inversions,
   "pages_perfectly_ordered": sum(1 for p in per_page if p["inversions"] == 0 and p["matched"] > 1),
   "pages_with_matches": sum(1 for p in per_page if p["matched"] > 1),
 },
 "teacher_ink": {
   "boxes_matching_no_pupil_line": extra,
   "of_those_containing_teacher_marking": teach_hit,
   "pct_of_extras_that_are_teacher": round(100*teach_hit/max(1, extra), 1),
   "pct_of_all_boxes_that_are_teacher": round(100*teach_hit/max(1, pred_tot), 1),
   "examples": teacher_examples,
 },
 "per_page": per_page,
}
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_page"}, ensure_ascii=False, indent=1))
print(f"\n-> {a.out}")
