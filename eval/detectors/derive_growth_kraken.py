#!/usr/bin/env python
"""E4.2 — fit LINE_GROW for KRAKEN on the school_notebooks_RU VAL split (never the exam). 2026-08-22

Kraken was benchmarked UNTUNED against RP's val-fitted growth (handoff §4e: "a floor, not a verdict").
It is the only detector that beats RP on misses — 237 vs 335 — so it deserves the same treatment RP got.

RP's own numbers (top 0.25 / bottom 0.365 / sides 0.02) are NOT transferable: they describe how RP's
word-union boxes sit against the truth. Kraken emits native baseline polygons with different geometry,
so the factors have to be re-fitted. Same PROCEDURE as derive_growth.py, different detector:

  run with growth OFF -> match each GT WORD-UNION line box to its best predicted box (IoU >= 0.2)
  -> median signed edge gap, as a fraction of the PREDICTED box's own height / width
  -> that median IS the growth factor, because _grow() applies it in exactly that form.

Median, not mean: a handful of wildly wrong boxes must not drag the factor.

  eval/.venv-det/bin/python eval/detectors/derive_growth_kraken.py --n 40

⚠ THE GROUND-TRUTH CONVENTION IS NOT OPTIONAL. school_notebooks_RU carries two kinds of line box:
  * `text_line` annotation polygons — the annotator's generous line REGION
  * word-union — the union of the line's `pupil_text` word boxes, tight around the ink (the exam
    convention that build_exam_gt.py produces)
Fitting against `text_line` gives (0, 0, 0) for BOTH RP and Kraken — every gap comes out negative,
i.e. "the boxes are already too big". Fitting against word-union gives RP (0.257, 0.377, 0.018),
which reproduces the shipped production constants (0.25, 0.365, 0.02) almost exactly. So word-union
is correct and text_line is wrong, and this script defaults to word-union.

That error was caught only by re-running the fit on RP as a CONTROL and noticing it disagreed with
what production ships. Always run the control.

RESULT (40 val pages, 2026-08-22): Kraken LINE_GROW = (0.05, 0.11, 0.0) — 5x less growth at the top
and 3.4x less at the bottom than RP needs, because Kraken emits native baseline polygons rather than
a union of word boxes. Applying it to the exam fixed the geometry (median IoU 0.76 -> 0.80, edges
centred) and changed nothing that matters: 6 fewer matched lines, 12 MORE merges. See
plans/CROP-GEOMETRY-2026-08-22.md §5. Kraken untuned was not being handicapped.
"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(ROOT / "data/raw/school_notebooks_RU/train"))
ap.add_argument("--split", default="val")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(ROOT / "eval/runs/kraken_growth_val.json"))
ap.add_argument("--boxes-out", default=str(ROOT / "eval/runs/det_kraken_val.json"))
ap.add_argument("--convention", default="word-union", choices=["word-union", "text_line"],
                help="word-union reproduces production RP's constants; text_line does NOT (see docstring)")
a = ap.parse_args()

from PIL import Image
from kraken import blla

d = json.load(open(Path(a.src) / f"annotations_{a.split}.json", encoding="utf-8"))
cats = {c["id"]: c["name"] for c in d["categories"]}
images = {im["id"]: im["file_name"] for im in d["images"]}
byimg = collections.defaultdict(list)                       # text_line convention
grp = collections.defaultdict(lambda: collections.defaultdict(list))   # word-union convention
for x in d["annotations"]:
    nm = cats[x["category_id"]]
    if nm == "text_line":
        byimg[x["image_id"]].append(x)
    elif nm == "pupil_text" and x.get("group_id") is not None:
        for seg in x.get("segmentation") or []:
            if len(seg) >= 6:
                xs, ys = seg[0::2], seg[1::2]
                grp[x["image_id"]][x["group_id"]].append([min(xs), min(ys), max(xs), max(ys)])
ids = sorted(grp if a.convention == "word-union" else byimg)
rng = np.random.RandomState(a.seed); rng.shuffle(ids); ids = ids[:a.n]


def gt_boxes(iid):
    if a.convention == "word-union":
        out = []
        for _, ws in grp[iid].items():
            B = np.array(ws, dtype=float)
            out.append([float(B[:, 0].min()), float(B[:, 1].min()), float(B[:, 2].max()), float(B[:, 3].max())])
        return out
    out = []
    for x in byimg[iid]:
        for seg in x["segmentation"]:
            if len(seg) >= 6:
                xs, ys = seg[0::2], seg[1::2]
                out.append([min(xs), min(ys), max(xs), max(ys)])
    return out


def img_path(iid):
    fp = Path(a.src) / "images" / images[iid]
    return fp if fp.exists() else Path(a.src) / images[iid]


def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / max(1e-9, (p[2]-p[0])*(p[3]-p[1]) + (g[2]-g[0])*(g[3]-g[1]) - i)


gaps = {"top": [], "bottom": [], "left": [], "right": []}
matched = total = 0
saved = {}
for k, iid in enumerate(ids):
    fp = img_path(iid)
    if not fp.exists():
        print(f"[{k+1}/{len(ids)}] MISSING {fp.name}", flush=True); continue
    im = Image.open(fp).convert("RGB")
    seg = blla.segment(im, model=None)
    lines = getattr(seg, "lines", None) or seg.get("lines", [])
    pred = []
    for l in lines:
        pg = getattr(l, "boundary", None) or (l.get("boundary") if isinstance(l, dict) else None)
        if not pg: continue
        xs = [p[0] for p in pg]; ys = [p[1] for p in pg]
        pred.append([float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))])
    saved[images[iid]] = [b + [1.0] for b in pred]
    G = gt_boxes(iid); total += len(G)
    for g in G:
        best = max(pred, key=lambda p: iou(p, g), default=None)
        if best is None or iou(best, g) < 0.2: continue
        matched += 1
        h = max(1.0, best[3] - best[1]); w = max(1.0, best[2] - best[0])
        gaps["top"].append((best[1] - g[1]) / h)      # positive -> pred starts BELOW truth -> grow up
        gaps["bottom"].append((g[3] - best[3]) / h)   # positive -> truth ends BELOW pred  -> grow down
        gaps["left"].append((best[0] - g[0]) / w)
        gaps["right"].append((g[2] - best[2]) / w)
    print(f"[{k+1}/{len(ids)}] {images[iid]}: pred {len(pred)} gt {len(G)}", flush=True)

med = {k: float(np.median(v)) for k, v in gaps.items() if v}
p25 = {k: float(np.percentile(v, 25)) for k, v in gaps.items() if v}
p75 = {k: float(np.percentile(v, 75)) for k, v in gaps.items() if v}
print(f"\nmatched {matched}/{total} GT lines (IoU>=0.2)  [convention: {a.convention}]")
print("median signed gaps (fraction of the PREDICTED box's own size; positive = grow needed):")
for k in ("top", "bottom", "left", "right"):
    if k in med: print(f"  {k:<7} median {med[k]:+.3f}   p25 {p25[k]:+.3f}   p75 {p75[k]:+.3f}")
grow = (max(0.0, round(med.get("top", 0), 3)), max(0.0, round(med.get("bottom", 0), 3)),
        max(0.0, round((med.get("left", 0) + med.get("right", 0)) / 2, 3)))
print(f"\nLINE_GROW for KRAKEN : top {grow[0]}  bottom {grow[1]}  sides {grow[2]}")
print(f"LINE_GROW for RP     : top 0.25   bottom 0.365  sides 0.02   (for contrast — do NOT reuse)")
json.dump({"n_pages": len(ids), "matched": matched, "total": total,
           "median": med, "p25": p25, "p75": p75, "line_grow": list(grow)},
          open(a.out, "w"), indent=1)
json.dump(saved, open(a.boxes_out, "w"))
print(f"\n-> {a.out}\n-> {a.boxes_out}")
