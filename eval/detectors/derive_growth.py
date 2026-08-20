#!/usr/bin/env python
"""E4.2 — derive LINE_GROW box-growth fractions for a segm model on the school_notebooks_RU VAL split (never the exam).

Runs RPDetector with growth OFF over N val pages, matches predicted line boxes to GT text_line boxes (best IoU ≥ 0.2),
and prints the median signed edge gaps as fractions of the predicted box's own height/width — exactly the form _grow()
applies. Use for models trained on shrunk masks together with --unclip (train_segm SHRINK_R 0.4 ↔ unclip 1.5).

  eval/.venv/bin/python eval/detectors/derive_growth.py --onnx models/segm_ft_v3/segm_ft.onnx --unclip 1.5 --n 40
"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import cv2
from bilimai.detector import RPDetector

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True); ap.add_argument("--src", default=str(ROOT / "data/raw/school_notebooks_RU/train"))
ap.add_argument("--split", default="val"); ap.add_argument("--n", type=int, default=40)
ap.add_argument("--unclip", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gt", default="", help="exam-convention ground_truth.json (build_exam_gt.py output) — line bbox = word-union; overrides --src/--split")
ap.add_argument("--images", default="", help="images dir for --gt (default: <src>/images)")
a = ap.parse_args()

if a.gt:
    g = json.load(open(a.gt, encoding="utf-8"))
    images = {fn: fn for fn in g}
    byimg = {fn: [l["bbox"] for l in pg["lines"]] for fn, pg in g.items()}
    ids = sorted(byimg)[:a.n]
    IMD = Path(a.images) if a.images else Path(a.src) / "images"
    def gt_boxes(iid): return byimg[iid]
    def img_path(iid): return IMD / iid
else:
    d = json.load(open(Path(a.src) / f"annotations_{a.split}.json", encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    byimg = collections.defaultdict(list)
    for x in d["annotations"]:
        if cats[x["category_id"]] == "text_line": byimg[x["image_id"]].append(x)
    images = {im["id"]: im["file_name"] for im in d["images"]}
    ids = sorted(byimg); rng = np.random.RandomState(a.seed); rng.shuffle(ids); ids = ids[:a.n]
    def gt_boxes(iid):
        out = []
        for x in byimg[iid]:
            for seg in x["segmentation"]:
                if len(seg) >= 6:
                    xs, ys = seg[0::2], seg[1::2]; out.append([min(xs), min(ys), max(xs), max(ys)])
        return out
    def img_path(iid):
        fp = Path(a.src) / "images" / images[iid]
        return fp if fp.exists() else Path(a.src) / images[iid]

det = RPDetector(a.onnx, line_grow=(0, 0, 0), word_grow=(0, 0, 0), unclip=a.unclip)


def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / max(1e-9, (p[2] - p[0]) * (p[3] - p[1]) + (g[2] - g[0]) * (g[3] - g[1]) - i)


gaps = {"top": [], "bottom": [], "left": [], "right": []}
matched = total = 0
for k, iid in enumerate(ids):
    img = cv2.imread(str(img_path(iid)))
    if img is None: continue
    pred = det.detect(img)["lines"]
    G = gt_boxes(iid)
    total += len(G)
    for g in G:
        best = max(pred, key=lambda p: iou(p, g), default=None)
        if best is None or iou(best, g) < 0.2: continue
        matched += 1
        h = max(1.0, best[3] - best[1]); w = max(1.0, best[2] - best[0])
        gaps["top"].append((best[1] - g[1]) / h); gaps["bottom"].append((g[3] - best[3]) / h)
        gaps["left"].append((best[0] - g[0]) / w); gaps["right"].append((g[2] - best[2]) / w)
    print(f"[{k + 1}/{len(ids)}] {images[iid]}: pred {len(pred)} gt {len(G)}", flush=True)

med = {k: float(np.median(v)) for k, v in gaps.items()}
print(f"\nmatched {matched}/{total} GT lines (IoU≥0.2)")
print("median gaps as fractions of the predicted box's own size (positive = grow needed):", json.dumps(med, indent=1))
grow = (max(0.0, round(med["top"], 3)), max(0.0, round(med["bottom"], 3)),
        max(0.0, round((med["left"] + med["right"]) / 2, 3)))
print(f"LINE_GROW for this model: top {grow[0]} bottom {grow[1]} x {grow[2]}")
