#!/usr/bin/env python3
"""E4.2 — benchmark open-source line detectors against our production ReadingPipeline (2026-08-22).

Only two detectors have ever been scored here (RP, Surya, both on 2026-08-17). Today's failure analysis showed RP's
ceiling is set by what its MASK captures — thin descender strokes fall under its 0.8 threshold, and no box-edge
post-processing recovers ink the mask never saw. So the next honest move is a different detector, not more tuning.

Emits our standard format so the existing scorers work unchanged:
    eval/runs/det_<name>.json        {file: [[x0,y0,x1,y1,conf], ...]}
then score with:
    det_failure.py --pred eval/runs/det_<name>.json --tag <name>     (matched / missed / merged / split, h_ratio)
    miss_profile.py --pred ...                                        (what it misses)
    crop_cost.py   --pred ...                                         (what its crops cost the reader)

Each backend is optional — run what is installed:
    eval/.venv-det/bin/python eval/detectors/bench_detectors.py --which kraken,doctr,craft

NOTE ON GEOMETRY: Kraken and CRAFT emit POLYGONS. We store the axis-aligned bounding box so the scorers stay
comparable with RP, but we ALSO keep the raw polygon in <out>.poly.json — that is what a polygon-aware crop
experiment would need, and it is the whole reason these detectors are interesting.
"""
import argparse, json, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--images", default=str(ROOT / "eval/testset_v2/ru_pages/images"))
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--outdir", default=str(ROOT / "eval/runs"))
ap.add_argument("--which", default="kraken,doctr,craft,paddle")
ap.add_argument("--limit", type=int, default=0, help="first N pages only (smoke test)")
a = ap.parse_args()

gt = json.load(open(a.gt, encoding="utf-8"))
files = sorted(gt)[: a.limit] if a.limit else sorted(gt)
IMG = Path(a.images)
print(f"pages: {len(files)}", flush=True)


def bbox(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


# ---------------------------------------------------------------- backends
def run_kraken(files):
    """Baseline-based segmentation built for handwritten/historical documents; returns line polygons."""
    from kraken import blla
    from kraken.lib import vgsl
    from PIL import Image
    model = None
    out, poly = {}, {}
    for i, fn in enumerate(files):
        im = Image.open(IMG / fn).convert("RGB")
        seg = blla.segment(im, model=model)
        lines = getattr(seg, "lines", None) or seg.get("lines", [])
        bs, ps = [], []
        for l in lines:
            pg = getattr(l, "boundary", None) or (l.get("boundary") if isinstance(l, dict) else None)
            if not pg: continue
            b = bbox(pg); bs.append(b + [1.0]); ps.append([[float(x), float(y)] for x, y in pg])
        out[fn], poly[fn] = bs, ps
        if i % 10 == 0: print(f"  kraken {i+1}/{len(files)}", flush=True)
    return out, poly


def run_doctr(files):
    """docTR DBNet detector (word boxes) — grouped into lines by vertical proximity."""
    from doctr.models import detection_predictor
    from PIL import Image
    import numpy as np
    det = detection_predictor(arch="db_resnet50", pretrained=True, assume_straight_pages=False)
    out, poly = {}, {}
    for i, fn in enumerate(files):
        im = np.array(Image.open(IMG / fn).convert("RGB"))
        H, W = im.shape[:2]
        res = det([im])[0]
        arr = res["words"] if isinstance(res, dict) else res
        bs, ps = [], []
        for w in arr:
            w = np.array(w)
            if w.ndim == 2 and w.shape[0] >= 4:                    # polygon (assume_straight_pages=False)
                pg = [[float(p[0]) * W, float(p[1]) * H] for p in w[:4, :2]]   # docTR appends a 5th row (confidence), not a corner
            else:                                                   # x0,y0,x1,y1[,conf]
                v = w.reshape(-1)
                pg = [[float(v[0]) * W, float(v[1]) * H], [float(v[2]) * W, float(v[1]) * H],
                      [float(v[2]) * W, float(v[3]) * H], [float(v[0]) * W, float(v[3]) * H]]
            bs.append(bbox(pg) + [1.0]); ps.append(pg)
        out[fn], poly[fn] = group_words_to_lines(bs), ps
        if i % 10 == 0: print(f"  doctr {i+1}/{len(files)}", flush=True)
    return out, poly


def run_craft(files):
    """CRAFT character-region detector; craft-text-detector returns word polygons."""
    from craft_text_detector import Craft
    craft = Craft(output_dir=None, crop_type="poly", cuda=False)
    out, poly = {}, {}
    for i, fn in enumerate(files):
        r = craft.detect_text(str(IMG / fn))
        ps = [[[float(x), float(y)] for x, y in p] for p in r.get("polys", [])]
        out[fn] = group_words_to_lines([bbox(p) + [1.0] for p in ps]); poly[fn] = ps
        if i % 10 == 0: print(f"  craft {i+1}/{len(files)}", flush=True)
    return out, poly


def run_paddle(files):
    """PP-OCR DBNet text detector; returns quadrilaterals per text region."""
    from paddleocr import PaddleOCR
    try:                                  # PaddleOCR >=3 dropped show_log/use_gpu
        ocr = PaddleOCR(lang="ru", use_textline_orientation=False)
    except TypeError:
        try: ocr = PaddleOCR(lang="ru")
        except TypeError: ocr = PaddleOCR(use_angle_cls=False, lang="ru", show_log=False, use_gpu=False)
    out, poly = {}, {}
    for i, fn in enumerate(files):
        try:
            res = ocr.ocr(str(IMG / fn), det=True, rec=False, cls=False)
            boxes = res[0] if res and res[0] else []
        except TypeError:                 # v3 API: predict() returns dicts with dt_polys
            r = ocr.predict(str(IMG / fn))
            boxes = []
            for item in (r or []):
                dp = item.get("dt_polys") if isinstance(item, dict) else getattr(item, "dt_polys", None)
                if dp is not None: boxes.extend([[[float(x), float(y)] for x, y in q] for q in dp])
        ps = [[[float(x), float(y)] for x, y in b] for b in boxes]
        out[fn] = group_words_to_lines([bbox(p) + [1.0] for p in ps]); poly[fn] = ps
        if i % 10 == 0: print(f"  paddle {i+1}/{len(files)}", flush=True)
    return out, poly


def group_words_to_lines(boxes, tol=0.6):
    """Word/region boxes -> line boxes: sort by y, merge while vertical centres are within tol * median height."""
    if not boxes: return []
    bs = sorted([b[:4] for b in boxes], key=lambda b: ((b[1] + b[3]) / 2, b[0]))
    hs = sorted(b[3] - b[1] for b in bs); med = hs[len(hs) // 2] or 1.0
    lines, cur = [], [bs[0]]
    for b in bs[1:]:
        cy = (b[1] + b[3]) / 2; ccy = sum((x[1] + x[3]) / 2 for x in cur) / len(cur)
        if abs(cy - ccy) <= tol * med: cur.append(b)
        else: lines.append(cur); cur = [b]
    lines.append(cur)
    return [[min(x[0] for x in g), min(x[1] for x in g), max(x[2] for x in g), max(x[3] for x in g), 1.0] for g in lines]


def run_easyocr(files):
    """EasyOCR bundles CRAFT; detail=1 with the detector only returns word quadrilaterals."""
    import easyocr
    rd = easyocr.Reader(["ru"], gpu=False, verbose=False)
    out, poly = {}, {}
    for i, fn in enumerate(files):
        res = rd.detect(str(IMG / fn))
        horiz, free = (res + ([], []))[:2]
        ps = []
        for grp in (horiz or []):
            for b in grp:
                if len(b) == 4 and not isinstance(b[0], (list, tuple)):
                    x0, x1, y0, y1 = [float(v) for v in b]
                    ps.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        for grp in (free or []):
            for pg in grp:
                if pg: ps.append([[float(x), float(y)] for x, y in pg])
        out[fn] = group_words_to_lines([bbox(pp) + [1.0] for pp in ps]); poly[fn] = ps
        if i % 10 == 0: print(f"  easyocr {i+1}/{len(files)}", flush=True)
    return out, poly


BACKENDS = {"kraken": run_kraken, "doctr": run_doctr, "craft": run_craft, "paddle": run_paddle, "easyocr": run_easyocr}

for name in [w.strip() for w in a.which.split(",") if w.strip()]:
    fn_ = BACKENDS.get(name)
    if not fn_: print(f"unknown backend {name}"); continue
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    try:
        boxes, polys = fn_(files)
    except Exception as e:
        print(f"  {name} FAILED: {type(e).__name__}: {str(e)[:200]}"); continue
    op = Path(a.outdir) / f"det_{name}.json"
    json.dump(boxes, open(op, "w"), indent=None)
    json.dump(polys, open(str(op) + ".poly.json", "w"), indent=None)
    n = sum(len(v) for v in boxes.values())
    print(f"  {name}: {n} line boxes over {len(files)} pages in {time.time()-t0:.0f}s → {op}", flush=True)
