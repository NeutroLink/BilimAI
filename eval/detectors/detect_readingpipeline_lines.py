#!/usr/bin/env python3
"""E4.1/E4.2 — run the product detector (bilimai.detector.RPDetector: ReadingPipeline segmenter + val-derived box growth)
over an exam set and write line boxes for score_detector.py / e2e_exam.py, plus a raw dump for det_failure.py.
  <out>            {file: [[x0,y0,x1,y1,conf], ...]}
  <out>.raw.json   {file: {"words": [...ungrown...], "polylines": [...], "groups": [[word_idx,...],...], "words_grown": [...]}}
Growth defaults = the module's val-derived factors; pass --grow-top/--grow-bottom/--grow-x 0 0 0 to reproduce the 2026-08-17
baseline (F1@0.5 0.70 on exam v2; with growth 0.89). 2026-08-19: thin wrapper over the module.
"""
import argparse, json, sys, time
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from bilimai.detector import RPDetector, LINE_GROW, WORD_GROW, DEFAULT_ONNX
ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--images", default="", help="default: <gt dir>/images")
ap.add_argument("--out", default=str(ROOT / "eval/runs/rp_det_lines_v2g.json"))
ap.add_argument("--thr-word", type=float, default=0.8); ap.add_argument("--thr-line", type=float, default=0.5)
ap.add_argument("--dilate", type=int, default=3); ap.add_argument("--min-area", type=int, default=10); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--grow-top", type=float, default=LINE_GROW[0]); ap.add_argument("--grow-bottom", type=float, default=LINE_GROW[1]); ap.add_argument("--grow-x", type=float, default=LINE_GROW[2])
a = ap.parse_args()
IMG = Path(a.images) if a.images else Path(a.gt).parent / "images"
det = RPDetector(a.onnx, a.thr_word, a.thr_line, a.dilate, a.min_area, line_grow=(a.grow_top, a.grow_bottom, a.grow_x), word_grow=WORD_GROW, threads=a.threads)
gt = json.load(open(a.gt, encoding="utf-8")); out = {}; raw = {}; t0 = time.time()
for n, fn in enumerate(gt):
    r = det.detect(cv2.imread(str(IMG / fn)))
    out[fn] = [b + [1.0] for b in r["lines"]]
    raw[fn] = {"words": [[round(v, 1) for v in w] for w in r["raw_words"]], "polylines": [[[round(v, 1) for v in p] for p in pl] for pl in r["polylines"]],
               "groups": r["line_words"], "words_grown": [[round(v, 1) for v in w] for w in r["words"]]}
    print(f"[{n+1}/{len(gt)}] {fn}: lines {len(out[fn])} (gt {len(gt[fn]['lines'])}) words {len(r['words'])}", flush=True)
json.dump(out, open(a.out, "w")); json.dump(raw, open(a.out.replace(".json", ".raw.json"), "w"))
print(f"→ {a.out} (+ .raw.json) | {time.time()-t0:.0f}s")
