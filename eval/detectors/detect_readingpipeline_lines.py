#!/usr/bin/env python3
"""E4.1/E4.2 — ReadingPipeline segmenter → line boxes (+ raw words and text_line polylines) for the sealed exam.

Runs ai-forever/ReadingPipeline-notebooks segm_model.onnx (LinkNet, 896×896, sigmoid; channels 0 shrinked_text = words,
1 bordered_text, 2 text_line) and groups words into lines with a re-implementation of the pipeline's LineFinder idea:
each word joins the text_line polyline nearest to it vertically. Writes
  <out>            {file: [[x0,y0,x1,y1,conf], ...]}                     line boxes (score_detector.py / e2e_exam.py format)
  <out>.raw.json   {file: {"words": [[x0,y0,x1,y1],...], "polylines": [[[x,y],...],...], "groups": [[word_idx,...],...]}}
so the failure analysis (det_failure.py) can see what the network saw. Knobs: --thr-word, --thr-line, --dilate, --min-area.
2026-08-19: CLI + raw dump; defaults = the 2026-08-17 baseline (thr 0.8 / 0.5, dilate 3, min area 10).
"""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort
ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default=str(ROOT / "models/readingpipeline/segm/segm_model.onnx"))
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--images", default="", help="default: <gt dir>/images")
ap.add_argument("--out", default=str(ROOT / "eval/runs/rp_det_lines_v2.json"))
ap.add_argument("--thr-word", type=float, default=0.8); ap.add_argument("--thr-line", type=float, default=0.5)
ap.add_argument("--dilate", type=int, default=3); ap.add_argument("--min-area", type=int, default=10)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--grow-top", type=float, default=0.0); ap.add_argument("--grow-bottom", type=float, default=0.0); ap.add_argument("--grow-x", type=float, default=0.0,
                help="grow each final line box by these fractions of its own height (top/bottom) / width (each side); derived on the val split 2026-08-19: 0.25 / 0.365 / 0.02 — the network's word-union boxes are ~61 %% of the annotators' line height")
a = ap.parse_args()
IMG = Path(a.images) if a.images else Path(a.gt).parent / "images"
so = ort.SessionOptions(); so.intra_op_num_threads = a.threads; so.inter_op_num_threads = a.threads
sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"]); name = sess.get_inputs()[0].name
gt = json.load(open(a.gt, encoding="utf-8")); out = {}; raw = {}; t0 = time.time()
for n, fn in enumerate(gt):
    img = cv2.imread(str(IMG / fn)); H, W = img.shape[:2]
    x = np.transpose(cv2.resize(img, (896, 896)).astype(np.float32) / 255, (2, 0, 1))[None]
    pred = sess.run(None, {name: x})[0][0]; sx, sy = W / 896, H / 896
    m = (pred[0] > a.thr_word).astype(np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    for c in cs:
        if cv2.contourArea(c) < a.min_area: continue
        x0, y0, w, h = cv2.boundingRect(c); words.append([x0 * sx, y0 * sy, (x0 + w) * sx, (y0 + h) * sy])
    ml = (pred[2] > a.thr_line).astype(np.uint8)
    if a.dilate > 0: ml = cv2.dilate(ml, np.ones((a.dilate, a.dilate), np.uint8))
    cl, _ = cv2.findContours(ml, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in cl:
        if len(c) < 5: continue
        pts = c[:, 0, :].astype(np.float32); pts[:, 0] *= sx; pts[:, 1] *= sy; lines.append(pts)
    groups = [[] for _ in lines]
    for wi, wb in enumerate(words):
        cx, cy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2; best = None; bd = 1e9
        for li, pts in enumerate(lines):
            near = pts[np.abs(pts[:, 0] - cx) < max(30, (wb[2] - wb[0]))]
            if len(near) == 0: continue
            d = np.min(np.abs(near[:, 1] - cy))
            if d < bd: bd, best = d, li
        if best is not None and bd < 1.2 * (wb[3] - wb[1]): groups[best].append(wi)
    boxes = []
    for g in groups:
        if not g: continue
        gb = np.array([words[i] for i in g]); x0, y0, x1, y1 = float(gb[:, 0].min()), float(gb[:, 1].min()), float(gb[:, 2].max()), float(gb[:, 3].max())
        h, w = y1 - y0, x1 - x0
        boxes.append([max(0.0, x0 - a.grow_x * w), max(0.0, y0 - a.grow_top * h), min(float(W), x1 + a.grow_x * w), min(float(H), y1 + a.grow_bottom * h), 1.0])
    out[fn] = boxes; raw[fn] = {"words": [[round(v, 1) for v in w] for w in words], "polylines": [p.round(1).tolist() for p in lines], "groups": groups}
    print(f"[{n+1}/{len(gt)}] {fn}: lines {len(boxes)} (gt {len(gt[fn]['lines'])}) words {len(words)} polylines {len(lines)}", flush=True)
json.dump(out, open(a.out, "w")); json.dump(raw, open(a.out.replace(".json", ".raw.json"), "w"))
print(f"→ {a.out} (+ .raw.json) | {time.time()-t0:.0f}s")
