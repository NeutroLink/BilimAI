#!/usr/bin/env python
"""Score a line detector against ground-truth line boxes: precision/recall/F1 at IoU >= thr, per page + overall.
Usage: score_detector.py --gt eval/testset_v1/ru_pages/ground_truth.json --pred eval/runs/surya_det.json [--iou 0.5]
pred: {file: [[x0,y0,x1,y1(,conf)], ...]}
"""
import argparse, json
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if not inter: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gt", required=True); ap.add_argument("--pred", required=True); ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args(); gt = json.load(open(a.gt)); pr = json.load(open(a.pred))
    TP = FP = FN = 0; rows = []
    for fn, pg in gt.items():
        G = [l["bbox"] for l in pg["lines"]]; P = [b[:4] for b in pr.get(fn, [])]
        if G and P:
            M = np.array([[iou(g, p) for p in P] for g in G]); r, c = linear_sum_assignment(-M)
            tp = int(sum(M[i, j] >= a.iou for i, j in zip(r, c)))
        else: tp = 0
        fp, fn_ = len(P) - tp, len(G) - tp; TP += tp; FP += fp; FN += fn_
        rows.append((fn, len(G), len(P), tp, fp, fn_))
    prec = TP / max(TP + FP, 1); rec = TP / max(TP + FN, 1); f1 = 2*prec*rec/max(prec+rec, 1e-9)
    print(f"{'page':10} {'gt':>4} {'pred':>5} {'tp':>4} {'fp':>4} {'fn':>4}")
    for r in rows: print(f"{r[0]:10} {r[1]:4d} {r[2]:5d} {r[3]:4d} {r[4]:4d} {r[5]:4d}")
    print(f"IoU>={a.iou}: precision {prec:.3f} recall {rec:.3f} F1 {f1:.3f}   (TP {TP} FP {FP} FN {FN})")

if __name__ == "__main__": main()
