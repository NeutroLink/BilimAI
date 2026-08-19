#!/usr/bin/env python3
"""E4.2 step 1 — failure analysis of a line detector against GT line boxes (sealed exam).

For every GT line: best-IoU prediction; category
  matched  IoU ≥ 0.5
  cut      best IoU in [0.1, 0.5) and the prediction lies mostly inside the GT box (pred area ≥ 70 % inside) → box too small
  merged   the prediction that best matches this GT also covers ≥ 40 % of another GT line → one box spans two lines
  split    ≥ 2 predictions each cover ≥ 25 % of this GT line → line broken into pieces
  loose    best IoU in [0.1, 0.5), none of the above (offset / too big)
  missed   no prediction with IoU ≥ 0.1
plus, for pairs with IoU ≥ 0.3, the signed edge errors (pred − GT) normalised by GT height (top/bottom) or width
(left/right): negative top = pred starts above GT; positive bottom = pred ends below GT. Tells whether boxes are too tight
vertically (ascender/descender cut), too wide, or shifted. Writes eval/runs/det_failure_<tag>.json and prints a summary.
"""
import argparse, json, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--pred", default=str(ROOT / "eval/runs/rp_det_lines_v2.json")); ap.add_argument("--tag", default="r0")
a = ap.parse_args()
def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3]); inter = max(0, x1 - x0) * max(0, y1 - y0)
    ap_ = (p[2] - p[0]) * (p[3] - p[1]); ag = (g[2] - g[0]) * (g[3] - g[1]); return inter / max(1e-9, ap_ + ag - inter), inter, ap_, ag
gt = json.load(open(a.gt, encoding="utf-8")); pred = json.load(open(a.pred, encoding="utf-8"))
cats = collections.Counter(); edges = collections.defaultdict(list); per_page = {}; ious = []
for fn, page in gt.items():
    G = [l["bbox"] for l in page["lines"]]; P = [p[:4] for p in pred.get(fn, [])]
    if not P: cats["missed"] += len(G); per_page[fn] = {"missed": len(G)}; continue
    M = np.zeros((len(G), len(P))); INTER = np.zeros_like(M); AP = np.zeros(len(P)); AG = np.zeros(len(G))
    for i, g in enumerate(G):
        for j, p in enumerate(P):
            M[i, j], INTER[i, j], AP[j], AG[i] = iou(p, g)
    pc = collections.Counter()
    for i, g in enumerate(G):
        j = int(M[i].argmax()); best = M[i, j]; ious.append(best)
        if best >= 0.5: c = "matched"
        elif best < 0.1: c = "missed"
        else:
            cover_other = [INTER[k, j] / AG[k] for k in range(len(G)) if k != i]
            n_parts = sum(1 for jj in range(len(P)) if INTER[i, jj] / AG[i] >= 0.25)
            if max(cover_other, default=0) >= 0.4: c = "merged"
            elif n_parts >= 2: c = "split"
            elif INTER[i, j] / AP[j] >= 0.7: c = "cut"
            else: c = "loose"
        cats[c] += 1; pc[c] += 1
        if best >= 0.3:
            p = P[j]; h = g[3] - g[1]; w = g[2] - g[0]
            edges["top"].append((p[1] - g[1]) / h); edges["bottom"].append((p[3] - g[3]) / h)
            edges["left"].append((p[0] - g[0]) / w); edges["right"].append((p[2] - g[2]) / w)
            edges["h_ratio"].append((p[3] - p[1]) / h); edges["w_ratio"].append((p[2] - p[0]) / w)
    per_page[fn] = dict(pc)
n = sum(cats.values()); print(f"GT lines {n} | pred boxes {sum(len(v) for v in pred.values())}")
for c in ("matched", "cut", "loose", "merged", "split", "missed"): print(f"  {c:8} {cats[c]:5} ({cats[c]/n:.0%})")
ious = np.array(ious); print(f"best-IoU per GT line: median {np.median(ious):.2f} | ≥0.5 {np.mean(ious>=0.5):.0%} | 0.3–0.5 {np.mean((ious>=0.3)&(ious<0.5)):.0%} | <0.3 {np.mean(ious<0.3):.0%}")
print("edge errors on IoU≥0.3 pairs (pred − GT, ÷ GT height/width; mean ± sd, median):")
summ = {}
for k in ("top", "bottom", "left", "right", "h_ratio", "w_ratio"):
    v = np.array(edges[k]); summ[k] = {"mean": float(v.mean()), "sd": float(v.std()), "median": float(np.median(v))}
    print(f"  {k:8} {v.mean():+.3f} ± {v.std():.3f}  median {np.median(v):+.3f}")
json.dump({"pred": a.pred, "categories": dict(cats), "n_gt": n, "iou_median": float(np.median(ious)), "edges": summ, "per_page": per_page},
          open(ROOT / f"eval/runs/det_failure_{a.tag}.json", "w"), indent=1)
print(f"→ eval/runs/det_failure_{a.tag}.json")
