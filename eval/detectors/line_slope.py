#!/usr/bin/env python3
"""E4.2 — HOW MUCH DO LINES SLOPE, AND WHAT DOES A RIGID BOX COST? (2026-08-22)

Founder's premise: lines with high peaks / low troughs, and lines that slope up or down, do not fit an
axis-aligned rectangle. Either the box shaves ink, or it must grow so tall it swallows the neighbours.

This measures the premise geometrically, with no reader and no GPU, from data already on disk:
  * GT word boxes per line              -> the true ink extent (eval/testset_v2/ru_pages/ground_truth.json)
  * RP detector line boxes              -> what production actually crops (eval/runs/rp_det_lines_v2g.json)

Per line, from the word boxes:
  slope_deg  : least-squares fit through word-box centres
  h_rect     : height of the axis-aligned box around all words   (what we crop today)
  h_band     : ink height measured RELATIVE to the fitted sloping baseline (what a slope-following strip needs)
  waste      : h_rect / h_band  -- 1.0 means slope costs nothing, 1.5 means the rigid box is 50 % dead paper
  bulge      : max |residual| of word centres from the straight fit, in units of median word height (curvature)

And against the detector box:
  shaved     : fraction of GT word-box AREA that falls OUTSIDE the matched RP box  (the founder's "shaved off")
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
gt = json.load(open(ROOT / "eval/testset_v2/ru_pages/ground_truth.json", encoding="utf-8"))
rp = json.load(open(ROOT / "eval/runs/rp_det_lines_v2g.json", encoding="utf-8"))


def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return i / max(1e-9, (p[2]-p[0])*(p[3]-p[1]) + (g[2]-g[0])*(g[3]-g[1]) - i)


rows = []
for fn, page in gt.items():
    P = [b[:4] for b in rp.get(fn, [])]
    for L in page["lines"]:
        ws = L.get("words") or []
        if len(ws) < 2:
            continue
        B = np.array([w["bbox"] for w in ws], dtype=float)          # x0,y0,x1,y1
        cx = (B[:, 0] + B[:, 2]) / 2
        cy = (B[:, 1] + B[:, 3]) / 2
        hs = B[:, 3] - B[:, 1]
        if cx.max() - cx.min() < 1:
            continue
        m, c = np.polyfit(cx, cy, 1)                                 # baseline through word centres
        slope_deg = float(np.degrees(np.arctan(m)))
        resid = cy - (m * cx + c)
        bulge = float(np.max(np.abs(resid)) / max(1e-6, np.median(hs)))

        h_rect = float(B[:, 3].max() - B[:, 1].min())
        # ink extent measured relative to the sloping baseline: de-slope every word box, then take the span
        top_rel = B[:, 1] - (m * cx + c)
        bot_rel = B[:, 3] - (m * cx + c)
        h_band = float(bot_rel.max() - top_rel.min())
        waste = h_rect / max(1e-6, h_band)

        best, shaved = 0.0, None
        if P:
            best_i = int(np.argmax([iou(p, L["bbox"]) for p in P]))
            best = iou(P[best_i], L["bbox"])
            if best >= 0.5:
                d = P[best_i]
                area = np.maximum(0, B[:, 2]-B[:, 0]) * np.maximum(0, B[:, 3]-B[:, 1])
                ix = np.maximum(0, np.minimum(B[:, 2], d[2]) - np.maximum(B[:, 0], d[0]))
                iy = np.maximum(0, np.minimum(B[:, 3], d[3]) - np.maximum(B[:, 1], d[1]))
                shaved = float(1 - (ix*iy).sum() / max(1e-6, area.sum()))

        rows.append(dict(fn=fn, gid=L.get("group_id"), nw=len(ws), chars=len(L.get("text") or ""),
                         slope=slope_deg, bulge=bulge, h_rect=h_rect, h_band=h_band, waste=waste,
                         med_wh=float(np.median(hs)), iou=best, shaved=shaved))

A = rows
sl = np.abs([r["slope"] for r in A]); wa = np.array([r["waste"] for r in A]); bu = np.array([r["bulge"] for r in A])
print(f"lines with >=2 words: {len(A)}  (of {sum(len(p['lines']) for p in gt.values())} GT lines)\n")

def q(v, name, unit=""):
    print(f"  {name:<12} median {np.median(v):7.3f}{unit}   p75 {np.percentile(v,75):7.3f}   "
          f"p90 {np.percentile(v,90):7.3f}   p99 {np.percentile(v,99):7.3f}   max {v.max():7.3f}")

print("SLOPE / CURVATURE of real lines")
q(sl, "|slope| deg", "")
q(bu, "bulge (xh)")
print(f"  |slope| > 2 deg : {100*np.mean(sl>2):5.1f} %      > 5 deg : {100*np.mean(sl>5):5.1f} %"
      f"      > 10 deg : {100*np.mean(sl>10):5.1f} %")
print(f"  bulge   > 0.25  : {100*np.mean(bu>0.25):5.1f} %     > 0.5   : {100*np.mean(bu>0.5):5.1f} %\n")

print("COST OF RIGIDITY  (waste = rigid-box height / slope-following band height)")
q(wa, "waste")
print(f"  waste > 1.2 : {100*np.mean(wa>1.2):5.1f} %   > 1.5 : {100*np.mean(wa>1.5):5.1f} %   "
      f"> 2.0 : {100*np.mean(wa>2.0):5.1f} %")
print(f"  dead paper in the average rigid crop: {100*np.mean(1 - 1/wa):.1f} % of its height\n")

print("SLOPE vs WASTE")
print(f"  {'|slope| band':<16}{'n':>6}{'med waste':>11}{'med bulge':>11}")
for lo, hi in [(0,1),(1,2),(2,4),(4,8),(8,90)]:
    k = (sl>=lo)&(sl<hi)
    if k.sum(): print(f"  {f'{lo}-{hi} deg':<16}{k.sum():>6}{np.median(wa[k]):>11.3f}{np.median(bu[k]):>11.3f}")

sh = [r for r in A if r["shaved"] is not None]
if sh:
    s = np.array([r["shaved"] for r in sh]); ss = np.abs([r["slope"] for r in sh]); ww = np.array([r["waste"] for r in sh])
    print(f"\nDETECTOR SHAVING  (matched lines, IoU>=0.5, n={len(sh)})")
    print(f"  GT word-box area falling OUTSIDE the RP box: mean {100*s.mean():.2f} %  median {100*np.median(s):.2f} %")
    print(f"  lines losing >5 % of ink : {100*np.mean(s>0.05):5.1f} %     >20 % : {100*np.mean(s>0.20):5.1f} %")
    print(f"  {'|slope| band':<16}{'n':>6}{'mean shaved %':>15}")
    for lo, hi in [(0,1),(1,2),(2,4),(4,8),(8,90)]:
        k = (ss>=lo)&(ss<hi)
        if k.sum(): print(f"  {f'{lo}-{hi} deg':<16}{k.sum():>6}{100*s[k].mean():>15.2f}")
    print(f"  {'waste band':<16}{'n':>6}{'mean shaved %':>15}")
    for lo, hi in [(1.0,1.1),(1.1,1.25),(1.25,1.5),(1.5,99)]:
        k = (ww>=lo)&(ww<hi)
        if k.sum(): print(f"  {f'{lo}-{hi}':<16}{k.sum():>6}{100*s[k].mean():>15.2f}")

json.dump(A, open(ROOT / "eval/runs/det_line_slope.json", "w"), indent=1)
print(f"\n-> eval/runs/det_line_slope.json  ({len(A)} lines)")
