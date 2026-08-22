#!/usr/bin/env python3
"""E4.2 — page-relative line growth (2026-08-22).

Problem measured today: line boxes whose overlap with the human box is 0.50–0.70 capture only **0.714** of the true
line height (83 % of them are too small, clipped worse at the bottom), and they are short lines — median 2 words vs 5.
The current rule grows each box by a fraction of ITS OWN height (LINE_GROW = top .25 / bottom .365 / side .02), so a
box that is already 30 % too short receives a correspondingly small correction and never catches up.

Fix under test: grow against a REFERENCE height that cannot collapse — `ref = max(own_h, k * page_median_line_h)`.
k=0 reproduces production exactly (sanity check). Larger k pulls squashed boxes up toward their neighbours' size while
leaving already-normal boxes untouched.

Runs entirely off the saved raw dump (ungrown word boxes + groupings), so no network pass is needed.

    eval/.venv/bin/python eval/detectors/grow_pagerel.py
"""
import argparse, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=str(ROOT / "eval/runs/rp_det_lines_v2g.raw.json"))
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--ks", default="0,0.5,0.7,0.8,0.9,1.0")
ap.add_argument("--dump", default="", help="write the boxes for this k to eval/runs/rp_det_lines_pagerel.json")
ap.add_argument("--dump-k", type=float, default=0.8)
a = ap.parse_args()

LINE_GROW = (0.25, 0.365, 0.02)     # top, bottom, side — must match bilimai.detector
raw = json.load(open(a.raw, encoding="utf-8"))
gt = json.load(open(a.gt, encoding="utf-8"))


def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return i / max(1e-9, (p[2] - p[0]) * (p[3] - p[1]) + (g[2] - g[0]) * (g[3] - g[1]) - i)


def build(page_raw, W, H, k):
    """Union member word boxes per group, then grow against max(own_h, k * page median line height)."""
    words, groups = page_raw["words"], page_raw["groups"]
    base = []
    for g in groups:
        if not g: continue
        xs0 = min(words[i][0] for i in g); ys0 = min(words[i][1] for i in g)
        xs1 = max(words[i][2] for i in g); ys1 = max(words[i][3] for i in g)
        base.append([xs0, ys0, xs1, ys1])
    if not base: return []
    med_h = statistics.median([b[3] - b[1] for b in base])
    t, bt, sx = LINE_GROW
    out = []
    for b in base:
        h = b[3] - b[1]; w = b[2] - b[0]
        ref = max(h, k * med_h)                       # k=0 → ref == h → identical to production
        out.append([max(0.0, b[0] - sx * w), max(0.0, b[1] - t * ref),
                    min(float(W), b[2] + sx * w), min(float(H), b[3] + bt * ref)])
    return out


print(f"{'k':>5} {'matched':>9} {'missed':>8} {'merged*':>8} {'h_ratio':>9} {'h_ratio bad-band':>17} {'boxes':>7}")
print("-" * 70)
for k in [float(x) for x in a.ks.split(",")]:
    n_match = n_miss = n_boxes = 0; hr_all = []; hr_bad = []; n_gt = 0; over = 0
    for fn, page in gt.items():
        if fn not in raw: continue
        W, H = float(page["width"]), float(page["height"])
        P = build(raw[fn], W, H, k); n_boxes += len(P)
        G = [l["bbox"] for l in page["lines"] if (l.get("text") or "").strip()]
        n_gt += len(G)
        for g in G:
            best, bi = max(((iou(p, g), i) for i, p in enumerate(P)), default=(0.0, -1))
            if best >= 0.5:
                n_match += 1
                hr = (P[bi][3] - P[bi][1]) / (g[3] - g[1]); hr_all.append(hr)
                if best < 0.7: hr_bad.append(hr)
            elif best < 0.1:
                n_miss += 1
        # crude over-grow signal: a predicted box covering >=40% of two different GT lines
        for p in P:
            if sum(1 for g in G if iou(p, g) > 0 and (min(p[3], g[3]) - max(p[1], g[1])) > 0.4 * (g[3] - g[1])) >= 2:
                over += 1
    hb = statistics.median(hr_bad) if hr_bad else float("nan")
    print(f"{k:>5.2f} {n_match:>6} ({100*n_match/n_gt:.0f}%) {n_miss:>8} {over:>8} {statistics.median(hr_all):>9.3f} {hb:>17.3f} {n_boxes:>7}")
print("\n* merged = predicted box vertically covering >=40% of two GT lines (over-growth warning signal)")

if a.dump:
    out = {}
    for fn, page in gt.items():
        if fn not in raw: continue
        out[fn] = [b + [1.0] for b in build(raw[fn], float(page["width"]), float(page["height"]), a.dump_k)]
    json.dump(out, open(a.dump, "w"), indent=None)
    print(f"→ {a.dump} (k={a.dump_k})")
