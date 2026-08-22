#!/usr/bin/env python3
"""E4.2 — re-group detector WORD polygons into LINE boxes (2026-08-22).

The first grouping in bench_detectors.py chained boxes against a drifting centroid and collapsed 11,416 docTR words
into 145 lines. This replaces it with 1-D gap clustering on y-centres: sort word centres, start a new line wherever
the gap to the previous centre exceeds `gap` x the median word height. Works off the saved *.poly.json, so no
detector re-run is needed.

    eval/.venv/bin/python eval/detectors/regroup_lines.py --dets doctr,easyocr,paddle
"""
import argparse, json, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--dets", default="doctr,easyocr,paddle")
ap.add_argument("--runs", default=str(ROOT / "eval/runs"))
ap.add_argument("--gap", type=float, default=0.6, help="new line when centre gap > gap * median word height")
a = ap.parse_args()
gtd = json.load(open(ROOT / "eval/testset_v2/ru_pages/ground_truth.json", encoding="utf-8"))

def bbox(pg):
    xs=[p[0] for p in pg]; ys=[p[1] for p in pg]
    return [float(min(xs)),float(min(ys)),float(max(xs)),float(max(ys))]

for det in [d.strip() for d in a.dets.split(",") if d.strip()]:
    pp = Path(a.runs)/f"det_{det}.json.poly.json"
    if not pp.is_file(): print(f"[{det}] no polygons on disk — skip"); continue
    polys = json.load(open(pp)); out = {}
    for fn, ps in polys.items():
        bs = [bbox(p) for p in ps if len(p) >= 3]
        if not bs: out[fn] = []; continue
        # 2026-08-22: these pages are two-page SPREADS (e.g. 4000x3000). Clustering y across the whole spread merges
        # the left page's line N with the right page's line N. Split by column first, exactly as RPDetector does.
        W = gtd[fn]["width"] if fn in gtd else max(b[2] for b in bs)
        H = gtd[fn]["height"] if fn in gtd else max(b[3] for b in bs)
        cols = [bs] if (W / max(H, 1) <= 1.15) else [[b for b in bs if (b[0]+b[2])/2 <= W/2],
                                                     [b for b in bs if (b[0]+b[2])/2 >  W/2]]
        lines = []
        for cb in cols:
            if not cb: continue
            med = statistics.median([b[3]-b[1] for b in cb]) or 1.0
            cb.sort(key=lambda b: (b[1]+b[3])/2)
            cur, prev = [cb[0]], (cb[0][1]+cb[0][3])/2
            for b in cb[1:]:
                c = (b[1]+b[3])/2
                if c - prev > a.gap*med: lines.append(cur); cur = []
                cur.append(b); prev = c
            lines.append(cur)
        out[fn] = [[min(x[0] for x in g), min(x[1] for x in g), max(x[2] for x in g), max(x[3] for x in g), 1.0]
                   for g in lines if g]
    op = Path(a.runs)/f"det_{det}.json"
    json.dump(out, open(op,"w"), indent=None)
    n = sum(len(v) for v in out.values())
    print(f"[{det}] {sum(len(v) for v in polys.values())} words -> {n} lines ({n/max(len(out),1):.0f}/page)  -> {op}")
