#!/usr/bin/env python3
"""E4.2 — profile the lines the production detector NEVER finds (2026-08-22).

det_failure.py says WHAT happens to each GT line; this says WHAT THOSE LINES ARE LIKE. On exam v2 with the production
config (RP + val-derived growth) the matched boxes are geometrically near-exact (h_ratio 1.02, w_ratio 0.99) — so the
whole remaining gap is the ~13 % of lines that get no prediction at all. Fixing detection means understanding those.

For every GT line we recompute the miss/match verdict (best IoU < 0.1 = missed, matching det_failure.py) and attach
descriptive features, then contrast missed vs matched on each feature so the over-represented ones stand out.

    eval/.venv/bin/python eval/detectors/miss_profile.py
    ... --pred eval/runs/rp_det_lines_v2g.json --gt eval/testset_v2/ru_pages/ground_truth.json

Writes eval/runs/det_miss_profile.json and prints the contrast table.
"""
import argparse, json, collections, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--pred", default=str(ROOT / "eval/runs/rp_det_lines_v2g.json"))
ap.add_argument("--out", default=str(ROOT / "eval/runs/det_miss_profile.json"))
ap.add_argument("--examples", type=int, default=12)
a = ap.parse_args()


def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / max(1e-9, (p[2] - p[0]) * (p[3] - p[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter)


gt = json.load(open(a.gt, encoding="utf-8"))
pred = json.load(open(a.pred, encoding="utf-8"))

rows = []
for fn, page in gt.items():
    W, H = float(page["width"]), float(page["height"])
    P = [p[:4] for p in pred.get(fn, [])]
    lines = page["lines"]
    heights = [l["bbox"][3] - l["bbox"][1] for l in lines] or [1.0]
    med_h = statistics.median(heights)
    widths = [l["bbox"][2] - l["bbox"][0] for l in lines] or [1.0]
    med_w = statistics.median(widths)
    for l in lines:
        b = l["bbox"]; h = b[3] - b[1]; w = b[2] - b[0]
        best = max((iou(p, b) for p in P), default=0.0)
        txt = (l.get("text") or "").strip()
        nwords = len(l.get("words") or []) or len(txt.split())
        rows.append({
            "file": fn, "text": txt, "bbox": b, "best_iou": round(best, 3),
            "missed": best < 0.1,
            "n_words": nwords,
            "n_chars": len(txt),
            "h": h, "w": w,
            "h_rel": h / max(med_h, 1e-9),          # vs this page's median line height
            "w_rel": w / max(med_w, 1e-9),
            "y_rel": ((b[1] + b[3]) / 2) / max(H, 1e-9),   # 0 = top of page, 1 = bottom
            "x_rel": ((b[0] + b[2]) / 2) / max(W, 1e-9),
            "aspect": w / max(h, 1e-9),
            "near_top": (b[1] / max(H, 1e-9)) < 0.06,
            "near_bottom": (b[3] / max(H, 1e-9)) > 0.94,
            "near_left": (b[0] / max(W, 1e-9)) < 0.04,
            "near_right": (b[2] / max(W, 1e-9)) > 0.96,
            "column": l.get("column"),
        })

miss = [r for r in rows if r["missed"]]
match = [r for r in rows if not r["missed"]]
print(f"GT lines {len(rows)} | missed {len(miss)} ({100*len(miss)/max(len(rows),1):.1f} %) | matched {len(match)}")
print()

def med(rs, k): return statistics.median([r[k] for r in rs]) if rs else float("nan")
def share(rs, pred_fn): return 100 * sum(1 for r in rs if pred_fn(r)) / max(len(rs), 1)

print(f"{'feature':<26} {'missed':>10} {'matched':>10}   {'lift':>6}")
print("-" * 58)
for label, key in (("median words/line", "n_words"), ("median chars/line", "n_chars"),
                   ("median height (px)", "h"), ("median height ÷ page median", "h_rel"),
                   ("median width ÷ page median", "w_rel"), ("median aspect (w/h)", "aspect"),
                   ("median y position (0=top)", "y_rel")):
    m, t = med(miss, key), med(match, key)
    lift = (m / t) if t else float("nan")
    print(f"{label:<26} {m:>10.2f} {t:>10.2f}   {lift:>5.2f}x")

print()
print(f"{'bucket (share of group)':<26} {'missed':>10} {'matched':>10}   {'lift':>6}")
print("-" * 58)
BUCKETS = [
    ("1-word lines",          lambda r: r["n_words"] <= 1),
    ("≤2-word lines",         lambda r: r["n_words"] <= 2),
    ("≤5 chars",              lambda r: r["n_chars"] <= 5),
    ("short: w < 30% of page",lambda r: r["w_rel"] < 0.3),
    ("small: h < 70% median", lambda r: r["h_rel"] < 0.7),
    ("tall: h > 150% median", lambda r: r["h_rel"] > 1.5),
    ("top 6 % of page",       lambda r: r["near_top"]),
    ("bottom 6 % of page",    lambda r: r["near_bottom"]),
    ("left margin",           lambda r: r["near_left"]),
    ("right margin",          lambda r: r["near_right"]),
    ("empty/blank text",      lambda r: r["n_chars"] == 0),
]
buckets = {}
for label, fn in BUCKETS:
    m, t = share(miss, fn), share(match, fn)
    buckets[label] = {"missed_pct": round(m, 1), "matched_pct": round(t, 1),
                      "lift": round(m / t, 2) if t else None}
    lift = f"{m/t:>5.2f}x" if t else "   inf"
    print(f"{label:<26} {m:>9.1f}% {t:>9.1f}%   {lift}")

# how much of the miss mass do the top buckets actually explain?
print()
covered = [r for r in miss if r["n_words"] <= 2 or r["w_rel"] < 0.3]
print(f"share of misses that are ≤2 words OR <30 % page width: {100*len(covered)/max(len(miss),1):.1f} %")
by_page = collections.Counter(r["file"] for r in miss)
worst = by_page.most_common(5)
print(f"misses concentrated? top-5 pages hold {100*sum(n for _,n in worst)/max(len(miss),1):.1f} % of all misses: {worst}")

print()
print("examples of missed lines (text | words | h÷med | w÷med | y):")
for r in sorted(miss, key=lambda r: -r["n_chars"])[:a.examples // 2]:
    print(f"  LONG  «{r['text'][:44]:<44}» {r['n_words']:>2}w  h{r['h_rel']:.2f} w{r['w_rel']:.2f} y{r['y_rel']:.2f}")
for r in sorted(miss, key=lambda r: r["n_chars"])[:a.examples // 2]:
    print(f"  SHORT «{r['text'][:44]:<44}» {r['n_words']:>2}w  h{r['h_rel']:.2f} w{r['w_rel']:.2f} y{r['y_rel']:.2f}")

json.dump({"n_gt": len(rows), "n_missed": len(miss), "buckets": buckets,
           "missed": [{k: v for k, v in r.items() if k != "bbox"} for r in miss]},
          open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n→ {a.out}")
