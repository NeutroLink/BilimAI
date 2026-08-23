#!/usr/bin/env python
"""Score Qwen's own line boxes against ai-forever's human boxes, with the SAME growth-fitting the
production detector gets. 2026-08-22.

Qwen zero-shot scored 80 % matched at median IoU 0.676 against RP's 84 % / 0.810. RP only reaches
0.810 because its raw boxes are corrected by val-fitted growth factors (top 0.25 / bottom 0.365 /
sides 0.02). Qwen's boxes have never had that treatment, so the comparison is not like-for-like —
exactly the mistake the Kraken benchmark made (plans/CROP-GEOMETRY-2026-08-22.md §5).

Growth is fitted here on a HELD-OUT half of the pages and applied to the other half, so the number
reported is not the number we tuned on.

  eval/.venv/bin/python eval/detectors/score_qwen_boxes.py eval/runs/qwen_sweep_*.json
"""
import json, sys, unicodedata
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
gt = json.load(open(ROOT / "eval/testset_v2/ru_pages/ground_truth.json", encoding="utf-8"))


def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    i = max(0., x1 - x0) * max(0., y1 - y0)
    return i / max(1e-9, (p[2]-p[0])*(p[3]-p[1]) + (g[2]-g[0])*(g[3]-g[1]) - i)


def cer(ref, hyp):
    ref, hyp = unicodedata.normalize("NFC", ref), unicodedata.normalize("NFC", hyp)
    d = list(range(len(hyp) + 1))
    for i, cr in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, ch in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (cr != ch))
    return d[-1] / max(1, len(ref))


def to_page(rec):
    """Qwen emits 0-1000 normalised coords inside the region we sent (measured: 80 % vs 4 % for
    absolute pixels, so this is not a guess)."""
    sw, sh = rec["sub_size"]; ox, oy = rec["sub_origin"]
    out = []
    for it in rec["items"]:
        b = it["bbox"]
        q = [b[0]/1000*sw + ox, b[1]/1000*sh + oy, b[2]/1000*sw + ox, b[3]/1000*sh + oy]
        if q[2] > q[0] and q[3] > q[1]:
            out.append((q, it["text"]))
    return out


def grow(b, g, W=1e9, H=1e9):
    t, bt, x = g; h = b[3] - b[1]; w = b[2] - b[0]
    return [max(0., b[0] - x*w), max(0., b[1] - t*h), min(W, b[2] + x*w), min(H, b[3] + bt*h)]


def fit_growth(recs):
    """Median signed edge gap as a fraction of the PREDICTED box's own size — identical procedure to
    eval/detectors/derive_growth.py, against word-union GT (the convention that reproduces RP's
    shipped constants; text_line polygons give zero growth and are wrong)."""
    gaps = {"t": [], "b": [], "l": [], "r": []}
    for r in recs:
        P = [q for q, _ in to_page(r)]
        if not P: continue
        for L in gt[r["file"]]["lines"]:
            g = L["bbox"]
            ox, oy = r["sub_origin"]; sw, _ = r["sub_size"]
            if not (ox <= (g[0]+g[2])/2 <= ox+sw): continue
            best = max(P, key=lambda p: iou(p, g))
            if iou(best, g) < 0.2: continue
            h = max(1., best[3]-best[1]); w = max(1., best[2]-best[0])
            gaps["t"].append((best[1]-g[1])/h); gaps["b"].append((g[3]-best[3])/h)
            gaps["l"].append((best[0]-g[0])/w); gaps["r"].append((g[2]-best[2])/w)
    if not gaps["t"]: return (0., 0., 0.), 0
    m = {k: float(np.median(v)) for k, v in gaps.items()}
    return (max(0., round(m["t"], 3)), max(0., round(m["b"], 3)),
            max(0., round((m["l"]+m["r"])/2, 3))), len(gaps["t"])


def evaluate(recs, g):
    matched = total = 0; ious = []; cers = []; ch = 0
    for r in recs:
        pairs = to_page(r)
        if not pairs: continue
        P = [grow(q, g) for q, _ in pairs]; T = [t for _, t in pairs]
        ox, oy = r["sub_origin"]; sw, _ = r["sub_size"]
        for L in gt[r["file"]]["lines"]:
            gb = L["bbox"]; txt = (L.get("text") or "").strip()
            if not (ox <= (gb[0]+gb[2])/2 <= ox+sw): continue
            total += 1
            j = int(np.argmax([iou(p, gb) for p in P])); best = iou(P[j], gb)
            ious.append(best)
            if best >= 0.5:
                matched += 1
                if txt: cers.append((cer(txt, T[j].strip()), len(txt))); ch += len(txt)
    cw = sum(c*n for c, n in cers)/max(ch, 1) if cers else float("nan")
    return matched, total, (np.median(ious) if ious else 0.0), cw


print(f"{'run':<34}{'growth (t/b/x)':>22}{'matched':>16}{'med IoU':>9}{'text CER':>10}")
for path in sys.argv[1:]:
    recs = json.load(open(path, encoding="utf-8"))
    files = sorted({r["file"] for r in recs})
    half = set(files[: max(1, len(files)//2)])                     # fit here
    fit = [r for r in recs if r["file"] in half]
    hold = [r for r in recs if r["file"] not in half] or recs      # report here
    g, n = fit_growth(fit)
    for label, gg in (("raw", (0., 0., 0.)), (f"fitted n={n}", g)):
        m, t, mi, cw = evaluate(hold, gg)
        print(f"{Path(path).stem:<34}{str(gg):>22}{f'{m}/{t} ({100*m/max(t,1):.0f} %)':>16}{mi:>9.3f}{cw:>10.3f}"
              + ("" if label == "raw" else "   <- growth fitted on held-out pages"))
print(f"\n{'RP production (reference)':<34}{'(0.25,0.365,0.02)':>22}{'84 %':>16}{0.810:>9.3f}{0.066:>10.3f}")
