#!/usr/bin/env python3
"""E4.2 — does polygon masking help when the RECTANGLE IS A DETECTOR BOX? (2026-08-22)

polygon_crop.py masked inside GROUND-TRUTH line boxes and found nothing (p=1.000, 243/294 lines identical). But GT
boxes are already tight. A detector box is looser, so it swallows MORE of the neighbouring lines' ink — masking a bad
box is a different question from masking a good one, and that is what this measures.

For each GT line we take the detector's best-matching box and read it twice:
  A "plain"  : the detector box, cropped as production does (pad 12, resize h=128)
  B "masked" : identical rectangle, pixels outside the DETECTOR'S OWN polygons filled with paper tone
Reference text is the GT line. Fully paired; the only difference between arms is the mask.

Kraken emits one polygon per line; docTR / EasyOCR / PaddleOCR emit word quads that this script assigns to a box by
centre containment. If a detector produced no usable polygons its arm is skipped rather than faked.

    eval/.venv/bin/python eval/detectors/mask_detector_crops.py --dets kraken,doctr,easyocr,paddle --n 200
"""
import argparse, json, random, sys, time, unicodedata
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--images", default=str(ROOT / "eval/testset_v2/ru_pages/images"))
ap.add_argument("--dets", default="kraken,doctr,easyocr,paddle")
ap.add_argument("--runs", default=str(ROOT / "eval/runs"))
ap.add_argument("--base", default=str(ROOT / "models/GLM-OCR"))
ap.add_argument("--adapter", default=str(ROOT / "models/adapters/glm-ocr-lora-ru-r5c"))
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--dilate", type=int, default=6)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--out", default=str(ROOT / "eval/runs/det_mask_ab.json"))
ap.add_argument("--iou-lo", type=float, default=0.5, help="only lines whose detector box overlaps GT in [lo,hi)")
ap.add_argument("--iou-hi", type=float, default=1.01)
a = ap.parse_args()

sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw
from bilimai.reader import make_reader


def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return i / max(1e-9, (p[2] - p[0]) * (p[3] - p[1]) + (g[2] - g[0]) * (g[3] - g[1]) - i)


def cer(ref, hyp):
    ref, hyp = unicodedata.normalize("NFC", ref), unicodedata.normalize("NFC", hyp)
    d = list(range(len(hyp) + 1))
    for i, cr in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, ch in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (cr != ch))
    return d[-1] / max(1, len(ref))


gt = json.load(open(a.gt, encoding="utf-8"))
reader = None
results = {}

for det in [d.strip() for d in a.dets.split(",") if d.strip()]:
    bp = Path(a.runs) / f"det_{det}.json"; pp = Path(str(bp) + ".poly.json")
    if not bp.is_file() or not pp.is_file():
        print(f"[{det}] SKIP — no boxes/polygons on disk"); continue
    boxes = json.load(open(bp)); polys = json.load(open(pp))

    cands = []
    for fn, page in gt.items():
        P = [b[:4] for b in boxes.get(fn, [])]
        PG = polys.get(fn, [])
        if not P or not PG: continue
        for l in page["lines"]:
            t = (l.get("text") or "").strip()
            if len(t) < 6: continue
            best, bi = max(((iou(p, l["bbox"]), i) for i, p in enumerate(P)), default=(0.0, -1))
            if not (a.iou_lo <= best < a.iou_hi): continue
            b = P[bi]
            inside = [pg for pg in PG
                      if b[0] <= sum(x for x, _ in pg) / len(pg) <= b[2] and b[1] <= sum(y for _, y in pg) / len(pg) <= b[3]]
            if inside: cands.append((fn, t, b, inside))
    if not cands:
        print(f"[{det}] SKIP — no line matched a box with polygons inside"); continue
    random.Random(1).shuffle(cands); sample = cands[: a.n]
    print(f"\n[{det}] usable lines {len(cands)} | sampling {len(sample)} -> {2*len(sample)} reads", flush=True)

    PAD = 12; crops, meta, cache = [], [], {}
    for fn, t, b, inside in sample:
        if fn not in cache: cache[fn] = Image.open(Path(a.images) / fn).convert("RGB")
        im = cache[fn]; W, H = im.size
        x0, y0 = max(0, int(b[0] - PAD)), max(0, int(b[1] - PAD))
        x1, y1 = min(W, int(b[2] + PAD)), min(H, int(b[3] + PAD))
        if x1 - x0 < 8 or y1 - y0 < 8: continue
        plain = im.crop((x0, y0, x1, y1))
        arr = np.asarray(plain); paper = tuple(int(v) for v in np.median(arr.reshape(-1, 3), axis=0))
        mask = Image.new("L", plain.size, 0); md = ImageDraw.Draw(mask)
        for pg in inside:
            pts = [(px - x0, py - y0) for px, py in pg]
            if len(pts) >= 3:
                md.polygon(pts, fill=255)
                if a.dilate: md.line(pts + [pts[0]], fill=255, width=2 * a.dilate)
        masked = Image.composite(plain, Image.new("RGB", plain.size, paper), mask)
        crops += [plain, masked]; meta += [("plain", t), ("masked", t)]

    if reader is None:
        reader = make_reader(a.base, a.adapter, line_h=128, max_new_tokens=96)
        print(f"reader {reader.name}", flush=True)
    t0 = time.time()
    texts, _ = reader.read(crops, batch_size=a.batch_size, with_conf=False)
    print(f"  read {len(crops)} crops in {time.time()-t0:.0f}s", flush=True)

    rows = {}
    for (k, t), hyp in zip(meta, texts): rows.setdefault(t, {})[k] = (hyp or "").strip()
    recs = [{"n": len(t), "plain": cer(t, v["plain"]), "masked": cer(t, v["masked"])} for t, v in rows.items() if len(v) == 2]
    ch = sum(r["n"] for r in recs) or 1
    p_ = sum(r["plain"] * r["n"] for r in recs) / ch
    m_ = sum(r["masked"] * r["n"] for r in recs) / ch
    bt = sum(1 for r in recs if r["masked"] < r["plain"] - 1e-9)
    ws = sum(1 for r in recs if r["masked"] > r["plain"] + 1e-9)
    pv = None
    try:
        from scipy.stats import binomtest
        if bt + ws: pv = binomtest(bt, bt + ws, 0.5).pvalue
    except Exception: pass
    print(f"  n={len(recs)}  plain {p_:.5f} | masked {m_:.5f} | diff {m_-p_:+.5f} "
          f"({'BETTER' if m_ < p_ else 'WORSE' if m_ > p_ else 'tie'})  better {bt} worse {ws} "
          f"p={'%.3f' % pv if pv is not None else 'n/a'}")
    results[det] = {"n": len(recs), "plain": p_, "masked": m_, "diff": m_ - p_, "better": bt, "worse": ws, "p": pv}

print("\n=== summary ===")
print(f"{'detector':<10}{'n':>5}{'plain':>10}{'masked':>10}{'diff':>10}{'p':>8}")
for k, v in results.items():
    print(f"{k:<10}{v['n']:>5}{v['plain']:>10.5f}{v['masked']:>10.5f}{v['diff']:>+10.5f}{(('%.3f' % v['p']) if v['p'] is not None else 'n/a'):>8}")
json.dump(results, open(a.out, "w"), indent=1)
print(f"\n→ {a.out}")
