#!/usr/bin/env python3
"""E4.2 — does POLYGON-MASKED cropping beat plain rectangular cropping? (2026-08-22)

Founder's observation from the dataset's annotation viewer: the hand-drawn word polygons INTERLOCK. A descender from
the line above dives down into the row below, and the polygon keeps that stroke owned by the line above. An
axis-aligned rectangle around a line cannot do that — it swallows the neighbour's ink.

Measured support: our axis-aligned line box is 1.6x the height of a typical word even on perfectly flat lines, so a
line crop is mostly NOT this line.

This isolates that effect, using the SAME rectangle in both arms so nothing else changes:
  A "plain"  : crop the line's bounding rectangle (what we do today)
  B "masked" : identical rectangle, but every pixel outside THIS line's own word polygons is filled with paper tone
Both are read by the production reader with the identical recipe (pad 12, resize h=128). Fully paired.

If B wins, the fix is polygon-aware cropping, and it is worth a LoRA round on masked strips (~$9) to remove the
train/test mismatch (the reader was trained on unmasked strips, so this measures the ceiling pessimistically).

    eval/.venv/bin/python eval/detectors/polygon_crop.py --n 260
"""
import argparse, json, random, statistics, sys, time, unicodedata
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--ann", default=str(ROOT / "data/raw/school_notebooks_RU/exam/annotations_test.json"))
ap.add_argument("--images", default=str(ROOT / "eval/testset_v2/ru_pages/images"))
ap.add_argument("--base", default=str(ROOT / "models/GLM-OCR"))
ap.add_argument("--adapter", default=str(ROOT / "models/adapters/glm-ocr-lora-ru-r5c"))
ap.add_argument("--out", default=str(ROOT / "eval/runs/det_polygon_crop.json"))
ap.add_argument("--n", type=int, default=260)
ap.add_argument("--dilate", type=int, default=6, help="grow each polygon by this many px so we never shave our own ink")
ap.add_argument("--batch-size", type=int, default=16)
a = ap.parse_args()

sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw
from bilimai.reader import make_reader


def cer(ref, hyp):
    ref, hyp = unicodedata.normalize("NFC", ref), unicodedata.normalize("NFC", hyp)
    d = list(range(len(hyp) + 1))
    for i, cr in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, ch in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (cr != ch))
    return d[-1] / max(1, len(ref))


gt = json.load(open(a.gt, encoding="utf-8"))
ann = json.load(open(a.ann, encoding="utf-8"))
imgs = {i["id"]: i["file_name"] for i in ann["images"]}

# word polygons per (file, group_id) — group_id is exactly how our GT lines were built
polys = {}
for x in ann["annotations"]:
    fn = imgs.get(x["image_id"])
    if fn not in gt or x["category_id"] != 0: continue
    seg = x.get("segmentation") or []
    if not seg or not isinstance(seg[0], list): continue
    pts = [(seg[0][i], seg[0][i + 1]) for i in range(0, len(seg[0]) - 1, 2)]
    polys.setdefault((fn, str(x.get("group_id"))), []).append(pts)

cands = []
for fn, page in gt.items():
    for l in page["lines"]:
        t = (l.get("text") or "").strip()
        gid = str(l.get("group_id"))
        if t and (fn, gid) in polys and len(t) >= 6:
            cands.append((fn, gid, t, l["bbox"]))
random.Random(1).shuffle(cands)
sample = cands[: a.n]
print(f"lines with polygons: {len(cands)} | sampling {len(sample)} -> {2*len(sample)} reads", flush=True)

PAD = 12
crops, meta, cache = [], [], {}
for fn, gid, txt, b in sample:
    if fn not in cache: cache[fn] = Image.open(Path(a.images) / fn).convert("RGB")
    im = cache[fn]; W, H = im.size
    x0, y0 = max(0, int(b[0] - PAD)), max(0, int(b[1] - PAD))
    x1, y1 = min(W, int(b[2] + PAD)), min(H, int(b[3] + PAD))
    plain = im.crop((x0, y0, x1, y1))

    # paper tone = per-channel median of the crop (dominated by paper, not ink)
    arr = np.asarray(plain)
    paper = tuple(int(v) for v in np.median(arr.reshape(-1, 3), axis=0))

    mask = Image.new("L", plain.size, 0)
    md = ImageDraw.Draw(mask)
    for pg in polys[(fn, gid)]:
        pts = [(px - x0, py - y0) for px, py in pg]
        if len(pts) >= 3:
            md.polygon(pts, fill=255)
            if a.dilate:                       # widen the outline so we never clip our own strokes
                md.line(pts + [pts[0]], fill=255, width=2 * a.dilate)
    masked = Image.composite(plain, Image.new("RGB", plain.size, paper), mask)

    crops += [plain, masked]; meta += [("plain", txt), ("masked", txt)]

reader = make_reader(a.base, a.adapter, line_h=128, max_new_tokens=96)
print(f"reader {reader.name} | reading {len(crops)} crops...", flush=True)
t0 = time.time()
texts, _ = reader.read(crops, batch_size=a.batch_size, with_conf=False)
print(f"read in {time.time()-t0:.0f}s", flush=True)

rows = {}
for (kind, txt), hyp in zip(meta, texts):
    rows.setdefault(txt, {})[kind] = (hyp or "").strip()
recs = [{"ref": t, "n": len(t), "plain": cer(t, v["plain"]), "masked": cer(t, v["masked"])}
        for t, v in rows.items() if len(v) == 2]

def cw(k):
    ed = sum(r[k] * r["n"] for r in recs); ch = sum(r["n"] for r in recs)
    return ed / max(ch, 1)

p, m = cw("plain"), cw("masked")
better = sum(1 for r in recs if r["masked"] < r["plain"] - 1e-9)
worse = sum(1 for r in recs if r["masked"] > r["plain"] + 1e-9)
print(f"\nn paired lines: {len(recs)}")
print(f"  PLAIN  rectangle crop : {p:.5f}")
print(f"  MASKED polygon crop   : {m:.5f}")
print(f"  masked - plain        : {m-p:+.5f}  -> {'BETTER' if m < p else 'WORSE' if m > p else 'no change'}")
print(f"  per line: masked better {better} | masked worse {worse} | identical {len(recs)-better-worse}")
try:
    from scipy.stats import binomtest
    if better + worse:
        print(f"  McNemar exact p = {binomtest(better, better+worse, 0.5).pvalue:.3f}")
except Exception:
    pass
json.dump({"n": len(recs), "plain": p, "masked": m, "better": better, "worse": worse},
          open(a.out, "w"), indent=1)
print(f"\n→ {a.out}")
