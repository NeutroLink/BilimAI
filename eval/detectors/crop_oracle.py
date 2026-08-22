#!/usr/bin/env python3
"""E4.3 — what does correcting the detector's VERTICAL extent recover? (2026-08-22)

crop_cost.py measured that swapping the detector box for the human box is worth +0.0074 CER.
This script splits that gap into its vertical and horizontal components.

The detector shaves 4.98 % of line ink vertically (65 % below = descenders) but only 2.09 %
horizontally. This asks: if the detector's y-range (top/bottom) were perfect but x-range stayed
wrong, how much closer to human accuracy would we get?

Three arms, same lines, paired, read by the same reader in ONE batch:
  A "det"     : the production detector box           (x0,y0,x1,y1 from rp_det_lines_v2g.json)
  B "detvert" : detector x-range, GT y-range         ([det_x0, gt_y0, det_x1, gt_y1])
  C "human"   : the full GT line box                 (gt bbox)

Per-line inclusion: best-IoU detector box >= --iou (default 0.5), len(text.strip()) >= 6.
Sample --n (default 400) with random.Random(1).

    eval/.venv/bin/python eval/detectors/crop_oracle.py --n 400 --dry-run

Writes eval/runs/det_crop_oracle.json. Keeps all per-line records, not a sample.
"""
import argparse, json, random, statistics, sys, time, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--images", default=str(ROOT / "eval/testset_v2/ru_pages/images"))
ap.add_argument("--pred", default=str(ROOT / "eval/runs/rp_det_lines_v2g.json"))
ap.add_argument("--base", default=str(ROOT / "models/GLM-OCR"))
ap.add_argument("--adapter", default=str(ROOT / "models/adapters/glm-ocr-lora-ru-r5c"))
ap.add_argument("--out", default=str(ROOT / "eval/runs/det_crop_oracle.json"))
ap.add_argument("--n", type=int, default=400, help="matched lines to sample (paired: 3 reads each)")
ap.add_argument("--iou", type=float, default=0.5, help="min best-IoU to count a line as matched")
ap.add_argument("--iou-lo", type=float, default=0.0, help="min IoU for filtering")
ap.add_argument("--iou-hi", type=float, default=1.01, help="max IoU for filtering")
ap.add_argument("--batch-size", type=int, default=24)   # measured peak on M1 Pro MPS: 16-32 flat, 48+ worse
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--dry-run", action="store_true", help="build and print stats, do not load reader")
a = ap.parse_args()

sys.path.insert(0, str(ROOT))
from PIL import Image


def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / max(1e-9, (p[2] - p[0]) * (p[3] - p[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter)


def cer(ref, hyp):
    ref, hyp = unicodedata.normalize("NFC", ref), unicodedata.normalize("NFC", hyp)
    d = list(range(len(hyp) + 1))
    for i, cr in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, ch in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (cr != ch))
    return d[-1] / max(1, len(ref))


def vertical_ink_lost(det_box, gt_words):
    """Fraction of GT word-box area lost vertically by detector box (y-axis only)."""
    if not gt_words:
        return 0.0
    # Bounding box of all words
    x0 = min(w["bbox"][0] for w in gt_words)
    y0 = min(w["bbox"][1] for w in gt_words)
    x1 = max(w["bbox"][2] for w in gt_words)
    y1 = max(w["bbox"][3] for w in gt_words)
    word_area = (x1 - x0) * (y1 - y0)

    # Intersection: x-range is full word extent, y-range is detector intersection
    inter_y0 = max(y0, det_box[1])
    inter_y1 = min(y1, det_box[3])
    inter_area = (x1 - x0) * max(0.0, inter_y1 - inter_y0)

    lost = word_area - inter_area
    return lost / max(word_area, 1e-9)


gt = json.load(open(a.gt, encoding="utf-8"))
pred = json.load(open(a.pred, encoding="utf-8"))

candidates = []  # (file, text, gt_box, rp_box, words, iou)
for fn, page in gt.items():
    P = [p[:4] for p in pred.get(fn, [])]
    if not P:
        continue
    for l in page["lines"]:
        g = l["bbox"]
        txt = (l.get("text") or "").strip()
        if not txt or len(txt) < 6:
            continue
        best, bi = max(((iou(p, g), i) for i, p in enumerate(P)), default=(0.0, -1))
        if best >= a.iou and a.iou_lo <= best < a.iou_hi:
            candidates.append((fn, txt, g, P[bi], l.get("words", []), best))

random.Random(a.seed).shuffle(candidates)
sample = candidates[:a.n]

print(f"candidate lines: {len(candidates)} | sampling {len(sample)} (paired → {4*len(sample)} reads)", flush=True)

# Build crops for all three arms
from bilimai.inkcrop import grow_box_to_ink
import cv2
import numpy as np

PAD = 12
imgs, meta, ckeys = [], [], []
cache = {}
gray_cache = {}


def _crop(im, b, W, H, fn):
    """Crop with the production recipe, but round to ints FIRST so the cache key and the pixels agree.
    PIL rounds internally anyway; doing it here makes two arms that land on the same rectangle provably
    the same image, which is what lets us skip the duplicate read."""
    x0, y0 = int(round(max(0, b[0] - PAD))), int(round(max(0, b[1] - PAD)))
    x1, y1 = int(round(min(W, b[2] + PAD))), int(round(min(H, b[3] + PAD)))
    return (fn, x0, y0, x1, y1), im.crop((x0, y0, x1, y1))


for fn, txt, gt_box, det_box, words, iou_val in sample:
    if fn not in cache:
        cache[fn] = Image.open(Path(a.images) / fn).convert("RGB")
    im = cache[fn]
    W, H = im.size

    if fn not in gray_cache:
        gray_cache[fn] = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)

    arms = [
        # A: what production crops today
        ("det", det_box),
        # B: ORACLE — detector x-range, GT y-range. Isolates the vertical term: how much of the
        #    det->human gap is purely the top and bottom edges being in the wrong place?
        ("detvert", [det_box[0], gt_box[1], det_box[2], gt_box[3]]),
        # C: the full human box. Anchor — the det->human gap should reproduce crop_cost.py's 0.0074.
        ("human", gt_box),
        # D: BUILDABLE — bilimai/inkcrop.py grows the box until it stops cutting this line's ink,
        #    capped at the neighbours' cores. Monotone: only ever adds, never masks or shrinks.
        ("ink", grow_box_to_ink(gray_cache[fn], det_box, [q[:4] for q in pred.get(fn, [])])),
    ]
    for kind, b in arms:
        k, img = _crop(im, b, W, H, fn)
        ckeys.append(k); imgs.append(img)
        meta.append((kind, txt, iou_val, gt_box, det_box, words))

# Dry-run: print stats and exit
if a.dry_run:
    ious = [iou_val for _, _, iou_val, _, _, _ in meta[::4]]
    vert_losses = [vertical_ink_lost(det_box, words) for _, _, _, _, det_box, words in meta[::4]]
    print(f"crops built: {len(imgs)}")
    print(f"median IoU: {statistics.median(ious):.3f}")
    print(f"median vertical ink lost: {statistics.median(vert_losses):.4f}")
    print(f"vertical ink lost distribution:")
    for pct in [10, 25, 50, 75, 90]:
        val = statistics.quantiles(vert_losses, n=100)[pct-1] if len(vert_losses) > 1 else vert_losses[0]
        print(f"  p{pct}: {val:.4f}")
    sys.exit(0)

# Load reader and read all three arms in one batch
from bilimai.reader import make_reader
reader = make_reader(a.base, a.adapter, line_h=128, max_new_tokens=96)
# DEDUP: on ~54 % of lines the ink-aware crop leaves the box untouched, so arm D is byte-identical to
# arm A. Reading the same pixels twice buys nothing. Read each distinct rectangle once, then fan the
# answer back out. Measured saving on the full pool: 1,166 of 8,556 reads (13.6 %).
seen, uniq, slot = {}, [], []
for k, img in zip(ckeys, imgs):
    if k not in seen:
        seen[k] = len(uniq); uniq.append(img)
    slot.append(seen[k])
print(f"reader {reader.name} | {len(imgs)} crops -> {len(uniq)} distinct "
      f"({len(imgs)-len(uniq)} duplicate reads skipped, {100*(len(imgs)-len(uniq))/max(1,len(imgs)):.1f} %)", flush=True)
t0 = time.time()
utexts, _ = reader.read(uniq, batch_size=a.batch_size, with_conf=False)
texts = [utexts[i] for i in slot]
dt = time.time() - t0
print(f"read in {dt:.0f}s  ({len(uniq)/max(dt,1e-9):.2f} reads/s)", flush=True)

# Collect results by (reference, iou, words) key; also track boxes
rows = {}
box_map = {}  # (txt, iou_val, words_key) -> (det_box, gt_box)
for (kind, txt, iou_val, gt_box, det_box, words), hyp in zip(meta, texts):
    key = (txt, iou_val, tuple(tuple(w["bbox"]) for w in words))
    rows.setdefault(key, {})[kind] = (hyp or "").strip()
    if key not in box_map:
        box_map[key] = (det_box, gt_box)

# Build records only for lines with all three arms
recs = []
for (txt, iou_val, words_key), v in rows.items():
    if not all(k in v for k in ("det", "detvert", "human", "ink")):
        continue
    det_cer = cer(txt, v["det"])
    detvert_cer = cer(txt, v["detvert"])
    human_cer = cer(txt, v["human"])
    ink_cer = cer(txt, v["ink"])

    # Reconstruct words and boxes from key
    words = [{"bbox": list(bbox)} for bbox in words_key]
    det_box, gt_box = box_map[(txt, iou_val, words_key)]
    vert_loss = vertical_ink_lost(det_box, words)

    recs.append({
        "ref": txt,
        "iou": iou_val,
        "n_chars": len(txt),
        "det_cer": det_cer,
        "detvert_cer": detvert_cer,
        "human_cer": human_cer,
        "ink_cer": ink_cer,
        "det_hyp": v["det"],
        "detvert_hyp": v["detvert"],
        "human_hyp": v["human"],
        "ink_hyp": v["ink"],
        "vertical_ink_lost": vert_loss
    })

def cw(key):
    ed = sum(r[key] * r["n_chars"] for r in recs)
    ch = sum(r["n_chars"] for r in recs)
    return ed / max(ch, 1)

det_cw = cw("det_cer")
detvert_cw = cw("detvert_cer")
human_cw = cw("human_cer")
ink_cw = cw("ink_cer")

# Per-line deltas
det_v_deltas = [r["detvert_cer"] - r["det_cer"] for r in recs]
det_h_deltas = [r["human_cer"] - r["det_cer"] for r in recs]
det_i_deltas = [r["ink_cer"] - r["det_cer"] for r in recs]

# Better/worse counts
detv_worse = sum(1 for d in det_v_deltas if d > 1e-9)
detv_better = sum(1 for d in det_v_deltas if d < -1e-9)
deth_worse = sum(1 for d in det_h_deltas if d > 1e-9)
deth_better = sum(1 for d in det_h_deltas if d < -1e-9)
deti_worse = sum(1 for d in det_i_deltas if d > 1e-9)
deti_better = sum(1 for d in det_i_deltas if d < -1e-9)

# McNemar test
from scipy.stats import binomtest
mc_detv = binomtest(detv_better, detv_better + detv_worse, 0.5, alternative="two-sided") if (detv_better + detv_worse) > 0 else None
mc_deth = binomtest(deth_better, deth_better + deth_worse, 0.5, alternative="two-sided") if (deth_better + deth_worse) > 0 else None
mc_deti = binomtest(deti_better, deti_better + deti_worse, 0.5, alternative="two-sided") if (deti_better + deti_worse) > 0 else None

print()
print(f"n paired lines            : {len(recs)}")
print(f"char-weighted CER, det       : {det_cw:.5f}")
print(f"char-weighted CER, detvert   : {detvert_cw:.5f}")
print(f"char-weighted CER, human     : {human_cw:.5f}")
print(f"char-weighted CER, inkcrop   : {ink_cw:.5f}   <- the buildable one")
print(f"delta detvert - det          : {detvert_cw - det_cw:+.5f}")
print(f"delta human - det            : {human_cw - det_cw:+.5f}  [sanity anchor: should be ~-0.0074]")
print(f"delta inkcrop - det          : {ink_cw - det_cw:+.5f}")
print()
print(f"detvert vs det: worse {detv_worse} | better {detv_better} | identical {len(recs)-detv_worse-detv_better}")
if mc_detv:
    print(f"  McNemar p: {mc_detv.pvalue:.4f}")
print(f"human vs det: worse {deth_worse} | better {deth_better} | identical {len(recs)-deth_worse-deth_better}")
if mc_deth:
    print(f"  McNemar p: {mc_deth.pvalue:.4f}")
print(f"inkcrop vs det: worse {deti_worse} | better {deti_better} | identical {len(recs)-deti_worse-deti_better}")
if mc_deti:
    print(f"  McNemar p: {mc_deti.pvalue:.4f}")
print()

# Fraction of gap recovered by detvert
gap_det_h = human_cw - det_cw
gap_det_dv = detvert_cw - det_cw
if gap_det_h > 1e-9:
    recovered = (gap_det_h - gap_det_dv) / gap_det_h
    print(f"vertical fix recovers {recovered:.1%} of det→human gap (({gap_det_h:+.5f} − {gap_det_dv:+.5f}) / {gap_det_h:+.5f})")
else:
    recovered = 0.0
    print(f"gap too small for ratio (det→human = {gap_det_h:+.5f})")

json.dump({
    "n": len(recs),
    "det_cer_charweighted": det_cw,
    "detvert_cer_charweighted": detvert_cw,
    "human_cer_charweighted": human_cw,
    "delta_detvert_minus_det": detvert_cw - det_cw,
    "delta_human_minus_det": human_cw - det_cw,
    "detvert_vs_det_worse": detv_worse,
    "detvert_vs_det_better": detv_better,
    "human_vs_det_worse": deth_worse,
    "human_vs_det_better": deth_better,
    "detvert_vs_det_mcnemar_p": mc_detv.pvalue if mc_detv else None,
    "human_vs_det_mcnemar_p": mc_deth.pvalue if mc_deth else None,
    "vertical_recovery_fraction": recovered,
    "sample": recs
}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n→ {a.out}")
