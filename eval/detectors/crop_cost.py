#!/usr/bin/env python3
"""E4.2 — what does the detector's CROP cost, on lines it already found? (2026-08-22)

miss_profile.py showed the 335 missed lines are only 1.7 % of exam characters, while the measured E2E gap
(real boxes 0.0658 vs human boxes 0.0227) is 4.3 points. So most of the detector's cost is NOT misses — it is
lines that ARE matched being cropped differently from the human box (median best IoU 0.81, not 1.0).

This isolates that term. For each MATCHED GT line (best IoU >= --iou) we read the SAME line twice with the
production reader — once cropped by the human box, once by the detector box — and compare CER against the same
reference. Identical reader, identical prompt, identical crop recipe (pad 12, resize h=128, as e2e_exam.py);
the ONLY difference is the rectangle. Paired by construction, so per-line deltas are meaningful.

    eval/.venv/bin/python eval/detectors/crop_cost.py --n 400

Writes eval/runs/det_crop_cost.json.
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
ap.add_argument("--out", default=str(ROOT / "eval/runs/det_crop_cost.json"))
ap.add_argument("--n", type=int, default=400, help="matched lines to sample (paired: 2 reads each)")
ap.add_argument("--iou", type=float, default=0.5, help="min best-IoU to count a line as matched")
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

sys.path.insert(0, str(ROOT))
from PIL import Image
from bilimai.reader import make_reader


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


gt = json.load(open(a.gt, encoding="utf-8"))
pred = json.load(open(a.pred, encoding="utf-8"))

pairs = []                                    # (file, text, gt_box, rp_box, iou)
for fn, page in gt.items():
    P = [p[:4] for p in pred.get(fn, [])]
    if not P:
        continue
    for l in page["lines"]:
        g = l["bbox"]; txt = (l.get("text") or "").strip()
        if not txt:
            continue
        best, bi = max(((iou(p, g), i) for i, p in enumerate(P)), default=(0.0, -1))
        if best >= a.iou:
            pairs.append((fn, txt, g, P[bi], best))

random.Random(a.seed).shuffle(pairs)
sample = pairs[:a.n]
print(f"matched lines available {len(pairs)} | sampling {len(sample)} (paired → {2*len(sample)} reads)", flush=True)

PAD = 12
imgs, meta = [], []
cache = {}
for fn, txt, g, p, ov in sample:
    if fn not in cache:
        cache[fn] = Image.open(Path(a.images) / fn).convert("RGB")
    im = cache[fn]; W, H = im.size
    for kind, b in (("gt", g), ("rp", p)):
        imgs.append(im.crop((max(0, b[0] - PAD), max(0, b[1] - PAD), min(W, b[2] + PAD), min(H, b[3] + PAD))))
        meta.append((kind, txt, ov))

reader = make_reader(a.base, a.adapter, line_h=128, max_new_tokens=96)
print(f"reader {reader.name} | reading {len(imgs)} crops...", flush=True)
t0 = time.time()
texts, _ = reader.read(imgs, batch_size=a.batch_size, with_conf=False)
print(f"read in {time.time()-t0:.0f}s", flush=True)

rows = {}
for (kind, txt, ov), hyp in zip(meta, texts):
    rows.setdefault((txt, ov), {})[kind] = (hyp or "").strip()

recs = [{"ref": t, "iou": ov, "gt_cer": cer(t, v["gt"]), "rp_cer": cer(t, v["rp"]),
         "gt_hyp": v["gt"], "rp_hyp": v["rp"], "n_chars": len(t)}
        for (t, ov), v in rows.items() if "gt" in v and "rp" in v]

def cw(key):
    ed = sum(r[key] * r["n_chars"] for r in recs); ch = sum(r["n_chars"] for r in recs)
    return ed / max(ch, 1)

g_cw, r_cw = cw("gt_cer"), cw("rp_cer")
deltas = [r["rp_cer"] - r["gt_cer"] for r in recs]
worse = sum(1 for d in deltas if d > 1e-9); better = sum(1 for d in deltas if d < -1e-9)

print()
print(f"n paired lines            : {len(recs)}")
print(f"char-weighted CER, human box : {g_cw:.5f}")
print(f"char-weighted CER, detector  : {r_cw:.5f}")
print(f"CROP COST (detector − human) : {r_cw - g_cw:+.5f}  ({100*(r_cw-g_cw)/max(g_cw,1e-9):+.1f} % relative)")
print(f"per-line: worse {worse} | better {better} | identical {len(recs)-worse-better}")
print(f"median per-line delta     : {statistics.median(deltas):+.5f}")
print(f"median IoU of sampled pairs: {statistics.median([r['iou'] for r in recs]):.3f}")

# does the damage concentrate on the loosest boxes?
print()
print("crop cost by box overlap:")
for lo, hi in ((0.5, 0.7), (0.7, 0.85), (0.85, 1.01)):
    sub = [r for r in recs if lo <= r["iou"] < hi]
    if not sub: continue
    ed_g = sum(r["gt_cer"] * r["n_chars"] for r in sub); ed_r = sum(r["rp_cer"] * r["n_chars"] for r in sub)
    ch = sum(r["n_chars"] for r in sub)
    print(f"  IoU {lo:.2f}–{hi:.2f}: n={len(sub):>4}  human {ed_g/ch:.4f}  detector {ed_r/ch:.4f}  cost {(ed_r-ed_g)/ch:+.4f}")

json.dump({"n": len(recs), "gt_cer_charweighted": g_cw, "rp_cer_charweighted": r_cw,
           "crop_cost": r_cw - g_cw, "worse": worse, "better": better,
           "sample": recs[:60]}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n→ {a.out}")
