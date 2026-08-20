#!/usr/bin/env python
"""E2.10 step 1 — math baseline (2026-08-20, free, Mac): can the current reader read handwritten math at all?

Renders the 300 human-written expressions of the MathWriting excerpt (InkML strokes → line strips) and runs the product
reader (GLM-OCR + v5 LoRA). Scoring is honest about what the reader can even express: it outputs plain text, so
- subset A (linear-ASCII labels — no \frac, ^, _, {}): CER vs the label with all spaces removed — a real number;
- subset B (2-D layout): qualitative only — predictions recorded so we see WHAT it does with fractions/powers.

  eval/.venv/bin/python eval/math/baseline_mathwriting.py [--n 300] [--adapter models/adapters/glm-ocr-lora-ru-v5]

Writes eval/runs/math_baseline_v0.json, strips + a 12-example sheet under out/math_baseline/.
"""
import argparse, json, re, sys
from pathlib import Path
from xml.etree import ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(ROOT / "data/raw/mathwriting/mathwriting-2024-excerpt"))
ap.add_argument("--n", type=int, default=300); ap.add_argument("--h", type=int, default=128)
ap.add_argument("--adapter", default=str(ROOT / "models/adapters/glm-ocr-lora-ru-v5")); ap.add_argument("--base", default=str(ROOT / "models/GLM-OCR"))
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()

NS = {"i": "http://www.w3.org/2003/InkML"}
LINEAR = re.compile(r"^[0-9A-Za-z+\-*/=().,:;<>!| ]+$")


def parse_inkml(fp):
    r = ET.parse(fp).getroot()
    lab = {x.get("type"): (x.text or "") for x in r.findall("i:annotation", NS)}
    traces = []
    for t in r.findall("i:trace", NS):
        pts = []
        for p in (t.text or "").strip().split(","):
            v = p.split()
            if len(v) >= 2: pts.append((float(v[0]), float(v[1])))
        if len(pts) >= 2: traces.append(np.array(pts, np.float32))
    return lab.get("normalizedLabel") or lab.get("label"), traces


def render(traces, H):
    allp = np.vstack(traces)
    x0, y0 = allp.min(0); x1, y1 = allp.max(0)
    ih = max(1.0, y1 - y0); s = (H * 0.72) / ih
    W = int((x1 - x0) * s) + 40
    if W > 3200: s *= 3200 / W; W = 3200
    img = np.full((H, max(64, W), 3), 250, np.uint8)
    for t in traces:
        q = ((t - (x0, y0)) * s + (20, H * 0.14)).astype(np.int32)
        cv2.polylines(img, [q.reshape(-1, 1, 2)], False, (40, 35, 30), 3, cv2.LINE_AA)
    return img


def cer(hyp, ref):
    h, r = hyp.replace(" ", ""), ref.replace(" ", "")
    d = np.zeros((len(h) + 1, len(r) + 1), np.int32); d[:, 0] = np.arange(len(h) + 1); d[0, :] = np.arange(len(r) + 1)
    for i in range(1, len(h) + 1):
        for j in range(1, len(r) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (h[i - 1] != r[j - 1]))
    return d[-1, -1] / max(1, len(r))


files = []
for sp in ("train", "valid", "test"):
    files += sorted((Path(a.src) / sp).glob("*.inkml"))
files = files[:a.n]
outd = ROOT / "out/math_baseline"; (outd / "strips").mkdir(parents=True, exist_ok=True)
recs = []; imgs = []
for fp in files:
    lab, traces = parse_inkml(fp)
    if not lab or not traces: continue
    img = render(traces, a.h)
    name = fp.stem + ".jpg"
    cv2.imwrite(str(outd / "strips" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    recs.append({"file": name, "label": lab, "linear": bool(LINEAR.match(lab))}); imgs.append(img)
print(f"{len(recs)} expressions rendered ({sum(r['linear'] for r in recs)} linear-ASCII)", flush=True)

from PIL import Image
from bilimai.reader import GLMBatchReader
reader = GLMBatchReader(a.base, a.adapter, max_new_tokens=64)
crops = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in imgs]
texts, _ = reader.read(crops, batch_size=a.batch, prompt="[school] Text Recognition:")
for r, t in zip(recs, texts): r["pred"] = t

lin = [r for r in recs if r["linear"]]
for r in lin: r["cer"] = round(cer(r["pred"], r["label"]), 4)
lin_cer = float(np.mean([r["cer"] for r in lin])) if lin else None
exact = sum(r["pred"].replace(" ", "") == r["label"].replace(" ", "") for r in lin)
res = {"n": len(recs), "linear_n": len(lin), "linear_mean_cer": lin_cer, "linear_exact": exact,
       "reader": reader.name, "examples_2d": [{"label": r["label"], "pred": r["pred"]} for r in recs if not r["linear"]][:15]}
json.dump({"summary": res, "records": recs}, open(ROOT / "eval/runs/math_baseline_v0.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res["examples_2d"][:5], ensure_ascii=False, indent=1))
print(f"\nlinear-ASCII subset: n={len(lin)} mean CER {lin_cer:.3f} exact {exact}/{len(lin)}" if lin else "no linear subset")

tiles = []
for r in (lin[:6] + [x for x in recs if not x["linear"]][:6]):
    im = cv2.imread(str(outd / "strips" / r["file"]))
    s = min(1.0, 1200 / im.shape[1]); im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
    im = cv2.copyMakeBorder(im, 0, 0, 0, max(0, 1202 - im.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cap = np.full((30, 1202, 3), 246, np.uint8)
    tag = "LIN" if r["linear"] else "2D "
    cv2.putText(cap, f"{tag} label: {r['label'][:50]}  ||  pred: {r['pred'][:50]}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA)
    tiles += [cap, im]
cv2.imwrite(str(outd / "sheet.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 92])
print("sheet:", outd / "sheet.jpg")
