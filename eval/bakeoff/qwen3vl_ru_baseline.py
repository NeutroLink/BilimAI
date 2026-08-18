#!/usr/bin/env python3
"""Baseline for a second reader base: gbull25/qwen3-vl-russian-handwriting (Qwen3-VL-4B-Instruct fine-tune, fp16,
Apache-2.0) — scored exactly like our GLM adapters so the numbers are comparable:
  * sealed exam, oracle line boxes (pad 12, no resize — same as the chukhrovns bake-off kernel) → glm-format lines JSON,
    scored on the Mac with eval/score.py
  * HWR200 held-out: the SAME 1,500 strips round 4 used (taken from replay40/holdout_preds.json) → CER median / by cond
Runs on a Vast box: MODEL dir + exam + hwr200 strips in the Kaggle layout. --part exam|holdout|probe; --prompt overrides.
Probe = 30 exam lines × 3 candidate prompts (the model card documents none) — pick the best, then run the parts.
"""
import argparse, json, os, sys, time, statistics
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--part", required=True, choices=["probe", "exam", "holdout"])
ap.add_argument("--model", default="/kaggle/models/qwen3vl-ru-hw")
ap.add_argument("--prompt", default="Распознай рукописный текст на изображении. Выведи только текст, без пояснений.")
ap.add_argument("--out", default="/kaggle/working/results/qwen3vl_ru_hw")
ap.add_argument("--holdout-list", default="/kaggle/working/results/replay40/holdout_preds.json")
ap.add_argument("--max-new", type=int, default=96)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
EXAM = Path("/kaggle/input/datasets/jahongir713/bilimai-exam-ru-v1"); HWR = Path("/kaggle/input/datasets/jahongir713/bilimai-strips-hwr200-v1")

from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent / "pkg"))
from bilimai.reader import GLMBatchReader           # works for any HF image-text-to-text model with a chat template
reader = GLMBatchReader(a.model, None, device="cuda", prompt=a.prompt, line_h=10**6, max_new_tokens=a.max_new)
BS = int(os.environ.get("EVAL_BS", "16"))
def read_many(imgs, prompt): return reader.read(imgs, batch_size=BS, prompt=prompt, with_conf=False)[0]
def read(img, prompt, max_new=None): return read_many([img], prompt)[0]
print("loaded", a.model, "| part", a.part, "| batched", BS, flush=True)

def cer(ref, hyp):
    m, n = len(ref), len(hyp)
    if m == 0: return float(n > 0)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[n] / m

def crop_line(img, bbox, pad=12):
    x0, y0, x1, y1 = bbox
    return img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.size[0], x1 + pad), min(img.size[1], y1 + pad)))

gt = json.load(open(EXAM / "ground_truth.json", encoding="utf-8"))

if a.part == "probe":
    cands = [a.prompt, "Text Recognition:", "Transcribe the handwritten Russian text in the image. Output only the text."]
    sample = [(fn, l) for fn in list(gt)[:6] for l in gt[fn]["lines"][:5]]          # 30 lines, 6 pages
    imgs = {fn: Image.open(EXAM / "images" / fn).convert("RGB") for fn, _ in sample}
    res = {}
    for p in cands:
        t0 = time.time(); cs = [cer(l["text"], read(crop_line(imgs[fn], l["bbox"]), p)) for fn, l in sample]
        res[p] = {"cer_median": statistics.median(cs), "cer_mean": statistics.mean(cs), "sec_per_line": (time.time() - t0) / len(sample)}
        print(json.dumps({p[:40]: res[p]}, ensure_ascii=False), flush=True)
    best = min(res, key=lambda p: res[p]["cer_median"])
    json.dump({"results": res, "best": best}, open(OUT / "probe.json", "w"), ensure_ascii=False, indent=1)
    print("BEST PROMPT:", best)

elif a.part == "exam":
    t0 = time.time(); crops = []; owners = []
    for fn in gt:
        img = Image.open(EXAM / "images" / fn).convert("RGB")
        for l in gt[fn]["lines"]: crops.append(crop_line(img, l["bbox"])); owners.append((fn, l["bbox"]))
    texts = read_many(crops, a.prompt); preds = {fn: [] for fn in gt}
    for (fn, bb), txt in zip(owners, texts): preds[fn].append({"text": txt, "bbox": bb})
    json.dump(preds, open(OUT / "qwen3vl_ru_hw_lines.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"prompt": a.prompt, "seconds": time.time() - t0, "lines": len(crops)}, open(OUT / "exam_meta.json", "w"), ensure_ascii=False, indent=1)
    print(f"exam done {time.time()-t0:.0f}s ({len(crops)} lines, batched)")

else:  # holdout — same 1500 strips as round 4
    rows = json.load(open(a.holdout_list, encoding="utf-8"))
    t0 = time.time(); hyps = read_many([Image.open(HWR / "images" / r["file"]).convert("RGB") for r in rows], a.prompt); out = []; cs = []
    for r, hyp in zip(rows, hyps):
        c = cer(r["ref"], hyp); cs.append(c); out.append({"file": r["file"], "cond": r["cond"], "ref": r["ref"], "hyp": hyp, "cer": round(c, 4)})
    by = {}
    for o in out: by.setdefault(o["cond"], []).append(o["cer"])
    score = {"n": len(out), "cer_median": statistics.median(cs), "cer_mean": statistics.mean(cs),
             "by_cond_median": {k: round(statistics.median(v), 4) for k, v in by.items()}, "prompt": a.prompt, "seconds": time.time() - t0}
    json.dump(out, open(OUT / "holdout_preds.json", "w"), ensure_ascii=False, indent=1)
    json.dump(score, open(OUT / "holdout_score.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(score, ensure_ascii=False))
