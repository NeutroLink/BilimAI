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

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
proc = AutoProcessor.from_pretrained(a.model)
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.float16, device_map="cuda").eval()
print("loaded", a.model, "| part", a.part, flush=True)

def read(img, prompt, max_new=None):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new or a.max_new, do_sample=False)
    return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")

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
    preds = {}; t0 = time.time()
    for i, fn in enumerate(gt):
        img = Image.open(EXAM / "images" / fn).convert("RGB"); L = []
        for l in gt[fn]["lines"]:
            try: txt = read(crop_line(img, l["bbox"]), a.prompt)
            except Exception as e: txt = ""; print("  err", fn, e, flush=True)
            L.append({"text": txt, "bbox": l["bbox"]})
        preds[fn] = L; json.dump(preds, open(OUT / "qwen3vl_ru_hw_lines.json", "w"), ensure_ascii=False, indent=1)
        print(f"[exam {i+1}/{len(gt)}] {fn} {len(L)} lines | {time.time()-t0:.0f}s", flush=True)
    json.dump({"prompt": a.prompt, "seconds": time.time() - t0, "lines": sum(len(v) for v in preds.values())}, open(OUT / "exam_meta.json", "w"), ensure_ascii=False, indent=1)
    print(f"exam done {time.time()-t0:.0f}s")

else:  # holdout — same 1500 strips as round 4
    rows = json.load(open(a.holdout_list, encoding="utf-8"))
    t0 = time.time(); out = []; cs = []
    for i, r in enumerate(rows):
        img = Image.open(HWR / "images" / r["file"]).convert("RGB")
        try: hyp = read(img, a.prompt)
        except Exception as e: hyp = ""; print("  err", r["file"], e, flush=True)
        c = cer(r["ref"], hyp); cs.append(c); out.append({"file": r["file"], "cond": r["cond"], "ref": r["ref"], "hyp": hyp, "cer": round(c, 4)})
        if (i + 1) % 250 == 0: print(f"[holdout {i+1}/{len(rows)}] median so far {statistics.median(cs):.4f} | {time.time()-t0:.0f}s", flush=True)
    by = {}
    for o in out: by.setdefault(o["cond"], []).append(o["cer"])
    score = {"n": len(out), "cer_median": statistics.median(cs), "cer_mean": statistics.mean(cs),
             "by_cond_median": {k: round(statistics.median(v), 4) for k, v in by.items()}, "prompt": a.prompt, "seconds": time.time() - t0}
    json.dump(out, open(OUT / "holdout_preds.json", "w"), ensure_ascii=False, indent=1)
    json.dump(score, open(OUT / "holdout_score.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(score, ensure_ascii=False))
