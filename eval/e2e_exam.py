#!/usr/bin/env python3
"""BilimAI — END-TO-END exam evaluation: detector boxes → line crops → reader (GLM-OCR + LoRA) → eval/score.py.

The oracle number (glm_lora_*_lines.json) reads human-drawn line boxes, i.e. it measures the *reader* alone. In
production nobody draws boxes: a detector finds the lines, and every detector mistake (missed / merged / split / clipped
line) reaches the reader. This script mimics that path on the sealed exam so the two numbers can be compared:

    e2e CER − oracle CER  =  what the detector costs us today   (→ decides reader training vs E4.2 detector work)

Inputs
  --det      detector output on the exam pages: {"<image>": [[x0,y0,x1,y1,conf], ...]}  (eval/runs/surya_det.json,
             eval/runs/rp_det_lines.json — produced by eval/detectors/*, scored by score_detector.py)
  --adapter  LoRA adapter dir (or "none" for zero-shot base)
  --exam     dir with images/ + ground_truth.json (default: Kaggle layout, then eval/testset_v1/ru_pages)
  --base     GLM-OCR dir (default: /kaggle/models/GLM-OCR, then models/GLM-OCR)
  --out      predictions JSON in score.py's line format {"<image>": [{"text","bbox"}]}
  --min-conf drop detector boxes below this confidence (default 0.0 = keep all)
  --score    also run eval/score.py and print the summary (needs jiwer; on the box: pip install jiwer)

Crop recipe is identical to the oracle path in round4_hwr200_replay.py (pad 12 px, resize to height 128) so the ONLY
difference between the two runs is where the boxes come from. Boxes are read in top-to-bottom, left-to-right order;
score.py matches predicted lines to GT lines by IoU, so order does not affect the metrics.

Runs on CUDA (Vast/Kaggle), MPS (Mac, slow ~1 s/line) or CPU. ~5 min per (detector × adapter) on one RTX 5090.
"""
import argparse, json, os, sys, time, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--det", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--exam"); ap.add_argument("--base"); ap.add_argument("--min-conf", type=float, default=0.0)
ap.add_argument("--score", action="store_true"); ap.add_argument("--max-new", type=int, default=96)
ap.add_argument("--no-resize", action="store_true", help="keep crop resolution (Qwen3-VL runs; GLM runs resize to h=128 like training)")
ap.add_argument("--prompt", default="Text Recognition:")
a = ap.parse_args()

HERE = Path(__file__).resolve().parent
EXAM = Path(a.exam) if a.exam else next((p for p in (Path("/kaggle/input/datasets/jahongir713/bilimai-exam-ru-v1/ru_pages"),
        Path("/kaggle/input/datasets/jahongir713/bilimai-exam-ru-v1"), HERE / "testset_v1" / "ru_pages") if (p / "ground_truth.json").exists()), None)
assert EXAM, "exam dir with ground_truth.json not found — pass --exam"
BASE = a.base or next((str(p) for p in (Path("/kaggle/models/GLM-OCR"), HERE.parent / "models" / "GLM-OCR") if (p / "config.json").exists()), "zai-org/GLM-OCR")
PROMPT = a.prompt

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float16 if dev != "cpu" else torch.float32
proc = AutoProcessor.from_pretrained(BASE)
model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=dtype).to(dev).eval()
if a.adapter.lower() != "none":
    from peft import PeftModel
    # GATE: never evaluate a truncated adapter (2026-08-17)
    _va = next(p for p in (HERE / "train" / "vast" / "verify_artifacts.py", HERE / "verify_artifacts.py") if p.exists())   # repo layout vs box layout
    subprocess.run([sys.executable, str(_va), "adapter", a.adapter], check=True)
    model = PeftModel.from_pretrained(model, a.adapter).eval()
print(f"device {dev} | base {BASE} | adapter {a.adapter} | detector {a.det} | exam {EXAM}", flush=True)

def read(img):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(dev)
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=a.max_new, do_sample=False)
    return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

det = json.load(open(a.det, encoding="utf-8"))
gt = json.load(open(EXAM / "ground_truth.json", encoding="utf-8"))
preds, t0, n = {}, time.time(), 0
for i, fn in enumerate(gt):
    boxes = [b for b in det.get(fn, []) if len(b) < 5 or b[4] >= a.min_conf]
    boxes.sort(key=lambda b: (round(b[1] / 40), b[0]))                       # reading order: rows of ~40 px, then x
    img = Image.open(EXAM / "images" / fn).convert("RGB"); L = []
    for b in boxes:
        x0, y0, x1, y1 = [float(v) for v in b[:4]]; pad = 12
        crop = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.size[0], x1 + pad), min(img.size[1], y1 + pad)))
        if crop.size[0] < 4 or crop.size[1] < 4: continue
        if crop.size[1] > 128 and not a.no_resize:
            s = 128 / crop.size[1]; crop = crop.resize((max(8, int(crop.size[0] * s)), 128))
        try: txt = read(crop)
        except Exception as e: txt = ""; print("  err", fn, e, flush=True)
        L.append({"text": txt.replace("\n", " ").strip(), "bbox": [x0, y0, x1, y1]}); n += 1
    preds[fn] = L
    print(f"[e2e {i+1}/{len(gt)}] {fn}: {len(L)} detected lines (GT {len(gt[fn]['lines'])}) | {time.time()-t0:.0f}s", flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(preds, open(a.out, "w"), ensure_ascii=False, indent=1)
print(f"wrote {a.out}: {n} lines in {time.time()-t0:.0f}s")
if a.score:
    subprocess.run([sys.executable, str(HERE / "score.py"), "--gt", str(EXAM / "ground_truth.json"), "--pred", a.out, "--quiet"])
