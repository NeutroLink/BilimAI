# v2 — Chandra OCR 2 zero-shot (fixed generate_hf call: processor attached to model) + Surya OCR 2 line detection on the sealed RU exam (Kaggle T4)
"""Outputs (/kaggle/working):
  chandra_zero_lines.json   oracle line crops → text
  chandra_zero_pages.json   left/right page halves → text
  surya_det.json            {file: [[x0,y0,x1,y1,conf], ...]} line boxes in spread coords  (E4.1 detector candidate)
  timings.json
Score locally: eval/score.py for the text files; eval/detectors/score_detector.py for surya_det.json.
"""
import json, os, subprocess, sys, time, glob
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chandra-ocr[hf]", "surya-ocr", "--upgrade"], check=True)

import torch
from PIL import Image

_gt = glob.glob("/kaggle/input/**/ground_truth.json", recursive=True); assert _gt
DATA = Path(_gt[0]).parent; IMG_DIR = DATA / "images"
GT = json.load(open(DATA / "ground_truth.json", encoding="utf-8"))
OUT = Path("/kaggle/working"); timings = {}
print("GPU:", torch.cuda.get_device_name(0), "| n:", torch.cuda.device_count(), flush=True)

def halves(img):
    w, h = img.size; return [("L", img.crop((0, 0, w // 2, h))), ("R", img.crop((w // 2, 0, w, h)))]
def save(name, obj): json.dump(obj, open(OUT / name, "w"), ensure_ascii=False, indent=1)

# ------------------------------------------------------------ Surya line detection (pure torch)
t0 = time.time()
try:
    from surya.detection import DetectionPredictor
    det = DetectionPredictor()
    out = {}
    for i, fn in enumerate(GT):
        img = Image.open(IMG_DIR / fn).convert("RGB")
        preds = det([img])[0]
        boxes = []
        for b in preds.bboxes:
            bb = list(b.bbox) if hasattr(b, "bbox") else b["bbox"]
            conf = float(getattr(b, "confidence", 1.0) if hasattr(b, "confidence") else b.get("confidence", 1.0))
            boxes.append([float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]), conf])
        out[fn] = boxes; save("surya_det.json", out)
        print(f"[surya-det {i+1}/20] {fn}: {len(boxes)} lines", flush=True)
    del det; torch.cuda.empty_cache()
except Exception as e:
    import traceback; traceback.print_exc(); print("!!! surya detection failed:", e, flush=True)
timings["surya_det"] = time.time() - t0; save("timings.json", timings)

# ------------------------------------------------------------ Chandra OCR 2 (HF path)
t0 = time.time()
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    name = "datalab-to/chandra-ocr-2"
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.float16, device_map="auto").eval()   # T4: no bf16
    proc = AutoProcessor.from_pretrained(name); proc.tokenizer.padding_side = "left"
    model.processor = proc                                   # generate_hf reads model.processor (see chandra/model/hf.py)

    def read(img, prompt_type="ocr"):
        res = generate_hf([BatchInputItem(image=img, prompt_type=prompt_type)], model, max_output_tokens=256 if img.size[1] < 400 else 2048)[0]
        txt = getattr(res, "markdown", None) or getattr(res, "raw", None) or getattr(res, "text", None) or str(res)
        return str(txt).strip()

    lines_out, pages_out = {}, {}
    for i, fn in enumerate(GT):
        img = Image.open(IMG_DIR / fn).convert("RGB")
        t = time.time(); L = []
        for l in GT[fn]["lines"]:
            x0, y0, x1, y1 = l["bbox"]; pad = 12
            crop = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.size[0], x1 + pad), min(img.size[1], y1 + pad)))
            try: txt = read(crop)
            except Exception as e: txt = ""; print("  line err", fn, e, flush=True)
            L.append({"text": txt.replace("\n", " ").strip(), "bbox": l["bbox"]})
        lines_out[fn] = L; save("chandra_zero_lines.json", lines_out); ta = time.time() - t
        t = time.time(); parts = []
        for side, half in halves(img):
            try: parts.append(read(half))
            except Exception as e: parts.append(""); print("  page err", fn, side, e, flush=True)
        pages_out[fn] = "\n".join(parts); save("chandra_zero_pages.json", pages_out)
        print(f"[chandra {i+1}/20] {fn}: lines {len(L)} in {ta:.0f}s | pages {time.time()-t:.0f}s", flush=True)
except Exception as e:
    import traceback; traceback.print_exc(); print("!!! chandra failed:", e, flush=True)
timings["chandra"] = time.time() - t0; save("timings.json", timings)
print("ALL DONE", timings)
