# v2 — community Qwen3-VL-4B Russian-handwriting fine-tune (chukhrovns only; gbull25 dropped per user), zero-shot on the sealed RU exam (Kaggle T4, bnb 4-bit)
"""Candidate (user-supplied, 2026-08-16):
  chukhrovns/qwen3-vl-4b-russian-handwriting-merged-4bit   (already bnb 4-bit, fp16)
The card names no training data and reports no CER, so treat results with care: if they were trained on
ai-forever/school_notebooks_RU *test* split, our exam pages are not unseen for them.
Outputs (/kaggle/working):
  qwenhw_<tag>_zero_lines.json / qwenhw_<tag>_zero_pages.json / timings.json
Score locally with eval/score.py.
"""
import json, subprocess, sys, time, glob
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes", "accelerate", "--upgrade"], check=True)

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

_gt = glob.glob("/kaggle/input/**/ground_truth.json", recursive=True); assert _gt
DATA = Path(_gt[0]).parent; IMG_DIR = DATA / "images"
GT = json.load(open(DATA / "ground_truth.json", encoding="utf-8"))
OUT = Path("/kaggle/working"); timings = {}
print("GPU:", torch.cuda.get_device_name(0), flush=True)
def save(name, obj): json.dump(obj, open(OUT / name, "w"), ensure_ascii=False, indent=1)

MODELS = {
    "chukhrovns": ("chukhrovns/qwen3-vl-4b-russian-handwriting-merged-4bit", False),   # already quantised
}
PROMPT = "Распознай рукописный текст на изображении. Выведи только текст, без пояснений."

for tag, (name, quantise) in MODELS.items():
    t0 = time.time()
    try:
        kw = dict(device_map="auto", dtype=torch.float16)
        if quantise:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        model = AutoModelForImageTextToText.from_pretrained(name, **kw).eval()
        proc = AutoProcessor.from_pretrained(name)
        print(f"[{tag}] loaded in {time.time()-t0:.0f}s", flush=True)

        def read(img, max_new=96):
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
            inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

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
            lines_out[fn] = L; save(f"qwenhw_{tag}_zero_lines.json", lines_out); ta = time.time() - t
            t = time.time(); w, h = img.size; parts = []
            for half in (img.crop((0, 0, w // 2, h)), img.crop((w // 2, 0, w, h))):
                try: parts.append(read(half, max_new=1024))
                except Exception as e: parts.append(""); print("  page err", fn, e, flush=True)
            pages_out[fn] = "\n".join(parts); save(f"qwenhw_{tag}_zero_pages.json", pages_out)
            print(f"[{tag} {i+1}/20] {fn}: lines {len(L)} in {ta:.0f}s | pages {time.time()-t:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"!!! {tag} failed:", e, flush=True)
    timings[tag] = time.time() - t0; save("timings.json", timings)
print("ALL DONE", timings)
