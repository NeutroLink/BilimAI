# v4 — install vllm first, then surya-ocr --no-deps (+light deps): pip cannot co-resolve surya's pillow<11 / hf-hub<2 pins with vllm 0.27
# v3 — Surya OCR 2 *reader* (v2: Kaggle resolved vllm 0.11.0 → no Qwen3_5 arch; now pin vllm>=0.27.1)
# v2 — Surya OCR 2 *reader* zero-shot on the sealed RU exam (Kaggle T4).
#      v1 failed: Surya's vllm backend spawns `docker run` (no Docker on Kaggle). v2 starts `vllm serve` ourselves
#      and attaches via SURYA_INFERENCE_URL (surya/inference/backends/spawn.py: external_url path, no docker).
"""Outputs (/kaggle/working):
  surya_zero_lines.json   oracle line crops → text
  surya_zero_pages.json   left/right page halves → text (block texts joined by newlines)
  timings.json
Score locally with eval/score.py.
"""
import json, os, re, subprocess, sys, time, glob, html
from pathlib import Path

MODEL = "datalab-to/surya-ocr-2"; PORT = 8000
os.environ["SURYA_INFERENCE_BACKEND"] = "vllm"
os.environ["SURYA_INFERENCE_URL"] = f"http://127.0.0.1:{PORT}/v1"
os.environ["VLLM_GPU_TYPE"] = "t4"
os.environ["VLLM_DTYPE"] = "half"
t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm>=0.27.1"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "surya-ocr"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "beautifulsoup4", "filetype", "pypdfium2", "pydantic-settings", "python-dotenv", "platformdirs", "opencv-python-headless", "openai", "httpx"], check=True)
print(f"install {time.time()-t0:.0f}s", flush=True)
subprocess.run([sys.executable, "-m", "pip", "list"], stdout=open("/kaggle/working/pip_list.txt","w"))
import importlib.metadata as _md; print("vllm", _md.version("vllm"), "torch", _md.version("torch"), "transformers", _md.version("transformers"), flush=True)

# ---- start vLLM server ourselves (mirrors the args surya's docker spawn would pass; T4: fp16, smaller ctx)
srv_log = open("/kaggle/working/vllm_server.log", "w")
srv = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
                        "--served-model-name", MODEL, "--port", str(PORT), "--dtype", "half",
                        "--max-model-len", "8192", "--max-num-seqs", "8", "--gpu-memory-utilization", "0.85",
                        "--enable-prefix-caching", "--mm-processor-kwargs", json.dumps({"min_pixels": 3136, "max_pixels": 6291456})],
                       stdout=srv_log, stderr=subprocess.STDOUT)
import urllib.request
t = time.time(); up = False
while time.time() - t < 1500:
    if srv.poll() is not None: break
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3); up = True; break
    except Exception: time.sleep(10)
print(f"vllm server up={up} in {time.time()-t:.0f}s", flush=True)
if not up:
    srv_log.flush(); print(open("/kaggle/working/vllm_server.log").read()[-6000:], flush=True); sys.exit(1)

import torch
from PIL import Image

_gt = glob.glob("/kaggle/input/**/ground_truth.json", recursive=True); assert _gt
DATA = Path(_gt[0]).parent; IMG_DIR = DATA / "images"
GT = json.load(open(DATA / "ground_truth.json", encoding="utf-8"))
OUT = Path("/kaggle/working"); timings = {}
print("GPU:", torch.cuda.get_device_name(0), flush=True)

def save(name, obj): json.dump(obj, open(OUT / name, "w"), ensure_ascii=False, indent=1)
def strip_html(s):
    return html.unescape(re.sub(r"<[^>]+>", " ", s or "")).replace("\xa0", " ").strip()

from surya.inference import SuryaInferenceManager
from surya.recognition import RecognitionPredictor
t = time.time()
manager = SuryaInferenceManager()
rec = RecognitionPredictor(manager)
print(f"backend up in {time.time()-t:.0f}s", flush=True)

def read_many(images):
    """images -> list of texts (blocks joined in reading order)."""
    preds = rec(images)
    outs = []
    for p in preds:
        blocks = getattr(p, "text_lines", None) or getattr(p, "blocks", None) or []
        try: blocks = sorted(blocks, key=lambda b: getattr(b, "reading_order", 0))
        except Exception: pass
        parts = []
        for b in blocks:
            t = getattr(b, "html", None) or getattr(b, "text", None) or ""
            parts.append(strip_html(t))
        outs.append("\n".join(x for x in parts if x))
    return outs

t0 = time.time()
lines_out, pages_out = {}, {}
for i, fn in enumerate(GT):
    img = Image.open(IMG_DIR / fn).convert("RGB")
    crops = []
    for l in GT[fn]["lines"]:
        x0, y0, x1, y1 = l["bbox"]; pad = 12
        crops.append(img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.size[0], x1 + pad), min(img.size[1], y1 + pad))))
    t = time.time()
    try:
        texts = []
        for k in range(0, len(crops), 16):
            texts += read_many(crops[k:k+16])
    except Exception as e:
        import traceback; traceback.print_exc(); texts = [""] * len(crops); print("  line err", fn, e, flush=True)
    lines_out[fn] = [{"text": t.replace("\n", " ").strip(), "bbox": l["bbox"]} for t, l in zip(texts, GT[fn]["lines"])]
    save("surya_zero_lines.json", lines_out); ta = time.time() - t
    t = time.time()
    w, h = img.size
    try:
        parts = read_many([img.crop((0, 0, w // 2, h)), img.crop((w // 2, 0, w, h))])
    except Exception as e:
        import traceback; traceback.print_exc(); parts = ["", ""]; print("  page err", fn, e, flush=True)
    pages_out[fn] = "\n".join(parts); save("surya_zero_pages.json", pages_out)
    print(f"[surya {i+1}/20] {fn}: lines {len(crops)} in {ta:.0f}s | pages {time.time()-t:.0f}s", flush=True)
timings["surya_read"] = time.time() - t0; save("timings.json", timings)
print("ALL DONE", timings)
srv.terminate()
