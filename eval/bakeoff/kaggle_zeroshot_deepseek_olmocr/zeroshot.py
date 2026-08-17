# v4 — request T4 accelerator (P100 unsupported by Kaggle torch)
"""BilimAI M1 zero-shot on Kaggle GPU: DeepSeek-OCR-2 and olmOCR-2 over the sealed RU exam.

Runs on Kaggle (T4 x2, transformers 4.x). Inputs: dataset jahongir713/bilimai-exam-ru-v1
(20 spreads + ground_truth.json). Outputs (in /kaggle/working):
  deepseek_zero_lines.json   oracle line crops, "Free OCR"
  deepseek_zero_pages.json   left/right page halves, "Free OCR"
  deepseek_zero_ground.json  page halves with <|grounding|> -> lines with bboxes (spread coords)
  olmocr_zero_lines.json / olmocr_zero_pages.json
  timings.json
Score locally with eval/score.py.
"""
import json, os, re, subprocess, sys, time, glob, io, contextlib
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "accelerate", "bitsandbytes",
                "einops", "addict", "easydict", "matplotlib", "qwen-vl-utils"], check=True)

import torch
from PIL import Image

_gt = glob.glob("/kaggle/input/**/ground_truth.json", recursive=True)
assert _gt, f"ground_truth.json not found under /kaggle/input: {glob.glob('/kaggle/input/**', recursive=True)[:20]}"
DATA = Path(_gt[0]).parent
IMG_DIR = DATA / "images"
GT = json.load(open(DATA / "ground_truth.json", encoding="utf-8"))
OUT = Path("/kaggle/working")
DEV = "cuda"
print("GPU:", torch.cuda.get_device_name(0), "| n:", torch.cuda.device_count(), flush=True)
timings = {}


def halves(img):
    w, h = img.size
    return [("L", img.crop((0, 0, w // 2, h))), ("R", img.crop((w // 2, 0, w, h)))]


def line_crops(fn, img, pad=12):
    for l in GT[fn]["lines"]:
        x0, y0, x1, y1 = l["bbox"]
        yield l, img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.size[0], x1 + pad), min(img.size[1], y1 + pad)))


def save(name, obj):
    json.dump(obj, open(OUT / name, "w"), ensure_ascii=False, indent=1)


def run_protocols(tag, read_fn, ground_fn=None):
    """read_fn(img)->str ; ground_fn(img)->list[{text,bbox}] (optional)."""
    lines_out, pages_out, ground_out = {}, {}, {}
    t0 = time.time()
    for i, fn in enumerate(GT):
        img = Image.open(IMG_DIR / fn).convert("RGB")
        # protocol A: oracle line crops
        t = time.time(); L = []
        for l, crop in line_crops(fn, img):
            try:
                txt = read_fn(crop)
            except Exception as e:
                txt = ""; print("  line err", fn, e, flush=True)
            L.append({"text": txt.replace("\n", " ").strip(), "bbox": l["bbox"]})
        lines_out[fn] = L; save(f"{tag}_zero_lines.json", lines_out)
        ta = time.time() - t
        # protocol B: page halves, plain text
        t = time.time(); parts = []
        for side, half in halves(img):
            try:
                parts.append(read_fn(half))
            except Exception as e:
                parts.append(""); print("  page err", fn, side, e, flush=True)
        pages_out[fn] = "\n".join(parts); save(f"{tag}_zero_pages.json", pages_out)
        tb = time.time() - t
        # protocol C: grounding (boxes) on halves
        tc = 0
        if ground_fn:
            t = time.time(); G = []
            off = img.size[0] // 2
            for side, half in halves(img):
                try:
                    r = ground_fn(half)
                except Exception as e:
                    r = []; print("  ground err", fn, side, e, flush=True)
                for d in r:
                    b = d["bbox"]
                    if side == "R": b = [b[0] + off, b[1], b[2] + off, b[3]]
                    G.append({"text": d["text"], "bbox": b})
            ground_out[fn] = G; save(f"{tag}_zero_ground.json", ground_out)
            tc = time.time() - t
        print(f"[{tag} {i+1}/20] {fn}: lines {len(L)} in {ta:.0f}s | pages {tb:.0f}s | ground {tc:.0f}s", flush=True)
    timings[tag] = time.time() - t0; save("timings.json", timings)


# ============================================================ DeepSeek-OCR-2
def deepseek():
    from transformers import AutoModel, AutoTokenizer
    name = "unsloth/DeepSeek-OCR-2"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(name, trust_remote_code=True, use_safetensors=True,
                                      _attn_implementation="eager").eval().cuda().to(torch.float16)  # T4: no bf16, no flash-attn
    tmp = "/kaggle/working/_ds"; os.makedirs(tmp, exist_ok=True)
    REF = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>", re.S)

    def _infer(img, prompt, crop_mode):
        p = f"{tmp}/in.png"; img.save(p)
        buf = io.StringIO()
        with torch.no_grad(), contextlib.redirect_stdout(buf):
            res = model.infer(tok, prompt=prompt, image_file=p, output_path=tmp, base_size=1024, image_size=768,
                              crop_mode=crop_mode, save_results=False, eval_mode=True)
        return res if isinstance(res, str) else buf.getvalue()

    def read(img):
        small = img.size[1] < 400          # line crop: no tiling
        return re.sub(r"<\|.*?\|>", "", _infer(img, "<image>\nFree OCR. ", crop_mode=not small)).strip()

    def ground(img):
        text = _infer(img, "<image>\n<|grounding|>OCR this image. ", crop_mode=True)
        w, h = img.size; out = []
        for m in REF.finditer(text):
            x0, y0, x1, y1 = (int(m.group(k)) for k in range(2, 6))
            out.append({"text": m.group(1).strip(), "bbox": [x0 / 999 * w, y0 / 999 * h, x1 / 999 * w, y1 / 999 * h]})
        return out

    run_protocols("deepseek", read, ground)
    del model; torch.cuda.empty_cache()


# ============================================================ olmOCR-2 (Qwen2.5-VL-7B), 4-bit on T4
def olmocr():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLM
    except ImportError:
        VLM = Qwen2VLForConditionalGeneration
    name = "allenai/olmOCR-2-7B-1025"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    proc = AutoProcessor.from_pretrained(name)
    model = VLM.from_pretrained(name, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16).eval()
    PROMPT = ("Below is an image of a handwritten school notebook page. Transcribe the handwritten text exactly as "
              "written, in Russian, line by line. Output only the text.")

    def read(img, max_new=None):
        # olmOCR expects the long side rendered at ~1288 px
        w, h = img.size; s = 1288 / max(w, h)
        if s < 1: img = img.resize((int(w * s), int(h * s)))
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new or (96 if img.size[1] < 400 else 1536), do_sample=False)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    run_protocols("olmocr", read)
    del model; torch.cuda.empty_cache()


for fn in (deepseek, olmocr):
    try:
        fn()
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"!!! {fn.__name__} failed: {e}", flush=True)
print("ALL DONE", timings)
