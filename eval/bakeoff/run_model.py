#!/usr/bin/env python
"""Run an OCR model over the sealed test set and save predictions for score.py.

Protocol: every notebook spread is split into LEFT and RIGHT page images (like a
single-page scan from a school printer). Each half is transcribed separately; the two
outputs are concatenated into one prediction per spread. Whole-spread mode is available
with --no-split for comparison.

Usage:
  run_model.py --model glm      --out eval/runs/glm_zero.json
  run_model.py --model deepseek --out eval/runs/deepseek_zero.json [--prompt free|ground]
  run_model.py --model glm --limit 2   # smoke test on 2 pages
"""
import argparse, json, os, re, sys, time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "eval/testset_v1/ru_pages/ground_truth.json"
IMG_DIR = ROOT / "eval/testset_v1/ru_pages/images"
MODELS = ROOT / "models"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def halves(img: Image.Image):
    w, h = img.size
    return [("L", img.crop((0, 0, w // 2, h))), ("R", img.crop((w // 2, 0, w, h)))]


# ----------------------------------------------------------------------------- GLM-OCR
def load_glm():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    path = str(MODELS / "GLM-OCR")
    proc = AutoProcessor.from_pretrained(path)
    dtype = torch.float16 if DEVICE == "mps" else "auto"
    model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=dtype).to(DEVICE).eval()
    return proc, model


def run_glm(bundle, img: Image.Image, prompt="Text Recognition:", max_new_tokens=2048):
    proc, model = bundle
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


# ------------------------------------------------------------------------ DeepSeek-OCR-2
def load_deepseek():
    from transformers import AutoModel, AutoTokenizer
    # transformers >= 5 renamed DeepseekV2MoE -> DeepseekV2Moe; the model's custom code imports the old name.
    import transformers.models.deepseek_v2.modeling_deepseek_v2 as _dsv2
    if not hasattr(_dsv2, "DeepseekV2MoE") and hasattr(_dsv2, "DeepseekV2Moe"):
        _dsv2.DeepseekV2MoE = _dsv2.DeepseekV2Moe
    path = str(MODELS / "DeepSeek-OCR-2")
    if DEVICE != "cuda":
        # The model's infer() hard-codes .cuda(); route those calls to our device instead.
        torch.Tensor.cuda = lambda self, *a, **k: self.to(DEVICE)  # type: ignore[assignment]
        torch.nn.Module.cuda = lambda self, *a, **k: self.to(DEVICE)  # type: ignore[assignment]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, use_safetensors=True,
                                      _attn_implementation="eager" if DEVICE != "cuda" else "flash_attention_2")
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float16
    model = model.eval().to(DEVICE).to(dtype)
    return tok, model


DS_PROMPTS = {"free": "<image>\nFree OCR. ", "ground": "<image>\n<|grounding|>OCR this image. "}
REF_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>", re.S)


def run_deepseek(bundle, img: Image.Image, prompt="free", tmp_dir="/tmp/bilimai_ds"):
    tok, model = bundle
    os.makedirs(tmp_dir, exist_ok=True)
    p = os.path.join(tmp_dir, "in.png"); img.save(p)
    import io, contextlib
    buf = io.StringIO()
    with torch.no_grad(), contextlib.redirect_stdout(buf):
        res = model.infer(tok, prompt=DS_PROMPTS[prompt], image_file=p, output_path=tmp_dir,
                          base_size=1024, image_size=768, crop_mode=True, save_results=False, eval_mode=True)
    text = res if isinstance(res, str) else buf.getvalue()
    if prompt == "ground":
        # grounding output: <|ref|>text<|/ref|><|det|>[[x0,y0,x1,y1]]<|/det|> with coords on a 0-999 grid
        w, h = img.size
        lines = []
        for m in REF_RE.finditer(text):
            t = m.group(1).strip(); x0, y0, x1, y1 = (int(m.group(i)) for i in range(2, 6))
            lines.append({"text": t, "bbox": [x0 / 999 * w, y0 / 999 * h, x1 / 999 * w, y1 / 999 * h]})
        return lines if lines else re.sub(r"<\|.*?\|>", "", text).strip()
    return re.sub(r"<\|.*?\|>", "", text).strip()


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["glm", "deepseek"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=None, help="deepseek: free|ground ; glm: prompt text")
    ap.add_argument("--no-split", action="store_true", help="feed the whole spread instead of halves")
    ap.add_argument("--lines", action="store_true",
                    help="oracle-layout mode: read each ground-truth line crop separately (pure reading ability)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    gt = json.load(open(GT, encoding="utf-8"))
    files = list(gt.keys())[: a.limit or None]
    print(f"device={DEVICE} model={a.model} pages={len(files)} split={not a.no_split}", flush=True)

    t0 = time.time()
    if a.model == "glm":
        bundle = load_glm(); run = lambda im: run_glm(bundle, im, a.prompt or "Text Recognition:")
    else:
        bundle = load_deepseek(); run = lambda im: run_deepseek(bundle, im, a.prompt or "free")
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    out_path = Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    preds = json.load(open(out_path)) if out_path.exists() else {}
    for i, fn in enumerate(files):
        if fn in preds:
            continue
        img = Image.open(IMG_DIR / fn).convert("RGB")
        t1 = time.time()
        if a.lines:
            pad = 12; out_lines = []
            for l in gt[fn]["lines"]:
                x0, y0, x1, y1 = l["bbox"]
                crop = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.size[0], x1 + pad), min(img.size[1], y1 + pad)))
                r = run(crop)
                txt = r if isinstance(r, str) else " ".join(d["text"] for d in r)
                out_lines.append({"text": txt.replace("\n", " ").strip(), "bbox": l["bbox"]})
            preds[fn] = out_lines
            json.dump(preds, open(out_path, "w"), ensure_ascii=False, indent=1)
            print(f"[{i+1}/{len(files)}] {fn}: {len(out_lines)} lines in {time.time()-t1:.0f}s", flush=True)
            continue
        parts = [("ALL", img)] if a.no_split else halves(img)
        pieces = []
        for tag, part in parts:
            r = run(part)
            if isinstance(r, list) and tag == "R":       # shift right-half boxes back into spread coords
                off = img.size[0] // 2
                r = [{"text": d["text"], "bbox": [d["bbox"][0] + off, d["bbox"][1], d["bbox"][2] + off, d["bbox"][3]]} for d in r]
            pieces.append(r)
        if all(isinstance(p, list) for p in pieces):
            pred = [d for p in pieces for d in p]
        else:
            pred = "\n".join(p if isinstance(p, str) else "\n".join(d["text"] for d in p) for p in pieces)
        preds[fn] = pred
        json.dump(preds, open(out_path, "w"), ensure_ascii=False, indent=1)   # checkpoint every page
        n = len(pred) if isinstance(pred, list) else pred.count("\n") + 1
        print(f"[{i+1}/{len(files)}] {fn}: {n} lines in {time.time()-t1:.0f}s", flush=True)
    print(f"done in {time.time()-t0:.0f}s → {out_path}")


if __name__ == "__main__":
    main()
