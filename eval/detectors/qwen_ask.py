#!/usr/bin/env python
"""Ask Qwen anything about a page, and see exactly what it says. 2026-08-23.

    eval/.venv-mlx/bin/python eval/detectors/qwen_ask.py <image> "<your prompt>"

    # a prompt too long for the shell:
    eval/.venv-mlx/bin/python eval/detectors/qwen_ask.py <image> --prompt-file my_prompt.txt

    # split a two-page spread (usually WORSE — measured 2026-08-23; whole page won on 2806)
    eval/.venv-mlx/bin/python eval/detectors/qwen_ask.py <image> "..." --half

Prints the model's raw reply. If the reply contains boxes, also writes an annotated PNG next to the
image so you can see where it thinks the lines are, and reports how many boxes it found.

Options worth knowing:
  --max-side N     how big an image the model gets (default 2800). More pixels = finer detail, slower.
  --max-tokens N   how long a reply it may produce (default 4000). Too low truncates a dense page.
  --temp T         0.0 = deterministic (default). Raise only if you want variety.
  --save-json P    keep the parsed boxes for scoring later.
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("prompt", nargs="?", default=None)
ap.add_argument("--prompt-file", default=None)
ap.add_argument("--model", default=str(ROOT / "models/Qwen3-VL-4B-Instruct-MLX-8bit"))
ap.add_argument("--max-side", type=int, default=2800)
ap.add_argument("--max-tokens", type=int, default=4000)
ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--rep-penalty", type=float, default=1.05)
ap.add_argument("--half", action="store_true", help="split a two-page spread down the middle")
ap.add_argument("--save-json", default=None)
a = ap.parse_args()

prompt = Path(a.prompt_file).read_text(encoding="utf-8") if a.prompt_file else a.prompt
if not prompt:
    sys.exit("give a prompt, either as an argument or with --prompt-file")

from PIL import Image, ImageDraw
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

print(f"model  : {a.model}")
print(f"image  : {a.image}")
print(f"prompt : {prompt.strip()[:300]}{'...' if len(prompt) > 300 else ''}\n", flush=True)
model, processor = load(a.model)

im0 = Image.open(a.image).convert("RGB")
W, H = im0.size
regions = [("full", 0, 0, W, H)] if not a.half else [("left", 0, 0, W // 2, H), ("right", W // 2, 0, W, H)]

all_boxes, replies = [], []
for tag, rx0, ry0, rx1, ry1 in regions:
    sub = im0.crop((rx0, ry0, rx1, ry1)); sw, sh = sub.size
    sc = min(1.0, a.max_side / max(sw, sh))
    snd = sub.resize((int(sw * sc), int(sh * sc)))
    t0 = time.time()
    fmt = apply_chat_template(processor, model.config, prompt, num_images=1)
    res = generate(model, processor, fmt, [snd], max_tokens=a.max_tokens, verbose=False,
                   temperature=a.temp, repetition_penalty=a.rep_penalty, repetition_context_size=256)
    raw = res if isinstance(res, str) else getattr(res, "text", str(res))
    replies.append((tag, raw))
    print("=" * 70)
    print(f"[{tag}]  {time.time()-t0:.0f}s   {len(raw)} characters")
    print("=" * 70)
    print(raw)
    print()
    # if it looks like boxes, map them back to page pixels (Qwen emits 0-1000 normalised — measured)
    i, j = raw.find("["), raw.rfind("]")
    items = []
    if i != -1 and j > i:
        try:
            items = [x for x in json.loads(raw[i:j + 1]) if isinstance(x, dict)]
        except Exception:
            items = [json.loads(m.group(0)) for m in re.finditer(r"\{[^{}]*\}", raw)
                     if m.group(0).count(":") >= 2] if "bbox" in raw else []
    for it in items:
        b = it.get("bbox_2d") or it.get("bbox")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            all_boxes.append(([b[0] / 1000 * sw + rx0, b[1] / 1000 * sh + ry0,
                               b[2] / 1000 * sw + rx0, b[3] / 1000 * sh + ry0], str(it.get("text", ""))))

if all_boxes:
    vis = im0.copy(); d = ImageDraw.Draw(vis)
    for b, _ in all_boxes:
        d.rectangle(b, outline=(0, 90, 235), width=max(3, W // 900))
    out = Path(a.image).with_name(Path(a.image).stem + "_qwen.png")
    vis.save(out)
    print(f"{len(all_boxes)} boxes -> {out}")
    if a.save_json:
        json.dump([{"bbox": b, "text": t} for b, t in all_boxes],
                  open(a.save_json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"parsed boxes -> {a.save_json}")
else:
    print("(no boxes in the reply — that is fine if you asked a question rather than for boxes)")
