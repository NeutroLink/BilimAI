#!/usr/bin/env python
"""Can one VLM replace BOTH the detector and the reader? (2026-08-22)

Every Qwen test in this project so far was handed pre-cut line strips from ORACLE boxes — the model's
own layout ability was never exercised. Qwen3-VL-4B-Instruct's card claims "stronger 2D grounding" and
"improved long-document structure parsing", which is exactly a one-pass replacement for
RPDetector + GLM-OCR. This measures whether that survives contact with children's Russian cursive.

Asks the model for every text line as {"bbox_2d": [x0,y0,x1,y1], "text": "..."} and then scores:
  BOXES  against ai-forever's human line boxes (matched / missed / merged / split, det_failure rules)
  TEXT   char-weighted CER on the lines it matched

COORDINATE CONVENTION IS MEASURED, NOT ASSUMED. Qwen2-VL emitted 0-1000 normalised coordinates,
Qwen2.5-VL absolute pixels, and the Qwen3-VL card does not state which. So the script scores BOTH
interpretations and reports which one fits — guessing here would silently produce "the model can't
detect anything" when the truth is that the boxes were in the wrong units.

  eval/.venv-mlx/bin/python eval/detectors/qwen_grounding_probe.py --pages 3 --max-side 1600

PRE-FILTERING WAS TRIED AND REMOVED (2026-08-23). A --clean flag fed the model a filtered image —
pupil ink in its original colours, paper ruling / printed margin / page-edge shadow / teacher's red pen
whitened out (bilimai/ink.py). Measured on three separate sets, it never helped and mildly hurt:

    5 hardest pages, 335 lines   raw 186 (55.5 %)   filtered 181 (54.0 %)
    2718 (45 lines)              raw  45 (100 %)    filtered  44 (98 %),  text CER 0.488 -> 0.513
    my_notebook p-01             raw  19 lines      filtered  19 lines,   reading visibly worse

Plausible reason: whitening discards information the model uses — paper texture, and the faint stroke
fragments the filter does not quite catch. bilimai/ink.py stays, because the guardrail in
bilimai/guards.py needs those masks; it just no longer touches what the model sees.
"""
import argparse, json, re, sys, time, unicodedata
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "models/Qwen3-VL-4B-Instruct-MLX-8bit"))
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--images", default=str(ROOT / "eval/testset_v2/ru_pages/images"))
ap.add_argument("--pages", type=int, default=3)
ap.add_argument("--files", default="", help="comma-separated page filenames instead of the first --pages")
ap.add_argument("--max-side", type=int, default=1600, help="downscale before sending; coords are mapped back")
ap.add_argument("--half", action="store_true", help="send left/right halves separately (these are two-page spreads)")
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--prompt-variant", default="base", choices=["base", "fragments", "complete"])
ap.add_argument("--rep-penalty", type=float, default=1.05,
                help="without this the model degenerates into a repetition loop on dense handwriting "
                     "(observed 2026-08-22: 8,000 chars of the same invented phrase from one page)")
ap.add_argument("--out", default=str(ROOT / "eval/runs/qwen_grounding_probe.json"))
a = ap.parse_args()

# The prompts live in their own module so the probe, the R7 trainer and serving cannot drift apart
# (a reworded instruction moved this script's own headline 2.4x). Byte-identical to what was measured.
from qwen_grounding_probe_prompts import PROMPTS
try:
    from bilimai.guards import prompt_fingerprint as _pf
    _FP = _pf(PROMPTS[a.prompt_variant])
except Exception:
    _FP = None

def cer(ref, hyp):
    ref, hyp = unicodedata.normalize("NFC", ref), unicodedata.normalize("NFC", hyp)
    d = list(range(len(hyp) + 1))
    for i, cr in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, ch in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (cr != ch))
    return d[-1] / max(1, len(ref))

def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return i / max(1e-9, (p[2]-p[0])*(p[3]-p[1]) + (g[2]-g[0])*(g[3]-g[1]) - i)

def parse_items(txt):
    r"""Pull out [{bbox_2d, text}, ...].

    v2: the first version used a NON-GREEDY r"\[.*?\]", which matches the inner coordinate array
    [701, 103, 998, 146] and never the outer list — it reported 0 items from perfectly good output.
    Take the outermost brackets instead, and fall back to salvaging individual {...} objects so a
    reply truncated by max_tokens still yields every complete line it managed to emit."""
    items = []
    i, j = txt.find("["), txt.rfind("]")
    if i != -1 and j > i:
        try:
            v = json.loads(txt[i:j + 1])
            if isinstance(v, list):
                items = [x for x in v if isinstance(x, dict)]
        except Exception:
            items = []
    if not items:                                    # truncated or malformed -> salvage per object
        for m in re.finditer(r"\{[^{}]*\}", txt, re.S):
            try:
                x = json.loads(m.group(0))
                if isinstance(x, dict):
                    items.append(x)
            except Exception:
                pass
    out = []
    for it in items:
        b = it.get("bbox_2d") or it.get("bbox") or it.get("box")
        t = it.get("text") or it.get("text_content") or it.get("label") or ""
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                out.append(([float(v) for v in b], str(t)))
            except Exception:
                pass
    return out, items


from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

print(f"loading {a.model} ...", flush=True)
model, processor = load(a.model)
cfg = model.config
gt = json.load(open(a.gt, encoding="utf-8"))
files = [f.strip() for f in a.files.split(",") if f.strip()] or sorted(gt)[: a.pages]
records = []

for fn in files:
    im0 = Image.open(Path(a.images) / fn).convert("RGB")
    W0, H0 = im0.size
    regions = [("full", 0, 0, W0, H0)] if not a.half else [
        ("left", 0, 0, W0 // 2, H0), ("right", W0 // 2, 0, W0, H0)]
    for tag, rx0, ry0, rx1, ry1 in regions:
        sub = im0.crop((rx0, ry0, rx1, ry1))
        sw, sh = sub.size
        sc = min(1.0, a.max_side / max(sw, sh))
        snd = sub.resize((int(sw * sc), int(sh * sc)))
        t0 = time.time()
        fmt = apply_chat_template(processor, cfg, PROMPTS[a.prompt_variant], num_images=1)
        res = generate(model, processor, fmt, [snd], max_tokens=a.max_tokens, verbose=False,
                       repetition_penalty=a.rep_penalty, repetition_context_size=256, temperature=0.0)
        raw = res if isinstance(res, str) else getattr(res, "text", str(res))
        dt = time.time() - t0
        items, _ = parse_items(raw)
        print(f"{fn} [{tag}] {dt:.0f}s -> {len(items)} items from {len(raw)} chars", flush=True)
        if not items:
            print("   RAW (first 400):", raw[:400].replace("\n", " "), flush=True)
        # PROVENANCE (2026-08-23): the prompt is part of the model — `complete` vs `fragments` moved this
        # script's own headline 2.4x. Runs saved before today recorded neither, so eval/runs/qwen_hard20.json
        # can no longer be attributed to a prompt and its junk-box rate is uninterpretable. Never again.
        records.append({"file": fn, "region": tag, "sent_size": snd.size, "sub_origin": [rx0, ry0],
                        "sub_size": [sw, sh], "scale": sc, "seconds": dt,
                        "prompt_variant": a.prompt_variant, "prompt_fingerprint": _FP,
                        "model": str(a.model), "max_side": a.max_side, "rep_penalty": a.rep_penalty,
                        "items": [{"bbox": b, "text": t} for b, t in items], "raw": raw[:4000]})

json.dump(records, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"\n-> {a.out}")

# ---- score under BOTH coordinate conventions ---------------------------------------------------
print("\n=== which coordinate convention fits? ===")
for conv in ("absolute-in-sent-image", "normalised-0-1000"):
    tot_m = tot_g = 0; ious = []; cers = []; chars = 0
    for r in records:
        page = gt[r["file"]]
        G = [l["bbox"] for l in page["lines"]]
        T = [(l.get("text") or "").strip() for l in page["lines"]]
        sw, sh = r["sub_size"]; ox, oy = r["sub_origin"]; sc = r["scale"]
        P, PT = [], []
        for it in r["items"]:
            b = it["bbox"]
            if conv == "absolute-in-sent-image":
                q = [b[0] / sc + ox, b[1] / sc + oy, b[2] / sc + ox, b[3] / sc + oy]
            else:
                q = [b[0] / 1000 * sw + ox, b[1] / 1000 * sh + oy, b[2] / 1000 * sw + ox, b[3] / 1000 * sh + oy]
            if q[2] > q[0] and q[3] > q[1]:
                P.append(q); PT.append(it["text"])
        if not P: continue
        for gi, g in enumerate(G):
            if not (ox <= (g[0] + g[2]) / 2 <= ox + sw): continue
            tot_g += 1
            j = int(np.argmax([iou(p, g) for p in P])); best = iou(P[j], g)
            ious.append(best)
            if best >= 0.5:
                tot_m += 1
                if T[gi]:
                    cers.append((cer(T[gi], PT[j].strip()), len(T[gi]))); chars += len(T[gi])
    if tot_g:
        cw = sum(c * n for c, n in cers) / max(chars, 1) if cers else float("nan")
        print(f"  {conv:<26} matched {tot_m}/{tot_g} ({100*tot_m/tot_g:4.0f} %)  "
              f"median IoU {np.median(ious):.3f}  text CER on matched {cw:.4f}")
print("\nreference — RP production on this exam: matched 84 %, median best-IoU 0.81, "
      "reader CER 0.023 with human boxes")
