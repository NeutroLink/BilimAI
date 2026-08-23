#!/usr/bin/env python
"""Can Qwen read handwritten maths, where GLM is blind? 2026-08-23.

E2.10 step 1 measured the product reader (GLM-OCR + v5) on the MathWriting excerpt and found it math-blind:
**mean CER 0.491, 2/14 exact** on the linear-ASCII subset, and on 2-D layout it both collapses structure
(fractions and roots vanish) and TRANSLITERATES maths into Russian letters — «\\Phi=\\int E(...)» came back
as «ф= I (ёс. дд)». That is the RU fine-tune's bias: GLM was trained to emit Russian text and cannot express
a fraction at all.

Qwen3-VL is a general model that was never narrowed to Russian, and it can emit LaTeX. So the interesting
question is not "is Qwen's CER lower" but "can Qwen express 2-D maths at all" — a capability question GLM
could not even be asked.

Two arms, on the SAME 300 strips and the SAME labels as the GLM run (out/math_baseline/strips/,
eval/runs/math_baseline_v0.json), so the comparison is paired, not two independent samples:
  A. plain transcription, scored on the 14 linear-ASCII labels — directly comparable to GLM's 0.491.
  B. LaTeX instruction, scored on all 300 — CER on the LaTeX string plus exact-match after normalisation.

  eval/.venv-mlx/bin/python eval/math/qwen_math_baseline.py [--n 300]
"""
import argparse, json, re, sys, time, unicodedata
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "models/Qwen3-VL-4B-Instruct-MLX-8bit"))
ap.add_argument("--strips", default=str(ROOT / "out/math_baseline/strips"))
ap.add_argument("--base-run", default=str(ROOT / "eval/runs/math_baseline_v0.json"))
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--out", default=str(ROOT / "eval/runs/qwen_math_baseline.json"))
a = ap.parse_args()

PLAIN = "Text Recognition:"
LATEX = ("Look at this handwritten mathematical expression.\n"
         "Write it out in LaTeX, exactly as written. Return ONLY the LaTeX, no explanation, no $ delimiters.")


def cer(ref, hyp):
    ref, hyp = unicodedata.normalize("NFC", ref), unicodedata.normalize("NFC", hyp)
    d = list(range(len(hyp) + 1))
    for i, cr in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, ch in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (cr != ch))
    return d[-1] / max(1, len(ref))


def norm_latex(s):
    """Whitespace and delimiter noise only — never rewrite the maths itself."""
    s = s.strip()
    s = re.sub(r"^\$+|\$+$", "", s.strip())
    s = re.sub(r"^\\\(|\\\)$", "", s.strip())
    s = re.sub(r"^```(?:latex)?|```$", "", s.strip())
    s = s.replace("\\left", "").replace("\\right", "")
    return re.sub(r"\s+", "", s)


base = json.load(open(a.base_run, encoding="utf-8"))
recs = base["records"][: a.n]
print(f"{len(recs)} strips | GLM reference: linear CER {base['summary']['linear_mean_cer']:.4f}, "
      f"{base['summary']['linear_exact']}/{base['summary']['linear_n']} exact", flush=True)

from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
print("loading", a.model, flush=True)
model, processor = load(a.model)
cfg = model.config
THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


def ask(img, prompt, max_tokens=192):
    fmt = apply_chat_template(processor, cfg, prompt, num_images=1)
    o = generate(model, processor, fmt, [img], max_tokens=max_tokens, verbose=False,
                 repetition_penalty=1.05, repetition_context_size=256, temperature=0.0)
    return THINK.sub("", (o if isinstance(o, str) else getattr(o, "text", str(o))), count=1).strip()


out = []
t0 = time.time()
linear = [r for r in recs if r["linear"]]
# ---- ARM A: the 14 linear-ASCII labels, plain prompt — like-for-like with GLM -------------------
a_cers = []
for r in linear:
    img = Image.open(Path(a.strips) / r["file"]).convert("RGB")
    hyp = ask(img, PLAIN)
    ref = re.sub(r"\s+", "", r["label"])
    c = cer(ref, re.sub(r"\s+", "", hyp))
    a_cers.append(c)
    out.append({"arm": "plain", "file": r["file"], "label": r["label"], "pred": hyp, "cer": round(c, 4)})
print(f"ARM A (plain, n={len(a_cers)}): mean CER {np.mean(a_cers):.4f}  exact "
      f"{sum(1 for c in a_cers if c == 0)}/{len(a_cers)}   [GLM: 0.4912, 2/14]", flush=True)

# ---- ARM B: LaTeX instruction on all 300 --------------------------------------------------------
b_cers, b_exact, b_lin = [], 0, []
for k, r in enumerate(recs):
    img = Image.open(Path(a.strips) / r["file"]).convert("RGB")
    hyp = ask(img, LATEX, max_tokens=256)
    ref_n, hyp_n = norm_latex(r["label"]), norm_latex(hyp)
    c = cer(ref_n, hyp_n)
    b_cers.append(c); b_exact += (ref_n == hyp_n)
    if r["linear"]: b_lin.append(c)
    out.append({"arm": "latex", "file": r["file"], "label": r["label"], "pred": hyp,
                "linear": r["linear"], "cer": round(c, 4), "exact": ref_n == hyp_n})
    if k % 50 == 0:
        print(f"  latex {k+1}/{len(recs)} cer {c:.3f} | {hyp[:60]}", flush=True)

summary = {"n": len(recs), "seconds": round(time.time()-t0, 1),
           "arm_plain_linear": {"n": len(a_cers), "mean_cer": round(float(np.mean(a_cers)), 4),
                                "exact": int(sum(1 for c in a_cers if c == 0))},
           "arm_latex_all": {"n": len(b_cers), "mean_cer": round(float(np.mean(b_cers)), 4),
                             "median_cer": round(float(np.median(b_cers)), 4), "exact": int(b_exact),
                             "exact_pct": round(100*b_exact/max(1, len(b_cers)), 1)},
           "arm_latex_linear_subset": {"n": len(b_lin),
                                       "mean_cer": round(float(np.mean(b_lin)), 4) if b_lin else None},
           "glm_reference": {"linear_mean_cer": base["summary"]["linear_mean_cer"],
                             "linear_exact": base["summary"]["linear_exact"],
                             "linear_n": base["summary"]["linear_n"],
                             "note": "GLM cannot express a fraction at all — no LaTeX arm exists for it"}}
json.dump({"summary": summary, "records": out}, open(a.out, "w"), ensure_ascii=False, indent=1)
print("\n=== QWEN MATH BASELINE ===")
print(json.dumps(summary, indent=1, ensure_ascii=False))
print(f"\n-> {a.out}")
