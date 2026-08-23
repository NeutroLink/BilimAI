#!/usr/bin/env python
"""Uzbek: does Qwen need our detector, or can it read a whole page by itself? 2026-08-23.

Founder question before committing money to an Uzbek round: "test qwen locally for how it does with
uzbek strips vs whole pages, then decide how to train it in uzbek".

Uzbek has no annotated pages and no real pupil pages at all — only the 239,905 font-rendered line strips
(E2.6b) plus 59,997 misspelled ones. So this composes pages from freshly generated strips. That is not a
workaround, it is the point: a COMPOSED page carries its own ground-truth boxes for free, which means the
page-level route (R7) can be extended to Uzbek without anyone labelling a single page.

FAIRNESS — the two conditions must differ ONLY in framing, or the comparison is worthless (the standing
lesson: "a negative result is only as good as the thing you tested"). So both arms see the SAME ink:
  * strips are rendered with PLAIN paper (`_paper` is temporarily forced to plain),
  * the page is one consistently ruled sheet, and each line's ink is transferred with a DARKEN blend.
Rendering each strip on its own ruled paper and stacking them would double the ruling and put seams
between lines — the page arm would then lose to an artefact of my own making, not to difficulty.

Two steps, because the two halves need different environments — rendering needs fontTools (eval/.venv) and
inference needs mlx_vlm (eval/.venv-mlx), and neither venv has both:

  eval/.venv/bin/python     eval/detectors/uz_page_probe.py --make            # build the pages, no model
  eval/.venv-mlx/bin/python eval/detectors/uz_page_probe.py --run --reuse     # score both arms
"""
import argparse, json, random, sys, time, unicodedata
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval" / "util"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "models/Qwen3-VL-4B-Instruct-MLX-8bit"))
ap.add_argument("--pages", type=int, default=4)
ap.add_argument("--lines", type=int, default=16, help="lines per composed page")
ap.add_argument("--strip-sample", type=int, default=48, help="how many single strips to read in the strip arm")
ap.add_argument("--fonts", default=str(ROOT / "data/fonts/uz_bridge"))
ap.add_argument("--corpus", default=str(ROOT / "data/raw/uzbek_news"))
ap.add_argument("--out", default=str(ROOT / "out/uz_page_probe"))
ap.add_argument("--runs", default=str(ROOT / "eval/runs/uz_page_probe.json"))
ap.add_argument("--max-side", type=int, default=1600)
ap.add_argument("--max-tokens", type=int, default=3072)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--make", action="store_true", help="generate the pages only, load no model")
ap.add_argument("--run", action="store_true", help="run both arms through the model")
ap.add_argument("--reuse", action="store_true",
                help="score the pages already in --out instead of regenerating (rendering needs fontTools,\n                      which lives in eval/.venv; inference needs mlx_vlm, which lives in eval/.venv-mlx)")
a = ap.parse_args()

import cv2
if not a.reuse:                       # fontTools/font rendering is only needed to BUILD the pages
    import make_uz_font_strips as G
    from bilimai.uzbek import normalize
sys.path.insert(0, str(ROOT / "eval" / "detectors"))
from qwen_grounding_probe_prompts import PAGE_PROMPT

OUT = Path(a.out); (OUT / "pages").mkdir(parents=True, exist_ok=True); (OUT / "strips").mkdir(exist_ok=True)

# ---- force plain paper for the strips so the page supplies the only ruling -----------------------
_orig_paper = G._paper if not a.reuse else None
def _white_paper(W, H, rng, baseline, th):
    """PURE white, no noise, no ruling — so the multiply blend below is exactly neutral outside the ink.

    First attempt used a near-white noisy paper and a darken blend. It left a faint rectangular halo around
    every line: a free, machine-visible hint of exactly where each box was. The page arm would have scored
    on an artefact I introduced. White + multiply leaves nothing behind."""
    return np.full((H, W, 3), 255, np.uint8), "plain"


def ruled_page(W, H, rng, spacing):
    base = (rng.randint(236, 248), rng.randint(242, 251), rng.randint(245, 253))
    pg = np.full((H, W, 3), base, np.uint8)
    noise = np.random.RandomState(rng.randrange(1 << 30)).normal(0, 2.5, (H, W, 1))
    pg = np.clip(pg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    col = (rng.randint(180, 215), rng.randint(140, 178), rng.randint(125, 165))
    for y in range(spacing, H, spacing):
        cv2.line(pg, (0, y), (W, y), col, 1)
    mx = int(W * rng.uniform(0.07, 0.11))
    cv2.line(pg, (mx, 0), (mx, H), (70, 70, 205), 2)                       # red margin, as in real notebooks
    return pg, mx


def make_pages():
    rng = random.Random(a.seed)
    fonts = G.load_fonts(Path(a.fonts))
    assert fonts, f"no fonts under {a.fonts}"
    want = a.pages * a.lines + 40
    lines = [t for t in G.corpus_lines(Path(a.corpus), random.Random(a.seed), want)][:want]
    lines = [t for t in lines if t and normalize(t)]
    print(f"corpus lines: {len(lines)}  fonts: {len(fonts)}", flush=True)

    G._paper = _white_paper
    pages, strips = [], []
    li = 0
    try:
        for p in range(a.pages):
            PW, PH = 1600, 1200
            sp = PH // (a.lines + 3)
            pg, mx = ruled_page(PW, PH, rng, sp)
            # one page = one pupil: a single writer, as a real notebook page would be
            writer = G.make_writer(rng.randrange(4000), fonts)
            got = []
            y = sp + int(sp * 0.15)
            while len(got) < a.lines and li < len(lines):
                txt = normalize(lines[li]); li += 1
                if not txt:
                    continue
                r = G.render_line(txt, fonts, random.Random(rng.randrange(1 << 30)), writer)
                if r is None:
                    continue
                strip, meta = r
                sh_ = int(sp * rng.uniform(0.80, 0.95))
                sc = sh_ / strip.shape[0]
                sw_ = max(8, int(strip.shape[1] * sc))
                if sw_ > PW - mx - 30:                                     # too wide — trim words until it fits
                    keep = txt
                    while sw_ > PW - mx - 30 and len(keep.split()) > 3:
                        keep = " ".join(keep.split()[:-1])
                        r2 = G.render_line(keep, fonts, random.Random(rng.randrange(1 << 30)), writer)
                        if r2 is None: break
                        strip, meta = r2; sc = sh_ / strip.shape[0]; sw_ = max(8, int(strip.shape[1] * sc))
                    txt = keep
                strip = cv2.resize(strip, (sw_, sh_), interpolation=cv2.INTER_AREA)
                x = mx + rng.randint(8, 30)
                if y + sh_ >= PH - sp: break
                reg = pg[y:y + sh_, x:x + sw_].astype(np.float32)
                # multiply blend: white (255) is exactly neutral, ink scales the page down. No halo, and the
                # page's own ruling and texture stay visible THROUGH the ink, as on real paper.
                pg[y:y + sh_, x:x + sw_] = np.clip(reg * (strip.astype(np.float32) / 255.0), 0, 255).astype(np.uint8)
                got.append({"bbox": [x, y, x + sw_, y + sh_], "text": txt})
                sname = f"uz_p{p:02d}_l{len(got):02d}.jpg"
                cv2.imwrite(str(OUT / "strips" / sname), strip)
                strips.append({"file": sname, "text": txt, "page": p})
                y += sp
            fn = f"uz_page_{p:02d}.jpg"
            cv2.imwrite(str(OUT / "pages" / fn), pg, [cv2.IMWRITE_JPEG_QUALITY, 92])
            pages.append({"file": fn, "w": PW, "h": PH, "lines": got, "writer": writer["id"],
                          "font": writer["font"]["name"]})
            print(f"{fn}: {len(got)} lines, writer {writer['id']} ({writer['font']['name']})", flush=True)
    finally:
        G._paper = _orig_paper
    json.dump({"pages": pages, "strips": strips}, open(OUT / "gt.json", "w"), ensure_ascii=False, indent=1)
    return pages, strips


# ---- scoring (same rules as eval/train/vast/round7_page.py) --------------------------------------
def cer(ref, hyp):
    ref, hyp = unicodedata.normalize("NFC", ref), unicodedata.normalize("NFC", hyp)
    d = list(range(len(hyp) + 1))
    for i, cr in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, ch in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (cr != ch))
    return d[-1] / max(1, len(ref))

# The oʻ/gʻ mark cannot be DRAWN by 58 of the 59 bridge fonts, so the renderer substitutes ‘/’ while the
# label keeps the official letter. Scoring strictly therefore charges the model for a character the ink does
# not contain — 65 % of this probe's whole error. Both numbers are reported: `_folded` is reading ability,
# the strict one is convention compliance, and `mark_*` says whether the distinction survives at all.
sys.path.insert(0, str(ROOT))
from bilimai.uzbek import fold_marks, mark_agreement


def cer_folded(ref, hyp):
    return cer(fold_marks(ref), fold_marks(hyp))


def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    i = max(0., x1 - x0) * max(0., y1 - y0)
    return i / max(1e-9, (p[2]-p[0])*(p[3]-p[1]) + (g[2]-g[0])*(g[3]-g[1]) - i)

import re
THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)
def parse_items(txt):
    txt = THINK.sub("", txt, count=1)
    items, i, j = [], txt.find("["), txt.rfind("]")
    if i != -1 and j > i:
        try:
            v = json.loads(txt[i:j+1]); items = [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []
        except Exception: items = []
    if not items:
        for m in re.finditer(r"\{[^{}]*\}", txt, re.S):
            try:
                x = json.loads(m.group(0))
                if isinstance(x, dict): items.append(x)
            except Exception: pass
    out = []
    for it in items:
        b = it.get("bbox_2d") or it.get("bbox") or it.get("box"); t = it.get("text") or ""
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try: out.append(([float(v) for v in b], str(t)))
            except Exception: pass
    return out


def main():
    if a.reuse:
        g = json.load(open(OUT / "gt.json", encoding="utf-8"))
        pages, strips = g["pages"], g["strips"]
        print(f"reusing {len(pages)} pages / {len(strips)} strips from {OUT}", flush=True)
    else:
        pages, strips = make_pages()
    print(f"\n-> {OUT}/pages ({len(pages)} pages), {OUT}/strips ({len(strips)} strips), {OUT}/gt.json")
    if a.make and not a.run:
        return

    from PIL import Image
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    print(f"loading {a.model} ...", flush=True)
    model, processor = load(a.model)
    cfg = model.config
    res = {"model": Path(a.model).name, "pages": [], "strips": []}

    # ---- ARM 1: strips, the way the production reader is called ---------------------------------
    STRIP_PROMPT = "Text Recognition:"
    sample = strips[: a.strip_sample]
    t0 = time.time(); scers = []; scers_f = []; mk_ok = mk_tot = 0
    for k, s in enumerate(sample):
        im = Image.open(OUT / "strips" / s["file"]).convert("RGB")
        fmt = apply_chat_template(processor, cfg, STRIP_PROMPT, num_images=1)
        o = generate(model, processor, fmt, [im], max_tokens=128, verbose=False,
                     repetition_penalty=1.05, repetition_context_size=256, temperature=0.0)
        hyp = THINK.sub("", (o if isinstance(o, str) else getattr(o, "text", str(o))), count=1).strip()
        c = cer(s["text"], hyp); cf = cer_folded(s["text"], hyp)
        mo, mt = mark_agreement(s["text"], hyp); mk_ok += mo; mk_tot += mt
        scers.append((c, len(s["text"]))); scers_f.append((cf, len(s["text"])))
        res["strips"].append({"file": s["file"], "ref": s["text"], "hyp": hyp,
                              "cer": round(c, 4), "cer_folded": round(cf, 4)})
        if k % 10 == 0: print(f"  strip {k+1}/{len(sample)} cer {c:.3f} (folded {cf:.3f}) | {hyp[:60]}", flush=True)
    ch = sum(n for _, n in scers)
    strip_cw = sum(c*n for c, n in scers)/max(ch, 1)
    strip_cw_f = sum(c*n for c, n in scers_f)/max(ch, 1)
    print(f"\nSTRIPS: n={len(scers)}  char-weighted CER {strip_cw:.4f} strict / {strip_cw_f:.4f} FOLDED"
          f"  median {np.median([c for c,_ in scers_f]):.4f}  mark agreement {mk_ok}/{mk_tot}"
          f"  ({time.time()-t0:.0f}s)", flush=True)

    # ---- ARM 2: whole page, boxes + text ---------------------------------------------------------
    matched = tot = pred_n = 0; ious = []; pcers = []; pcers_f = []; pch = 0; pmk_ok = pmk_tot = 0
    for p in pages:
        im = Image.open(OUT / "pages" / p["file"]).convert("RGB")
        W0, H0 = im.size
        sc = min(1.0, a.max_side / max(W0, H0))
        snd = im.resize((int(W0*sc), int(H0*sc))) if sc < 1.0 else im
        t1 = time.time()
        fmt = apply_chat_template(processor, cfg, PAGE_PROMPT, num_images=1)
        o = generate(model, processor, fmt, [snd], max_tokens=a.max_tokens, verbose=False,
                     repetition_penalty=1.05, repetition_context_size=256, temperature=0.0)
        raw = o if isinstance(o, str) else getattr(o, "text", str(o))
        items = parse_items(raw)
        res["pages"].append({"file": p["file"], "seconds": round(time.time()-t1, 1),
                             "n_items": len(items), "raw": raw[:6000]})
        G_ = [l["bbox"] for l in p["lines"]]; T_ = [l["text"] for l in p["lines"]]
        P, PT = [], []
        for b, t in items:
            q = [b[0]/1000*W0, b[1]/1000*H0, b[2]/1000*W0, b[3]/1000*H0]
            if q[2] > q[0] and q[3] > q[1]: P.append(q); PT.append(t)
        pred_n += len(P); tot += len(G_)
        print(f"  {p['file']}: {len(items)} items in {time.time()-t1:.0f}s", flush=True)
        if not P: continue
        for gi, g in enumerate(G_):
            j = int(np.argmax([iou(q, g) for q in P])); best = iou(P[j], g); ious.append(best)
            if best >= 0.5:
                matched += 1
                h = PT[j].strip()
                pcers.append((cer(T_[gi], h), len(T_[gi])))
                pcers_f.append((cer_folded(T_[gi], h), len(T_[gi])))
                mo, mt = mark_agreement(T_[gi], h); pmk_ok += mo; pmk_tot += mt
                pch += len(T_[gi])
    page_cw = sum(c*n for c, n in pcers)/max(pch, 1) if pcers else None
    page_cw_f = sum(c*n for c, n in pcers_f)/max(pch, 1) if pcers_f else None
    summary = {"strip_cer_charweighted": round(strip_cw, 4),
               "strip_cer_charweighted_folded": round(strip_cw_f, 4),
               "strip_mark_agreement": f"{mk_ok}/{mk_tot}", "strip_n": len(scers),
               "page_gt_lines": tot, "page_pred_lines": pred_n, "page_matched": matched,
               "page_matched_pct": round(100*matched/max(1, tot), 1),
               "page_median_iou": round(float(np.median(ious)), 3) if ious else None,
               "page_cer_charweighted": round(page_cw, 4) if page_cw is not None else None,
               "page_cer_charweighted_folded": round(page_cw_f, 4) if page_cw_f is not None else None,
               "page_mark_agreement": f"{pmk_ok}/{pmk_tot}",
               "headline": "quote the FOLDED CER for reading ability; the strict one also charges for the "
                           "okina/tutuq convention, which 58 of 59 fonts cannot even draw"}
    res["summary"] = summary
    Path(a.runs).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.runs, "w"), ensure_ascii=False, indent=1)
    print("\n=== UZBEK: strips vs whole pages (Qwen3-VL-4B zero-shot) ===")
    print(json.dumps(summary, indent=1))
    print(f"\n-> {a.runs}")


if __name__ == "__main__":
    main()
