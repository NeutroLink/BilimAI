#!/usr/bin/env python
"""E4.2 step 3 — train our own page segmenter (word / teacher / line masks) to replace the RP ONNX whose network misses
1-2-word fragments (13 % of exam GT lines; 99 % of all misses). 2026-08-20, runs on one 24 GB GPU (PAI-DSW free quota).

Fresh LinkNet (resnet34) at 896 px — the RP checkpoint exists only as ONNX, so per the plan's fallback we train from scratch
on the same masks: ch0 = word ink (pupil_text + pupil_comment polygons), ch1 = teacher ink (teacher_comment), ch2 = text_line.
Input convention matches bilimai/detector.py exactly (resize 896x896, /255, no mean-std; output = sigmoid probabilities),
so the exported ONNX is a drop-in for RPDetector(onnx=...). The fragment fix is in the SAMPLING: 40 % of training crops are
zoomed views centred on a line with ≤ FRAG_WORDS words (the network then sees fragments at readable scale).
v4 (2026-08-20, after the v3 exam verdict F1 0.694 with 96 % of misses = ≤2-word lines): fragment crops 65 % with
stronger zoom, per-instance loss weights up to 3× on small word/line polygons, --resume from a prior checkpoint,
14 epochs, and the in-script box gate now uses deployment post-processing (unclip 1.5 + val-fitted growth) against
exam-convention word-union GT — the old gate mis-ranked v1 vs v2 and was 20× pessimistic on v3.

  python train_segm.py --src <school_notebooks_RU/train> --out runs_segm [--epochs 8 --batch 6]

Outputs in --out: segm_ft.onnx (best val epoch), best.pt, log.txt, val_metrics.json (pixel F1 per channel + box-level line
F1@0.5 on 40 val pages with the production post-processing constants).
"""
import argparse, collections, json, os, random, sys, time
from pathlib import Path
import numpy as np

def build_pages(src, split):
    import json as J
    d = J.load(open(Path(src) / f"annotations_{split}.json", encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    byimg = collections.defaultdict(list)
    for x in d["annotations"]: byimg[x["image_id"]].append(x)
    pages = []
    for im in d["images"]:
        anns = byimg[im["id"]]
        polys = {"word": [], "teacher": [], "line": []}
        for x in anns:
            n = cats[x["category_id"]]
            k = "word" if n in ("pupil_text", "pupil_comment") else "teacher" if n == "teacher_comment" else "line" if n == "text_line" else None
            if k:
                for seg in x["segmentation"]:
                    if len(seg) >= 6: polys[k].append(np.array(seg, dtype=np.float32).reshape(-1, 2))
        # word groups per line for the fragment sampler (same grouping as make_line_strips: group_id, else containing text_line)
        words = [x for x in anns if cats[x["category_id"]] in ("pupil_text", "pupil_comment")]
        tl = [x for x in anns if cats[x["category_id"]] == "text_line"]
        def box(a):
            s = a["segmentation"][0]; xs, ys = s[0::2], s[1::2]; return [min(xs), min(ys), max(xs), max(ys)]
        tlb = [box(a) for a in tl]
        groups = collections.defaultdict(list)
        for w in words:
            g = w.get("group_id")
            if g is None:
                b = box(w); c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                hit = [i for i, t in enumerate(tlb) if t[0] <= c[0] <= t[2] and t[1] <= c[1] <= t[3]]
                g = f"tl{hit[0]}" if hit else f"solo{id(w)}"
            groups[g].append(box(w))
        frags = []
        for g, bs in groups.items():
            if 1 <= len(bs) <= FRAG_WORDS:
                frags.append([min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)])
        pages.append({"file": str(Path(src) / "images" / im["file_name"]), "w": im["width"], "h": im["height"], "polys": polys, "frags": frags})
    return pages

FRAG_WORDS = 2
S = 896

SHRINK_R = {"word": 0.4, "teacher": 0.4, "line": 0.4}      # v3 (2026-08-20): DBNet-style INWARD BOUNDARY OFFSET d = A(1-r^2)/L
def _shrink_poly(p):
    """Offset polygon `p` (N,2) inward by d = A(1-r^2)/L (pyclipper, like DBNet). Preserves length of thin shapes — the v2
    centroid scaling collapsed long text_line polygons to stubs and the line channel learned to predict nothing."""
    import pyclipper
    r = 0.4
    a = abs(pyclipper.Area(p.tolist()))
    l = np.sqrt(((p - np.roll(p, 1, 0)) ** 2).sum(1)).sum()
    if a < 4 or l < 4: return [p]
    d = a * (1 - r * r) / l
    pc = pyclipper.PyclipperOffset(); pc.AddPath([tuple(q) for q in p.astype(int)], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    out = pc.Execute(-d)
    return [np.array(q, dtype=np.float32) for q in out] if out else []

def rasterize(polys, w, h, sx, sy, ox=0.0, oy=0.0):
    """Per-polygon masks offset inward (instances separate; post-processing grows boxes back — the RP contract).
    v4: also a per-pixel LOSS WEIGHT map — small word/line instances get up to 3× weight (w = clip(√(median_area/area),
    1, 3) per polygon): 96 % of v3's exam misses were ≤2-word fragment lines the net never fired on."""
    import cv2
    m = np.zeros((3, S, S), dtype=np.uint8)
    wm = np.ones((3, S, S), dtype=np.float32)
    for ci, k in enumerate(("word", "teacher", "line")):
        shr = []
        for p0 in polys[k]:
            p = (p0 - [ox, oy]) * [sx, sy]
            shr += [(q, abs(cv2.contourArea(q.astype(np.float32)))) for q in _shrink_poly(p)]
        areas = [aa for _, aa in shr if aa > 2]
        med = float(np.median(areas)) if areas else 0.0
        for q, aa in shr:
            cv2.fillPoly(m[ci], [q.astype(np.int32)], 1)
            if ci != 1 and med > 0 and aa > 2:
                wgt = float(np.clip(np.sqrt(med / aa), 1.0, float(os.environ.get("WMAX", 3.0))))
                if wgt > 1.05: cv2.fillPoly(wm[ci], [q.astype(np.int32)], wgt)
    return m, wm
_SYNTH_FONTS = None
_SYNTH_VOCAB = ("Упражнение Задача Домашняя работа Классная Диктант Правило Ответ Проверка Словарь Число"
                " осень зима весна лето снег дождь ветер солнце небо земля вода лес поле река дом село город"
                " мама папа брат сестра друг школа класс урок книга ручка тетрадь доска стол окно дверь"
                " бежит идёт стоит лежит поёт живёт растёт цветёт падает светит хорошо плохо быстро медленно").split()

def synth_paste(img, p, rng):
    """v6: render synthetic cursive Cyrillic words (6 school-script fonts + shear/rotate variety), colour them with ink
    SAMPLED FROM THIS PAGE'S OWN WRITING, and sit them on the page's real ruled rows — the empty x-stretches of GT
    text-line bands — so unlike v5's floating crops they land exactly where real writing would. SYNTH_P / SYNTH_N
    env-tunable; fonts from eval/detectors/fonts_ru or data/fonts/ru_school."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    global _SYNTH_FONTS
    if _SYNTH_FONTS is None:
        from pathlib import Path as _P
        for d in (_P(__file__).parent / "fonts_ru", _P(__file__).resolve().parents[2] / "data/fonts/ru_school"):
            if d.is_dir() and list(d.glob("*.ttf")): _SYNTH_FONTS = sorted(str(f) for f in d.glob("*.ttf")); break
        if _SYNTH_FONTS is None: _SYNTH_FONTS = []
    lines = p["polys"]["line"]
    if not _SYNTH_FONTS or not lines: return img, p
    H, W = img.shape[:2]
    boxes = [[q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()]
             for k in ("word", "teacher", "line") for q in p["polys"][k]]
    # ink samples from this page's real words (dark pixels inside word polys)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); samples = []
    for q in p["polys"]["word"][:12]:
        x0, y0 = int(q[:, 0].min()), int(q[:, 1].min()); x1, y1 = int(q[:, 0].max()), int(q[:, 1].max())
        if x1 - x0 < 4 or y1 - y0 < 4: continue
        g = gray[y0:y1, x0:x1]; c = img[y0:y1, x0:x1]
        ink = c[g < max(40, int(np.median(g)) - 35)]
        if len(ink): samples.append(ink[:: max(1, len(ink) // 200)])
    if not samples: return img, p
    samples = np.concatenate(samples).astype(np.float32)
    samples = samples[samples[:, 2] < samples[:, 0] + 25]        # drop reddish pixels — teacher ink bleeding into word boxes
    if len(samples) < 50: return img, p
    dark = samples[samples.sum(1) <= np.percentile(samples.sum(1), 30)]   # ink CORE, not the pale anti-aliased fringe
    # classic blue ballpoint (founder reference sample 2026-08-21): saturated blue core, BGR ≈ (150, 62, 42)
    base = np.median(dark, 0) * 0.35 + np.float32([150, 62, 42]) * 0.65
    img = img.copy(); polys = {k: list(v) for k, v in p["polys"].items()}; frags = list(p["frags"])
    rows = [[q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()] for q in lines]
    rows = [r for r in rows if 18 <= r[3] - r[1] <= H * 0.08]
    if not rows: return img, p
    med_h = float(np.median([r[3] - r[1] for r in rows]))       # page-typical text height; thin outlier rows mislead
    lo, hi = (int(v) for v in os.environ.get("SYNTH_N", "4,12").split(","))
    for _ in range(rng.randint(lo, hi)):
        rb = rows[rng.randrange(len(rows))]                     # row gives the Y (sits on a real rule); X roams the spread
        rh = med_h * rng.uniform(0.9, 1.3)
        word = _SYNTH_VOCAB[rng.randrange(len(_SYNTH_VOCAB))]
        if rng.random() < 0.15: word = str(rng.randint(1, 599)) + ("." if rng.random() < 0.5 else "")
        f = ImageFont.truetype(_SYNTH_FONTS[rng.randrange(len(_SYNTH_FONTS))], int(rh * 1.1))
        tw = int(f.getlength(word)) + 20
        tim = Image.new("L", (tw, int(rh * 2.2)), 0)
        ImageDraw.Draw(tim).text((10, int(rh * 0.2)), word, fill=255, font=f)
        m0 = np.array(tim)
        sh = rng.uniform(-0.05, 0.30); rot = rng.uniform(-0.03, 0.03)   # school slant leans right
        A = np.float32([[1, sh + rot, 0], [0, 1, 0]])
        m0 = cv2.warpAffine(m0, A, (m0.shape[1] + int(m0.shape[0] * abs(sh + rot)) + 4, m0.shape[0]))
        ys, xs = np.nonzero(m0 > 32)
        if not len(ys): continue
        m0 = m0[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        s = (rh * rng.uniform(0.85, 1.15)) / m0.shape[0]     # GT line boxes span ascender→descender; match them
        m0 = cv2.resize(m0, (max(2, int(m0.shape[1] * s)), max(2, int(m0.shape[0] * s))))
        dh, dw = m0.shape
        # find an empty x-stretch in this row band (12 px margins), baseline on the rule
        ty = int(rb[3] - dh - rh * rng.uniform(0.02, 0.10))
        if ty < 0 or ty + dh > H: continue                   # ascender-tall word on a top-of-page rule
        x_lo, x_hi = int(W * 0.04), int(W * 0.96 - dw)
        if x_hi <= x_lo: continue
        spot = None
        for _t in range(25):
            tx = rng.randrange(x_lo, x_hi)
            if all(tx + dw + 12 <= b[0] or b[2] + 12 <= tx or ty + dh + 12 <= b[1] or b[3] + 12 <= ty for b in boxes):
                spot = tx; break
        if spot is None: continue
        tx = spot
        a = cv2.GaussianBlur(m0.astype(np.float32) / 255.0, (3, 3), 0)[..., None] * rng.uniform(0.88, 1.0)
        # solid dark-blue ink: page-sampled core colour + SMOOTH low-frequency variation (per-pixel noise reads as speckle)
        field = cv2.GaussianBlur(np.random.normal(0, 18, (dh, dw, 3)).astype(np.float32), (0, 0), max(2, dh / 10))
        ink = np.clip(base[None, None, :] + field, 0, 255)
        reg = img[ty:ty + dh, tx:tx + dw].astype(np.float32)
        img[ty:ty + dh, tx:tx + dw] = (a * ink + (1 - a) * reg).astype(np.uint8)
        nq = np.array([[tx, ty], [tx + dw, ty], [tx + dw, ty + dh], [tx, ty + dh]], np.float32)
        polys["word"].append(nq); polys["line"].append(nq)
        boxes.append([tx, ty, tx + dw, ty + dh]); frags.append([tx, ty, tx + dw, ty + dh])
    p2 = dict(p); p2["polys"] = polys; p2["frags"] = frags
    return img, p2

def paste_frags(img, p, rng):
    """v5 synthetic fragment augmentation: cut 1–2-word snippets from THIS page (same pen/paper/lighting, so the paste
    is hard to tell from real writing) and feather-blend them onto empty ruled regions. Each paste is appended to the
    word AND line channels and to frags — the net then sees many more short lines per page, which is exactly what v3/v4
    miss (96 % of exam misses are ≤2-word lines). PASTE_P / PASTE_N env-tunable."""
    import cv2
    H, W = img.shape[:2]
    words = p["polys"]["word"]
    if not words: return img, p
    boxes = [[q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()]
             for k in ("word", "teacher", "line") for q in p["polys"][k]]
    donors = [q for q in words if 15 < (q[:, 1].max() - q[:, 1].min()) < H * 0.08
              and 15 < (q[:, 0].max() - q[:, 0].min()) < W * 0.3]
    if not donors: return img, p
    img = img.copy(); polys = {k: list(v) for k, v in p["polys"].items()}; frags = list(p["frags"])
    # landing zone = bbox of the page's existing writing (+3 % slack) — keeps pastes on the notebook, off the desk
    zx0 = max(0, min(b[0] for b in boxes) - W * 0.03); zy0 = max(0, min(b[1] for b in boxes) - H * 0.03)
    zx1 = min(W, max(b[2] for b in boxes) + W * 0.03); zy1 = min(H, max(b[3] for b in boxes) + H * 0.03)
    lo, hi = (int(v) for v in os.environ.get("PASTE_N", "4,12").split(","))
    for _ in range(rng.randint(lo, hi)):
        q = donors[rng.randrange(len(donors))]
        pad = 6
        x0 = max(0, int(q[:, 0].min()) - pad); y0 = max(0, int(q[:, 1].min()) - pad)
        x1 = min(W, int(q[:, 0].max()) + pad); y1 = min(H, int(q[:, 1].max()) + pad)
        dw, dh = x1 - x0, y1 - y0
        if dw < 10 or dh < 10 or zx1 - dw <= zx0 or zy1 - dh <= zy0: continue
        spot = None
        for _t in range(25):     # rejection-sample an empty landing spot (12 px margin from every existing box)
            tx = rng.randrange(int(zx0), int(zx1 - dw)); ty = rng.randrange(int(zy0), int(zy1 - dh))
            if all(tx + dw + 12 <= b[0] or b[2] + 12 <= tx or ty + dh + 12 <= b[1] or b[3] + 12 <= ty for b in boxes):
                spot = (tx, ty); break
        if spot is None: continue
        tx, ty = spot
        patch = img[y0:y1, x0:x1].astype(np.float32); tgt = img[ty:ty + dh, tx:tx + dw].astype(np.float32)
        patch = np.clip(patch * np.clip(np.median(tgt) / max(1.0, np.median(patch)), 0.85, 1.18), 0, 255)
        a = np.zeros((dh, dw), np.float32)
        cv2.fillPoly(a, [(q - [x0, y0]).astype(np.int32)], 1.0)
        a = cv2.GaussianBlur(cv2.dilate(a, np.ones((9, 9), np.uint8)), (21, 21), 0)[..., None]
        img[ty:ty + dh, tx:tx + dw] = (a * patch + (1 - a) * tgt).astype(np.uint8)
        nq = q + [tx - x0, ty - y0]                   # translate poly into the landing spot's frame
        polys["word"].append(nq); polys["line"].append(nq)
        boxes.append([tx, ty, tx + dw, ty + dh]); frags.append([tx, ty, tx + dw, ty + dh])
    p2 = dict(p); p2["polys"] = polys; p2["frags"] = frags
    return img, p2

class Ds:
    def __init__(self, pages, train):
        self.pages, self.train = pages, train
    def __len__(self): return len(self.pages)
    def sample(self, i, rng):
        import cv2
        p = self.pages[i]; img = cv2.imread(p["file"])
        if img is None: return None
        if self.train and rng.random() < float(os.environ.get("PASTE_P", 0.5)):
            img, p = paste_frags(img, p, rng)
        if self.train and rng.random() < float(os.environ.get("SYNTH_P", 0.0)):
            img, p = synth_paste(img, p, rng)
        H, W = img.shape[:2]
        if self.train and p["frags"] and rng.random() < float(os.environ.get("FRAG_P", 0.65)):   # v4: fragment-centred crop (was 40 %), stronger zoom; FRAG_P/WMAX env-overridable for the 4-GPU variant sweep
            fb = p["frags"][rng.randrange(len(p["frags"]))]
            cw = rng.uniform(0.25, 0.55) * W; ch = rng.uniform(0.25, 0.55) * H
            cx = min(max(fb[0] + (fb[2] - fb[0]) * rng.random(), cw / 2), W - cw / 2)
            cy = min(max(fb[1] + (fb[3] - fb[1]) * rng.random(), ch / 2), H - ch / 2)
            x0, y0 = cx - cw / 2, cy - ch / 2
            crop = img[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
            m, wm = rasterize(p["polys"], W, H, S / cw, S / ch, x0, y0)
        else:
            crop = img; m, wm = rasterize(p["polys"], W, H, S / W, S / H)
        x = cv2.resize(crop, (S, S)).astype(np.float32) / 255.0
        if self.train and rng.random() < 0.5: x = x * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08); x = np.clip(x, 0, 1)
        return np.transpose(x, (2, 0, 1)), m.astype(np.float32), wm

def collate(batch):
    import torch
    xs, ms, ws = zip(*batch)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ms)), torch.from_numpy(np.stack(ws))

def main():
    import torch, cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", default="runs_segm")
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 14))); ap.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", 6)))
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--val-pages", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="", help="load net weights from a prior best.pt/last.pt and continue (fresh optimizer)")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")
    def P(*s): print(*s, flush=True); print(*s, file=log, flush=True)
    import segmentation_models_pytorch as smp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try: net = smp.Linknet("resnet34", encoder_weights="imagenet", classes=3, activation=None).to(dev)
    except Exception as e: P("imagenet weights unavailable, training from random init:", str(e)[:100]); net = smp.Linknet("resnet34", encoder_weights=None, classes=3, activation=None).to(dev)
    if a.resume:
        net.load_state_dict(torch.load(a.resume, map_location=dev)); P(f"resumed weights from {a.resume}")
    tr = Ds(build_pages(a.src, "train"), True); va = Ds(build_pages(a.src, "val"), False)
    P(f"pages: train {len(tr.pages)} val {len(va.pages)} | frag-lines/page median {np.median([len(p['frags']) for p in tr.pages])} | device {dev}")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler(dev) if dev == "cuda" else None
    bce_none = torch.nn.BCEWithLogitsLoss(reduction="none")
    def dice(logit, m):
        p = torch.sigmoid(logit); num = 2 * (p * m).sum((2, 3)); den = p.sum((2, 3)) + m.sum((2, 3)) + 1e-6
        return 1 - (num / den).mean()
    rng = random.Random(a.seed); best = -1
    for ep in range(a.epochs):
        net.train(); idx = list(range(len(tr))); rng.shuffle(idx); t0 = time.time(); losses = []
        for s in range(0, len(idx), a.batch):
            batch = [b for b in (tr.sample(i, rng) for i in idx[s:s + a.batch]) if b]
            if not batch: continue
            x, m, wm = collate(batch); x, m, wm = x.to(dev), m.to(dev), wm.to(dev)
            opt.zero_grad()
            with torch.autocast(dev, enabled=dev == "cuda"):
                y = net(x); loss = (bce_none(y, m) * wm).mean() + dice(y, m)
            if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else: loss.backward(); opt.step()
            losses.append(float(loss))
            if (s // a.batch) % 40 == 0: P(f"  ep{ep} {s}/{len(idx)} loss {np.mean(losses[-40:]):.4f} ({(time.time()-t0)/(s+a.batch)*len(idx)/60:.0f} min/ep)")
        sched.step()
        # ---- val: pixel F1 per channel + production box-level line F1 on --val-pages pages
        net.eval(); tp = np.zeros(3); fp = np.zeros(3); fn = np.zeros(3)
        with torch.no_grad():
            for i in range(len(va)):
                b = va.sample(i, rng)
                if not b: continue
                x, m, _ = b; y = torch.sigmoid(net(torch.from_numpy(x[None]).to(dev)))[0].cpu().numpy()
                for c, thr in ((0, .8), (1, .5), (2, .5)):
                    pr = y[c] > thr; gt = m[c] > .5
                    tp[c] += (pr & gt).sum(); fp[c] += (pr & ~gt).sum(); fn[c] += (~pr & gt).sum()
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        box_f1 = box_line_f1(net, va, dev, n=a.val_pages)
        P(f"ep{ep}: loss {np.mean(losses):.4f} | val pixel F1 word/teacher/line {f1.round(3).tolist()} | box line F1@0.5 {box_f1:.3f} | {(time.time()-t0)/60:.1f} min")
        score = box_f1
        torch.save(net.state_dict(), out / "last.pt")
        if score > best:
            best = score; torch.save(net.state_dict(), out / "best.pt")
            export_onnx(net, out / "segm_ft.onnx", dev); P(f"  saved best (box F1 {best:.3f})")
        json.dump({"epoch": ep, "pixel_f1": f1.tolist(), "box_line_f1": box_f1, "best": best}, open(out / "val_metrics.json", "w"), indent=1)
    P("DONE best box line F1", best)

def boxes_from_mask(y, unclip=1.5, grow=(0.153, 0.293, 0.014)):
    """Deployment post-processing for shrunk-mask models (v4 gate fix: v3's in-script metric used the RP constants and no
    unclip → 0.030 in-script vs 0.694 on the exam, and mis-ranked v1 vs v2): word contours pyclipper-expanded by
    d = A·unclip/L, grouped by nearest line polyline, grown by the val-fitted fractions (derive_growth.py, 2026-08-20)."""
    import cv2, pyclipper
    LINE_GROW = grow
    m = (y[2] > 0.5).astype(np.uint8); m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    cl, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mw = (y[0] > 0.8).astype(np.uint8); cw, _ = cv2.findContours(mw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    for c in cw:
        if cv2.contourArea(c) < 4: continue
        if unclip > 0:
            d = cv2.contourArea(c) * unclip / max(1.0, cv2.arcLength(c, True))
            pco = pyclipper.PyclipperOffset(); pco.AddPath(c[:, 0, :].tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            ex = pco.Execute(d)
            if ex:
                pts = np.array(max(ex, key=lambda p: cv2.contourArea(np.array(p, dtype=np.int32))), dtype=np.int32)
                bx, by, bw, bh = cv2.boundingRect(pts.reshape(-1, 1, 2)); words.append([bx, by, bx + bw, by + bh]); continue
        bx, by, bw, bh = cv2.boundingRect(c); words.append([bx, by, bx + bw, by + bh])
    lines = [c[:, 0, :].astype(np.float32) for c in cl if len(c) >= 5]
    groups = [[] for _ in lines]
    for wi, wb in enumerate(words):
        cx, cy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2; bd, best = 1e9, None
        for li, pts in enumerate(lines):
            near = pts[np.abs(pts[:, 0] - cx) < max(30, wb[2] - wb[0])]
            if len(near) == 0: continue
            d = np.min(np.abs(near[:, 1] - cy))
            if d < bd: bd, best = d, li
        if best is not None and bd < 1.2 * (wb[3] - wb[1]): groups[best].append(wi)
    out = []
    for g in groups:
        if not g: continue
        gb = np.array([words[i] for i in g]); b = [gb[:, 0].min(), gb[:, 1].min(), gb[:, 2].max(), gb[:, 3].max()]
        t, bt, xg = LINE_GROW; h = b[3] - b[1]; w = b[2] - b[0]
        out.append([max(0, b[0] - xg * w), max(0, b[1] - t * h), min(896, b[2] + xg * w), min(896, b[3] + bt * h)])
    return out

def box_line_f1(net, va, dev, n=40, iou_thr=0.5):
    import torch, cv2
    rng = random.Random(1); ids = list(range(len(va.pages))); rng.shuffle(ids); ids = ids[:n]
    def iou(a, b):
        x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x1 - x0) * max(0, y1 - y0)
        return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)
    TP = FP = FN = 0
    with torch.no_grad():
        for i in ids:
            p = va.pages[i]; img = cv2.imread(p["file"])
            if img is None: continue
            H, W = img.shape[:2]
            x = np.transpose(cv2.resize(img, (S, S)).astype(np.float32) / 255.0, (2, 0, 1))
            y = torch.sigmoid(net(torch.from_numpy(x[None]).to(dev)))[0].cpu().numpy()
            pred = boxes_from_mask(y)
            # GT line boxes in the EXAM convention (v4 gate fix): union of the word polygons whose centroid falls in a
            # text_line polygon's bbox — build_exam_gt.py convention, not the tighter raw text_line bbox
            gt = []
            wcs = [(poly[:, 0].mean(), poly[:, 1].mean(), poly) for poly in p["polys"]["word"]]
            for poly in p["polys"]["line"]:
                lb = [poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()]
                grp = [wp for cx, cy, wp in wcs if lb[0] <= cx <= lb[2] and lb[1] <= cy <= lb[3]]
                if not grp: continue
                b = [min(w[:, 0].min() for w in grp) * S / W, min(w[:, 1].min() for w in grp) * S / H,
                     max(w[:, 0].max() for w in grp) * S / W, max(w[:, 1].max() for w in grp) * S / H]
                if b[2] - b[0] > 4 and b[3] - b[1] > 2: gt.append(b)
            used = set()
            for g in gt:
                m = [(iou(g, pb), k) for k, pb in enumerate(pred) if k not in used]
                m = max(m) if m else (0, -1)
                if m[0] >= iou_thr: TP += 1; used.add(m[1])
                else: FN += 1
            FP += len(pred) - len(used)
    return 2 * TP / max(1, 2 * TP + FP + FN)

def export_onnx(net, path, dev):
    import torch
    class Wrap(torch.nn.Module):
        def __init__(self, n): super().__init__(); self.n = n
        def forward(self, x): return torch.sigmoid(self.n(x))
    w = Wrap(net).eval()
    torch.onnx.export(w, torch.zeros(1, 3, S, S, device=dev), str(path), input_names=["input"], output_names=["pred"],
                      dynamic_axes={"input": {0: "batch"}, "pred": {0: "batch"}}, opset_version=17)

if __name__ == "__main__":
    main()
