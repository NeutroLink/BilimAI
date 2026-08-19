"""R5b — pre-rendered line-strip augmentation (TRAINING-PLAN §3: elastic, shear ±12°, stroke ±1 px, blur, JPEG 40–90, ink
colour, red-pen overlays, neighbour-line fragments, box jitter ±20 %). Pure OpenCV/NumPy so it runs on the Mac and on a rented
box without extra packages. Deterministic per (file, copy) seed. ~5–10 ms per strip on one core; the box pre-renders ×2 copies
of the school strips with multiprocessing before training (LLaMA-Factory loads images from disk, it has no augmentation hook).

    from bilimai.augment import augment
    out = augment(img_bgr, rng, neighbour=other_strip_bgr)      # same dtype, height may change by the jitter

QA sheet:  python -m bilimai.augment --labels data/derived/strips_ru_v1/labels.jsonl --n 10 --out out/r5b_aug_qa.jpg
"""
from __future__ import annotations
import random
import numpy as np
import cv2


def _elastic(img, rng, alpha=34.0, sigma=5.0):
    h, w = img.shape[:2]; rs = np.random.RandomState(rng.randrange(1 << 30))
    dx = cv2.GaussianBlur((rs.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rs.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, x + dx, y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _shear(img, rng, max_deg=12.0):
    h, w = img.shape[:2]; deg = rng.uniform(-max_deg, max_deg); sh = np.tan(np.radians(deg))
    M = np.float32([[1, sh, -sh * h / 2], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _stroke(img, rng):
    k = np.ones((2, 2), np.uint8)
    return cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)     # dark ink on light paper: erode = thicker


def _blur(img, rng):
    if rng.random() < 0.6: return cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.6))
    k = rng.choice([3, 5]); kern = np.zeros((k, k), np.float32); kern[k // 2, :] = 1.0 / k               # horizontal motion blur
    return cv2.filter2D(img, -1, kern)


def _jpeg(img, rng, lo=40, hi=90):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(lo, hi)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def _ink_colour(img, rng):
    """Recolour the dark (ink) pixels: blue / violet / green / brown-black pens."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); v = hsv[..., 2]
    mask = v < 140
    hue = rng.choice([110, 120, 130, 140, 60, 10])        # OpenCV hue 0–179: ~110–140 blue/violet, 60 green, 10 brown
    sat = rng.randint(60, 200)
    hsv[..., 0][mask] = hue; hsv[..., 1][mask] = np.maximum(hsv[..., 1][mask], sat)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _paper(img, rng):
    """Brightness / contrast / gamma / slight colour cast — phone photos vs scans."""
    a = rng.uniform(0.85, 1.15); b = rng.uniform(-20, 20)
    out = cv2.convertScaleAbs(img, alpha=a, beta=b)
    if rng.random() < 0.5:
        g = rng.uniform(0.8, 1.25); lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)]).astype(np.uint8); out = cv2.LUT(out, lut)
    if rng.random() < 0.4:
        cast = np.array([rng.randint(-12, 12) for _ in range(3)], dtype=np.int16); out = np.clip(out.astype(np.int16) + cast, 0, 255).astype(np.uint8)
    return out


def _red_pen(img, rng):
    """Teacher's red pen over the pupil's line: underline, tick, a stroke through a word, a circled bit."""
    out = img.copy(); h, w = out.shape[:2]; col = (rng.randint(20, 60), rng.randint(20, 60), rng.randint(170, 230)); t = max(1, h // 40)
    for _ in range(rng.randint(1, 2)):
        r = rng.random(); x0 = rng.randint(0, max(1, w - w // 4)); x1 = min(w - 1, x0 + rng.randint(w // 10, w // 3))
        if r < 0.4:   cv2.line(out, (x0, int(h * 0.85)), (x1, int(h * 0.85) + rng.randint(-3, 3)), col, t)          # underline
        elif r < 0.65:cv2.line(out, (x0, int(h * 0.5)), (x1, int(h * 0.5) + rng.randint(-4, 4)), col, t)             # strike
        elif r < 0.85:                                                                                                # tick
            cx, cy = x0, int(h * 0.6); s = h // 3
            cv2.polylines(out, [np.array([[cx, cy], [cx + s // 3, cy + s // 3], [cx + s, cy - s // 2]], np.int32)], False, col, t)
        else: cv2.ellipse(out, ((x0 + x1) // 2, h // 2), ((x1 - x0) // 2 + 4, h // 2 - 2), 0, 0, 360, col, t)       # circle
    a = rng.uniform(0.55, 0.9)
    return cv2.addWeighted(out, a, img, 1 - a, 0)


def _neighbour(img, rng, other):
    """Glue a slice of another strip above or below: ascenders/descenders of the adjacent line bleeding into the box."""
    h, w = img.shape[:2]
    if other is None: return img
    oh, ow = other.shape[:2]; frac = rng.uniform(0.1, 0.22); sh = max(2, int(oh * frac))
    ox = rng.randint(0, max(0, ow - w)) if ow > w else 0
    sl = other[oh - sh:, ox:ox + w] if rng.random() < 0.5 else other[:sh, ox:ox + w]
    if sl.shape[1] < w: sl = cv2.copyMakeBorder(sl, 0, 0, 0, w - sl.shape[1], cv2.BORDER_REPLICATE)
    top = rng.random() < 0.5
    out = np.vstack([sl, img]) if top else np.vstack([img, sl])
    return out


def _jitter(img, rng, frac=0.2):
    """Box jitter ±20 %: crop or pad each edge by a random fraction of the height (detector boxes are never exact)."""
    h, w = img.shape[:2]
    def d(): return int(rng.uniform(-frac, frac) * h)
    t, b = d(), d(); l, r = int(rng.uniform(-frac, frac) * h), int(rng.uniform(-frac, frac) * h)
    out = img
    # crop negatives
    y0 = max(0, -t); y1 = h - max(0, -b); x0 = max(0, -l); x1 = w - max(0, -r)
    if y1 - y0 < h // 2: y0, y1 = 0, h
    out = out[y0:y1, x0:x1]
    # pad positives (replicate the paper)
    out = cv2.copyMakeBorder(out, max(0, t), max(0, b), max(0, l), max(0, r), cv2.BORDER_REPLICATE)
    return out


def augment(img_bgr: np.ndarray, rng: random.Random, neighbour: np.ndarray | None = None, strength: float = 1.0) -> np.ndarray:
    """One augmented copy. Each op fires with its own probability (scaled by `strength`); geometry first, then ink, then
    capture artefacts, so the JPEG/blur land on top like a real photo would."""
    p = lambda x: rng.random() < x * strength
    out = img_bgr
    if p(0.5): out = _jitter(out, rng)
    if p(0.35): out = _neighbour(out, rng, neighbour)
    if p(0.5): out = _shear(out, rng)
    if p(0.5): out = _elastic(out, rng, alpha=rng.uniform(15, 40), sigma=rng.uniform(4, 7))
    if p(0.4): out = _stroke(out, rng)
    if p(0.35): out = _ink_colour(out, rng)
    if p(0.2): out = _red_pen(out, rng)
    if p(0.7): out = _paper(out, rng)
    if p(0.5): out = _blur(out, rng)
    if p(0.8): out = _jpeg(out, rng)
    return out


def _main():
    import argparse, json, random as _r
    from pathlib import Path
    ap = argparse.ArgumentParser(); ap.add_argument("--labels", required=True); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--copies", type=int, default=2); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rows = [json.loads(l) for l in open(a.labels, encoding="utf-8")]; rng = _r.Random(a.seed)
    pick = rng.sample(rows, a.n); root = Path(a.labels).parent / "images"; W = 1100; tiles = []
    for r in pick:
        img = cv2.imread(str(root / r["file"])); other = cv2.imread(str(root / rng.choice(rows)["file"]))
        row = [img] + [augment(img, _r.Random(f"{r['file']}:{c}"), neighbour=other) for c in range(a.copies)]
        for im in row:
            s = W / im.shape[1]; im = cv2.resize(im, (W, max(8, int(im.shape[0] * s)))); tiles.append(cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(200, 200, 200)))
    sheet = np.vstack(tiles); Path(a.out).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(a.out, sheet); print("wrote", a.out, sheet.shape)


if __name__ == "__main__":
    _main()
