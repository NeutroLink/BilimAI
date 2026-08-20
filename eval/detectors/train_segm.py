#!/usr/bin/env python
"""E4.2 step 3 — train our own page segmenter (word / teacher / line masks) to replace the RP ONNX whose network misses
1-2-word fragments (13 % of exam GT lines; 99 % of all misses). 2026-08-20, runs on one 24 GB GPU (PAI-DSW free quota).

Fresh LinkNet (resnet34) at 896 px — the RP checkpoint exists only as ONNX, so per the plan's fallback we train from scratch
on the same masks: ch0 = word ink (pupil_text + pupil_comment polygons), ch1 = teacher ink (teacher_comment), ch2 = text_line.
Input convention matches bilimai/detector.py exactly (resize 896x896, /255, no mean-std; output = sigmoid probabilities),
so the exported ONNX is a drop-in for RPDetector(onnx=...). The fragment fix is in the SAMPLING: 40 % of training crops are
zoomed views centred on a line with ≤ FRAG_WORDS words (the network then sees fragments at readable scale).

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
    """Per-polygon masks offset inward (instances separate; post-processing grows boxes back — the RP contract)."""
    import cv2
    m = np.zeros((3, S, S), dtype=np.uint8)
    for ci, k in enumerate(("word", "teacher", "line")):
        for p0 in polys[k]:
            p = (p0 - [ox, oy]) * [sx, sy]
            for q in _shrink_poly(p):
                cv2.fillPoly(m[ci], [q.astype(np.int32)], 1)
    return m
class Ds:
    def __init__(self, pages, train):
        self.pages, self.train = pages, train
    def __len__(self): return len(self.pages)
    def sample(self, i, rng):
        import cv2
        p = self.pages[i]; img = cv2.imread(p["file"])
        if img is None: return None
        H, W = img.shape[:2]
        if self.train and p["frags"] and rng.random() < 0.4:      # fragment-centred crop, 40-70 % of the page
            fb = p["frags"][rng.randrange(len(p["frags"]))]
            cw = rng.uniform(0.4, 0.7) * W; ch = rng.uniform(0.4, 0.7) * H
            cx = min(max(fb[0] + (fb[2] - fb[0]) * rng.random(), cw / 2), W - cw / 2)
            cy = min(max(fb[1] + (fb[3] - fb[1]) * rng.random(), ch / 2), H - ch / 2)
            x0, y0 = cx - cw / 2, cy - ch / 2
            crop = img[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
            m = rasterize(p["polys"], W, H, S / cw, S / ch, x0, y0)
        else:
            crop = img; m = rasterize(p["polys"], W, H, S / W, S / H)
        x = cv2.resize(crop, (S, S)).astype(np.float32) / 255.0
        if self.train and rng.random() < 0.5: x = x * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08); x = np.clip(x, 0, 1)
        return np.transpose(x, (2, 0, 1)), m.astype(np.float32)

def collate(batch):
    import torch
    xs, ms = zip(*batch); return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ms))

def main():
    import torch, cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", default="runs_segm")
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 8))); ap.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", 6)))
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--val-pages", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")
    def P(*s): print(*s, flush=True); print(*s, file=log, flush=True)
    import segmentation_models_pytorch as smp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try: net = smp.Linknet("resnet34", encoder_weights="imagenet", classes=3, activation=None).to(dev)
    except Exception as e: P("imagenet weights unavailable, training from random init:", str(e)[:100]); net = smp.Linknet("resnet34", encoder_weights=None, classes=3, activation=None).to(dev)
    tr = Ds(build_pages(a.src, "train"), True); va = Ds(build_pages(a.src, "val"), False)
    P(f"pages: train {len(tr.pages)} val {len(va.pages)} | frag-lines/page median {np.median([len(p['frags']) for p in tr.pages])} | device {dev}")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler(dev) if dev == "cuda" else None
    bce = torch.nn.BCEWithLogitsLoss()
    def dice(logit, m):
        p = torch.sigmoid(logit); num = 2 * (p * m).sum((2, 3)); den = p.sum((2, 3)) + m.sum((2, 3)) + 1e-6
        return 1 - (num / den).mean()
    rng = random.Random(a.seed); best = -1
    for ep in range(a.epochs):
        net.train(); idx = list(range(len(tr))); rng.shuffle(idx); t0 = time.time(); losses = []
        for s in range(0, len(idx), a.batch):
            batch = [b for b in (tr.sample(i, rng) for i in idx[s:s + a.batch]) if b]
            if not batch: continue
            x, m = collate(batch); x, m = x.to(dev), m.to(dev)
            opt.zero_grad()
            with torch.autocast(dev, enabled=dev == "cuda"):
                y = net(x); loss = bce(y, m) + dice(y, m)
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
                x, m = b; y = torch.sigmoid(net(torch.from_numpy(x[None]).to(dev)))[0].cpu().numpy()
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

def boxes_from_mask(y):
    """Production post-processing (bilimai/detector.py constants): line channel thr .5 + dilate 3, word channel thr .8;
    words grouped by nearest line polyline; grow by LINE_GROW. Returns grown line boxes at mask scale."""
    import cv2
    LINE_GROW = (0.25, 0.365, 0.02)
    m = (y[2] > 0.5).astype(np.uint8); m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    cl, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mw = (y[0] > 0.8).astype(np.uint8); cw, _ = cv2.findContours(mw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    words = [cv2.boundingRect(c) for c in cw if cv2.contourArea(c) >= 4]
    words = [[x, yy, x + w, yy + h] for x, yy, w, h in words]
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
            # GT line boxes = word-group boxes (frags file has only fragments; rebuild from polys quickly via line polygons)
            gt = []
            for poly in p["polys"]["line"]:
                b = [poly[:, 0].min() * S / W, poly[:, 1].min() * S / H, poly[:, 0].max() * S / W, poly[:, 1].max() * S / H]
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
