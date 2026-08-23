#!/usr/bin/env python
"""E4.2 — FINE-TUNE the production detector at the miss gap. 2026-08-22.

Starts from `ai-forever/ReadingPipeline-notebooks : segm/segm_model.ckpt` — the PyTorch weights behind
the ONNX we ship — instead of from scratch. Verified drop-in (see eval/detectors/segm_linkresnet.py).
Target: the 335 missed short fragments (<= 1.7 of the 4.3-point detector gap), WITHOUT losing RP's
37 merges / 13 splits. A merged line reads 89 % wrong, so trading merges for misses is a net loss —
plans/DETECTOR-FINETUNE-2026-08-22.md fixes the accept/kill thresholds before any number exists.

WHAT THIS FIXES vs eval/detectors/train_segm.py
  1. Right network. train_segm builds smp.Linknet/resnet34 (21.8 M) from ImageNet; the model it was
     trying to beat is LinkResNet/resnet50 (28.8 M). Six runs, best exam F1 0.694 vs RP's 0.892.
  2. Right init. RP's weights, not random.
  3. Real data pipeline. train_segm has a `--workers` flag that is NEVER USED: samples are produced
     inline in the training loop, single-threaded, with a full cv2.imread of a 4000x3000 JPEG PER
     SAMPLE. The GPU idles on JPEG decode. Here: a DataLoader with workers, and each decoded page
     yields --crops-per-page samples, so decode cost is amortised K-fold on top of the parallelism.
  4. Correct loss. LinkResNet ends in nn.Sigmoid(); BCEWithLogitsLoss on that applies sigmoid twice.
     We train with return_logits=True and re-attach sigmoid for export.
  5. Shrink ratio. train_segm._shrink_poly hardcodes r = 0.4 and IGNORES its own SHRINK_R dict, so
     that dict is dead config. RP's published segm_config.json used shrink_ratio 0.5 — matching it
     matters now that we inherit RP's weights. --shrink-r makes it real and defaults to 0.5.
  6. Throughput. AMP, channels_last, cudnn.benchmark, TF32, DataParallel across all visible GPUs,
     cv2 threads pinned off inside workers (they fight the DataLoader otherwise).

Kaggle (free T4; request it explicitly — the default P100 is sm_60 and Kaggle's torch has no kernel
image for it):  kaggle kernels push ... --accelerator NvidiaTeslaT4
Note: every box versions the SAME Kaggle dataset and each push replaces the whole file set — that is
how r6_ctrl and r6_hwr2 were lost. One run at a time, or distinct slugs.

  smoke (Mac, minutes):  eval/.venv/bin/python eval/detectors/finetune_rp_segm.py --src <train> --smoke
  real  (T4):            python eval/detectors/finetune_rp_segm.py --src <train> --out runs_ft --epochs 6
"""
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import train_segm as TS                      # reuse page building, rasterise, fragment sampling
from segm_linkresnet import LinkResNet

S = TS.S                                     # 896
SHRINK_R = [0.5]                             # set from --shrink-r in main(); RP's config used 0.5


def make_shrinker(r: float):
    """train_segm._shrink_poly hardcodes r=0.4. Same DBNet offset, but with r actually configurable."""
    import pyclipper, cv2  # noqa: F401
    def _shrink(p):
        a = abs(pyclipper.Area(p.tolist()))
        l = float(np.sqrt(((p - np.roll(p, 1, 0)) ** 2).sum(1)).sum())
        if a < 4 or l < 4:
            return [p]
        d = a * (1 - r * r) / l
        pc = pyclipper.PyclipperOffset()
        pc.AddPath([tuple(q) for q in p.astype(int)], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        out = pc.Execute(-d)
        return [np.array(q, dtype=np.float32) for q in out] if out else []
    return _shrink



# --- RP-matching mask construction -------------------------------------------------------------
# MEASURED 2026-08-22 (12 val pages, frozen RP checkpoint scored against candidate targets):
#
#   channel        RP's segm_config.json recipe        pixel-F1     train_segm.py's construction   F1
#   0 words        Shrink(pupil+teacher, r=0.5)          0.869      shrink(pupil+pupil_comment)   0.864
#   1 border       Border(pupil+teacher, r=0.5)          0.550      fill(teacher_comment)         0.008
#   2 text_line    PolylineToMask(text_line, th=2)       0.711      shrink(text_line)             0.335
#
# So train_segm's targets are wrong on ch1 (65x off) and ch2 (2x off). Channel 2 is the one
# bilimai/detector.py reads to GROUP WORDS INTO LINES — the source of RP's 37-merge advantage over
# Kraken's 150. Fine-tuning RP's weights against a fat shrunk-polygon line mask would destroy exactly
# the property we are trying to protect. These functions reproduce RP's recipe instead.
FROZEN_RP_VAL_F1 = (0.869, 0.550, 0.711)      # what the untouched checkpoint scores; epoch 0 must be near this


def _shrink_r(p, r):
    import pyclipper
    a = abs(pyclipper.Area(p.tolist()))
    l = float(np.sqrt(((p - np.roll(p, 1, 0)) ** 2).sum(1)).sum())
    if a < 4 or l < 4:
        return [p]
    pc = pyclipper.PyclipperOffset()
    pc.AddPath([tuple(q) for q in p.astype(int)], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    o = pc.Execute(-a * (1 - r * r) / l)
    return [np.array(q, np.float32) for q in o] if o else []


def rp_rasterize(polys, sx, sy, ox=0.0, oy=0.0, r=0.5, wmax=3.0):
    """RP's three channels + train_segm's small-instance loss weighting (the fragment lever we keep).

    Returns (m[3,S,S] float32, wm[3,S,S] float32). Weighting is applied to ch0 and ch2 only: ch1 is a
    border ring whose area is not a meaningful instance size."""
    import cv2
    m = np.zeros((3, S, S), np.uint8)
    wm = np.ones((3, S, S), np.float32)
    ink = [(p0 - [ox, oy]) * [sx, sy] for p0 in (polys["word"] + polys["teacher"])]

    shr = []
    for p in ink:
        shr += [(q, abs(cv2.contourArea(q.astype(np.float32)))) for q in _shrink_r(p, r)]
    areas = [aa for _, aa in shr if aa > 2]
    med = float(np.median(areas)) if areas else 0.0
    for q, aa in shr:
        cv2.fillPoly(m[0], [q.astype(np.int32)], 1)
        if med > 0 and aa > 2:
            w = float(np.clip(np.sqrt(med / aa), 1.0, wmax))
            if w > 1.05:
                cv2.fillPoly(wm[0], [q.astype(np.int32)], w)

    full = np.zeros((S, S), np.uint8)
    for p in ink:
        cv2.fillPoly(full, [p.astype(np.int32)], 1)
    m[1] = full & ~m[0]                                   # border ring = filled minus shrunk

    lines = [(p0 - [ox, oy]) * [sx, sy] for p0 in polys["line"]]
    lens = [float(np.sqrt(((p - np.roll(p, 1, 0)) ** 2).sum(1)).sum()) for p in lines]
    medl = float(np.median(lens)) if lens else 0.0
    for p, L in zip(lines, lens):
        cv2.polylines(m[2], [p.astype(np.int32)], True, 1, 2)      # thickness 2 — RP's PolylineToMask
        if medl > 0 and L > 2:
            w = float(np.clip(np.sqrt(medl / L), 1.0, wmax))
            if w > 1.05:
                cv2.polylines(wm[2], [p.astype(np.int32)], True, w, 6)
    return m.astype(np.float32), wm


class PageCrops:
    """One item = one DECODED PAGE -> K crops. That is the whole point: the JPEG is 4000x3000 and
    decoding dominates, so paying it once per K samples (on top of worker parallelism) is the single
    biggest throughput win available here."""

    def __init__(self, pages, train, k, seed=0):
        self.pages, self.train, self.k, self.seed = pages, train, k, seed

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, i):
        import cv2
        cv2.setNumThreads(0)                 # workers must not each spawn a thread pool
        rng = random.Random((self.seed * 1000003) ^ (i * 2654435761) ^ os.getpid())
        p0 = self.pages[i]
        img0 = cv2.imread(p0["file"])
        if img0 is None:
            return []
        out = []
        for _ in range(self.k if self.train else 1):
            img, p = img0, p0
            if self.train and rng.random() < float(os.environ.get("PASTE_P", 0.5)):
                img, p = TS.paste_frags(img0.copy(), p0, rng)
            H, W = img.shape[:2]
            if self.train and p["frags"] and rng.random() < float(os.environ.get("FRAG_P", 0.65)):
                fb = p["frags"][rng.randrange(len(p["frags"]))]
                cw = rng.uniform(0.25, 0.55) * W; ch = rng.uniform(0.25, 0.55) * H
                cx = min(max(fb[0] + (fb[2] - fb[0]) * rng.random(), cw / 2), W - cw / 2)
                cy = min(max(fb[1] + (fb[3] - fb[1]) * rng.random(), ch / 2), H - ch / 2)
                x0, y0 = cx - cw / 2, cy - ch / 2
                crop = img[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
                m, wm = rp_rasterize(p["polys"], S / cw, S / ch, x0, y0, r=SHRINK_R[0])
            else:
                crop = img
                m, wm = rp_rasterize(p["polys"], S / W, S / H, r=SHRINK_R[0])
            if crop.size == 0:
                continue
            x = cv2.resize(crop, (S, S)).astype(np.float32) / 255.0
            if self.train and rng.random() < 0.5:
                x = np.clip(x * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08), 0, 1)
            out.append((np.transpose(x, (2, 0, 1)), m, wm))
        return out


def collate(page_batches):
    import torch
    flat = [s for pg in page_batches for s in pg]
    if not flat:
        return None
    xs, ms, ws = zip(*flat)
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ms)), torch.from_numpy(np.stack(ws)))


def main():
    import torch
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="school_notebooks_RU/train (has annotations_train/val.json + images/)")
    ap.add_argument("--out", default="runs_ft")
    ap.add_argument("--ckpt", default="", help="RP checkpoint; default = download from the HF hub")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-pages", type=int, default=2, help="pages per step; samples per step = this x --crops-per-page")
    ap.add_argument("--crops-per-page", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4, help="LOW on purpose — nudging a champion, not raising a beginner")
    ap.add_argument("--freeze-encoder-epochs", type=int, default=1)
    ap.add_argument("--shrink-r", type=float, default=0.5, help="RP's published segm_config.json used 0.5")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (all cores)")
    ap.add_argument("--val-pages", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="", choices=["", "cuda", "mps", "cpu"],
                    help="force a device; default auto. Use cpu to avoid contending with a job already on the GPU")
    ap.add_argument("--smoke", action="store_true", help="tiny run to prove the pipeline end-to-end")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")
    def P(*s): print(*s, flush=True); print(*s, file=log, flush=True)

    SHRINK_R[0] = a.shrink_r
    P(f"shrink ratio = {a.shrink_r} | masks built RP-style (ch0 shrink, ch1 border ring, ch2 outline th2)")

    if a.device == "cpu":
        dev, ngpu = "cpu", 0
    elif a.device == "mps" or (not a.device and not torch.cuda.is_available() and torch.backends.mps.is_available()):
        dev, ngpu = "mps", 1
    elif torch.cuda.is_available():
        dev, ngpu = "cuda", torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True            # fixed 896x896 input -> autotuner pays off
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        dev, ngpu = "cpu", 0
    workers = a.workers or max(1, (os.cpu_count() or 4))
    P(f"device {dev} x{ngpu} | dataloader workers {workers} | cores {os.cpu_count()}")

    ckpt = a.ckpt
    if not ckpt:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download("ai-forever/ReadingPipeline-notebooks", "segm/segm_model.ckpt")
    net = LinkResNet(3, 3, pretrained=False, encoder="resnet50", return_logits=True)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net.load_state_dict(sd, strict=True)                 # strict: a silent partial load is the worst outcome
    P(f"loaded RP weights STRICT from {ckpt}")
    net = net.to(dev)
    if dev == "cuda":
        net = net.to(memory_format=torch.channels_last)
        if ngpu > 1:
            net = torch.nn.DataParallel(net); P(f"DataParallel across {ngpu} GPUs")

    tr_pages = TS.build_pages(a.src, "train")
    va_pages = TS.build_pages(a.src, "val")[: a.val_pages]
    if a.smoke:
        tr_pages, va_pages, a.epochs = tr_pages[:8], va_pages[:4], 1
    tr = PageCrops(tr_pages, True, a.crops_per_page, a.seed)
    va = PageCrops(va_pages, False, 1, a.seed)
    P(f"pages: train {len(tr_pages)} val {len(va_pages)} | samples/step {a.batch_pages * a.crops_per_page}")

    dl_kw = dict(num_workers=workers, collate_fn=collate, pin_memory=(dev == "cuda"))
    if workers > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=4)
    tl = DataLoader(tr, batch_size=a.batch_pages, shuffle=True, drop_last=False, **dl_kw)
    vl = DataLoader(va, batch_size=1, shuffle=False, **dl_kw)

    core = net.module if isinstance(net, torch.nn.DataParallel) else net
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, a.epochs))
    scaler = torch.amp.GradScaler("cuda") if dev == "cuda" else None
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def dice(logit, m):
        p = torch.sigmoid(logit)
        num = 2 * (p * m).sum((2, 3)); den = p.sum((2, 3)) + m.sum((2, 3)) + 1e-6
        return 1 - (num / den).mean()

    def set_encoder_frozen(flag):
        for nm in ("firstconv", "firstbn", "encoder1", "encoder2", "encoder3", "encoder4"):
            for prm in getattr(core, nm).parameters():
                prm.requires_grad = not flag

    best = -1.0
    for ep in range(a.epochs):
        frozen = ep < a.freeze_encoder_epochs
        set_encoder_frozen(frozen)
        net.train(); t0 = time.time(); losses = []; nseen = 0
        for step, batch in enumerate(tl):
            if batch is None:
                continue
            x, m, wm = batch
            x = x.to(dev, non_blocking=True); m = m.to(dev, non_blocking=True); wm = wm.to(dev, non_blocking=True)
            if dev == "cuda":
                x = x.to(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, enabled=(dev == "cuda")):
                y = net(x)
                loss = (bce(y, m) * wm).mean() + dice(y, m)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            losses.append(float(loss)); nseen += x.shape[0]
            if step % 40 == 0:
                r = nseen / max(1e-9, time.time() - t0)
                P(f"  ep{ep}{' [enc frozen]' if frozen else ''} step {step}/{len(tl)} "
                  f"loss {np.mean(losses[-40:]):.4f} | {r:.1f} samples/s | eta {(len(tl)-step)*a.batch_pages*a.crops_per_page/max(r,1e-9)/60:.1f} min")
        sched.step()

        net.eval(); inter = np.zeros(3); tot = np.zeros(3)
        with torch.no_grad():
            for batch in vl:
                if batch is None:
                    continue
                x, m, _ = batch
                x = x.to(dev); m = m.to(dev)
                if dev == "cuda":
                    x = x.to(memory_format=torch.channels_last)
                with torch.autocast(dev, enabled=(dev == "cuda")):
                    pr = (torch.sigmoid(net(x)) > 0.8).float()
                inter += (pr * m).sum((0, 2, 3)).float().cpu().numpy()
                tot += (pr.sum((0, 2, 3)) + m.sum((0, 2, 3))).float().cpu().numpy()
        f1 = 2 * inter / np.maximum(tot, 1e-6)
        P(f"ep{ep} done in {(time.time()-t0)/60:.1f} min | train loss {np.mean(losses):.4f} | "
          f"val pixel-F1 word {f1[0]:.4f} border {f1[1]:.4f} line {f1[2]:.4f}")
        if ep == 0:
            gap = [FROZEN_RP_VAL_F1[i] - f1[i] for i in range(3)]
            P(f"  epoch-0 vs frozen RP {FROZEN_RP_VAL_F1}: gap {[round(g, 3) for g in gap]}")
            if max(gap) > 0.15:
                P("  *** WARNING: epoch 0 is far below the untouched checkpoint. The TARGETS are probably "
                  "wrong, not the training. Do not spend GPU on this run — re-check rp_rasterize. ***")
        score = float(f1[0] + f1[2])                       # word + line channels are what detector.py reads
        if score > best:
            best = score
            torch.save(core.state_dict(), out / "best.pt")
            P(f"  new best ({score:.4f}) -> {out/'best.pt'}")
        torch.save(core.state_dict(), out / "last.pt")

    # export with the sigmoid RE-ATTACHED so it is a drop-in for bilimai/detector.py (thresholds probabilities at 0.8)
    core.load_state_dict(torch.load(out / "best.pt", map_location="cpu"))
    core.return_logits = False
    core.eval().to("cpu")
    onnx_path = out / "segm_ft.onnx"
    torch.onnx.export(core, torch.zeros(1, 3, S, S), str(onnx_path),
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "b"}, "output": {0: "b"}}, opset_version=17)
    # torch's exporter writes weights to a sibling .onnx.data by default. bilimai/detector.py is handed
    # a single path, and a copied .onnx without its .data file fails at load — inline it so the artifact
    # is self-contained like the production segm_model.onnx.
    try:
        import onnx
        mdl = onnx.load(str(onnx_path))
        onnx.save(mdl, str(onnx_path), save_as_external_data=False)
        side = Path(str(onnx_path) + ".data")
        if side.exists():
            side.unlink()
        P(f"inlined weights -> single-file {onnx_path} ({onnx_path.stat().st_size/1e6:.0f} MB)")
    except Exception as e:
        P(f"WARNING: could not inline ONNX weights ({e}); keep segm_ft.onnx.data beside the .onnx")
    P(f"exported {onnx_path}  — verify with det_failure.py, then the accept/kill gate in "
      f"plans/DETECTOR-FINETUNE-2026-08-22.md (merged <= 45 is the binding one)")
    json.dump({"best_pixel_f1_sum": best, "epochs": a.epochs, "shrink_r": a.shrink_r,
               "lr": a.lr, "samples_per_step": a.batch_pages * a.crops_per_page},
              open(out / "train_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
