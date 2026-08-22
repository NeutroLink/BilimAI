#!/usr/bin/env python3
"""E4.3 — what does the detector's MERGE/SPLIT cost, on lines already found? (2026-08-22)

crop_cost.py isolated the cost of boxes that are matched but cropped wrong (median IoU 0.81).
But the detector also makes structural errors: merging two lines into one box, or splitting one
line across multiple boxes. Nobody has measured what these cost in CER — this does.

A MERGE is one detector box covering >= 2 GT lines (each >= --cover of its area inside the box).
A SPLIT is one GT line covered by >= 2 detector boxes (each >= --cover of its area inside the line).

For each merge, read the merged box once and each line separately, paired on the joined reference.
For each split, read the fragments joined and the whole line, paired on the same reference.
Compute char-weighted CER and McNemar test; report per detector and "page-level CER points" scaled
by how much of the page is affected.

    eval/.venv/bin/python eval/detectors/merge_split_cost.py --dry-run
    eval/.venv/bin/python eval/detectors/merge_split_cost.py

Writes eval/runs/det_merge_split.json.
"""
import argparse, json, random, statistics, sys, time, unicodedata
from pathlib import Path
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--images", default=str(ROOT / "eval/testset_v2/ru_pages/images"))
ap.add_argument("--rp", default=str(ROOT / "eval/runs/rp_det_lines_v2g.json"))
ap.add_argument("--kraken", default=str(ROOT / "eval/runs/det_kraken.json"))
ap.add_argument("--base", default=str(ROOT / "models/GLM-OCR"))
ap.add_argument("--adapter", default=str(ROOT / "models/adapters/glm-ocr-lora-ru-r5c"))
ap.add_argument("--out", default=str(ROOT / "eval/runs/det_merge_split.json"))
ap.add_argument("--cover", type=float, default=0.6, help="min fractional overlap to count as covered")
ap.add_argument("--dets", default="rp,kraken", help="detectors to analyze (comma-sep)")
ap.add_argument("--batch-size", type=int, default=24)   # measured peak on M1 Pro MPS
ap.add_argument("--dry-run", action="store_true", help="build sample, print geometry, exit before reading")
a = ap.parse_args()

sys.path.insert(0, str(ROOT))
from PIL import Image


def bbox_area(b):
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))


def bbox_intersection(b1, b2):
    x0, y0, x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    return max(0.0, (x1 - x0) * (y1 - y0))


def cer(ref, hyp):
    ref, hyp = unicodedata.normalize("NFC", ref), unicodedata.normalize("NFC", hyp)
    d = list(range(len(hyp) + 1))
    for i, cr in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, ch in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (cr != ch))
    return d[-1] / max(1, len(ref))


gt = json.load(open(a.gt, encoding="utf-8"))
detectors = {}
for det_name in a.dets.split(","):
    det_name = det_name.strip()
    path = getattr(a, det_name, None)
    if path:
        detectors[det_name] = json.load(open(path, encoding="utf-8"))

# compute merges and splits per detector
all_results = {}

for det_name, pred in detectors.items():
    print(f"\n=== {det_name.upper()} ===", flush=True)

    merges = []  # [(fn, det_box, covered_line_indices)]
    splits = []  # [(fn, gt_line_idx, gt_box, covering_det_indices)]

    for fn, page in gt.items():
        P = pred.get(fn, [])
        gt_lines = page["lines"]
        if not P or not gt_lines:
            continue
        G = [l["bbox"] for l in gt_lines]
        Pb = [b[:4] for b in P]

        # AUTHORITATIVE CLASSIFICATION — identical to eval/detectors/det_failure.py, so the counts here
        # reconcile with the benchmark table (RP: 37 merged / 13 split). The gate that matters is
        # `best IoU < 0.5`: a GT line whose own box already matches well is MATCHED, and is not a merge
        # just because the box happens to overlap its neighbour too. Without that gate every box on a
        # densely written page looks like a merge (an earlier version of this script reported 2,014).
        import numpy as _np
        M = _np.zeros((len(G), len(Pb))); INTER = _np.zeros_like(M)
        AG = _np.array([max(1e-9, (g[2]-g[0])*(g[3]-g[1])) for g in G])
        for i, g in enumerate(G):
            for j, q in enumerate(Pb):
                x0, y0, x1, y1 = max(q[0], g[0]), max(q[1], g[1]), min(q[2], g[2]), min(q[3], g[3])
                inter = max(0.0, x1-x0) * max(0.0, y1-y0)
                INTER[i, j] = inter
                M[i, j] = inter / max(1e-9, (q[2]-q[0])*(q[3]-q[1]) + (g[2]-g[0])*(g[3]-g[1]) - inter)

        seen_merge_box = set()
        for i in range(len(G)):
            j = int(M[i].argmax()); best = float(M[i, j])
            if best >= 0.5 or best < 0.1:
                continue                                    # matched, or missed — neither is a merge/split
            cover_other = [INTER[k, j] / AG[k] for k in range(len(G)) if k != i]
            n_parts = [jj for jj in range(len(Pb)) if INTER[i, jj] / AG[i] >= 0.25]
            if max(cover_other, default=0.0) >= 0.4:
                if j in seen_merge_box:
                    continue                                # one case per detector box, not per GT line
                seen_merge_box.add(j)
                covered = sorted(k for k in range(len(G)) if INTER[k, j] / AG[k] >= 0.4)
                if len(covered) >= 2:
                    merges.append((fn, Pb[j], covered))
            elif len(n_parts) >= 2:
                splits.append((fn, i, G[i], n_parts))

    n_merges, n_splits = len(merges), len(splits)
    total_crops = sum(1 + len(covered) for _, _, covered in merges) + sum(1 + len(covering) for _, _, _, covering in splits)

    print(f"merges found: {n_merges} | splits found: {n_splits} | total crops that would be read: {total_crops}", flush=True)

    if a.dry_run:
        if merges:
            print(f"example merges:")
            for i, (fn, det_box, covered) in enumerate(merges[:3]):
                gt_texts = [gt[fn]["lines"][li].get("text", "").strip() for li in covered]
                joined = " ".join(gt_texts)
                print(f"  {i+1}. {fn}: {joined[:50]}", flush=True)
        continue

    if not merges and not splits:
        print("no merges or splits found; skipping reader", flush=True)
        all_results[det_name] = {"n_merges": 0, "n_splits": 0, "records": []}
        continue

    # prepare crops for reading
    imgs, meta = [], []
    cache = {}
    PAD = 12

    # merge crops: (merged, separate_1, separate_2, ...)
    for fn, det_box, covered in merges:
        if fn not in cache:
            cache[fn] = Image.open(Path(a.images) / fn).convert("RGB")
        im = cache[fn]
        W, H = im.size

        # merged: read the detector box
        x0 = max(0, int(det_box[0] - PAD))
        y0 = max(0, int(det_box[1] - PAD))
        x1 = min(W, int(det_box[2] + PAD))
        y1 = min(H, int(det_box[3] + PAD))
        imgs.append(im.crop((x0, y0, x1, y1)))

        gt_texts = [gt[fn]["lines"][li].get("text", "").strip() for li in covered]
        ref_text = " ".join(gt_texts)
        meta.append(("merge_merged", fn, covered, ref_text, len(ref_text)))

        # separate: read each line
        for li in covered:
            g = gt[fn]["lines"][li]["bbox"]
            x0 = max(0, int(g[0] - PAD))
            y0 = max(0, int(g[1] - PAD))
            x1 = min(W, int(g[2] + PAD))
            y1 = min(H, int(g[3] + PAD))
            imgs.append(im.crop((x0, y0, x1, y1)))
            meta.append(("merge_separate", fn, li, ref_text, len(ref_text)))

    # split crops: (whole, fragment_1, fragment_2, ...)
    for fn, li, gt_box, covering in splits:
        if fn not in cache:
            cache[fn] = Image.open(Path(a.images) / fn).convert("RGB")
        im = cache[fn]
        W, H = im.size

        ref_text = gt[fn]["lines"][li].get("text", "").strip()

        # whole: read the GT line
        x0 = max(0, int(gt_box[0] - PAD))
        y0 = max(0, int(gt_box[1] - PAD))
        x1 = min(W, int(gt_box[2] + PAD))
        y1 = min(H, int(gt_box[3] + PAD))
        imgs.append(im.crop((x0, y0, x1, y1)))
        meta.append(("split_whole", fn, li, ref_text, len(ref_text)))

        # split: read each fragment
        P = pred[fn]
        P_boxes = [p[:4] for p in P]
        for pi in covering:
            det_box = P_boxes[pi]
            x0 = max(0, int(det_box[0] - PAD))
            y0 = max(0, int(det_box[1] - PAD))
            x1 = min(W, int(det_box[2] + PAD))
            y1 = min(H, int(det_box[3] + PAD))
            imgs.append(im.crop((x0, y0, x1, y1)))
            meta.append(("split_fragment", fn, li, ref_text, len(ref_text)))

    # read all crops
    from bilimai.reader import make_reader
    reader = make_reader(a.base, a.adapter, line_h=128, max_new_tokens=96)
    print(f"reader {reader.name} | reading {len(imgs)} crops...", flush=True)
    t0 = time.time()
    texts, _ = reader.read(imgs, batch_size=a.batch_size, with_conf=False)
    print(f"read in {time.time()-t0:.0f}s", flush=True)

    # match outputs to arms and compute CER
    merge_records = []
    split_records = []

    # process merge records
    idx = 0
    for fn, det_box, covered in merges:
        merged_hyp = (texts[idx] or "").strip()
        idx += 1

        sep_hyps = []
        for li in covered:
            sep_hyps.append((texts[idx] or "").strip())
            idx += 1

        ref_text = " ".join([gt[fn]["lines"][li].get("text", "").strip() for li in covered])

        merged_cer_val = cer(ref_text, merged_hyp)
        sep_cer_val = cer(ref_text, " ".join(sep_hyps))

        merge_records.append({
            "type": "merge",
            "filename": fn,
            "n_covered": len(covered),
            "ref": ref_text,
            "merged_hyp": merged_hyp,
            "sep_hyps": sep_hyps,
            "merged_cer": merged_cer_val,
            "sep_cer": sep_cer_val,
            "delta_cer": merged_cer_val - sep_cer_val,
            "n_chars": len(ref_text)
        })

    # process split records
    for fn, li, gt_box, covering in splits:
        whole_hyp = (texts[idx] or "").strip()
        idx += 1

        frag_hyps = []
        for pi in covering:
            frag_hyps.append((texts[idx] or "").strip())
            idx += 1

        ref_text = gt[fn]["lines"][li].get("text", "").strip()

        whole_cer_val = cer(ref_text, whole_hyp)
        frag_cer_val = cer(ref_text, " ".join(frag_hyps))

        split_records.append({
            "type": "split",
            "filename": fn,
            "n_fragments": len(covering),
            "ref": ref_text,
            "whole_hyp": whole_hyp,
            "frag_hyps": frag_hyps,
            "whole_cer": whole_cer_val,
            "frag_cer": frag_cer_val,
            "delta_cer": frag_cer_val - whole_cer_val,
            "n_chars": len(ref_text)
        })

    # char-weighted aggregation
    all_records = merge_records + split_records

    # compute arm CERs separately then combine
    merge_ch = sum(r["n_chars"] for r in merge_records)
    split_ch = sum(r["n_chars"] for r in split_records)
    total_ch = merge_ch + split_ch

    if merge_ch > 0:
        merge_arm1_cer = sum(r["merged_cer"] * r["n_chars"] for r in merge_records) / merge_ch
        merge_arm2_cer = sum(r["sep_cer"] * r["n_chars"] for r in merge_records) / merge_ch
    else:
        merge_arm1_cer = merge_arm2_cer = 0

    if split_ch > 0:
        split_arm1_cer = sum(r["frag_cer"] * r["n_chars"] for r in split_records) / split_ch
        split_arm2_cer = sum(r["whole_cer"] * r["n_chars"] for r in split_records) / split_ch
    else:
        split_arm1_cer = split_arm2_cer = 0

    # overall char-weighted CER for each arm
    if total_ch > 0:
        arm1_overall = (merge_arm1_cer * merge_ch + split_arm1_cer * split_ch) / total_ch
        arm2_overall = (merge_arm2_cer * merge_ch + split_arm2_cer * split_ch) / total_ch
    else:
        arm1_overall = arm2_overall = 0

    # char-weighted mean cost per record
    deltas = [r["delta_cer"] for r in all_records]
    if total_ch > 0:
        mean_delta = sum(r["delta_cer"] * r["n_chars"] for r in all_records) / total_ch
    else:
        mean_delta = 0

    # McNemar: count how many records have arm1 worse than arm2
    n_worse = sum(1 for d in deltas if d > 1e-9)
    n_better = sum(1 for d in deltas if d < -1e-9)

    # McNemar test
    if n_worse + n_better > 0:
        mcnemar_p = binomtest(n_worse, n_worse + n_better, 0.5, alternative="two-sided").pvalue
    else:
        mcnemar_p = 1.0

    # total characters on exam set
    all_gt_chars = sum(sum(len(l.get("text", "").strip()) for l in page["lines"]) for page in gt.values())
    affected_chars = total_ch

    page_level_cer_points = mean_delta * (affected_chars / max(all_gt_chars, 1))

    print()
    print(f"n merge cases       : {len(merge_records)}")
    print(f"n split cases       : {len(split_records)}")
    print(f"total paired cases  : {len(all_records)}")
    print(f"char-weighted CER, arm1 (detector): {arm1_overall:.5f}")
    print(f"char-weighted CER, arm2 (correct) : {arm2_overall:.5f}")
    print(f"mean delta          : {mean_delta:+.5f}")
    print(f"McNemar p           : {mcnemar_p:.4f}")
    print(f"page-level CER points: {page_level_cer_points:+.4f}")
    print(f"affected characters : {affected_chars} / {all_gt_chars} ({100*affected_chars/max(all_gt_chars, 1):.1f}%)")

    if len(all_records) < 9700 / 2:
        print(f"⚠ n={len(all_records)} cases is small; ~9700 pairs needed to resolve 2-point effect")

    all_results[det_name] = {
        "n_merges": len(merge_records),
        "n_splits": len(split_records),
        "n_total": len(all_records),
        "arm1_cer_charweighted": arm1_overall,
        "arm2_cer_charweighted": arm2_overall,
        "mean_delta": mean_delta,
        "mcnemar_p": mcnemar_p,
        "page_level_cer_points": page_level_cer_points,
        "affected_chars": affected_chars,
        "total_chars": all_gt_chars,
        "records": all_records[:100]
    }

if a.dry_run:
    print("\n[dry-run complete]", flush=True)
else:
    json.dump(all_results, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {a.out}")
