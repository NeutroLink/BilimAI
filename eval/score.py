#!/usr/bin/env python
"""Score a model's transcriptions of the sealed test set against ground truth.

Input: a predictions JSON  {"<image file>": <prediction>, ...}  where <prediction> is either
  - a plain string (whole-page transcription; lines separated by newlines), or
  - a list of {"text": ..., "bbox": [x0,y0,x1,y1]} dicts (line-level, with coordinates).

Metrics (per page and overall):
  page_cer / page_wer  — whole page as one text, ORDER-INSENSITIVE: predicted lines are
                         matched to ground-truth lines by text similarity (Hungarian
                         assignment), then concatenated in GT order. Punishes wrong/missing
                         lines, not left/right-page ordering differences.
  line_cer             — mean CER over matched line pairs (how well it reads a line it found)
  line_recall          — share of GT lines that got a matched prediction (did it find the line)
  box_hit_rate         — when bboxes are provided: share of matched lines whose predicted
                         box overlaps the GT line box (IoU > 0.3). Measures "can it point".

Usage:
  score.py --gt eval/testset_v1/ru_pages/ground_truth.json --pred runs/glm_zero.json [--csv out.csv]
"""
import argparse, json, re, unicodedata, sys
from difflib import SequenceMatcher

import jiwer
import numpy as np
from scipy.optimize import linear_sum_assignment


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("ё", "е").replace("Ё", "Е")          # dataset is inconsistent on ё; don't punish it
    s = s.replace("­", "")                          # soft hyphen
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def cer(ref: str, hyp: str) -> float:
    ref, hyp = norm(ref), norm(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.cer(ref, hyp)


def wer(ref: str, hyp: str) -> float:
    ref, hyp = norm(ref), norm(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.wer(ref, hyp)


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def to_lines(pred):
    """Normalise a prediction into a list of {'text', 'bbox' or None}."""
    if isinstance(pred, str):
        return [{"text": t, "bbox": None} for t in pred.splitlines() if norm(t)]
    out = []
    for p in pred:
        if isinstance(p, str):
            out.append({"text": p, "bbox": None})
        else:
            out.append({"text": p.get("text", ""), "bbox": p.get("bbox")})
    return [o for o in out if norm(o["text"])]


def match_lines(gt_lines, pred_lines):
    """Hungarian match on text similarity; returns list of (gt_idx, pred_idx or None)."""
    if not pred_lines:
        return [(i, None) for i in range(len(gt_lines))]
    sim = np.zeros((len(gt_lines), len(pred_lines)))
    for i, g in enumerate(gt_lines):
        gt = norm(g["text"])
        for j, p in enumerate(pred_lines):
            sim[i, j] = SequenceMatcher(None, gt, norm(p["text"])).ratio()
    rows, cols = linear_sum_assignment(-sim)
    assigned = {}
    for r, c in zip(rows, cols):
        if sim[r, c] >= 0.35:            # below this it's not the same line, treat GT line as missed
            assigned[r] = c
    return [(i, assigned.get(i)) for i in range(len(gt_lines))]


def score_page(gt_page, pred):
    gt_lines = gt_page["lines"]
    pred_lines = to_lines(pred)
    pairs = match_lines(gt_lines, pred_lines)

    ref_concat, hyp_concat = [], []
    line_cers, box_hits, box_total = [], 0, 0
    matched = 0
    for gi, pj in pairs:
        ref_concat.append(gt_lines[gi]["text"])
        if pj is None:
            hyp_concat.append("")
            continue
        matched += 1
        hyp = pred_lines[pj]["text"]
        hyp_concat.append(hyp)
        line_cers.append(cer(gt_lines[gi]["text"], hyp))
        if pred_lines[pj]["bbox"] is not None:
            box_total += 1
            if iou(pred_lines[pj]["bbox"], gt_lines[gi]["bbox"]) > 0.3:
                box_hits += 1
    # unmatched predictions (hallucinated / split lines) count as insertions
    used = {pj for _, pj in pairs if pj is not None}
    extra = [pred_lines[j]["text"] for j in range(len(pred_lines)) if j not in used]
    ref_text = "\n".join(ref_concat)
    hyp_text = "\n".join(hyp_concat + extra)
    return {
        "page_cer": cer(ref_text, hyp_text),
        "page_wer": wer(ref_text, hyp_text),
        "line_cer": float(np.mean(line_cers)) if line_cers else 1.0,
        "line_recall": matched / len(gt_lines) if gt_lines else 1.0,
        "extra_lines": len(extra),
        "box_hit_rate": (box_hits / box_total) if box_total else None,
        "n_gt_lines": len(gt_lines),
        "n_gt_chars": gt_page["n_chars"],
        "teacher_marks": gt_page.get("teacher_marks", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    gt = json.load(open(a.gt, encoding="utf-8"))
    pred = json.load(open(a.pred, encoding="utf-8"))

    rows = []
    for fn, gpage in gt.items():
        if fn not in pred:
            print(f"WARNING: no prediction for {fn}", file=sys.stderr)
            continue
        r = score_page(gpage, pred[fn]); r["file"] = fn
        rows.append(r)
    if not rows:
        sys.exit("no pages scored")

    # character-weighted overall CER: total edits / total GT chars
    tot_chars = sum(r["n_gt_chars"] for r in rows)
    overall = {
        "pages": len(rows),
        "page_cer_charweighted": sum(r["page_cer"] * r["n_gt_chars"] for r in rows) / tot_chars,
        "page_wer_mean": float(np.mean([r["page_wer"] for r in rows])),
        "line_cer_mean": float(np.mean([r["line_cer"] for r in rows])),
        "line_recall_mean": float(np.mean([r["line_recall"] for r in rows])),
        "extra_lines_total": int(sum(r["extra_lines"] for r in rows)),
    }
    bh = [r["box_hit_rate"] for r in rows if r["box_hit_rate"] is not None]
    overall["box_hit_rate_mean"] = float(np.mean(bh)) if bh else None
    clean = [r for r in rows if r["teacher_marks"] == 0]; marked = [r for r in rows if r["teacher_marks"] > 0]
    if clean and marked:
        overall["page_cer_clean_pages"] = float(np.mean([r["page_cer"] for r in clean]))
        overall["page_cer_marked_pages"] = float(np.mean([r["page_cer"] for r in marked]))

    if not a.quiet:
        print(f"{'file':>10} {'CER':>6} {'WER':>6} {'lineCER':>8} {'recall':>7} {'extra':>5} {'boxes':>6} {'lines':>5} {'red':>4}")
        for r in sorted(rows, key=lambda r: r["page_cer"]):
            bx = f"{r['box_hit_rate']:.2f}" if r["box_hit_rate"] is not None else "  -  "
            print(f"{r['file']:>10} {r['page_cer']:6.3f} {r['page_wer']:6.3f} {r['line_cer']:8.3f} {r['line_recall']:7.2f} {r['extra_lines']:5d} {bx:>6} {r['n_gt_lines']:5d} {r['teacher_marks']:4d}")
        print("-" * 70)
        for k, v in overall.items():
            print(f"{k:>26}: {v:.4f}" if isinstance(v, float) else f"{k:>26}: {v}")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps(overall))


if __name__ == "__main__":
    main()
