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
  line_cer_charweighted — PRIMARY (2026-08-18): total char edits ÷ total GT chars over matched lines
                         (big lines count more; the median-of-pages headline cannot resolve < 0.045).
                         *_all counts missed lines as fully wrong (the honest end-to-end number).
  word_accuracy        — share of GT words produced correctly (what a grader feels; 1 − word errors)
  lines_perfect_pct    — share of GT lines read with zero edits
  ci95_*               — page-bootstrap 95 % CI (1000 resamples of pages, seed 0)
  novel_*/seen_*       — split of the exam lines by whether a text twin (rapidfuzz ratio ≥ 80) exists in
                         the training labels (--train-labels); lines < 8 chars excluded from the split; novel is honest.
  --baseline pred.json — paired page-bootstrap on the difference of the primary metric vs another run

Usage:
  score.py --gt eval/testset_v1/ru_pages/ground_truth.json --pred runs/glm_zero.json [--csv out.csv]
           [--train-labels data/derived/strips_ru_v1/labels.jsonl] [--baseline runs/other.json] [--boot 1000]
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


def word_hits(ref: str, hyp: str) -> int:
    """Number of GT words produced correctly (jiwer alignment hits)."""
    ref, hyp = norm(ref), norm(hyp)
    if not ref: return 0
    if not hyp: return 0
    try:
        return int(jiwer.process_words(ref, hyp).hits)
    except Exception:
        return sum(1 for a, b in zip(ref.split(), hyp.split()) if a == b)


def seen_lines(gt, train_labels, cache=None, cutoff=80, min_chars=8, gt_path=None):
    """Which exam lines have a text twin (rapidfuzz ratio ≥ cutoff) in the training labels.
    Definition (2026-08-18): lines shorter than `min_chars` normalised characters ("ед.ч.", dates, single words) are EXCLUDED
    from the split (value None) — they match short training labels by chance and inflated "seen" from 39 % to 57 %.
    Cache is keyed by BOTH the training-labels dir and a hash of the GT file, so exam v1/v2 never share a cache."""
    import os, hashlib
    gt_key = hashlib.sha1(json.dumps({fn: [l["text"] for l in p["lines"]] for fn, p in gt.items()}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:10]
    cache = cache or os.path.join(os.path.dirname(train_labels), f"seen_lines_{os.path.basename(os.path.dirname(train_labels) or 'train')}_gt{gt_key}_min{min_chars}_r{cutoff}.json")
    if os.path.isfile(cache):
        return json.load(open(cache, encoding="utf-8"))
    from rapidfuzz import process, fuzz
    train = list({norm(json.loads(l)["text"]) for l in open(train_labels, encoding="utf-8") if json.loads(l).get("split", "train") == "train"})
    seen = {}
    for fn, page in gt.items():
        for i, l in enumerate(page["lines"]):
            t = norm(l["text"])
            if len(t) < min_chars: seen[f"{fn}#{i}"] = None; continue
            m = process.extractOne(t, train, scorer=fuzz.ratio, score_cutoff=cutoff)
            seen[f"{fn}#{i}"] = bool(m)
    json.dump(seen, open(cache, "w"), ensure_ascii=False)
    return seen


def aggregate(rows, only=None):
    """Char-weighted / word / perfect metrics over line records (optionally filtered by a predicate on (file, gi))."""
    L = [(r["file"], x) for r in rows for x in r["lines"] if only is None or only(r["file"], x["gi"])]
    ch_all = sum(x["n_chars"] for _, x in L); ed_all = sum(x["edits"] for _, x in L)
    M = [x for _, x in L if x["hyp"] is not None]
    ch_m = sum(x["n_chars"] for x in M); ed_m = sum(x["edits"] for x in M)
    nw = sum(x["n_words"] for _, x in L); hits = sum(x["word_hits"] for _, x in L)
    return {"lines": len(L), "line_cer_charweighted": ed_m / ch_m if ch_m else None, "line_cer_charweighted_all": ed_all / ch_all if ch_all else None,
            "word_accuracy": hits / nw if nw else None, "lines_perfect_pct": sum(x["perfect"] for _, x in L) / len(L) if L else None}


def _page_sums(rows, key):
    """Per-page (numerator, denominator) so a char-weighted metric of any page subset is a ratio of sums → vectorised bootstrap."""
    num, den = [], []
    for r in rows:
        L = r["lines"]
        if key == "line_cer_charweighted":
            M = [x for x in L if x["hyp"] is not None]; num.append(sum(x["edits"] for x in M)); den.append(sum(x["n_chars"] for x in M))
        elif key == "line_cer_charweighted_all":
            num.append(sum(x["edits"] for x in L)); den.append(sum(x["n_chars"] for x in L))
        elif key == "word_accuracy":
            num.append(sum(x["word_hits"] for x in L)); den.append(sum(x["n_words"] for x in L))
        else: raise KeyError(key)
    return np.array(num, dtype=float), np.array(den, dtype=float)


def page_bootstrap(rows, key, n=1000, seed=0, rows_b=None):
    """95 % CI of a char-weighted metric by resampling pages (vectorised: n × k index draws, sums via take);
    with rows_b (same pages, other run) → CI of the paired difference."""
    rng = np.random.default_rng(seed); k = len(rows)
    num, den = _page_sums(rows, key)
    picks = rng.integers(0, k, (n, k))
    vals = num[picks].sum(1) / np.maximum(den[picks].sum(1), 1e-9)
    if rows_b:
        idx_b = {r["file"]: r for r in rows_b}; rb = [idx_b[r["file"]] for r in rows]
        num_b, den_b = _page_sums(rb, key)
        vals = vals - num_b[picks].sum(1) / np.maximum(den_b[picks].sum(1), 1e-9)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


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
    """Match predicted lines to GT lines; returns list of (gt_idx, pred_idx or None).

    If every prediction carries a bbox, match by geometry (IoU) — the fair choice when the
    model was given the line positions or reports them. Otherwise match by text similarity
    (Hungarian assignment) so page ordering differences are not punished.
    """
    if not pred_lines:
        return [(i, None) for i in range(len(gt_lines))]
    if all(p.get("bbox") is not None for p in pred_lines):
        sim = np.zeros((len(gt_lines), len(pred_lines)))
        for i, g in enumerate(gt_lines):
            for j, p in enumerate(pred_lines):
                sim[i, j] = iou(g["bbox"], p["bbox"])
        thresh = 0.3
    else:
        sim = np.zeros((len(gt_lines), len(pred_lines)))
        for i, g in enumerate(gt_lines):
            gt = norm(g["text"])
            for j, p in enumerate(pred_lines):
                sim[i, j] = SequenceMatcher(None, gt, norm(p["text"])).ratio()
        thresh = 0.35                    # below this it's not the same line, treat GT line as missed
    rows, cols = linear_sum_assignment(-sim)
    assigned = {r: c for r, c in zip(rows, cols) if sim[r, c] >= thresh}
    return [(i, assigned.get(i)) for i in range(len(gt_lines))]


def score_page(gt_page, pred):
    gt_lines = gt_page["lines"]
    pred_lines = to_lines(pred)
    pairs = match_lines(gt_lines, pred_lines)

    ref_concat, hyp_concat = [], []
    line_cers, box_hits, box_total = [], 0, 0
    matched = 0; recs = []
    for gi, pj in pairs:
        ref_concat.append(gt_lines[gi]["text"])
        nref = norm(gt_lines[gi]["text"]); nchar = len(nref); nword = len(nref.split())
        if pj is None:
            hyp_concat.append("")
            recs.append({"gi": gi, "ref": gt_lines[gi]["text"], "hyp": None, "cer": None, "edits": nchar, "n_chars": nchar, "n_words": nword, "word_hits": 0, "perfect": False})
            continue
        matched += 1
        hyp = pred_lines[pj]["text"]
        hyp_concat.append(hyp)
        c = cer(gt_lines[gi]["text"], hyp); line_cers.append(c)
        recs.append({"gi": gi, "ref": gt_lines[gi]["text"], "hyp": hyp, "cer": c, "edits": c * nchar, "n_chars": nchar,
                     "n_words": nword, "word_hits": word_hits(gt_lines[gi]["text"], hyp), "perfect": c == 0.0})
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
        "lines": recs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--train-labels", help="labels.jsonl of the training strips → novel/seen split of the exam lines")
    ap.add_argument("--baseline", help="another predictions JSON → paired page-bootstrap CI of the difference (primary metric)")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    if not a.train_labels:
        import os
        _d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "derived", "strips_ru_v1", "labels.jsonl")
        if os.path.isfile(_d): a.train_labels = _d
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
        "page_cer_median": float(np.median([r["page_cer"] for r in rows])),
        "page_wer_mean": float(np.mean([r["page_wer"] for r in rows])),
        "line_cer_mean": float(np.mean([r["line_cer"] for r in rows])),
        "line_cer_median": float(np.median([r["line_cer"] for r in rows])),
        "line_recall_mean": float(np.mean([r["line_recall"] for r in rows])),
        "extra_lines_total": int(sum(r["extra_lines"] for r in rows)),
    }
    bh = [r["box_hit_rate"] for r in rows if r["box_hit_rate"] is not None]
    overall["box_hit_rate_mean"] = float(np.mean(bh)) if bh else None
    # ---- primary metrics (2026-08-18): char-weighted line CER, word accuracy, perfect lines, CI, novel-text subset
    overall.update(aggregate(rows))
    lo, hi = page_bootstrap(rows, "line_cer_charweighted", n=a.boot); overall["ci95_line_cer_charweighted"] = [lo, hi]
    lo, hi = page_bootstrap(rows, "line_cer_charweighted_all", n=a.boot); overall["ci95_line_cer_charweighted_all"] = [lo, hi]
    if a.train_labels:
        try:
            seen = seen_lines(gt, a.train_labels)
            nov = aggregate(rows, only=lambda f, gi: seen.get(f"{f}#{gi}") is False); sn = aggregate(rows, only=lambda f, gi: seen.get(f"{f}#{gi}") is True)
            overall.update({"novel_lines": nov["lines"], "novel_line_cer_charweighted": nov["line_cer_charweighted"], "novel_word_accuracy": nov["word_accuracy"],
                            "seen_lines": sn["lines"], "seen_line_cer_charweighted": sn["line_cer_charweighted"],
                            "short_lines_excluded": sum(1 for v in seen.values() if v is None)})
        except Exception as e:
            print(f"WARNING: novel/seen split skipped ({e})", file=sys.stderr)
    if a.baseline:
        base = json.load(open(a.baseline, encoding="utf-8")); rows_b = []
        for fn, gpage in gt.items():
            if fn in base: rb = score_page(gpage, base[fn]); rb["file"] = fn; rows_b.append(rb)
        common = {r["file"] for r in rows_b}; rows_p = [r for r in rows if r["file"] in common]
        d = aggregate(rows_p)["line_cer_charweighted"] - aggregate(rows_b)["line_cer_charweighted"]
        lo, hi = page_bootstrap(rows_p, "line_cer_charweighted", n=a.boot, rows_b=rows_b)
        overall["delta_vs_baseline"] = {"line_cer_charweighted": d, "ci95": [lo, hi], "significant": bool(hi < 0 or lo > 0), "baseline": a.baseline}
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
            print(f"{k:>30}: {v:.4f}" if isinstance(v, float) else f"{k:>30}: {v}")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            cols = [k for k in rows[0].keys() if k != "lines"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(json.dumps(overall, ensure_ascii=False))


if __name__ == "__main__":
    main()
