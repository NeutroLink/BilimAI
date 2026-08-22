#!/usr/bin/env python3
"""Measure reading-order cost in detector boxes: permutation error when lines are matched.

The E2E gap between real detector boxes and human boxes is 4.3 CER points. Misses account
for 1.7 (crop_cost.py measures that term). This isolates the READING ORDER term alone.

For each page:
  1. Match every detector box to its best-IoU GT line (threshold --iou, default 0.5).
     One GT line can only be claimed once — resolve greedily by descending IoU.
  2. Restrict to the set of GT lines that were matched.
  3. reference  = "\n".join(GT text of those lines, in GT order)
     hypothesis = "\n".join(the SAME texts, in the detector's box order)
  4. Page-level CER of hypothesis vs reference (using GT text for both).

Reports char-weighted CER across pages, plus diagnostic breakdown:
  * Kendall tau / inversion count per page
  * How many pages have non-identity order, and CER for that subset only
  * Worst 5 pages with specific line texts (truncated to 40 chars)
  * Breakdown by spread ratio (width/height > 1.15) — column risk indicator

Run for BOTH detectors via --dets rp,kraken. No reader or GPU needed — purely structural.

    eval/.venv/bin/python eval/detectors/reading_order_cost.py --dry-run
    eval/.venv/bin/python eval/detectors/reading_order_cost.py
"""
import argparse, json, statistics, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--dets", default="rp,kraken", help="comma-separated detector names")
ap.add_argument("--rp", default=str(ROOT / "eval/runs/rp_det_lines_v2g.json"))
ap.add_argument("--kraken", default=str(ROOT / "eval/runs/det_kraken.json"))
ap.add_argument("--out", default=str(ROOT / "eval/runs/det_reading_order.json"))
ap.add_argument("--iou", type=float, default=0.5, help="min best-IoU to count a line as matched")
ap.add_argument("--dry-run", action="store_true", help="build everything, print counts, exit")
a = ap.parse_args()

sys.path.insert(0, str(ROOT))


def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / max(1e-9, (p[2] - p[0]) * (p[3] - p[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter)


def cer(ref, hyp):
    ref, hyp = unicodedata.normalize("NFC", ref), unicodedata.normalize("NFC", hyp)
    d = list(range(len(hyp) + 1))
    for i, cr in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, ch in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (cr != ch))
    return d[-1] / max(1, len(ref))


def inversion_count(perm):
    """Count inversions in a permutation (Kendall tau measure)."""
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inv += 1
    return inv


# Load data
gt = json.load(open(a.gt, encoding="utf-8"))

dets_map = {}
for det_name in a.dets.split(","):
    det_name = det_name.strip()
    det_file = getattr(a, det_name, None)
    if det_file:
        dets_map[det_name] = json.load(open(det_file, encoding="utf-8"))

# Build per-detector results
results = {}

for det_name, det_data in dets_map.items():
    print(f"\n{det_name.upper()}:")
    pages_data = []

    for fn, page in sorted(gt.items()):
        W, H = page["width"], page["height"]
        spread = W / H

        gt_lines = page["lines"]  # already in GT reading order
        P = det_data.get(fn, [])

        if not P or not gt_lines:
            continue

        # Build list of (IoU, gt_idx, det_idx) for all valid pairings
        candidates = []
        for gt_idx, gt_line in enumerate(gt_lines):
            txt = (gt_line.get("text") or "").strip()
            if not txt:
                continue
            gt_box = gt_line["bbox"]

            for det_idx, det_box in enumerate(P):
                o = iou(det_box[:4], gt_box)
                if o >= a.iou:
                    candidates.append((o, gt_idx, det_idx))

        if not candidates:
            continue

        # Greedy matching: sort by descending IoU, assign each match if both are unassigned
        candidates.sort(reverse=True)
        assigned_gt = set()
        assigned_det = set()
        final_matches = {}  # gt_idx -> det_idx

        for o, gt_idx, det_idx in candidates:
            if gt_idx not in assigned_gt and det_idx not in assigned_det:
                final_matches[gt_idx] = det_idx
                assigned_gt.add(gt_idx)
                assigned_det.add(det_idx)

        if not final_matches:
            continue

        # Build GT order and detector order sequences
        matched_gt_indices = sorted(final_matches.keys())
        gt_texts = [gt_lines[i]["text"].strip() for i in matched_gt_indices]

        # Detector order: sort matched lines by their assigned detector box index
        det_sorted = sorted([(final_matches[gt_idx], gt_idx) for gt_idx in matched_gt_indices])
        det_order_indices = [gt_idx for _, gt_idx in det_sorted]
        det_texts = [gt_lines[i]["text"].strip() for i in det_order_indices]

        # Build strings with newline separation
        ref_text = "\n".join(gt_texts)
        hyp_text = "\n".join(det_texts)

        # Calculate CER
        page_cer = cer(ref_text, hyp_text)

        # Kendall tau: permutation of matched_gt_indices in det_order_indices
        perm = [det_order_indices.index(i) for i in matched_gt_indices]
        inv_count = inversion_count(perm)

        # Is order identity?
        is_identity = (det_order_indices == matched_gt_indices)

        # Total chars in this page's matched lines (text + newlines, matching CER denominator)
        n_chars = sum(len(t) for t in gt_texts) + max(0, len(gt_texts) - 1)

        page_data = {
            "file": fn,
            "n_matched": len(matched_gt_indices),
            "cer": page_cer,
            "inversions": inv_count,
            "is_identity": is_identity,
            "spread": spread,
            "n_chars": n_chars,
            "texts": [t[:40] for t in gt_texts],  # truncate for diagnostics
        }
        pages_data.append(page_data)

    # Print dry-run info and exit
    if a.dry_run:
        total_matched = sum(p["n_matched"] for p in pages_data)
        total_chars = sum(p["n_chars"] for p in pages_data)
        print(f"  pages with matches: {len(pages_data)}")
        print(f"  total matched lines: {total_matched}")
        print(f"  total chars in matched lines: {total_chars}")
        continue

    # Calculate aggregate stats
    total_chars = sum(p["n_chars"] for p in pages_data)

    # Char-weighted CER
    if total_chars > 0:
        char_weighted_cer = sum(p["cer"] * p["n_chars"] for p in pages_data) / total_chars
    else:
        char_weighted_cer = 0.0

    # Pages with non-identity order
    non_identity = [p for p in pages_data if not p["is_identity"]]
    if non_identity:
        non_identity_cw_cer = sum(p["cer"] * p["n_chars"] for p in non_identity) / sum(p["n_chars"] for p in non_identity)
    else:
        non_identity_cw_cer = 0.0

    # Worst 5 pages by CER
    worst_5 = sorted(pages_data, key=lambda p: p["cer"], reverse=True)[:5]

    # Spread breakdown (width/height > 1.15)
    spreads = [p for p in pages_data if p["spread"] > 1.15]
    non_spreads = [p for p in pages_data if p["spread"] <= 1.15]

    spread_cw_cer = sum(p["cer"] * p["n_chars"] for p in spreads) / sum(p["n_chars"] for p in spreads) if spreads else 0.0
    non_spread_cw_cer = sum(p["cer"] * p["n_chars"] for p in non_spreads) / sum(p["n_chars"] for p in non_spreads) if non_spreads else 0.0

    # Print results
    print()
    print(f"pages with matches          : {len(pages_data)}")
    print(f"char-weighted CER           : {char_weighted_cer:.5f}")
    print(f"pages with non-identity order: {len(non_identity)} / {len(pages_data)}")
    if non_identity:
        print(f"  char-weighted CER (non-identity only): {non_identity_cw_cer:.5f}")

    print()
    print("worst 5 pages by CER:")
    for i, p in enumerate(worst_5, 1):
        print(f"  {i}. {p['file']:30s} CER={p['cer']:.4f} inv={p['inversions']:3d} n={p['n_matched']:3d}")
        for txt in p["texts"][:3]:
            print(f"      {txt}")

    print()
    print("order error by page spread (width/height):")
    print(f"  spreads >1.15 (two-column): n={len(spreads):3d} CER={spread_cw_cer:.5f}")
    print(f"  spreads ≤1.15 (normal)    : n={len(non_spreads):3d} CER={non_spread_cw_cer:.5f}")

    # Store results
    results[det_name] = {
        "n_pages": len(pages_data),
        "char_weighted_cer": char_weighted_cer,
        "n_non_identity_pages": len(non_identity),
        "non_identity_cer": non_identity_cw_cer if non_identity else None,
        "total_chars": total_chars,
        "total_matched_lines": sum(p["n_matched"] for p in pages_data),
        "spread_cer": spread_cw_cer,
        "non_spread_cer": non_spread_cw_cer,
        "pages": pages_data,
    }

if not a.dry_run:
    json.dump(results, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {a.out}")
else:
    print("\n--dry-run: no reader needed, counts printed above; remove flag to process fully.")
