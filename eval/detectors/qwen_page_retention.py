#!/usr/bin/env python
"""Does Qwen keep the pupil's spelling mistakes, or silently fix them? Measured PAGE-LEVEL. 2026-08-23.

Verbatim retention is the standing reader metric (2026-08-19): of the real pupil misspellings in the sealed
exam, how many does the model write AS WRITTEN (kept) vs quietly correct (corrected) vs mangle (other)?
A marking product cannot flag a mistake that its own reader already erased. GLM's best was 45 % (R5b) and
R6's best arm 58.6 % — both measured on ORACLE line crops.

R7 is page-level, so the deployment format has no crops at all. This scores retention the way the product
will actually work: run the page, match each prediction to a GT line by IoU, and read the pupil's word out
of whatever the model returned for that line.

Detection failure and auto-correction are reported SEPARATELY. A page-level model that never found the line
has not "corrected" anything — lumping the two together would flatter or damn it for the wrong reason.
`retention_on_found` is the reading behaviour; `found_rate` is the detection half.

  eval/.venv/bin/python eval/detectors/qwen_page_retention.py eval/runs/qwen_baseline_*.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval" / "dictation"))
from verbatim_retention import retention

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--pairs", default=str(ROOT / "eval/runs/dictation/real_misspellings_v2.json"))
ap.add_argument("--iou", type=float, default=0.5)
ap.add_argument("--out", default=str(ROOT / "eval/runs/qwen_page_retention.json"))
a = ap.parse_args()


def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    i = max(0., x1 - x0) * max(0., y1 - y0)
    return i / max(1e-9, (p[2]-p[0])*(p[3]-p[1]) + (g[2]-g[0])*(g[3]-g[1]) - i)


gt = json.load(open(a.gt, encoding="utf-8"))
pairs = json.load(open(a.pairs, encoding="utf-8"))
records = []
for f in a.runs:
    records += json.load(open(f, encoding="utf-8"))
variants = sorted({r.get("prompt_variant", "UNKNOWN") for r in records})
fps = sorted({r.get("prompt_fingerprint") for r in records})
print("prompt variants in these runs:", variants, "| fingerprints:", fps)
if "UNKNOWN" in variants:
    print("!! some records predate prompt provenance — their numbers cannot be attributed to a prompt")

# reads[fn] = one entry per GT line, in GT order; "" where nothing matched
reads, found = {}, {}
for r in records:
    fn = r["file"]
    page = gt.get(fn)
    if not page:
        continue
    sw, sh = r["sub_size"]; ox, oy = r["sub_origin"]
    P, PT = [], []
    for it in r["items"]:
        b = it["bbox"]
        q = [b[0]/1000*sw + ox, b[1]/1000*sh + oy, b[2]/1000*sw + ox, b[3]/1000*sh + oy]
        if q[2] > q[0] and q[3] > q[1]:
            P.append(q); PT.append(it.get("text", ""))
    cur = reads.setdefault(fn, [{"text": "", "bbox": l["bbox"]} for l in page["lines"]])
    fnd = found.setdefault(fn, [False] * len(page["lines"]))
    for gi, l in enumerate(page["lines"]):
        if fnd[gi] or not P:
            continue
        j = int(np.argmax([iou(q, l["bbox"]) for q in P]))
        if iou(P[j], l["bbox"]) >= a.iou:
            cur[gi] = {"text": PT[j].replace("\n", " ").strip(), "bbox": l["bbox"]}
            fnd[gi] = True

sub = {fn: gt[fn] for fn in reads}
strict = [p for p in pairs if p.get("strict") and p["file"] in sub]
full = retention(reads, sub, pairs)
full.pop("details", None)

# the same measurement restricted to pairs whose line was actually FOUND
pairs_found = [p for p in strict if found.get(p["file"], [False])[p["gi"]] if p["gi"] < len(found[p["file"]])]
sub2 = {fn: gt[fn] for fn in {p["file"] for p in pairs_found}}
on_found = retention({f: reads[f] for f in sub2}, sub2, pairs_found)
on_found.pop("details", None)

n_lines = sum(len(v) for v in found.values())
n_found = sum(sum(v) for v in found.values())
res = {"pages": len(sub), "prompt_variants": variants, "prompt_fingerprints": fps,
       "gt_lines": n_lines, "lines_found": n_found, "found_rate": round(n_found / max(1, n_lines), 3),
       "strict_pairs_on_these_pages": len(strict), "strict_pairs_on_found_lines": len(pairs_found),
       "retention_all_pairs": full, "retention_on_found": on_found,
       "reference_glm_oracle_crops": {"R5b": 0.45, "R6_best_qwen_ft": 0.586}}
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
print(f"\n-> {a.out}")
