#!/usr/bin/env python3
"""Does sending the page BIGGER make Qwen find and read more? — paired, stratified, no GPU.

WHY THIS DECIDES SOMETHING
R7 and the roadmap's region-finder round both pre-resize pages to max-side 1600. Measured from the
sealed ground truth, that gives one text line ~51 px of height, where the production line reader gets
each crop at **128 px**. Training cannot recover detail that was never in the picture, so if line-finding
is resolution-bound, a round at 1600 could fail a recall gate that a resize would have passed — and the
project would record "Qwen is not your reader" on the strength of a downscale.

The free evidence that motivated this: across the 18 zero-shot pages, line recall correlates **+0.72**
with how big the text is when sent. Reading CER correlates only −0.37 and is confounded (small-text pages
also carry the most lines), so reading was unproven either way. This resolves it by changing ONE thing.

DESIGN
Paired and stratified. The same 7 sealed pages through both arms, split by how small the writing is:
  * 4 SMALL-text pages — where resolution should matter most, if it matters at all
  * 3 LARGE-text pages — controls; if these move as much as the small ones, the effect is not resolution
Same prompt (`complete`, fingerprinted), same decoding, same scorer. Boxes are scored with ZERO growth
because Qwen's boxes need no correction (measured: box hit rate 1.00).

  eval/.venv/bin/python eval/detectors/resolution_ab.py            # after both probe arms have run
"""
import argparse, json, sys, unicodedata
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("--a", default=str(ROOT / "eval/runs/res_test_1600.json"), help="arm A run")
ap.add_argument("--b", default=str(ROOT / "eval/runs/res_test_2560.json"), help="arm B run")
ap.add_argument("--gt", default=str(ROOT / "eval/testset_v2/ru_pages/ground_truth.json"))
ap.add_argument("--out", default=str(ROOT / "eval/runs/resolution_ab.json"))
a = ap.parse_args()

gt = json.loads(Path(a.gt).read_text(encoding="utf-8"))


# ---- matching + CER, identical to score_qwen_boxes.py so the two cannot drift -----------------------
def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    i = max(0., x1 - x0) * max(0., y1 - y0)
    return i / max(1e-9, (p[2]-p[0])*(p[3]-p[1]) + (g[2]-g[0])*(g[3]-g[1]) - i)


def cer(ref, hyp):
    ref, hyp = unicodedata.normalize("NFC", ref), unicodedata.normalize("NFC", hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    d = list(range(len(hyp) + 1))
    for i, cr in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, ch in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (cr != ch))
    return d[len(hyp)] / len(ref)


def to_page(rec):
    """Probe items -> page-pixel boxes. Coordinates are 0-1000 normalised (measured, not assumed:
    scored as absolute pixels the same run reads 4 % matched instead of 80 %)."""
    ox, oy = rec["sub_origin"]; sw, sh = rec["sub_size"]
    out = []
    for it in rec.get("items", []):
        b = it.get("bbox") or it.get("bbox_2d")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        q = [ox + b[0]/1000*sw, oy + b[1]/1000*sh, ox + b[2]/1000*sw, oy + b[3]/1000*sh]
        if q[2] > q[0] and q[3] > q[1]:
            out.append((q, str(it.get("text") or "")))
    return out


def score_page(rec):
    pairs = to_page(rec)
    P = [q for q, _ in pairs]; T = [t for _, t in pairs]
    lines = gt[rec["file"]]["lines"]
    matched, ed, ch, ious, used = 0, 0, 0, [], set()
    for L in lines:
        gb = L["bbox"]; txt = (L.get("text") or "").strip()
        if not P:
            continue
        js = [iou(p, gb) for p in P]
        j = int(np.argmax(js)); best = js[j]
        ious.append(best)
        if best >= 0.5:
            matched += 1; used.add(j)
            if txt:
                ed += cer(txt, T[j].strip()) * len(txt); ch += len(txt)
    return {"file": rec["file"], "gt_lines": len(lines), "matched": matched,
            "recall": matched / max(1, len(lines)),
            "cer": ed / ch if ch else float("nan"), "chars": ch,
            "edits": ed, "predicted": len(P), "extra": len(P) - len(used),
            "median_iou": float(np.median(ious)) if ious else 0.0,
            "seconds": rec.get("seconds"), "max_side": rec.get("max_side"),
            "prompt": rec.get("prompt_variant"), "fingerprint": rec.get("prompt_fingerprint")}


def load(p):
    recs = json.loads(Path(p).read_text(encoding="utf-8"))
    recs = [r for r in recs if r.get("region") in (None, "full")]
    return {r["file"]: score_page(r) for r in recs}


A, B = load(a.a), load(a.b)
common = sorted(set(A) & set(B))
if not common:
    sys.exit("no pages in common between the two arms")

# stratify by how small the writing is, in the ORIGINAL page (independent of either arm)
def char_px(fn):
    g = gt[fn]
    w = [(x["bbox"][2]-x["bbox"][0])/max(1, len(x.get("text") or ""))
         for L in g["lines"] for x in L["words"] if x.get("text")]
    return float(np.median(w)) / g["width"] * 1600 if w else float("nan")


strata = {"small text": [f for f in common if char_px(f) < 20],
          "large text": [f for f in common if char_px(f) >= 20]}

fp = {A[f]["fingerprint"] for f in common} | {B[f]["fingerprint"] for f in common}
assert len(fp) == 1, f"prompt fingerprint differs between arms: {fp} — the comparison is void"

ms_a = {A[f]["max_side"] for f in common}.pop()
ms_b = {B[f]["max_side"] for f in common}.pop()
print(f"\n=== RESOLUTION A/B — max-side {ms_a} vs {ms_b}, prompt {A[common[0]]['prompt']} "
      f"({fp.pop()}), {len(common)} sealed pages ===\n")
print(f"{'page':<11}{'char px':>8}{'   lines found':>16}{'      CER':>16}{'   extra boxes':>16}{'  secs':>14}")
print(f"{'':<11}{'@1600':>8}{ms_a:>8}{ms_b:>8}{ms_a:>8}{ms_b:>8}{ms_a:>8}{ms_b:>8}{ms_a:>7}{ms_b:>7}")
print("-" * 82)

rows = []
for name, files in strata.items():
    if not files:
        continue
    for f in sorted(files, key=char_px):
        x, y = A[f], B[f]
        print(f"{f:<11}{char_px(f):>8.1f}{x['matched']:>8}{y['matched']:>8}"
              f"{x['cer']:>8.3f}{y['cer']:>8.3f}{x['extra']:>8}{y['extra']:>8}"
              f"{x['seconds'] or 0:>7.0f}{y['seconds'] or 0:>7.0f}")
        rows.append({"stratum": name, **{f"a_{k}": v for k, v in x.items()},
                     **{f"b_{k}": v for k, v in y.items()}, "char_px_at_1600": char_px(f)})
    print("-" * 82)

summary = {}
for name, files in list(strata.items()) + [("ALL", common)]:
    if not files:
        continue
    am = sum(A[f]["matched"] for f in files); bm = sum(B[f]["matched"] for f in files)
    tot = sum(A[f]["gt_lines"] for f in files)
    ae = sum(A[f]["edits"] for f in files); ac = sum(A[f]["chars"] for f in files)
    be = sum(B[f]["edits"] for f in files); bc = sum(B[f]["chars"] for f in files)
    s = {"pages": len(files), "gt_lines": tot,
         "recall_a": round(am/max(1, tot), 3), "recall_b": round(bm/max(1, tot), 3),
         "cer_a": round(ae/max(1, ac), 4), "cer_b": round(be/max(1, bc), 4),
         "extra_a": sum(A[f]["extra"] for f in files), "extra_b": sum(B[f]["extra"] for f in files),
         "secs_a": round(sum(A[f]["seconds"] or 0 for f in files)),
         "secs_b": round(sum(B[f]["seconds"] or 0 for f in files))}
    s["d_recall"] = round(s["recall_b"] - s["recall_a"], 3)
    s["d_cer"] = round(s["cer_b"] - s["cer_a"], 4)
    summary[name] = s
    print(f"{name:<12} recall {s['recall_a']:.3f} -> {s['recall_b']:.3f} ({s['d_recall']:+.3f})   "
          f"CER {s['cer_a']:.4f} -> {s['cer_b']:.4f} ({s['d_cer']:+.4f})   "
          f"extra {s['extra_a']} -> {s['extra_b']}   {s['secs_a']}s -> {s['secs_b']}s")

# The verdict has to compare the strata, not just the total: if the LARGE-text controls move as much as
# the SMALL-text pages, whatever changed is not resolution.
sm, lg = summary.get("small text"), summary.get("large text")
if sm and lg:
    verdict = ("RESOLUTION-BOUND — small text gains and the large-text controls do not"
               if sm["d_recall"] >= 0.03 and sm["d_recall"] > lg["d_recall"] + 0.02 else
               "NOT resolution-bound at this step — recall moves alike in both strata, or not at all")
    print(f"\nVERDICT: {verdict}")
    print(f"  small-text recall {sm['d_recall']:+.3f}   large-text control {lg['d_recall']:+.3f}")
    summary["verdict"] = verdict
    summary["cost"] = f"{summary['ALL']['secs_b']/max(1,summary['ALL']['secs_a']):.1f}x wall-clock at {ms_b}"
    print(f"  cost of the bigger page: {summary['cost']}")

Path(a.out).write_text(json.dumps({"arm_a": a.a, "arm_b": a.b, "max_side_a": ms_a, "max_side_b": ms_b,
                                   "summary": summary, "pages": rows}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
print(f"\n-> {a.out}")
