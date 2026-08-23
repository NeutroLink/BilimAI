#!/usr/bin/env python3
"""Re-score the saved Uzbek probe run with the oʻ/gʻ mark folded — no model, no GPU, seconds.

WHY. 58 of the 59 fonts in `data/fonts/uz_bridge` cannot draw the okina U+02BB and 50 cannot draw the
tutuq U+02BC (checked 2026-08-23), so `make_uz_font_strips.py:130` substitutes ‘/’ at draw time while the
LABEL keeps the official letter. The image shows one character and the label demands another, and no model
can be graded on a distinction the ink does not carry. `bilimai/uzbek.py`'s docstring has demanded this
fold since it was written — "score.py must apply the same fold before any UZ number is quoted" — and it
had never been implemented.

This corrects the numbers offline from the saved ref/hyp pairs, the same way R6's CER was corrected from
`school_val_preds.json`, and writes them back into the run's summary so the stale figure cannot be quoted.

Reading ability is the FOLDED number. Convention compliance is reported separately as mark agreement, so
folding hides nothing: the marker still needs to know whether the reader carries the distinction.

  eval/.venv/bin/python eval/detectors/uz_rescore_folded.py [--run eval/runs/uz_page_probe.json] [--dry]
"""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bilimai.uzbek import fold_marks, mark_agreement

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=str(ROOT / "eval/runs/uz_page_probe.json"))
ap.add_argument("--gt", default=str(ROOT / "out/uz_page_probe/gt.json"))
ap.add_argument("--dry", action="store_true", help="print only, do not write the summary back")
a = ap.parse_args()


def edits(ref, hyp):
    if not ref:
        return 0 if not hyp else len(hyp)
    d = list(range(len(hyp) + 1))
    for i, cr in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, ch in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (cr != ch))
    return d[len(hyp)]


def iou(p, g):
    x0, y0, x1, y1 = max(p[0], g[0]), max(p[1], g[1]), min(p[2], g[2]), min(p[3], g[3])
    i = max(0., x1 - x0) * max(0., y1 - y0)
    return i / max(1e-9, (p[2]-p[0])*(p[3]-p[1]) + (g[2]-g[0])*(g[3]-g[1]) - i)


THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


def parse_items(txt):
    """Byte-identical to uz_page_probe.parse_items, so the two cannot drift."""
    txt = THINK.sub("", txt, count=1)
    items, i, j = [], txt.find("["), txt.rfind("]")
    if i != -1 and j > i:
        try:
            v = json.loads(txt[i:j + 1]); items = [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []
        except Exception:
            items = []
    if not items:
        for m in re.finditer(r"\{[^{}]*\}", txt, re.S):
            try:
                x = json.loads(m.group(0))
                if isinstance(x, dict): items.append(x)
            except Exception: pass
    out = []
    for it in items:
        b = it.get("bbox_2d") or it.get("bbox") or it.get("box"); t = it.get("text") or ""
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try: out.append(([float(v) for v in b], str(t)))
            except Exception: pass
    return out


run = json.loads(Path(a.run).read_text(encoding="utf-8"))
report = {}

# ---------------------------------------------------------------- strips
e = ef = n = mo = mt = 0
per = []
for r in run["strips"]:
    ref, hyp = r["ref"], r["hyp"]
    e += edits(ref, hyp)
    f = edits(fold_marks(ref), fold_marks(hyp)); ef += f; n += len(ref)
    per.append(f / max(1, len(ref)))
    x, y = mark_agreement(ref, hyp); mo += x; mt += y
per.sort()
report["strips"] = {"n": len(run["strips"]), "cer_strict": round(e/n, 4), "cer_folded": round(ef/n, 4),
                    "median_folded": round(per[len(per)//2], 4),
                    "above_0.10_folded": sum(1 for c in per if c > 0.10),
                    "mark_agreement": f"{mo}/{mt}",
                    "pct_of_error_that_is_only_the_mark": round(100*(e-ef)/max(e, 1))}

# ---------------------------------------------------------------- pages
gt = {p["file"]: p for p in json.loads(Path(a.gt).read_text(encoding="utf-8"))["pages"]}
e = ef = n = mo = mt = matched = tot = 0
for pr in run["pages"]:
    g = gt[pr["file"]]; W0, H0 = g["w"], g["h"]
    P, PT = [], []
    for b, t in parse_items(pr["raw"]):
        q = [b[0]/1000*W0, b[1]/1000*H0, b[2]/1000*W0, b[3]/1000*H0]
        if q[2] > q[0] and q[3] > q[1]: P.append(q); PT.append(t)
    tot += len(g["lines"])
    for L in g["lines"]:
        if not P: continue
        js = [iou(q, L["bbox"]) for q in P]; j = js.index(max(js))
        if max(js) >= 0.5:
            matched += 1; h = PT[j].strip(); ref = L["text"]
            e += edits(ref, h); ef += edits(fold_marks(ref), fold_marks(h)); n += len(ref)
            x, y = mark_agreement(ref, h); mo += x; mt += y
report["pages"] = {"matched": matched, "gt_lines": tot, "cer_strict": round(e/n, 4),
                   "cer_folded": round(ef/n, 4), "mark_agreement": f"{mo}/{mt}",
                   "pct_of_error_that_is_only_the_mark": round(100*(e-ef)/max(e, 1))}

report["headline"] = ("Quote the FOLDED CER for reading ability. The strict number also charges for the "
                      "okina/tutuq convention, which 58 of 59 bridge fonts cannot draw — so the ink does "
                      "not carry it. Mark agreement is reported separately: the marker needs to know.")
print(json.dumps(report, indent=1, ensure_ascii=False))

if not a.dry:
    run.setdefault("summary", {})["folded_rescore"] = report
    Path(a.run).write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> wrote summary.folded_rescore into {a.run}")
