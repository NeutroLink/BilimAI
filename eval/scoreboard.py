#!/usr/bin/env python3
"""The ten-cell scoreboard — five assignment types x two languages, measured, not asserted.

WHY THIS EXISTS
"Qwen leading across all assignment types in both languages" is a claim about ten cells. Until each cell
has a TEST, the phrase has no meaning and every training round is a guess about what matters.

Every number here is COMPUTED from a file in this repo at the moment you run it. Nothing is typed in.
A cell that reports `no test` is telling you its next job is labelling, not training — which is the most
useful thing this table can say.

TWO QUESTIONS PER CELL, and they are not the same question:
  READ   — can the model get the words off the page?          (perception)
  CHECK  — of the in-scope deviations from a SUPPLIED key, how many does it flag, and at what
           false-flag rate?                                    (the product's actual job)
A perfect READ score with no CHECK score is a research result, not a product.

⚠ SCOPE CHANGE 2026-08-24. This column used to ask "does it mark like a teacher" and was scored by
agreement with teacher marks. That is SHELVED. Teachers mark selectively and mark things outside our
scope (punctuation without a key, style, grades), so agreement answers a question we are not asking —
and we do not have the labelled pages for it either. See plans/SCOPE.md.

  eval/.venv/bin/python eval/scoreboard.py            # table + eval/runs/scoreboard.json
  eval/.venv/bin/python eval/scoreboard.py --json     # machine-readable only
"""
import argparse, importlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

ap = argparse.ArgumentParser()
ap.add_argument("--json", action="store_true", help="print JSON only")
ap.add_argument("--out", default=str(ROOT / "eval/runs/scoreboard.json"))
a = ap.parse_args()

MISSING = "—"


def has_symbol(dotted):
    """'bilimai.dictation:grade_dictation' -> True if importable and defined."""
    mod, _, sym = dotted.partition(":")
    try:
        return hasattr(importlib.import_module(mod), sym)
    except Exception:
        return False


def exists(*rel):
    return all((ROOT / r).exists() for r in rel)


# ---------------------------------------------------------------- measurement handlers
# Each returns (value_str, provenance_str) or (None, why_not).

def read_ru_pages(pred_rel, label):
    """Score a page-level prediction file against the sealed exam with the project's canonical scorer."""
    gt_p = ROOT / "eval/testset_v2/ru_pages/ground_truth.json"
    pr_p = ROOT / pred_rel
    if not gt_p.is_file() or not pr_p.is_file():
        return None, f"no {pred_rel}"
    try:
        import score as S
    except Exception as e:                                            # pragma: no cover
        return None, f"scorer unavailable ({e})"
    gt = json.loads(gt_p.read_text(encoding="utf-8"))
    pred = json.loads(pr_p.read_text(encoding="utf-8"))
    rows = [S.score_page(gt[k], pred[k]) for k in gt if k in pred]
    if not rows:
        return None, "no overlapping pages"
    ed = sum(l.get("edits", 0) for r in rows for l in r["lines"])
    ch = sum(r["n_gt_chars"] for r in rows)
    rec = sum(r["line_recall"] for r in rows) / len(rows)
    return (f"CER_all {ed/ch:.4f} · recall {rec:.3f}",
            f"{label}, {len(rows)} pages, eval/score.py")


def read_uz(arm):
    p = ROOT / "eval/runs/uz_page_probe.json"
    if not p.is_file():
        return None, "no uz_page_probe.json"
    d = json.loads(p.read_text(encoding="utf-8")).get("summary", {}).get("folded_rescore")
    if not d:
        return None, "run not folded — see eval/detectors/uz_rescore_folded.py"
    r = d[arm]
    return (f"CER {r['cer_folded']:.4f} folded · marks {r['mark_agreement']}",
            f"Qwen zero-shot, {arm}, SYNTHETIC pages only")


def read_probe(fname, key, label, extra=None):
    """MCQ / markup probes — SYNTHETIC pages that carry their own ground truth because we drew the marks."""
    p = ROOT / "eval/runs" / fname
    if not p.is_file():
        return None, f"no {fname} — run eval/detectors/mcq_markup_probe.py"
    s = json.loads(p.read_text(encoding="utf-8"))["summary"]
    val = f"{label} {s[key]:.2f}"
    if extra:
        val += " · " + " ".join(f"{k}{s[e]}" for k, e in extra)
    return val, "synthetic pages, marks drawn by us — a PASS is suggestive, not proof"


def read_math(arm_key, label):
    p = ROOT / "eval/runs/qwen_math_baseline.json"
    if not p.is_file():
        return None, "no qwen_math_baseline.json"
    s = json.loads(p.read_text(encoding="utf-8"))["summary"].get(arm_key)
    if not s:
        return None, f"no arm {arm_key}"
    ex = s.get("exact_pct", round(100 * s.get("exact", 0) / max(1, s["n"]), 1))
    return f"CER {s['mean_cer']:.4f} · exact {ex:g}%", f"{label}, n={s['n']}"


# ---------------------------------------------------------------- the ten cells
def cell(atype, lang, engine, schema, read_gt, read_fn, note=""):
    return dict(type=atype, lang=lang, engine=engine, schema=schema,
                read_gt=read_gt, read_fn=read_fn, note=note)


CELLS = [
    cell("Диктанты", "ru", "bilimai.dictation:grade_dictation", "contracts/dictation.schema.json",
         ["eval/testset_v2/ru_pages/ground_truth.json"],
         lambda: read_ru_pages("eval/runs/qwen_zeroshot_pred_for_score.json", "Qwen zero-shot")),
    cell("Диктанты", "uz", "bilimai.dictation:grade_dictation", "contracts/dictation.schema.json",
         ["out/uz_page_probe/gt.json"], lambda: read_uz("pages"),
         note="synthetic pages only — ZERO real Uzbek pupil pages exist"),

    cell("Математика", "ru", "bilimai.mathcheck:grade_math", "contracts/math.schema.json",
         ["out/math_baseline/strips"], lambda: read_math("arm_latex_all", "Qwen zero-shot, LaTeX"),
         note="strips only; nothing routes LaTeX into grade_math yet"),
    cell("Математика", "uz", "bilimai.mathcheck:grade_math", "contracts/math.schema.json",
         [], lambda: (None, "never tested in Uzbek"),
         note="maths notation is language-neutral; the surrounding words are not"),

    cell("Тесты", "ru", "bilimai.mcq:grade_mcq", "contracts/mcq.schema.json",
         ["out/mcq_probe_ru/gt.json"],
         lambda: read_probe("mcq_probe_ru.json", "letter_accuracy", "right option"),
         note="ROW OPEN (2026-08-23): 14/18 on which option is marked, mark TYPE 18/18. "
              "Weak link is reading the option text (0.50), not seeing the tick. No engine yet."),
    cell("Тесты", "uz", "bilimai.mcq:grade_mcq", "contracts/mcq.schema.json",
         [], lambda: (None, "no probe, no labels")),

    cell("Изложение", "ru", "bilimai.retelling:grade_retelling", "contracts/retelling.schema.json",
         ["out/markup_probe_ru/gt.json"],
         lambda: read_probe("markup_probe_ru.json", "recall", "markup recall"),
         note="ROW OPEN (2026-08-23): finds 27/36 marked words. But it CANNOT tell a strike-through from "
              "an underline (strike named right 1/12, called underline 7/12) — and those mean opposite "
              "things when marking. 1 page in 6 derailed (136 items). Rubric grading still needs a TEXT model."),
    cell("Изложение", "uz", "bilimai.retelling:grade_retelling", "contracts/retelling.schema.json",
         [], lambda: (None, "no engine, no labels")),

    cell("Открытые вопросы", "ru", "bilimai.openq:grade_open", "contracts/open_question.schema.json",
         [], lambda: (None, "no engine, no labels"),
         note="mvp/app.py grades TYPED text via Ollama; it has never seen a photograph"),
    cell("Открытые вопросы", "uz", "bilimai.openq:grade_open", "contracts/open_question.schema.json",
         [], lambda: (None, "no engine, no labels")),
]

# SHELVED 2026-08-24 — teacher-agreement grading. Kept as a record of where those pages would live if
# the question is ever reopened; nothing reads it. See plans/SCOPE.md.
TEACHER_MARKS_SHELVED = {
    ("Диктанты", "ru"): "eval/testset_v2/ru_pages/teacher_marks.json",
    ("Математика", "ru"): "eval/testset_v2/ru_pages/teacher_marks_math.json",
}


def check_dictation_ru():
    """The in-scope flagging rate, GENERATED by eval/dictation/gate.py. Never typed."""
    g = ROOT / "eval/runs/dictation/gate.json"
    if not g.is_file():
        return None, "no gate.json — run eval/dictation/gate.py"
    d = json.loads(g.read_text(encoding="utf-8"))
    return (f"{d['caught']}/{d['honest_denom']} = {d['honest_pct']:.1f}% @ "
            f"{d['realised_per_100']:.2f} fp/100"), "eval/runs/dictation/gate.json"


CHECKS = {("Диктанты", "ru"): check_dictation_ru}

rows = []
for c in CELLS:
    val, prov = c["read_fn"]()
    gt_ok = bool(c["read_gt"]) and exists(*c["read_gt"])
    chk = CHECKS.get((c["type"], c["lang"]))
    cval, cprov = chk() if chk else (None, "no check test — needs a supplied key and a scored set")
    rows.append({
        "type": c["type"], "lang": c["lang"],
        "engine": has_symbol(c["engine"]),
        "contract": exists(c["schema"]),
        "read_gt": gt_ok,
        "read_score": val, "read_provenance": prov,
        "check_score": cval, "check_provenance": cprov,
        "note": c["note"],
    })

summary = {
    "cells": len(rows),
    "with_engine": sum(r["engine"] for r in rows),
    "with_read_test": sum(r["read_score"] is not None for r in rows),
    "with_check_test": sum(r["check_score"] is not None for r in rows),
}
out = {"summary": summary, "rows": rows,
       "headline": "READ says the words can be lifted off the page. CHECK says how many in-scope "
                   "deviations from a supplied key it flags, and at what false-flag rate — the "
                   "product's actual job. Teacher-agreement grading is SHELVED (plans/SCOPE.md)."}

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

if a.json:
    print(json.dumps(out, ensure_ascii=False, indent=1))
    sys.exit(0)

W = (19, 6, 8, 10, 9, 41, 30)
hdr = ("type", "lang", "engine", "contract", "read GT", "READ score (measured)", "CHECK")
print("\n=== BilimAI scoreboard — 5 assignment types x 2 languages ===\n")
print("".join(h.ljust(w) for h, w in zip(hdr, W)))
print("-" * sum(W))
for r in rows:
    print("".join(x.ljust(w) for x, w in zip((
        r["type"], r["lang"],
        "yes" if r["engine"] else MISSING,
        "yes" if r["contract"] else MISSING,
        "yes" if r["read_gt"] else MISSING,
        (r["read_score"] or f"no test — {r['read_provenance']}")[:40],
        r["check_score"] or "no test",
    ), W)))
print("-" * sum(W))
print(f"\n  engines built      : {summary['with_engine']} of {summary['cells']}")
print(f"  cells with a READ  : {summary['with_read_test']} of {summary['cells']}")
print(f"  cells with a CHECK : {summary['with_check_test']} of {summary['cells']}   <- what 'production grade' means")
print("\nnotes:")
for r in rows:
    if r["note"]:
        print(f"  {r['type']} / {r['lang']}: {r['note']}")
print(f"\n-> {a.out}")
