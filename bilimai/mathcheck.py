"""E5.2 — Math engine (T1 answer check with SymPy; T3 step check hooks).

Input : key problems (contracts/math.schema.json#/$defs/Problem) + the student's read lines
        (text or LaTeX, top to bottom, each with a bbox and confidence) — either one problem per
        line ("drills": 25+17=42) or a block of lines per problem ("worked": steps ending in x=…).
Output: contract-shaped `marks` + `score_card.problems` (verdict, points, read answer/steps,
        first wrong step, checker used, confidence).

Deterministic first: SymPy decides equality/equivalence of the FINAL answer against the key
(exact, simplified-equivalent, numeric with tolerance, sets, "x=4"-style solutions, values with
units). LLM partial credit is a hook (`llm_partial_credit`) — off by default. Reading errors are
the reader's problem; anything below the confidence threshold → needs_review.
"""
from __future__ import annotations
import re
from typing import Any, Callable

import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_UNIT_RE = re.compile(r"^\s*([-+]?\d+(?:[.,]\d+)?)\s*([a-zA-Zа-яА-Я/²³^0-9·\s]*)\s*$")
_ANS_PREFIX = re.compile(r"^\s*(javob|жавоб|ответ|answer|j|о)\s*[:=]\s*", re.I)


# ----------------------------------------------------------------------------- parsing
def normalize(s: str) -> str:
    """Make handwritten-ish math parseable: unicode operators, commas as decimals, ':' division."""
    s = s.strip()
    s = _ANS_PREFIX.sub("", s)
    s = (s.replace("×", "*").replace("·", "*").replace("∙", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
          .replace("√", "sqrt").replace("²", "^2").replace("³", "^3").replace("π", "pi"))
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)                    # 3,5 → 3.5
    s = re.sub(r"(?<=\d)\s*:\s*(?=\d)", "/", s)              # 12 : 4 → 12/4 (school division)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\,", "").replace("\\ ", "").replace("$", "")
    return s.strip().rstrip(".;")


def to_expr(s: str):
    """Parse LaTeX or plain text into a SymPy expression / relation. Raises on failure."""
    s = normalize(s)
    if "\\" in s:
        return parse_latex(s)
    if "=" in s and s.count("=") == 1:
        l, r = s.split("=")
        return sp.Eq(parse_expr(l, transformations=_TRANSFORMS), parse_expr(r, transformations=_TRANSFORMS), evaluate=False)  # keep as an equation even if trivially true (12/4=3)
    return parse_expr(s, transformations=_TRANSFORMS)


def _rhs_or_self(e):
    return e.rhs if isinstance(e, sp.Equality) else e


def _accepted_forms(answer: str) -> list[str]:
    return [a.strip() for a in answer.split("||") if a.strip()]


# ----------------------------------------------------------------------------- checkers
def check_answer(read: str, expected: str, kind: str = "expression", tolerance: float | None = None) -> tuple[bool, str]:
    """Return (correct, checker_name). Tries every accepted form of `expected`."""
    for exp in _accepted_forms(expected):
        ok, how = _check_one(read, exp, kind, tolerance)
        if ok:
            return True, how
    return False, how


def _check_one(read: str, exp: str, kind: str, tol: float | None) -> tuple[bool, str]:
    r, e = normalize(read), normalize(exp)
    if kind == "text":
        return (r.lower() == e.lower(), "text_exact")
    if kind == "unit_value":
        mr, me = _UNIT_RE.match(r), _UNIT_RE.match(e)
        if mr and me:
            vr, ve = float(mr.group(1)), float(me.group(1))
            ur, ue = re.sub(r"\s+", "", mr.group(2)).lower(), re.sub(r"\s+", "", me.group(2)).lower()
            return (abs(vr - ve) <= (tol or 1e-9) and (ur == ue or not ue), "numeric_tolerance")
        return (False, "numeric_tolerance")
    try:
        er, ee = to_expr(r), to_expr(e)
    except Exception:
        return (r.replace(" ", "") == e.replace(" ", ""), "string_fallback")
    if kind == "equation_solution":
        # accept "x=4", "4", or {4} for expected "x=4"; compare solution values
        vr, ve = _rhs_or_self(er), _rhs_or_self(ee)
        try:
            return (sp.simplify(vr - ve) == 0, "sympy_equivalent")
        except Exception:
            return (vr == ve, "sympy_exact")
    if kind == "set":
        try:
            sr = sp.FiniteSet(*[sp.nsimplify(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", r)])
            se = sp.FiniteSet(*[sp.nsimplify(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", e)])
            return (sr == se, "sympy_exact")
        except Exception:
            return (False, "sympy_exact")
    vr, ve = _rhs_or_self(er), _rhs_or_self(ee)
    if vr == ve:
        return (True, "sympy_exact")
    if kind == "number" or tol is not None:
        try:
            return (abs(float(vr) - float(ve)) <= (tol or 1e-9), "numeric_tolerance")
        except Exception:
            pass
    try:
        return (sp.simplify(vr - ve) == 0, "sympy_equivalent")
    except Exception:
        return (False, "sympy_equivalent")


def first_wrong_step(steps: list[str]) -> int | None:
    """For a chain of equations/expressions, return index of the first line that is not
    equivalent to the previous one (a wrong transformation). None if the chain holds or can't tell."""
    prev = None
    for i, s in enumerate(steps):
        try:
            e = to_expr(s)
        except Exception:
            continue
        val = _rhs_or_self(e) if not isinstance(e, sp.Equality) else e
        if prev is not None:
            try:
                if isinstance(prev, sp.Equality) and isinstance(val, sp.Equality):
                    # equations: solution sets should agree
                    same = sp.solveset(prev, list(prev.free_symbols)[0]) == sp.solveset(val, list(val.free_symbols)[0]) if prev.free_symbols and val.free_symbols else sp.simplify((prev.lhs - prev.rhs) - (val.lhs - val.rhs)) == 0
                else:
                    same = sp.simplify(_rhs_or_self(prev) - _rhs_or_self(val)) == 0
                if not same:
                    return i
            except Exception:
                pass
        prev = val
    return None


# ----------------------------------------------------------------------------- engine
def _find_answer(lines: list[dict]) -> dict | None:
    """Pick the student's final answer among the lines: an explicit 'Javob:/Ответ:' line wins,
    else the last line containing '=' or a bare value."""
    for ln in reversed(lines):
        if _ANS_PREFIX.match(ln.get("text", "")):
            return ln
    for ln in reversed(lines):
        if ln.get("text", "").strip():
            return ln
    return None


def grade_math(problems: list[dict], groups: dict[str, list[dict]], *, review_threshold: float = 0.6,
               language: str = "ru", llm_partial_credit: Callable | None = None) -> dict[str, Any]:
    """`groups`: problem id → the student's read lines for that problem (each {text,bbox,confidence})."""
    marks, results = [], []
    total = 0.0; maxp = 0.0
    for p in problems:
        pts = float(p.get("points", 1)); maxp += pts
        lines = groups.get(p["id"], [])
        res = {"id": p["id"], "points_max": pts, "read_steps": [l["text"] for l in lines]}
        ans = _find_answer(lines)
        if ans is None:
            res.update({"verdict": "missing", "points_awarded": 0.0, "checker": "sympy_exact", "confidence": 1.0})
            results.append(res); continue
        conf = float(ans.get("confidence", 0.9)); review = conf < review_threshold
        read_ans = _ANS_PREFIX.sub("", ans["text"]).strip()
        # for "a=b" drills the student's answer is the RHS; for worked problems the last line as a whole
        ok, how = check_answer(read_ans, p["answer"], p.get("answer_kind", "expression"), p.get("tolerance"))
        res.update({"read_answer": read_ans, "checker": how, "confidence": round(conf, 3), "needs_review": review})
        if ans.get("bbox"): res["answer_bbox"] = [float(v) for v in ans["bbox"]]
        if ok:
            res.update({"verdict": "correct", "points_awarded": pts}); total += pts
            if ans.get("bbox"): marks.append(_mark("check", ans["bbox"], "correct", conf, review, ""))
        else:
            fw = first_wrong_step([l["text"] for l in lines]) if len(lines) > 1 else None
            partial = 0.0
            if p.get("partial_credit", True) and llm_partial_credit is not None:
                try: partial = float(llm_partial_credit(p, lines) or 0.0)
                except Exception: partial = 0.0
            res.update({"verdict": "partial" if partial > 0 else "wrong", "points_awarded": min(pts, partial),
                        "reason": _reason(p, read_ans, language)})
            if fw is not None: res["first_wrong_step"] = fw
            total += min(pts, partial)
            if ans.get("bbox"): marks.append(_mark("cross", ans["bbox"], "wrong_answer", conf, review, _reason(p, read_ans, language)))
            if fw is not None and lines[fw].get("bbox"):
                marks.append(_mark("underline", lines[fw]["bbox"], "wrong_step", conf, review, _step_expl(language)))
        results.append(res)
    n_ok = sum(1 for r in results if r["verdict"] == "correct")
    needs_review = any(r.get("needs_review") for r in results)
    return {"marks": marks, "score_card": {"score": total, "max_score": maxp, "summary": _summary(n_ok, len(results), total, maxp, language),
                                           "confidence": round(sum(r.get("confidence", 1) for r in results) / max(len(results), 1), 3),
                                           "needs_review": needs_review, "problems": results}}


def _mark(kind, bbox, reason, conf, review, expl):
    m = {"kind": kind, "bbox": [float(v) for v in bbox], "reason": reason, "confidence": round(conf, 3), "needs_review": review}
    if expl: m["explanation"] = expl
    return m


def _reason(p, read, lang):
    exp = _accepted_forms(p["answer"])[0]
    return f"javob {exp} bo'lishi kerak, yozilgan {read}" if lang == "uz" else f"должно быть {exp}, записано {read}"


def _step_expl(lang):
    return "shu qatordan boshlab xato" if lang == "uz" else "ошибка начиная с этой строки"


def _summary(n_ok, n, total, maxp, lang):
    if lang == "uz":
        return f"{n} misoldan {n_ok} tasi to'g'ri. Ball: {total:g}/{maxp:g}."
    return f"Верно {n_ok} из {n}. Баллы: {total:g}/{maxp:g}."
