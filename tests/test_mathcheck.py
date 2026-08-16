import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bilimai.mathcheck import check_answer, first_wrong_step, grade_math

def L(texts, conf=0.9):
    return [{"text": t, "bbox": [100, 100 + 60*i, 800, 150 + 60*i], "confidence": conf} for i, t in enumerate(texts)]

def test_answer_forms():
    assert check_answer("5/6", r"\frac{5}{6}", "expression")[0]
    assert check_answer(r"\frac{5}{6}", "5/6 || 0.8333", "expression")[0]
    assert check_answer("x=4", "x=4", "equation_solution")[0]
    assert check_answer("4", "x=4", "equation_solution")[0]
    assert check_answer("2(x+1)", "2x+2", "expression")[0]                # equivalent, not identical
    assert check_answer("0,5", "1/2", "number")[0]                          # comma decimal
    assert check_answer("20 m/s", "20 m/s", "unit_value", 0.01)[0]
    assert not check_answer("3/9", "5/6", "expression")[0]
    assert check_answer("{-2, 3}", "3, -2", "set")[0]
    assert check_answer("12 : 4 = 3", "3", "expression")[0]               # school division sign, drill line

def test_first_wrong_step():
    assert first_wrong_step(["3x+5=20", "3x=15", "x=5"]) is None
    assert first_wrong_step(["3x+5=20", "3x=25", "x=25/3"]) == 1
    assert first_wrong_step([r"\frac{2}{3}+\frac{1}{6}", r"\frac{2+1}{3+6}", r"\frac{3}{9}"]) == 1

def test_grade_math_drills_and_worked():
    problems = [{"id": "1", "answer": "42", "answer_kind": "number", "points": 1},
                {"id": "2", "answer": "x=5", "answer_kind": "equation_solution", "points": 2},
                {"id": "3", "answer": r"\frac{5}{6}", "answer_kind": "expression", "points": 1},
                {"id": "4", "answer": "7", "answer_kind": "number", "points": 1}]
    groups = {"1": L(["25+17=42"]), "2": L(["3x+5=20", "3x=15", "Javob: x=5"]),
              "3": L([r"\frac{2}{3}+\frac{1}{6}", r"\frac{2+1}{3+6}", r"\frac{3}{9}"]), "4": []}
    r = grade_math(problems, groups)
    v = {p["id"]: p["verdict"] for p in r["score_card"]["problems"]}
    assert v == {"1": "correct", "2": "correct", "3": "wrong", "4": "missing"}, v
    assert r["score_card"]["score"] == 3 and r["score_card"]["max_score"] == 5
    p3 = next(p for p in r["score_card"]["problems"] if p["id"] == "3"); assert p3["first_wrong_step"] == 1
    kinds = [m["kind"] for m in r["marks"]]; assert kinds.count("check") == 2 and "cross" in kinds and "underline" in kinds

def test_low_confidence_needs_review():
    r = grade_math([{"id": "1", "answer": "42", "answer_kind": "number"}], {"1": L(["25+17=42"], conf=0.3)})
    assert r["score_card"]["needs_review"] and r["marks"][0]["needs_review"]

if __name__ == "__main__":
    for f in (test_answer_forms, test_first_wrong_step, test_grade_math_drills_and_worked, test_low_confidence_needs_review):
        f(); print("ok", f.__name__)
