"""Run: eval/.venv/bin/python -m pytest tests -q   (or plain python tests/test_dictation.py)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bilimai.dictation import grade_dictation

KEY = ("Странный, тихий, ни на что не похожий, прерывистый звук слышался где-то неподалёку. "
       "Хотя Петя и сидел на мельнице, но он никого и ничего не увидел.")
STUDENT = ["Странный, тихий ни на что не похожий, прерывистый звук слышался",
           "где-то неподалёку. Хотя петя и сидел на мельнице, но он никого",
           "и нечего вдруг увидел."]

def lines(texts, conf=0.9):
    return [{"id": f"L{i}", "text": t, "bbox": [100, 100 + 60*i, 1500, 150 + 60*i], "confidence": conf} for i, t in enumerate(texts)]

def test_planted_errors_found_exactly():
    sc = grade_dictation(KEY, lines(STUDENT))["score_card"]
    kinds = sorted(e["kind"] for e in sc["errors"])
    assert kinds == ["capitalization", "extra_word", "missing_word", "punctuation_missing", "spelling"], kinds
    assert sc["n_spelling"] == 4 and sc["n_punctuation"] == 1 and sc["school_grade"] == "3"

def test_identical_is_perfect():
    sc = grade_dictation(KEY, lines([KEY]))["score_card"]
    assert sc["n_spelling"] == 0 and sc["n_punctuation"] == 0 and sc["school_grade"] == "5" and sc["coverage"] == 1.0

def test_low_confidence_flags_review():
    r = grade_dictation(KEY, lines(STUDENT, conf=0.4))
    assert all(m["needs_review"] for m in r["marks"]) and r["score_card"]["needs_review"]

def test_marks_have_bboxes_and_kinds():
    r = grade_dictation(KEY, lines(STUDENT))
    assert all(len(m["bbox"]) == 4 for m in r["marks"])
    assert {m["kind"] for m in r["marks"]} <= {"underline", "insert", "strike", "circle"}

if __name__ == "__main__":
    for f in (test_planted_errors_found_exactly, test_identical_is_perfect, test_low_confidence_flags_review, test_marks_have_bboxes_and_kinds):
        f(); print("ok", f.__name__)
