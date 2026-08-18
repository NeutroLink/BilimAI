"""E6.1 — Pipeline skeleton: grade(request) -> response, per contracts/.

    scan ─▶ LOCATE (line boxes) ─▶ READ (reader on each line) ─▶ ENGINE (per type) ─▶ RENDER ─▶ response

Every stage is a small pluggable object so the reader, detector and engines can be swapped as
they improve without touching the rest. What is real today:
  LOCATE : `OracleLocator` (boxes given in request.options.oracle_lines — used for eval/demo)
           `ProjectionLocator` (naive ink-profile line finder — placeholder until E4.1/E4.2)
  READ   : `GLMReader` (GLM-OCR + optional LoRA adapter, transformers; MPS/CUDA/CPU)
  ENGINE : dictation (bilimai.dictation), math (bilimai.mathcheck); others → "unsupported"
  RENDER : bilimai.render
Provenance is attached for internal use; `to_external()` strips it (contract rule).

CLI:  python -m bilimai.pipeline --request contracts/examples/dictation.json --out out/ \
        [--adapter models/adapters/glm-ocr-lora-ru-v1] [--oracle-from-gt eval/testset_v1/ru_pages/ground_truth.json]
"""
from __future__ import annotations
import argparse, json, time, uuid
from pathlib import Path
from typing import Any

from PIL import Image

from . import __version__
from .dictation import grade_dictation
from .mathcheck import grade_math
from .render import render

ROOT = Path(__file__).resolve().parents[1]


# ============================================================================ LOCATE
class OracleLocator:
    """Use line boxes supplied by the caller (eval / demo). bboxes in original pixels."""
    name = "oracle"
    def __init__(self, boxes: list[list[float]]): self.boxes = boxes
    def __call__(self, img: Image.Image) -> list[list[float]]: return list(self.boxes)


class ProjectionLocator:
    """Placeholder detector: horizontal ink-profile line segmentation on a grayscale page.
    Works on clean single-column pages; will be replaced by the trained detector (E4.1/E4.2)."""
    name = "projection-v0"
    def __init__(self, min_h=18, pad=6): self.min_h, self.pad = min_h, pad
    def __call__(self, img: Image.Image) -> list[list[float]]:
        import numpy as np
        g = np.asarray(img.convert("L"), dtype=np.float32)
        ink = (g < g.mean() - 0.6 * g.std()).astype(np.float32)          # dark pixels
        rows = ink.mean(axis=1); thr = max(rows.mean() * 0.5, 0.004)
        on = rows > thr; boxes = []; y = 0; H, W = ink.shape
        while y < H:
            if on[y]:
                y0 = y
                while y < H and on[y]: y += 1
                y1 = y
                if y1 - y0 >= self.min_h:
                    cols = ink[y0:y1].mean(axis=0) > 0.002
                    xs = np.where(cols)[0]
                    if len(xs):
                        boxes.append([float(max(0, xs[0] - self.pad)), float(max(0, y0 - self.pad)),
                                      float(min(W, xs[-1] + self.pad)), float(min(H, y1 + self.pad))])
            else:
                y += 1
        return boxes


# ============================================================================ READ
class GLMReader:
    """GLM-OCR line reader (+ optional LoRA adapter). Thin wrapper over bilimai.reader.GLMBatchReader (batched, 2026-08-18)."""
    def __init__(self, base: str | Path = ROOT / "models/GLM-OCR", adapter: str | Path | None = None,
                 device: str | None = None, max_new_tokens: int = 96, line_h: int = 128):
        from .reader import GLMBatchReader
        self._r = GLMBatchReader(base, adapter, device=device, max_new_tokens=max_new_tokens, line_h=line_h)
        self.name = self._r.name

    def read_line(self, crop: Image.Image) -> tuple[str, float]:
        return self._r.read_line(crop)

    def read_lines(self, crops: list, batch_size: int = 16) -> list[tuple[str, float]]:
        t, c = self._r.read(crops, batch_size=batch_size); return list(zip(t, c))


# ============================================================================ PIPELINE
class Pipeline:
    def __init__(self, reader, locator=None):
        self.reader = reader; self.locator = locator

    def grade(self, request: dict, out_dir: str | Path | None = None) -> dict[str, Any]:
        t0 = time.time(); timings = {}
        job = request.get("job_id") or str(uuid.uuid4()); atype = request["type"]
        img = Image.open(request["image"]["uri"]).convert("RGB")

        # LOCATE
        t = time.time()
        loc = self.locator or (OracleLocator(request["options"]["oracle_lines"]) if request.get("options", {}).get("oracle_lines") else ProjectionLocator())
        boxes = loc(img); timings["detect"] = round((time.time() - t) * 1000)

        # READ — all line crops of the page in one batched call
        t = time.time(); crops = []
        for b in boxes:
            pad = 12; x0, y0, x1, y1 = b
            crops.append(img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.size[0], x1 + pad), min(img.size[1], y1 + pad))))
        reads = self.reader.read_lines(crops) if hasattr(self.reader, "read_lines") else [self.reader.read_line(c) for c in crops]
        transcript = [{"id": f"L{i:02d}", "text": text, "bbox": [float(v) for v in b], "confidence": round(conf, 3)} for i, (b, (text, conf)) in enumerate(zip(boxes, reads))]
        timings["read"] = round((time.time() - t) * 1000)

        # ENGINE
        t = time.time()
        if atype == "dictation":
            key = request["key"]
            res = grade_dictation(key["text"], transcript, ignore_case=key.get("ignore_case", False),
                                  count_punctuation=key.get("count_punctuation", True),
                                  scale=key.get("grading_scale", "classic-5point"), language=request.get("language", "ru"))
            status = "needs_review" if res["score_card"].get("needs_review") else "ok"; engine = "dictation-engine@0.1"
        elif atype == "math":
            key = request["key"]; probs = key["problems"]
            # grouping: request.options.problem_lines = {id: [line indices]}; else drills = one line per problem in order
            pl = request.get("options", {}).get("problem_lines")
            groups = ({pid: [transcript[i] for i in idxs if i < len(transcript)] for pid, idxs in pl.items()} if pl
                      else {p["id"]: ([transcript[i]] if i < len(transcript) else []) for i, p in enumerate(probs)})
            res = grade_math(probs, groups, language=request.get("language", "ru"))
            status = "needs_review" if res["score_card"].get("needs_review") else "ok"; engine = "math-engine@0.1"
        else:
            res = {"marks": [], "score_card": {"score": 0, "max_score": 0, "summary": f"type '{atype}' not implemented yet", "confidence": 0.0, "needs_review": True}}
            status = "unsupported"; engine = "none"
        timings["grade"] = round((time.time() - t) * 1000)

        # RENDER
        marked_uri = None
        if out_dir is not None and res["marks"]:
            t = time.time(); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            marked = render(img, res["marks"]); marked_uri = str(out_dir / f"{job}.marked.png"); marked.save(marked_uri)
            timings["render"] = round((time.time() - t) * 1000)
        timings["total"] = round((time.time() - t0) * 1000)

        resp = {"job_id": job, "type": atype, "status": status,
                "transcript": transcript if request.get("options", {}).get("return_transcript", True) else [],
                "marks": res["marks"], "score_card": res["score_card"],
                "provenance": {"pipeline_version": __version__, "reader_model": getattr(self.reader, "name", "?"),
                               "detector_model": getattr(loc, "name", "?"), "engine": engine, "timings_ms": timings}}
        if marked_uri: resp["marked_image_uri"] = marked_uri          # contract: string or absent, never null
        return resp


def to_external(resp: dict) -> dict:
    """Contract rule: provenance never leaves the building."""
    return {k: v for k, v in resp.items() if k != "provenance"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True, help="JSON with a contract request (or {request: ...})")
    ap.add_argument("--out", default="out")
    ap.add_argument("--adapter", default=str(ROOT / "models/adapters/glm-ocr-lora-ru-v1"))
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--oracle-from-gt", help="ground_truth.json: use its line boxes for the request image (eval/demo)")
    a = ap.parse_args()
    req = json.load(open(a.request, encoding="utf-8")); req = req.get("request", req)
    if a.oracle_from_gt:
        gt = json.load(open(a.oracle_from_gt, encoding="utf-8"))
        fn = Path(req["image"]["uri"]).name
        req.setdefault("options", {})["oracle_lines"] = [l["bbox"] for l in gt[fn]["lines"]]
    reader = GLMReader(adapter=None if a.no_adapter else a.adapter)
    resp = Pipeline(reader).grade(req, out_dir=a.out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(resp, open(Path(a.out) / f"{resp['job_id']}.response.json", "w"), ensure_ascii=False, indent=1)
    sc = resp["score_card"]
    print(f"[{resp['status']}] {resp['type']}: score {sc['score']}/{sc['max_score']} conf {sc['confidence']} | {len(resp['marks'])} marks | {resp['provenance']['timings_ms']}")
    print(f"→ {a.out}/{resp['job_id']}.response.json", f"| marked: {resp.get('marked_image_uri')}" if resp.get('marked_image_uri') else "")


if __name__ == "__main__":
    main()
