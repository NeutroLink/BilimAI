"""BilimAI — line & word detector (E4.2, 2026-08-19): ai-forever ReadingPipeline-notebooks segmenter + our post-processing.

    det = RPDetector()                       # models/readingpipeline/segm/segm_model.onnx (MIT), CPU ONNX, ~0.7 s/page
    r = det.detect(img)                      # PIL image or numpy BGR
    r["lines"]  -> [[x0,y0,x1,y1], ...]      reading order (column, then top-to-bottom)
    r["words"]  -> [[x0,y0,x1,y1], ...]      all word boxes (grown), original pixels
    r["line_words"] -> [[word_idx, ...], ...] words of each line (same order as lines)

Why post-processing matters (measured on the sealed exam + val split, eval/runs/det_failure_*.json): the network finds the
lines (its own line count matches the truth on most pages) but its word-union boxes are only ~61 % of the annotators' line
height and its word boxes ~59 % height / 81 % width — the reader was fed strips with ascenders/descenders shaved off and the
judges get cut words. Growing every box by fixed fractions of its own size (derived on the VAL split, never on the exam)
lifted line F1@IoU0.5 from 0.70 to 0.89 (val 0.85) and word recall@0.5 from 0.28 to 0.92; E2E reading with v5 on the exam
went from 90.4 → 91.4 letters in 100 on the 20 v1 pages (all lines) and 93.7 / 82.9 words on all 60 pages. What remains
missed are 1–2-word fragments (grammar shorthand, superscripts) — the network does not fire on them.

Defaults = the val-derived factors: lines top 0.25 / bottom 0.365 / sides 0.02; words top 0.344 / bottom 0.416 / sides 0.105.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = ROOT / "models/readingpipeline/segm/segm_model.onnx"
LINE_GROW = (0.25, 0.365, 0.02)          # top, bottom, each side — fractions of the box's own height / width
WORD_GROW = (0.344, 0.416, 0.105)


class RPDetector:
    name = "rp-segm+grow-v1"

    def __init__(self, onnx: str | Path = DEFAULT_ONNX, thr_word: float = 0.8, thr_line: float = 0.5, dilate: int = 3,
                 min_area: int = 10, line_grow=LINE_GROW, word_grow=WORD_GROW, threads: int = 8):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"]); self.inp = self.sess.get_inputs()[0].name
        self.thr_word, self.thr_line, self.dilate, self.min_area = thr_word, thr_line, dilate, min_area
        self.line_grow, self.word_grow = line_grow, word_grow

    @staticmethod
    def _grow(b, g, W, H):
        t, bt, x = g; h = b[3] - b[1]; w = b[2] - b[0]
        return [max(0.0, b[0] - x * w), max(0.0, b[1] - t * h), min(float(W), b[2] + x * w), min(float(H), b[3] + bt * h)]

    def _pass(self, img_bgr, ox=0, oy=0):
        """One network pass over a (sub)image; returns words and text_line polylines in full-image pixels."""
        import cv2
        H, W = img_bgr.shape[:2]
        x = np.transpose(cv2.resize(img_bgr, (896, 896)).astype(np.float32) / 255, (2, 0, 1))[None]
        pred = self.sess.run(None, {self.inp: x})[0][0]; sx, sy = W / 896, H / 896
        m = (pred[0] > self.thr_word).astype(np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        words = []
        for c in cs:
            if cv2.contourArea(c) < self.min_area: continue
            x0, y0, w, h = cv2.boundingRect(c); words.append([ox + x0 * sx, oy + y0 * sy, ox + (x0 + w) * sx, oy + (y0 + h) * sy])
        ml = (pred[2] > self.thr_line).astype(np.uint8)
        if self.dilate > 0: ml = cv2.dilate(ml, np.ones((self.dilate, self.dilate), np.uint8))
        cl, _ = cv2.findContours(ml, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        lines = []
        for c in cl:
            if len(c) < 5: continue
            pts = c[:, 0, :].astype(np.float32); pts[:, 0] = ox + pts[:, 0] * sx; pts[:, 1] = oy + pts[:, 1] * sy; lines.append(pts)
        return words, lines

    @staticmethod
    def _iou(a, b):
        x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x1 - x0) * max(0, y1 - y0)
        return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)

    def raw(self, img_bgr, tiles: int = 0):
        """Network pass + contour extraction. Returns ungrown words, text_line polylines, groups (word idx per polyline).
        tiles=2: additionally run on a 2×2 grid of overlapping tiles (each upscaled to 896 px → small writing 2× larger) and
        merge words (drop duplicates IoU ≥ 0.5) — for the 1–2-word fragments the full-page pass misses (E4.2, 2026-08-19)."""
        H, W = img_bgr.shape[:2]
        words, lines = self._pass(img_bgr)
        if tiles and tiles > 1:
            tw, th = W / tiles, H / tiles; ov = 0.1
            for i in range(tiles):
                for j in range(tiles):
                    x0 = int(max(0, (i - ov) * tw)); x1 = int(min(W, (i + 1 + ov) * tw)); y0 = int(max(0, (j - ov) * th)); y1 = int(min(H, (j + 1 + ov) * th))
                    tw_, _ = self._pass(img_bgr[y0:y1, x0:x1], x0, y0)      # tiles add WORDS only; lines stay full-page
                    for wb in tw_:
                        if all(self._iou(wb, w) < 0.3 for w in words): words.append(wb)
        groups = [[] for _ in lines]
        for wi, wb in enumerate(words):                      # each word joins the text_line polyline nearest vertically
            cx, cy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2; best = None; bd = 1e9
            for li, pts in enumerate(lines):
                near = pts[np.abs(pts[:, 0] - cx) < max(30, (wb[2] - wb[0]))]
                if len(near) == 0: continue
                d = np.min(np.abs(near[:, 1] - cy))
                if d < bd: bd, best = d, li
            if best is not None and bd < 1.2 * (wb[3] - wb[1]): groups[best].append(wi)
            elif tiles and tiles > 1: groups.append([wi]); lines.append(np.array([[cx, cy]], dtype=np.float32))   # unclaimed word → its own tiny line (margin/above-line insertions)
        return words, lines, groups, (W, H)

    def detect(self, img, tiles: int = 0) -> dict:
        import cv2
        if not isinstance(img, np.ndarray): img = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        words, polylines, groups, (W, H) = self.raw(img, tiles=tiles)
        boxes = []; line_words = []
        for g in groups:
            if not g: continue
            gb = np.array([words[i] for i in g])
            boxes.append(self._grow([float(gb[:, 0].min()), float(gb[:, 1].min()), float(gb[:, 2].max()), float(gb[:, 3].max())], self.line_grow, W, H))
            line_words.append(list(g))
        # reading order: two-page spreads → column (left/right half) then y; single page → y
        order = list(range(len(boxes)))
        if boxes:
            cx = np.array([(b[0] + b[2]) / 2 for b in boxes]); col = (cx > W / 2).astype(int)
            if W / H > 1.15 and 0.2 <= col.mean() <= 0.8: keys = [(int(col[i]), boxes[i][1]) for i in order]
            else: keys = [(0, boxes[i][1]) for i in order]
            order = sorted(order, key=lambda i: keys[i])
        boxes = [boxes[i] for i in order]; line_words = [line_words[i] for i in order]
        gw = [self._grow(w, self.word_grow, W, H) for w in words]
        return {"lines": boxes, "words": gw, "line_words": line_words, "raw_words": words, "polylines": [p.tolist() for p in polylines]}


class RPLocator:
    """Pipeline LOCATE stage: RPDetector line boxes (bboxes in original pixels)."""
    name = RPDetector.name
    def __init__(self, detector: RPDetector | None = None): self.det = detector or RPDetector(); self.last = None
    def __call__(self, img) -> list[list[float]]:
        self.last = self.det.detect(img); return [list(b) for b in self.last["lines"]]
