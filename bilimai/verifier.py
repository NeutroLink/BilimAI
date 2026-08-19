"""BilimAI — word verifier for dictation marks (E5.8, 2026-08-19): the CTC judge in the product.

    v = CTCWordVerifier()                                  # models/readingpipeline/ocr (MIT), CPU ONNX, ~30 ms/word
    out = v.judge(img, [{"bbox": ink_word_box, "key": "проявить", "read": "проевить"}, ...])
    out[i] -> {"margin": float, "best": "проевить", "verdict": "error" | "review" | "ok"}

Why: the GLM reader decodes with a language prior and cannot tell "the pupil misspelled" from "I misread"; the CTC word
reader has no prior (one letter per image column). For a word whose read differs from the key, we score the key spelling
and ~500 one-letter variants (bilimai.dictation.candidates wide) on the ink alone; margin = best variant − key. High
margin → the ink really deviates from the key → error; middle → review («на проверку», always drawn); low → the ink
supports the key → our misread → ok (mark removed).

Thresholds are MEASURED, not guessed (eval/runs/dictation/ctc_r5all_fused.json: 7,697 correct + 86 real misspellings,
all 60 sealed pages, design "judge only GLM-mismatch words"):
  τ_error 13.7 → ≈ 1.4 false red marks per 100 words (some are real errors our labels missed), catches ≈ 36 % of real ones
  τ_review 5.1 → review adds catch to ≈ 59 % at ≈ 3.7 review marks per 100 words; below → ok (≈ 60 % of false alarms removed)
Known limit: words the reader auto-corrected to the key never reach the judge (26 % of real misspellings) → R5b.
Preprocessing = the pipeline's own (BGR, h 64, w ≤ 512, zero pad); crops = ink box grown by bilimai.detector.WORD_GROW
(the CTC reader wants annotator-like word boxes — 2026-08-19 sweep). PMI (GLM key-conditioned) judge: same interface, later.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OCR = ROOT / "models/readingpipeline/ocr"
TAU_ERROR, TAU_REVIEW = 13.7, 5.1


class CTCWordVerifier:
    name = "ctc-word-verifier@rp-notebooks"

    def __init__(self, model_dir: str | Path = DEFAULT_OCR, tau_error: float = TAU_ERROR, tau_review: float = TAU_REVIEW,
                 threads: int = 8, max_cands: int = 0, batch: int = 64):
        import json, onnxruntime as ort
        cfg = json.load(open(Path(model_dir) / "ocr_config.json", encoding="utf-8"))
        self.alpha = cfg["alphabet"]; self.H, self.W = cfg["image"]["height"], cfg["image"]["width"]
        self.ch = {c: i + 2 for i, c in enumerate(self.alpha)}; self.BLANK, self.OOV = 0, 1
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = threads
        self.sess = ort.InferenceSession(str(Path(model_dir) / "ocr_model.onnx"), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.tau_error, self.tau_review, self.max_cands, self.batch = tau_error, tau_review, max_cands, batch

    # ---- image side ---------------------------------------------------------------------------------------------------
    def _crop(self, img_bgr, box):
        from .detector import RPDetector, WORD_GROW
        H, W = img_bgr.shape[:2]; b = RPDetector._grow([float(v) for v in box], WORD_GROW, W, H)
        x0, y0, x1, y1 = [int(round(v)) for v in b]; return img_bgr[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]

    def _prep(self, crop):
        import cv2
        h, w = crop.shape[:2]; nw = min(int(w * (self.H / max(1, h))), self.W)
        im = cv2.resize(crop, (max(1, nw), self.H), interpolation=cv2.INTER_LINEAR)
        if nw < self.W: im = np.pad(im, ((0, 0), (0, self.W - nw), (0, 0)), "constant", constant_values=0)
        return np.moveaxis(im, -1, 0).astype(np.float32) / 255

    def logprobs(self, crops):
        """(T, C) log-softmax matrix per crop; batched ONNX."""
        out = []
        for s in range(0, len(crops), self.batch):
            x = np.stack([self._prep(c) for c in crops[s:s + self.batch]])
            y = self.sess.run(None, {self.inp: x})[0]                          # (T, N, C)
            if y.shape[1] != x.shape[0]: y = np.transpose(y, (1, 0, 2))
            out += [y[:, i] for i in range(y.shape[1])]
        return out

    # ---- scoring -----------------------------------------------------------------------------------------------------
    def _encode(self, s): return [self.ch.get(c, self.OOV) for c in s]

    def scores(self, lp, strings):
        """log P(s | crop) for each string: −ctc_loss, all strings in one call."""
        import torch
        T = lp.shape[0]; K = len(strings); tg = [self._encode(s) for s in strings]
        logp = torch.from_numpy(np.ascontiguousarray(lp)).unsqueeze(1).expand(T, K, lp.shape[1]).contiguous()
        tgt = torch.tensor([t for s in tg for t in s], dtype=torch.long)
        nll = torch.nn.functional.ctc_loss(logp, tgt, torch.full((K,), T, dtype=torch.long), torch.tensor([len(s) for s in tg]),
                                           blank=self.BLANK, reduction="none", zero_infinity=True)
        return (-nll).tolist()

    def verdict(self, margin: float) -> str:
        return "error" if margin >= self.tau_error else "review" if margin >= self.tau_review else "ok"

    def judge(self, img, items: list[dict]) -> list[dict]:
        """items: {"bbox": ink-tight word box (orig px), "key": key word, "read": reader's word or None}. Returns one dict
        per item: margin (best candidate − key, log-prob), best candidate, verdict. Batched: one ONNX pass for all words."""
        import cv2
        from .dictation import candidates
        if not items: return []
        if not isinstance(img, np.ndarray): img = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        crops = [self._crop(img, it["bbox"]) for it in items]; lps = self.logprobs(crops); out = []
        for it, lp in zip(items, lps):
            key, read = it["key"], it.get("read"); cands = candidates(key, read, self.max_cands, wide=True)
            sc = self.scores(lp, [key] + cands); lk = sc[0]; lc = sc[1:]
            margin = (max(lc) - lk) if lc else 0.0; best = cands[int(np.argmax(lc))] if lc else None
            out.append({"margin": round(float(margin), 3), "best": best, "verdict": self.verdict(margin), "logp_key": round(float(lk), 3)})
        return out
