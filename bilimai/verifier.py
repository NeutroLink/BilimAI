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
(the CTC reader wants annotator-like word boxes — 2026-08-19 sweep). PMIWordVerifier + FusedWordVerifier below: the product
default is the fused judge (CTC top-20 → PMI, verdict from min of standardised margins — both judges must agree).
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
                 threads: int = 8, max_cands: int = 0, batch: int = 64, coreml: bool = False):
        """coreml=True (Mac only): run the CRNN on Apple's GPU/Neural Engine via onnxruntime's CoreML provider (MLProgram) —
        measured 2026-08-19: 5.7 ms/crop vs 190 ms on one CPU thread, max |Δlogprob| 2e-5. Off by default (product = CPU)."""
        import json, onnxruntime as ort
        cfg = json.load(open(Path(model_dir) / "ocr_config.json", encoding="utf-8"))
        self.alpha = cfg["alphabet"]; self.H, self.W = cfg["image"]["height"], cfg["image"]["width"]
        self.ch = {c: i + 2 for i, c in enumerate(self.alpha)}; self.BLANK, self.OOV = 0, 1
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = threads; so.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        if coreml and "CoreMLExecutionProvider" in ort.get_available_providers():
            providers = [("CoreMLExecutionProvider", {"MLComputeUnits": "ALL", "ModelFormat": "MLProgram"}), "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(Path(model_dir) / "ocr_model.onnx"), so, providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.tau_error, self.tau_review, self.max_cands, self.batch = tau_error, tau_review, max_cands, batch

    # ---- image side ---------------------------------------------------------------------------------------------------
    def _crop_box(self, img_bgr, box):
        from .detector import RPDetector, WORD_GROW
        H, W = img_bgr.shape[:2]; b = RPDetector._grow([float(v) for v in box], WORD_GROW, W, H)
        x0, y0, x1, y1 = [int(round(v)) for v in b]; x0, y0 = max(0, x0), max(0, y0); x1, y1 = max(x0 + 1, x1), max(y0 + 1, y1)
        return (x0, y0, x1, y1)

    def _crop(self, img_bgr, box):
        x0, y0, x1, y1 = self._crop_box(img_bgr, box); return img_bgr[y0:y1, x0:x1]

    # ---- letter geometry (CTC forced alignment) ----------------------------------------------------------------------
    def char_frames(self, lp, string):
        """Viterbi CTC alignment of `string` on the (T, C) log-prob matrix → per character the [first, last] frame it owns."""
        ids = self._encode(string); n = len(ids)
        if n == 0: return []
        ext = [self.BLANK]; [ext.extend([i, self.BLANK]) for i in ids]; S = len(ext); T = lp.shape[0]
        NEG = -1e9; D = np.full((T, S), NEG); B = np.zeros((T, S), dtype=np.int8)
        D[0, 0] = lp[0, ext[0]]
        if S > 1: D[0, 1] = lp[0, ext[1]]
        for t in range(1, T):
            for s_ in range(S):
                best, arg = D[t - 1, s_], 0
                if s_ >= 1 and D[t - 1, s_ - 1] > best: best, arg = D[t - 1, s_ - 1], 1
                if s_ >= 2 and ext[s_] != self.BLANK and ext[s_] != ext[s_ - 2] and D[t - 1, s_ - 2] > best: best, arg = D[t - 1, s_ - 2], 2
                D[t, s_] = best + lp[t, ext[s_]]; B[t, s_] = arg
        s_ = S - 1 if S == 1 or D[T - 1, S - 1] >= D[T - 1, S - 2] else S - 2
        path = [0] * T
        for t in range(T - 1, -1, -1):
            path[t] = s_; s_ -= B[t, s_]
        frames = [[None, None] for _ in ids]
        for t, st in enumerate(path):
            if st % 2 == 1:
                k = st // 2
                if frames[k][0] is None: frames[k][0] = t
                frames[k][1] = t
        return frames

    def letter_geometry(self, lp, ink, key, crop_box, crop_w):
        """Which letters of the ink differ from the key, with page-pixel boxes. ink = the spelling the ink supports (the judge's
        best candidate), key = the expected word. Frames → crop x: the crop was resized to height H (width nw ≤ W=512), the
        network emits T frames over the 512-px canvas (8 px per frame) → x_crop = frame·(512/T)·(crop_w/nw)."""
        import difflib
        x0, y0, x1, y1 = crop_box; T = lp.shape[0]; h = y1 - y0
        nw = min(int(crop_w * (self.H / max(1, h))), self.W); px_per_frame = (self.W / T) * (crop_w / max(1, nw))
        fr = self.char_frames(lp, ink)
        # CTC fires each letter on ~1 frame near the letter's START and blanks elsewhere → a letter's visual span runs from
        # its own firing to the next letter's firing (half a frame earlier); inter-firing distances track letter widths
        # (checked on «слушат» 2026-08-19: ш 664–741 vs true 667–730 page px)
        n = len(fr); ink_frames = max(1, min(T, int(round(nw / (self.W / T)))))
        spans = []
        for i in range(n):
            f0, f1 = fr[i]
            if f0 is None: spans.append(None); continue
            next_start = next((fr[j][0] for j in range(i + 1, n) if fr[j][0] is not None), None)
            a = max(0.0, f0 - 0.5)
            b = (next_start - 0.5) if next_start is not None else min(ink_frames, f1 + 2.5)
            spans.append((a, max(b, a + 1)))
        def fbox(i):                                        # page box of ink letter i
            sp = spans[i]
            if sp is None: return None
            return [x0 + sp[0] * px_per_frame, y0, x0 + sp[1] * px_per_frame, y1]
        letters = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ink, key, autojunk=False).get_opcodes():
            if tag == "equal": continue
            if tag in ("replace", "delete"):
                bs = [b for b in (fbox(i) for i in range(i1, i2)) if b]
                if not bs: continue
                corr = key[j1:j2] if tag == "replace" else ""
                letters.append({"ink": ink[i1:i2], "correct": corr, "bbox": [min(b[0] for b in bs), y0, max(b[2] for b in bs), y1]})
            elif tag == "insert":                          # key letters missing in the ink → caret between ink letters
                left = fbox(i1 - 1) if i1 > 0 else None; right = fbox(i1) if i1 < len(ink) else None
                x = left[2] if left else (right[0] if right else x0); w = max(4.0, px_per_frame)
                letters.append({"ink": "", "correct": key[j1:j2], "bbox": [x - w / 2, y0, x + w / 2, y1]})
        return letters

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
            rec = {"margin": round(float(margin), 3), "best": best, "verdict": self.verdict(margin), "logp_key": round(float(lk), 3)}
            if rec["verdict"] == "error" and best:
                cb = self._crop_box(img, it["bbox"]); rec["letters"] = self.letter_geometry(lp, best, key, cb, cb[2] - cb[0])
            out.append(rec)
        return out


# ---------------------------------------------------------------------------------------------------------------- PMI
class PMIWordVerifier:
    """The GLM reader as a judge: score the key and candidate spellings in the read line's context, with and without the
    picture, and subtract the no-picture score (λ=0.5) so the reader's language habit cancels. Needs a GLMBatchReader
    (bilimai.reader) — the same loaded model the pipeline reads with. Items need line context: {"line_bbox", "line_words",
    "wi"} besides key/read. ~30 ms/word on a 5090, ~1.5 s/word on the Mac; used in cascade on the CTC top-K."""
    name = "pmi-word-verifier@glm"
    def __init__(self, reader, lam: float = 0.5, pad: int = 12): self.reader, self.lam, self.pad = reader, lam, pad
    def line_crop(self, img_pil, bbox):
        x0, y0, x1, y1 = bbox; W, H = img_pil.size
        return img_pil.crop((max(0, x0 - self.pad), max(0, y0 - self.pad), min(W, x1 + self.pad), min(H, y1 + self.pad)))
    def pmi(self, img_pil, item, strings):
        crop = self.line_crop(img_pil, item["line_bbox"])
        return self.reader.pmi_word_scores(crop, item["line_words"], item["wi"], strings, self.lam)


# -------------------------------------------------------------------------------------------------------------- fused
# z-standardisation constants and thresholds from the full-exam measurement — re-fit for the R5c adapter 2026-08-20
# (eval/runs/dictation/ctc_r5c_all_fused.json via eval/dictation/fuse_refit.py; v5 constants: Z_PMI (-2.059, 4.067),
# τ 2.29/0.74). fused = min(z_pmi, z_ctc) — "both judges must agree". Design B (judge reader-mismatch words only):
#   τ_error 2.30 → ≈ 0.65 false red / 100 words, catches ≈ 27 % of all real misspellings (36 % of judged)
#   τ_review 0.68 → catch incl. review ≈ 71 % of all (95 % of judged) at ≈ 6.5 marks / 100 words
Z_PMI = (-2.095, 3.829); Z_CTC = (-5.413, 7.805); TAU_F_ERROR, TAU_F_REVIEW = 2.30, 0.68

class FusedWordVerifier:
    """CTC judge on all ~500 spellings → its top-K go to the PMI judge → fused = min of standardised margins → verdict.
    Falls back to the CTC verdict for an item without line context."""
    name = "fused-ctc+pmi-word-verifier"
    def __init__(self, ctc: CTCWordVerifier, pmi: PMIWordVerifier, topk: int = 20, tau_error=TAU_F_ERROR, tau_review=TAU_F_REVIEW):
        self.ctc, self.pmi, self.topk, self.tau_error, self.tau_review = ctc, pmi, topk, tau_error, tau_review
    def verdict(self, fused): return "error" if fused >= self.tau_error else "review" if fused >= self.tau_review else "ok"
    def judge(self, img, items):
        import cv2
        from .dictation import candidates
        if not items: return []
        pil = img if not isinstance(img, np.ndarray) else None
        bgr = img if isinstance(img, np.ndarray) else cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        crops = [self.ctc._crop(bgr, it["bbox"]) for it in items]; lps = self.ctc.logprobs(crops); out = []
        for it, lp in zip(items, lps):
            key, read = it["key"], it.get("read"); cands = candidates(key, read, 0, wide=True)
            sc = self.ctc.scores(lp, [key] + cands); lk = sc[0]; lc = sc[1:]
            ctc_margin = (max(lc) - lk) if lc else 0.0
            order = sorted(range(len(cands)), key=lambda i: -lc[i])[:self.topk]; top = [cands[i] for i in order]
            rec = {"ctc_margin": round(float(ctc_margin), 3), "best": cands[order[0]] if order else None}
            if pil is not None and it.get("line_words") and it.get("line_bbox") is not None and it.get("wi") is not None and top:
                try:
                    ps = self.pmi.pmi(pil, it, [key] + top); pmi_margin = max(ps[1:]) - ps[0]
                    zc = (ctc_margin - Z_CTC[0]) / Z_CTC[1]; zp = (pmi_margin - Z_PMI[0]) / Z_PMI[1]; fused = min(zc, zp)
                    rec.update({"pmi_margin": round(float(pmi_margin), 3), "fused": round(float(fused), 3), "verdict": self.verdict(fused), "margin": round(float(fused), 3)})
                    if rec["verdict"] == "error" and rec["best"]:
                        cb = self.ctc._crop_box(bgr, it["bbox"]); rec["letters"] = self.ctc.letter_geometry(lp, rec["best"], key, cb, cb[2] - cb[0])
                    out.append(rec); continue
                except Exception as e:
                    rec["pmi_error"] = str(e)[:120]
            rec.update({"verdict": self.ctc.verdict(ctc_margin), "margin": rec["ctc_margin"]})
            if rec["verdict"] == "error" and rec["best"]:
                cb = self.ctc._crop_box(bgr, it["bbox"]); rec["letters"] = self.ctc.letter_geometry(lp, rec["best"], key, cb, cb[2] - cb[0])
            out.append(rec)
        return out

