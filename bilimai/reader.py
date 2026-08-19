"""BilimAI — batched GLM-OCR line reader (2026-08-18). One class used by the product pipeline and every eval script, so
inference is batched everywhere instead of one line at a time (rule: scripts must use the hardware fully).

    r = GLMBatchReader(base, adapter, device=None, prompt="Text Recognition:", line_h=128)
    texts, confs = r.read(crops, batch_size=16)          # crops: list[PIL.Image] → same order

Batched generation with LEFT padding (decoder-only), same crop recipe as training (resize to height 128), greedy decoding,
per-line confidence = mean token probability. Deterministic and equal to single-line reads up to fp16 noise (see
eval/util/check_batch_reader.py). Multi-GPU: run one process per GPU on a shard of the crops (see eval/train/vast/jobs).
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image

class GLMBatchReader:
    def __init__(self, base: str | Path, adapter: str | Path | None = None, device: str | None = None,
                 prompt: str = "Text Recognition:", line_h: int = 128, max_new_tokens: int = 96):
        self.base, self.adapter, self.prompt, self.line_h, self.max_new = str(base), (str(adapter) if adapter else None), prompt, line_h, max_new_tokens
        self.device = device; self._m = None
        self.name = f"glm-ocr@{Path(self.adapter).name if self.adapter else 'base'}"

    def _load(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        dev = self.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if dev in ("mps", "cuda") else torch.float32
        self.proc = AutoProcessor.from_pretrained(self.base)
        self.proc.tokenizer.padding_side = "left"
        m = AutoModelForImageTextToText.from_pretrained(self.base, dtype=dtype).to(dev).eval()
        if self.adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, self.adapter).eval()
        self._m, self._dev, self._torch = m, dev, torch

    def prep(self, crop: Image.Image) -> Image.Image:
        if crop.mode != "RGB": crop = crop.convert("RGB")
        if crop.size[1] > self.line_h:
            s = self.line_h / crop.size[1]; crop = crop.resize((max(8, int(crop.size[0] * s)), self.line_h))
        return crop

    def read(self, crops: list, batch_size: int = 16, prompt: str | None = None, with_conf: bool = True):
        """Read many line crops; returns (texts, confs) in input order."""
        if self._m is None: self._load()
        torch = self._torch; prompt = prompt or self.prompt
        texts, confs = [None] * len(crops), [None] * len(crops)
        # sort by width so batches have similar token counts (less padding), restore order at the end
        order = sorted(range(len(crops)), key=lambda i: crops[i].size[0] / max(1, crops[i].size[1]))
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]; imgs = [self.prep(crops[i]) for i in idx]
            msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}] for im in imgs]
            inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                                                   processor_kwargs={"padding": True, "padding_side": "left"}).to(self._dev)
            inputs.pop("token_type_ids", None)
            with torch.no_grad():
                out = self._m.generate(**inputs, max_new_tokens=self.max_new, do_sample=False,
                                       output_scores=with_conf, return_dict_in_generate=True, pad_token_id=self.proc.tokenizer.pad_token_id)
            P = inputs["input_ids"].shape[1]
            for b, i in enumerate(idx):
                seq = out.sequences[b][P:]
                texts[i] = self.proc.decode(seq, skip_special_tokens=True).strip().replace("\n", " ")
                if with_conf:
                    probs = []
                    for t, sc in zip(seq, out.scores):
                        tid = int(t)
                        if tid == self.proc.tokenizer.pad_token_id: break
                        probs.append(torch.softmax(sc[b].float(), dim=-1)[tid].item())
                    confs[i] = float(sum(probs) / len(probs)) if probs else 0.0
        return texts, confs

    def read_line(self, crop: Image.Image):
        t, c = self.read([crop], batch_size=1); return t[0], c[0]

    # ------------------------------------------------------------------ key-conditioned (forced) scoring — E5.8 PMI judge
    # Ported 2026-08-19 from eval/dictation/keyed_verify_pmi.py (validated there: cached vs full-sequence Δ ≤ 0.018 logprob).
    PROMPT = "Text Recognition:"

    def _prefix_inputs(self, crop):
        msgs = [{"role": "user", "content": [{"type": "image", "image": self.prep(crop)}, {"type": "text", "text": self.PROMPT}]}]
        inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self._dev)
        inputs.pop("token_type_ids", None); return inputs

    def _null_inputs(self):
        """Same prompt without the picture (text-only chat) — the reader's language habit alone."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": self.PROMPT}]}]
        inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self._dev)
        inputs.pop("token_type_ids", None); return inputs

    def _word_tokens(self, words):
        tok = self.proc.tokenizer; ids, spans = [], []
        for i, w in enumerate(words):
            t = tok(("" if i == 0 else " ") + w, add_special_tokens=False)["input_ids"]
            spans.append((len(ids), len(ids) + len(t))); ids += t
        return ids, spans

    def _rope_holder(self):
        if not hasattr(self, "_rope"):
            self._rope = next((m for m in self._m.modules() if hasattr(m, "rope_deltas")), None)
        return self._rope

    def forced_logprobs(self, inputs, seqs, has_image=True):
        """Teacher-force several word sequences after the same (cached) prompt. Returns per seq (token logprobs, word spans).
        The prompt runs once; only the candidate tokens are scored against the expanded KV cache."""
        import copy; torch = self._torch; dev = self._dev; tok = self.proc.tokenizer
        PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
        rope = self._rope_holder()
        with torch.no_grad():
            if not has_image and rope is not None: rope.rope_deltas = None
            pre = self._m(**inputs, use_cache=True)
            pkv = pre.past_key_values; last = torch.log_softmax(pre.logits[0, -1].float(), dim=-1)
            packs = [self._word_tokens(ws) for ws in seqs]; L = max(len(p[0]) for p in packs); B = len(seqs); P = inputs["input_ids"].shape[1]
            tgt = torch.full((B, L), PAD, dtype=torch.long, device=dev); attn = torch.zeros((B, P + L), dtype=torch.long, device=dev)
            for i, (ids, _) in enumerate(packs): tgt[i, :len(ids)] = torch.tensor(ids, device=dev); attn[i, :P + len(ids)] = 1
            cache = copy.deepcopy(pkv)
            if hasattr(cache, "batch_repeat_interleave"): cache.batch_repeat_interleave(B)
            else:
                for layer in cache.layers: layer.keys = layer.keys.repeat_interleave(B, 0); layer.values = layer.values.repeat_interleave(B, 0)
            pos = None
            if has_image and rope is not None and rope.rope_deltas is not None:
                delta = rope.rope_deltas.to(dev).view(1, -1, 1)
                pos = torch.arange(P, P + L, device=dev).view(1, 1, L).expand(3, B, L) + delta
            logits = self._m(input_ids=tgt, attention_mask=attn, past_key_values=cache, position_ids=pos, use_cache=False).logits.float()
            out = []
            for i, (ids, spans) in enumerate(packs):
                n = len(ids); vals = [last[ids[0]].item()]
                if n > 1:
                    lp = torch.log_softmax(logits[i, :n - 1], dim=-1); vals += lp[torch.arange(n - 1), tgt[i, 1:n]].tolist()
                out.append((vals, spans))
        return out

    def pmi_word_scores(self, crop, line_words, wi, strings, lam=0.5):
        """For the word at index `wi` of `line_words` (the read line), score each spelling in `strings` in that line context:
        s = log p(word | image, context) − lam · log p(word | no image, context). Returns a list aligned with `strings`."""
        if self._m is None: self._load()
        seqs = [line_words[:wi] + [s_] + line_words[wi + 1:] for s_ in strings]
        R_img = self.forced_logprobs(self._prefix_inputs(crop), seqs, True)
        R_null = self.forced_logprobs(self._null_inputs(), seqs, False)
        out = []
        for (li, sp), (ln, _) in zip(R_img, R_null):
            a, b = sp[wi]; out.append(sum(li[a:b]) - lam * sum(ln[a:b]))
        return out

