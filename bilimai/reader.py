"""BilimAI — batched VLM line reader. One class used by the product pipeline and every eval script, so inference is
batched everywhere instead of one line at a time (rule: scripts must use the hardware fully).

    r = make_reader(base, adapter)                       # picks the class from config.json model_type
    texts, confs = r.read(crops, batch_size=16)          # crops: list[PIL.Image] → same order

2026-08-22: generalised to any image-text-to-text VLM — AutoProcessor + AutoModelForImageTextToText load every
supported family through this identical path; only the provenance name and the preferred dtype differ per family.
The production reader class keeps byte-identical behaviour and its provenance name (call sites depend on both).

Batched generation with LEFT padding (decoder-only), same crop recipe as training (resize to height 128), greedy decoding,
per-line confidence = mean token probability. Deterministic and equal to single-line reads up to fp16 noise (see
eval/util/check_batch_reader.py). Multi-GPU: run one process per GPU on a shard of the crops (see eval/train/vast/jobs).
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image

import hashlib
import re as _re

_THINK = _re.compile(r"^\s*<think>.*?</think>\s*", _re.S)


def strip_think(text: str) -> str:
    """Drop a leading <think>…</think> block from a generation.

    R6 (2026-08-22) cost a whole rented run to this: under `TEMPLATE=qwen3_vl` — Qwen3-VL's *thinking*
    template — 100 % of 1,770 predictions came back prefixed with an EMPTY `<think>  </think>`. That is 19
    characters against a 30-character median line, so character CER read 0.78266 instead of the true 0.02188,
    and the run still passed its VERIFIED gate. Word-aligned metrics were untouched, and that contradiction is
    the only thing that exposed it.

    Training with `qwen3_vl_nothink` is the primary fix; this is the belt-and-braces one, because the failure
    is silent, it is invisible to word-level metrics, and a thinking template can arrive from any future base.
    """
    if not text or "<think>" not in text:
        return text
    return _THINK.sub("", text, count=1)


# Suffix appended to a reader family when the adapter emits letter-separated targets. Kept here so
# the verifier, the refit script and any future arm all spell it the same way.
LETTERSEP_FAMILY_SUFFIX = "-lettertgt"


from .lettersep import target_is_encoded as _target_is_encoded


class VLMLineReader:
    """Any image-text-to-text VLM as a batched line reader. Subclasses only set `family` (the provenance name) and the
    preferred dtype; every method below is model-agnostic."""
    family = "vlm"
    prefer_bf16 = False          # set on families trained in bf16 (Qwen3-VL); fp16 can overflow on those weights

    def __init__(self, base: str | Path, adapter: str | Path | None = None, device: str | None = None,
                 prompt: str = "Text Recognition:", line_h: int = 128, max_new_tokens: int = 192,
                 dtype: str | None = None, name: str | None = None, letter_sep: bool | None = None,
                 family: str | None = None):
        """`letter_sep=True` (Arm 1, 2026-08-25): this adapter was trained to emit letter-separated
        targets, so `read()` decodes them back before returning. Default False — the shipped reader is
        untouched. See bilimai/lettersep.py and plans/exec/2026-08-25-arm1-letter-targets.md.

        ⚠ THIS IS THE ONLY PLACE THE DECODE MAY LIVE. The codebase graph confirms `strip_think` has a
        single caller (`VLMLineReader.read`), so decoding here covers every consumer — dictation,
        verbatim_retention, ctc_verify box mapping, the in-job evals. Decoding anywhere else means
        two implementations and a silent divergence."""
        self.base, self.adapter, self.prompt, self.line_h, self.max_new = str(base), (str(adapter) if adapter else None), prompt, line_h, max_new_tokens
        self.device = device; self._m = None; self.dtype = dtype
        # ⚠ DEFAULTS FROM THE SAME ENV VAR AS THE DATA BUILDER (r5b_data.py). It was dead config
        # before: nothing in the repo passed it, and bilimai/pipeline.py's wrapper has an explicit
        # signature with no **kw, so the product path could not enable it at all. The result would
        # have been an arm whose targets were letter-separated while every in-job eval scored the
        # RAW spaced string against plain labels — school-val CER ~0.9 against a 0.024 gate, i.e. the
        # arm reads as a catastrophe even if it improved retention. ONE switch now drives both sides.
        import os as _os
        self.letter_sep = (_os.environ.get("BILIMAI_LETTERSEP", "0") == "1") \
            if letter_sep is None else letter_sep
        self.n_no_separator = 0     # RAW count of separator-less lines — see lettersep.separator_rate
        # ⚠ A LETTER-SEPARATED READER IS ITS OWN FAMILY, DERIVED — NOT REMEMBERED.
        # `family` decides which verifier constants and fitted reader-edit model get loaded
        # (bilimai/verifier.py const_path/edit_prior_path). Those describe "the reader's confusion
        # habits", and a reader trained to emit one letter at a time errs DIFFERENTLY by construction
        # — that is the entire point of Arm 1. Leaving it as plain "glm-ocr" would silently load
        # R5c's thresholds and edit prior for a reader they do not describe.
        # It is DERIVED from letter_sep rather than passed, so it cannot be forgotten: the family
        # changes exactly when the behaviour it names changes. `family=` still overrides explicitly.
        # See plans/exec/2026-08-25-arm1-letter-targets.md §3b.
        self.family = family if family is not None else (
            type(self).family + LETTERSEP_FAMILY_SUFFIX if self.letter_sep else type(self).family)
        self.name = name or f"{self.family}@{Path(self.adapter).name if self.adapter else 'base'}"

    def _resolve_dtype(self, torch, dev):
        """Explicit `dtype` wins. Else bf16 for bf16-trained families on CUDA (fp16 can overflow their weights); MPS keeps
        fp16 because its bf16 support is partial; CPU stays fp32."""
        if self.dtype is not None: return getattr(torch, self.dtype)
        if dev == "cuda": return torch.bfloat16 if self.prefer_bf16 else torch.float16
        return torch.float16 if dev == "mps" else torch.float32

    def _load(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        dev = self.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = self._resolve_dtype(torch, dev)
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

    def read(self, crops: list, batch_size: int = 16, prompt: str | list[str] | None = None, with_conf: bool = True):
        """Read many line crops; returns (texts, confs) in input order. `prompt` may be one string for all crops or a list with
        one prompt per crop (R5b keyed mode: "[school] [verbatim] Key: …\\nText Recognition:" differs per line)."""
        if self._m is None: self._load()
        torch = self._torch
        prompts = list(prompt) if isinstance(prompt, (list, tuple)) else [prompt or self.prompt] * len(crops)
        assert len(prompts) == len(crops), "one prompt per crop"
        texts, confs = [None] * len(crops), [None] * len(crops)
        # sort by width so batches have similar token counts (less padding), restore order at the end
        order = sorted(range(len(crops)), key=lambda i: crops[i].size[0] / max(1, crops[i].size[1]))
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]; imgs = [self.prep(crops[i]) for i in idx]
            msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompts[i]}]}] for im, i in zip(imgs, idx)]
            inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                                                   processor_kwargs={"padding": True, "padding_side": "left"}).to(self._dev)
            inputs.pop("token_type_ids", None)
            with torch.no_grad():
                out = self._m.generate(**inputs, max_new_tokens=self.max_new, do_sample=False,
                                       output_scores=with_conf, return_dict_in_generate=True, pad_token_id=self.proc.tokenizer.pad_token_id)
            P = inputs["input_ids"].shape[1]
            for b, i in enumerate(idx):
                seq = out.sequences[b][P:]
                raw = strip_think(self.proc.decode(seq, skip_special_tokens=True)).strip().replace("\n", " ")
                # ⚠ PER-PROMPT, NOT PER-READER. One Arm 1 adapter emits letter-separated targets for
                # ink prompts and PLAIN text for [essay] prompts, because r5b_data.py only encodes the
                # former. Decoding an essay line welds it into a single word — silently, and the
                # 1500-line HWR200 hold-out score goes with it. See lettersep.target_is_encoded.
                texts[i] = (self._undo_letter_sep(raw)
                            if self.letter_sep and _target_is_encoded(prompts[i]) else raw)
                if with_conf:
                    probs = []
                    for t, sc in zip(seq, out.scores):
                        tid = int(t)
                        if tid == self.proc.tokenizer.pad_token_id: break
                        probs.append(torch.softmax(sc[b].float(), dim=-1)[tid].item())
                    confs[i] = float(sum(probs) / len(probs)) if probs else 0.0
        return texts, confs

    def _undo_letter_sep(self, raw: str) -> str:
        """Letter-separated generation -> ordinary text.

        `n_no_separator` counts separator-less lines. It is NOT a defect count — a correctly encoded
        single word has no separator, and 12.9 % of sealed-exam lines are single words. Judge a RUN
        by comparing `lettersep.separator_rate` against the same rate over its labels; a per-line
        verdict is impossible from the string alone (see lettersep.separator_rate's docstring).
        """
        from .lettersep import WORD_SEP, decode
        if WORD_SEP not in raw:
            self.n_no_separator += 1        # NOT a defect count: a single-word line has no separator
        return decode(raw)

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

    def _prompt_pass(self, inputs, has_image, cache_key=None):
        """Run the prompt once and keep its KV cache, because the PMI judge asks for the same prompt again and again.

        Measured 2026-09-16 on a rented 5090 (cProfile over one real page, `eval/runs/pilot/`): `grade` cost 3107 ms of
        a 3700 ms page, of which `PMIWordVerifier.pmi` was 2768 ms over 16 words — 173 ms each, against the 30 ms this
        class's docstring claimed. The profile showed 78 GLM forward passes for those 16 words: each word paid for a
        picture prompt AND a no-picture prompt before its candidates were scored, and BOTH are identical work. The
        no-image prompt is one fixed sentence, the same for every word on every page; the image prompt is the LINE crop,
        shared by every judged word on that line. So they are computed once and reused.

        The returned cache is never handed out for mutation — `forced_logprobs` deep-copies it before expanding it to
        the candidate batch, which it already did. `rope_deltas` is part of the prompt's result, not a global, so it is
        stored beside the cache and restored on a hit; getting that wrong would shift every candidate's positions.
        """
        memo = getattr(self, "_prompt_memo", None)
        if memo is None: memo = self._prompt_memo = {}
        rope = self._rope_holder()
        if cache_key is not None and cache_key in memo:
            pkv, last, delta = memo[cache_key]
            if rope is not None: rope.rope_deltas = delta
            return pkv, last
        if not has_image and rope is not None: rope.rope_deltas = None
        pre = self._m(**inputs, use_cache=True)
        pkv, last = pre.past_key_values, self._torch.log_softmax(pre.logits[0, -1].float(), dim=-1)
        if cache_key is not None:
            # one picture prompt plus the fixed no-picture one: a line's KV cache is tens of MB, and holding every
            # line of a page would grow VRAM for no gain, since the judge works through the page line by line
            for key in [k for k in memo if k != "null" and k != cache_key]: memo.pop(key)
            memo[cache_key] = (pkv, last, rope.rope_deltas if rope is not None else None)
        return pkv, last

    def forced_logprobs(self, inputs, seqs, has_image=True, cache_key=None):
        """Teacher-force several word sequences after the same (cached) prompt. Returns per seq (token logprobs, word spans).
        The prompt runs once; only the candidate tokens are scored against the expanded KV cache."""
        import copy; torch = self._torch; dev = self._dev; tok = self.proc.tokenizer
        PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
        rope = self._rope_holder()
        with torch.no_grad():
            pkv, last = self._prompt_pass(inputs, has_image, cache_key)
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
        s = log p(word | image, context) − lam · log p(word | no image, context). Returns a list aligned with `strings`.

        Both prompts are cached (see `_prompt_pass`): the line crop's own bytes key the picture prompt, so consecutive
        words on one line reuse it, and the no-picture prompt is keyed once for the life of the reader."""
        if self._m is None: self._load()
        seqs = [line_words[:wi] + [s_] + line_words[wi + 1:] for s_ in strings]
        line_key = hashlib.blake2b(crop.tobytes(), digest_size=16).hexdigest()
        R_img = self.forced_logprobs(self._prefix_inputs(crop), seqs, True, cache_key=line_key)
        R_null = self.forced_logprobs(self._null_inputs(), seqs, False, cache_key="null")
        out = []
        for (li, sp), (ln, _) in zip(R_img, R_null):
            a, b = sp[wi]; out.append(sum(li[a:b]) - lam * sum(ln[a:b]))
        return out


class GLMBatchReader(VLMLineReader):
    """GLM-OCR (the production reader). Unchanged behaviour: fp16 everywhere, name "glm-ocr@<adapter>"."""
    family = "glm-ocr"
    prefer_bf16 = False


class QwenVLLineReader(VLMLineReader):
    """Alternate reader family. Loads in bf16 on CUDA — these checkpoints ship bfloat16 and fp16 can overflow them."""
    family = "qwen3-vl"
    prefer_bf16 = True


_FAMILIES = {"glm_ocr": GLMBatchReader, "qwen3_vl": QwenVLLineReader}


def reader_class_for(base: str | Path):
    """Pick the reader class from the checkpoint's own config.json `model_type`. Unknown types fall back to the generic
    VLMLineReader rather than guessing a family — it still loads, it just carries a neutral provenance name."""
    import json
    cfg = Path(base) / "config.json"
    mt = None
    if cfg.is_file():
        try: mt = json.loads(cfg.read_text(encoding="utf-8")).get("model_type")
        except Exception: mt = None
    return _FAMILIES.get(mt, VLMLineReader)


def make_reader(base: str | Path, adapter: str | Path | None = None, **kw):
    """The one entry point callers should use: returns the right reader for whatever checkpoint `base` is."""
    return reader_class_for(base)(base, adapter, **kw)
