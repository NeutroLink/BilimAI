"""Prove the prompt cache changed the speed and NOT a single number.

A judge that is faster and slightly different is worse than a slow one: every threshold in
`bilimai/data/verifier_v6.json` is fitted against these exact PMI scores, so a drift of 0.01 would
silently move the product's catch rate and false-flag rate. So this compares the cached path
against a forced-uncached path word by word, on real crops from a real page, and reports the
largest difference alongside the timing.

Run inside the worker image on a GPU box: `python prove_prompt_cache.py`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

PAGE = Path("/app/page-42.jpg")


def main() -> int:
    from PIL import Image

    from bilimai.detector import RPLocator
    from bilimai.reader import reader_class_for
    from pilot.worker import MODEL_ADAPTER, MODEL_BASE

    page = Image.open(PAGE).convert("RGB")
    lines = RPLocator()(page)                         # RPLocator is callable: page -> line bboxes
    reader = reader_class_for(MODEL_BASE)(base=MODEL_BASE, adapter=MODEL_ADAPTER, device="cuda")
    reader._load()

    # three words on ONE line (the case the cache is for) plus a word on the next line (the miss)
    crops = [page.crop(tuple(lines[0])), page.crop(tuple(lines[1]))]
    words = ["мир", "поосок", "на", "цветной", "луг"]
    candidates = ["поосок", "посок", "поесок", "поосак", "паосок", "поосокъ", "просок", "поособ"]
    jobs = [(crops[0], 1), (crops[0], 2), (crops[0], 3), (crops[1], 1)]

    def run(use_cache: bool) -> tuple[list[float], float]:
        reader._prompt_memo = {}
        scores: list[float] = []
        began = time.perf_counter()
        for crop, index in jobs:
            if use_cache:
                scores += reader.pmi_word_scores(crop, words, index, candidates)
            else:
                seqs = [words[:index] + [s] + words[index + 1:] for s in candidates]
                img = reader.forced_logprobs(reader._prefix_inputs(crop), seqs, True)
                null = reader.forced_logprobs(reader._null_inputs(), seqs, False)
                for (li, sp), (ln, _) in zip(img, null):
                    a, b = sp[index]
                    scores.append(sum(li[a:b]) - 0.5 * sum(ln[a:b]))
        return scores, time.perf_counter() - began

    run(True)                                     # warm the kernels so neither path pays for them
    cached, cached_seconds = run(True)
    plain, plain_seconds = run(False)

    worst = max(abs(a - b) for a, b in zip(cached, plain))
    print(json.dumps({
        "words_judged": len(jobs),
        "candidates_each": len(candidates),
        "uncached_ms_per_word": round(1000 * plain_seconds / len(jobs)),
        "cached_ms_per_word": round(1000 * cached_seconds / len(jobs)),
        "speedup": round(plain_seconds / cached_seconds, 2),
        "worst_score_difference": worst,
        "identical": worst == 0.0,
        "sample_cached": [round(s, 4) for s in cached[:4]],
        "sample_uncached": [round(s, 4) for s in plain[:4]],
    }, indent=1))
    return 0 if worst == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
