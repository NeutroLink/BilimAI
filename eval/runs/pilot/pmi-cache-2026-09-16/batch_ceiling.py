"""Size the win still left in `grade` before designing for it.

The prompt cache removed the repeated prompt passes (grade 2300 -> 2022 ms, scores bit-identical).
What remains is one candidate forward pass per judged word per branch: 16 words x (picture, no
picture). The no-picture branch shares ONE prompt across the whole page, so in principle every
word's candidates could ride in a single batched pass instead of sixteen.

Whether that is worth a refactor of the judge loop depends on how a forward pass scales with the
batch: if B=128 costs about what B=8 costs, the page's sixteen null passes collapse into one and
the saving is real. If it scales linearly, there is nothing there and the loop should be left
alone. This measures that curve rather than assuming it.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/app")

WORDS = ["мир", "поосок", "на", "цветной", "луг"]
CANDIDATES = ["поосок", "посок", "поесок", "поосак", "паосок", "поосокъ", "просок", "поособ"]


def main() -> int:
    from bilimai.reader import reader_class_for
    from pilot.worker import MODEL_ADAPTER, MODEL_BASE

    reader = reader_class_for(MODEL_BASE)(base=MODEL_BASE, adapter=MODEL_ADAPTER, device="cuda")
    reader._load()
    null_inputs = reader._null_inputs()

    def timed(batch: int) -> float:
        seqs = [WORDS[:1] + [CANDIDATES[i % len(CANDIDATES)]] + WORDS[2:] for i in range(batch)]
        reader.forced_logprobs(null_inputs, seqs, False, cache_key="null")   # warm
        began = time.perf_counter()
        for _ in range(3):
            reader.forced_logprobs(null_inputs, seqs, False, cache_key="null")
        return 1000 * (time.perf_counter() - began) / 3

    curve = {batch: round(timed(batch)) for batch in (8, 16, 32, 64, 128)}
    per_word_today = curve[8]
    print(json.dumps({
        "ms_by_batch": curve,
        "page_today_16_words_x8": round(16 * per_word_today),
        "page_if_batched_one_pass_of_128": curve[128],
        "null_branch_saving_ms": round(16 * per_word_today - curve[128]),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
