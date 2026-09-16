"""Find the hot code inside `grade`, which is now 2300 ms of a 3700 ms page.

Two candidates are already ruled out by measurement: the ink checker runs on CUDA (the image gate
asserts it) and costs 9.9 ms per judged word on this box, so ~40 words is ~400 ms of the 2300.
That leaves ~1.9 s of CPU-side Python — `dictation.candidates()` generates ~520 plausible
misspellings per flagged word, `align` runs twice when `line_align` is on, and every word is
tokenised and normalised. cProfile over one real page says which of those it is, by cumulative
time, rather than by anyone's instinct.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, "/app")

PAGE = Path("/app/page-42.jpg")
KEY = Path("/app/page-42.txt")


def main() -> int:
    from bilimai.pipeline import GLMReader, Pipeline
    from pilot.worker import MODEL_ADAPTER, MODEL_BASE

    reader = GLMReader(base=MODEL_BASE, adapter=MODEL_ADAPTER, device="cuda")
    pipeline = Pipeline(reader, read_guards=False)
    request = {
        "job_id": uuid.uuid4().hex,
        "type": "dictation",
        "image": {"uri": str(PAGE)},
        "key": {"text": KEY.read_text(encoding="utf-8"), "ignore_case": False,
                "count_punctuation": True},
        "options": {"return_transcript": True},
    }
    work = Path("/tmp/profile-grade")
    work.mkdir(exist_ok=True)

    warm = pipeline.grade(request, work)          # pay model load and kernel compilation first
    print("warm timings:", warm["provenance"]["timings_ms"], flush=True)
    print("marks:", len(warm.get("marks") or []),
          "| verifier:", warm["provenance"].get("verifier"), flush=True)

    profiler = cProfile.Profile()
    began = time.perf_counter()
    profiler.enable()
    response = pipeline.grade(request, work)
    profiler.disable()
    print(f"profiled page: {1000 * (time.perf_counter() - began):.0f} ms",
          response["provenance"]["timings_ms"], flush=True)

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats("cumulative")
    stats.print_stats(28)
    # only the rows matter, and only ours plus the obvious library hot spots
    for line in buffer.getvalue().splitlines():
        if "/app/bilimai" in line or "/app/pilot" in line or "cumtime" in line.lower() \
                or "onnxruntime" in line or "{method" in line or "transformers" in line:
            print(line.rstrip()[:190], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
