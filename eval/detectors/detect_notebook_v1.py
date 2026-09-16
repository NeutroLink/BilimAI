#!/usr/bin/env python3
"""A5 — run the PRODUCTION detector (bilimai.detector.RPDetector) over the my_notebook_v1 masked
evaluation pages and freeze its output as a PREDICTION artifact.

There is no box ground truth. These boxes are what the shipped detector produces on evaluation-only
pages; they must never be used as label geometry or scored against an invented reference. The only
cross-reference taken here is a CRUDE line-count sanity check against the corpus transcript
(canonical/page-NN.json -> page_lines, scored flag).

Driver, not a new engine: the module's own default ONNX and default growth factors are used and the
thresholds keep the class defaults. `RPDetector.__init__` pins `CPUExecutionProvider`, so to honour
the accelerator mandate the driver rebuilds the InferenceSession with identical SessionOptions and
the same model under the requested EP list; the provider the session ACTUALLY reports is stamped in
the run artifact (never the requested one). Measured reasons for any fallback are recorded verbatim.

Writes atomically (tmp + os.replace). Never touches the images, the canonical transcript or the ONNX.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RUN_DIR = ROOT / "eval/runs/notebook_v1"
DEFAULT_IMAGES = ROOT / "data/derived/my_notebook_v1/masked"
DEFAULT_CANONICAL = ROOT / "data/derived/my_notebook_v1/canonical"
DEFAULT_SPLIT = ROOT / "data/derived/my_notebook_v1/SPLIT.json"
DISCLAIMER = ("PREDICTIONS, NOT GROUND TRUTH - these boxes are the production detector's output on "
              "evaluation-only pages and must never be used as label geometry.")
# The FIRST A5 pass (2026-09-12) was produced by a hand-rebuilt session with PLAIN CoreML providers (no EP options).
# That config disagreed with CPU on 46/57 pages (max coordinate delta 2054 px; page-29 also gained a word), so it was
# superseded by the class default (explicit MLProgram/ALL), which matches CPU bit-for-bit. Recorded so nobody compares
# the two configurations silently. Evidence: eval/runs/notebook_v1/detector_provider_ab.json
SUPERSEDED_CONFIG = {
    "date": "2026-09-12",
    "superseded_artifact": "detector_v1_boxes.json first pass (deleted, not kept as a second artifact)",
    "superseded_provider_config": ("plain providers=['CoreMLExecutionProvider','CPUExecutionProvider'] with no EP "
                                   "options, built by hand in the driver"),
    "why_discarded": ("plain-EP CoreML disagreed with CPU on 46/57 pages, max coordinate delta 2054.374 px, page 29 "
                      "also gained one word (137 -> 138); a regenerable prediction dump that disagrees with the "
                      "shipped configuration is a trap, not evidence"),
    "accepted_provider_config": ("RPDetector class default: explicit {'MLComputeUnits':'ALL','ModelFormat':'MLProgram'} "
                                 "- matches CPU bit-for-bit"),
    "evidence": "eval/runs/notebook_v1/detector_provider_ab.json",
}


def peak_rss_mb() -> float:
    """Process peak RSS in MiB (macOS ru_maxrss is bytes; Linux is KiB)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def parse_pages(spec: str, all_pages: list[int]) -> list[int]:
    if spec.strip().lower() in ("", "all", "*"):
        return list(all_pages)
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [p for p in out if p not in all_pages]
    if bad:
        raise SystemExit(f"pages not in corpus: {bad}")
    return sorted(set(out))


def build_detector(onnx: Path, threads: int, coreml: bool):
    """RPDetector with its own defaults; the class selects the accelerator (CTCWordVerifier convention)."""
    from bilimai.detector import RPDetector
    return RPDetector(onnx, threads=threads, coreml=coreml)   # class defaults for thr/dilate/grow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="all", help="'all' or e.g. '1' / '1,2' / '2-57'")
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    ap.add_argument("--canonical", default=str(DEFAULT_CANONICAL))
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--out", default=str(RUN_DIR / "detector_v1_boxes.json"))
    ap.add_argument("--run-out", default=str(RUN_DIR / "detector_v1_run.json"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--provider", default="coreml", choices=["coreml", "cpu"])
    ap.add_argument("--probe-page", type=int, default=1, help="page run alone first for sizing")
    a = ap.parse_args()

    import cv2
    import onnxruntime as ort
    from bilimai.detector import DEFAULT_ONNX, LINE_GROW, WORD_GROW, COREML_EP

    onnx = Path(DEFAULT_ONNX)
    all_pages = list(range(1, 58))
    pages = parse_pages(a.pages, all_pages)
    threads = max(1, min(a.threads, os.cpu_count() or 1))

    avail = ort.get_available_providers()
    want_coreml = a.provider == "coreml"
    providers_requested = (["CoreMLExecutionProvider", "CPUExecutionProvider"] if want_coreml
                           else ["CPUExecutionProvider"])
    det = build_detector(onnx, threads, want_coreml)
    provider_reported = list(det.providers)
    fallback_reason = det.provider_reason or ""

    # ---- one-page sizing probe, run FIRST -------------------------------------------------
    probe_page = a.probe_page
    probe_img = Path(a.images) / f"page-{probe_page:02d}.jpg"
    img = cv2.imread(str(probe_img))
    if img is None:
        raise SystemExit(f"cannot read {probe_img}")
    try:
        t0 = time.perf_counter()
        det.detect(img)                              # warm the session + EP
        probe_s = time.perf_counter() - t0
    except Exception as exc:                         # a failed EP is a measured finding, not a crash
        if want_coreml:
            fallback_reason = f"{type(exc).__name__} running page-{probe_page:02d} on {provider_reported}: {exc}"
            det = build_detector(onnx, threads, False)
            provider_reported = list(det.providers)
            t0 = time.perf_counter()
            det.detect(img)
            probe_s = time.perf_counter() - t0
        else:
            raise
    probe_rss = peak_rss_mb()
    print(f"PROBE page-{probe_page:02d} | providers={provider_reported} | "
          f"wall {probe_s:.2f}s | peak RSS {probe_rss:.0f} MB", flush=True)

    # ---- full run -------------------------------------------------------------------------
    split = json.load(open(a.split, encoding="utf-8"))
    nb2fold = {nb: fold for fold, d in split["folds"].items() for nb in d["notebooks"]}
    page_fold = {int(pg): nb2fold[nb] for pg, nb in split["page_notebook"].items()}

    records: list[dict] = []
    per_page_s: dict[str, float] = {}
    violations: list[str] = []
    counts_rows: list[dict] = []
    t_full0 = time.perf_counter()
    for pg in pages:
        img_p = Path(a.images) / f"page-{pg:02d}.jpg"
        image = cv2.imread(str(img_p))
        if image is None:
            raise SystemExit(f"cannot read {img_p}")
        h, w = image.shape[:2]
        t0 = time.perf_counter()
        r = det.detect(image)
        dt = time.perf_counter() - t0
        per_page_s[f"{pg:02d}"] = round(dt, 4)
        lines = [[float(v) for v in b] for b in r["lines"]]
        words = [[float(v) for v in b] for b in r["words"]]
        if not lines:
            violations.append(f"page-{pg:02d}: zero line boxes")
        for bi, b in enumerate(lines):
            x0, y0, x1, y1 = b
            if not (x0 < x1 and y0 < y1):
                violations.append(f"page-{pg:02d} line {bi}: degenerate box {b}")
            if not (0 <= x0 and 0 <= y0 and x1 <= w and y1 <= h):
                violations.append(f"page-{pg:02d} line {bi}: outside {w}x{h}: {b}")
        records.append({
            "page": pg,
            "image": str(img_p.relative_to(ROOT)),
            "image_size": [w, h],
            "n_lines": len(lines),
            "n_words": len(words),
            "line_boxes": lines,
            "word_boxes": words,
            "line_words": [list(g) for g in r["line_words"]],
        })
        canon = json.load(open(Path(a.canonical) / f"page-{pg:02d}.json", encoding="utf-8"))
        scored = sum(1 for ln in canon["page_lines"] if ln.get("scored"))
        counts_rows.append({
            "page": pg, "fold": page_fold[pg],
            "detected_lines": len(lines), "transcript_scored_lines": scored,
            "signed_diff": len(lines) - scored,
        })
        print(f"[{len(records)}/{len(pages)}] page-{pg:02d}: lines {len(lines)} vs scored {scored} "
              f"({len(lines) - scored:+d}) words {len(words)} {dt:.2f}s", flush=True)
    full_s = time.perf_counter() - t_full0
    full_rss = peak_rss_mb()

    det_total = sum(r["detected_lines"] for r in counts_rows)
    sc_total = sum(r["transcript_scored_lines"] for r in counts_rows)
    agg = {"detected_lines": det_total, "transcript_scored_lines": sc_total, "signed_diff": det_total - sc_total}
    folds: dict[str, dict] = {}
    for row in counts_rows:
        f = folds.setdefault(row["fold"], {"pages": 0, "detected_lines": 0, "transcript_scored_lines": 0})
        f["pages"] += 1
        f["detected_lines"] += row["detected_lines"]
        f["transcript_scored_lines"] += row["transcript_scored_lines"]
    for f in folds.values():
        f["signed_diff"] = f["detected_lines"] - f["transcript_scored_lines"]

    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        git_head = f"unavailable: {exc}"

    onnx_sha = sha256_file(onnx)
    boxes = {
        "dataset": "my_notebook_v1",
        "artifact": "detector_v1_predictions",
        "provider": provider_reported,
        "onnx_sha256": onnx_sha,
        "image_source": str(Path(a.images).relative_to(ROOT)),
        "n_pages": len(records),
        "validation_violations": violations,
        "disclaimer": DISCLAIMER,
        "pages": records,
    }
    run = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": git_head,
        "onnx_path": str(onnx.relative_to(ROOT)),
        "onnx_sha256": onnx_sha,
        "provider_requested": a.provider,
        "providers_requested": providers_requested,
        "providers_session_reports": provider_reported,
        "coreml_ep_options": dict(COREML_EP[1]) if want_coreml else None,
        "superseded_config": SUPERSEDED_CONFIG,
        "provider_used": provider_reported[0] if provider_reported else "unknown",
        "coreml_available": "CoreMLExecutionProvider" in avail,
        "available_providers": avail,
        "provider_fallback_reason": fallback_reason or None,
        "threads": threads,
        "thresholds": {"thr_word": 0.8, "thr_line": 0.5, "dilate": 3, "min_area": 10, "unclip": 0.0},
        "growth": {"line_grow": list(LINE_GROW), "word_grow": list(WORD_GROW)},
        "sizing_probe": {
            "page": probe_page,
            "wall_clock_s": round(probe_s, 4),
            "peak_rss_mb": round(probe_rss, 1),
            "note": "run FIRST, alone, before the remaining pages",
        },
        "per_page_wall_clock_s": per_page_s,
        "full_run": {
            "pages": len(records),
            "wall_clock_s": round(full_s, 4),
            "pages_per_second": round(len(records) / full_s, 3) if full_s else None,
            "peak_rss_mb": round(full_rss, 1),
        },
        "counts": {"per_page": counts_rows, "aggregate": agg, "by_fold": folds},
        "counts_note": ("Crude line-count sanity only: no recall/precision/F1/IoU is computed - there is no "
                        "box ground truth, and inventing one is the defect this artifact avoids."),
        "validation_violations": violations,
        "scripts": [{"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha256_file(Path(__file__))}],
        "disclaimer": DISCLAIMER,
    }
    write_atomic(Path(a.out), boxes)
    write_atomic(Path(a.run_out), run)

    print(f"\nprobe page-{probe_page:02d}: {probe_s:.2f}s, {probe_rss:.0f} MB peak RSS")
    print(f"full run {len(records)} pages: {full_s:.1f}s ({len(records) / full_s:.2f} p/s), "
          f"{full_rss:.0f} MB peak RSS, providers={provider_reported}")
    print(f"lines detected {det_total} vs scored transcript {sc_total} ({det_total - sc_total:+d}); "
          f"folds {folds}")
    print(f"violations: {len(violations)}" + (f" {violations}" if violations else ""))
    print(f"-> {a.out}\n-> {a.run_out}")


if __name__ == "__main__":
    main()
