"""Kaggle kernel — fine-tune the production line detector at the miss gap. 2026-08-22.

Runs eval/detectors/finetune_rp_segm.py on a free T4. Starts from ai-forever/ReadingPipeline-notebooks
segm_model.ckpt (the PyTorch weights behind the ONNX we ship, verified bit-identical) rather than from
scratch, and aims the fragment-focused sampler at the 335 missed short fragments (<= 1.7 of the
4.3-point detector gap).

ACCELERATOR: request the T4 EXPLICITLY when pushing —
    kaggle kernels push -p eval/detectors/kaggle --accelerator NvidiaTeslaT4
Kaggle's default is a P100 (sm_60) which Kaggle's own PyTorch build has no kernel image for; that
already cost this project three failed runs.

OUTPUT: /kaggle/working/runs_ft/{best.pt, segm_ft.onnx, log.txt, train_meta.json}. Pull it down and
gate it with eval/detectors/det_failure.py against the accept/kill thresholds in
plans/DETECTOR-FINETUNE-2026-08-22.md — merged <= 45 is the binding one, NOT the miss count.
"""
import os, subprocess, sys, time, zipfile
from pathlib import Path

T0 = time.time()
def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

sh("pip install -q pyclipper onnx 'huggingface_hub>=0.23'")

WORK = Path("/kaggle/working"); WORK.mkdir(exist_ok=True)
TMP = Path("/kaggle/tmp"); TMP.mkdir(exist_ok=True)

# ---- code -------------------------------------------------------------------------------------
# pushed as a Kaggle dataset by tools/kaggle_push.sh (see kernel-metadata.json dataset_sources)
CODE = None
for c in sorted(Path("/kaggle/input").rglob("finetune_rp_segm.py")):   # rglob: works whether the
    CODE = c.parent; break                                            # dataset nests the files or not
assert CODE is not None, "finetune_rp_segm.py not found in /kaggle/input — attach the code dataset"
print("code from", CODE, "->", sorted(p.name for p in CODE.glob('*.py')))
sys.path.insert(0, str(CODE))

# ---- data: school_notebooks_RU straight from the Hub (3 GB, kernels have internet) --------------
from huggingface_hub import hf_hub_download
SRC = TMP / "snru"; SRC.mkdir(parents=True, exist_ok=True)
for f in ("annotations_train.json", "annotations_val.json"):
    p = hf_hub_download("ai-forever/school_notebooks_RU", f, repo_type="dataset", local_dir=str(SRC))
    print("  got", f, round(os.path.getsize(p) / 1e6), "MB", flush=True)
zp = hf_hub_download("ai-forever/school_notebooks_RU", "images.zip", repo_type="dataset", local_dir=str(SRC))
print("  got images.zip", round(os.path.getsize(zp) / 1e9, 2), "GB — unzipping", flush=True)
with zipfile.ZipFile(zp) as z:
    z.extractall(SRC / "_img")
os.remove(zp)

# The archive nests images in subdirectories (train/ val/ test/ or similar) and the exact layout is not
# documented. build_pages() expects a FLAT <src>/images/<file_name>, so index every image by basename
# and link the ones the annotations actually reference. Do not guess at the layout — v1 of this kernel
# renamed the "common parent" and produced a directory holding 2 subdirectories instead of 2,000 pages;
# the assert below is what caught it.
import json as _json, collections as _c
# VERIFIED 2026-08-22 by reading the zip's central directory over HTTP: it holds images/ (1,857 real
# pages) AND __MACOSX/images/ (1,857 AppleDouble stubs like ._17_191.JPG). Their common parent is the
# archive root — which is exactly why v1's "rename the common parent" produced a folder containing 2
# subdirectories instead of the pages. No basename collisions; all 1,557 train-referenced pages present.
found = [q for q in (SRC / "_img").rglob("*")
         if q.suffix.lower() in (".jpg", ".jpeg", ".png")
         and "__MACOSX" not in q.parts and not q.name.startswith("._")]
print(f"  extracted {len(found)} image files across {len({q.parent for q in found})} directories", flush=True)
by_name = _c.defaultdict(list)
for q in found:
    by_name[q.name].append(q)
dupes = {k: v for k, v in by_name.items() if len(v) > 1}
if dupes:
    print(f"  WARNING: {len(dupes)} basenames appear more than once, e.g. "
          f"{list(dupes)[:3]} — keeping the first of each", flush=True)

IMD = SRC / "images"; IMD.mkdir(exist_ok=True)
wanted = set()
for split in ("train", "val"):
    with open(SRC / f"annotations_{split}.json", encoding="utf-8") as fh:
        wanted |= {im["file_name"] for im in _json.load(fh)["images"]}
linked = missing = 0
for nm in wanted:
    src_q = by_name.get(nm)
    if not src_q:
        missing += 1; continue
    dst = IMD / nm
    if not dst.exists():
        os.link(src_q[0], dst) if src_q[0].stat().st_dev == os.stat(IMD).st_dev else __import__("shutil").copy2(src_q[0], dst)
    linked += 1
n_img = linked
print(f"images ready: {n_img} of {len(wanted)} referenced by annotations ({missing} missing) in {IMD}", flush=True)
assert n_img > 1000, f"expected thousands of pages, got {n_img}"
assert missing < 0.05 * len(wanted), f"{missing} of {len(wanted)} annotated pages are absent from images.zip"

# ---- train ------------------------------------------------------------------------------------
import torch
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
      f"{torch.cuda.device_count()} x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}", flush=True)
assert torch.cuda.is_available(), "no GPU — push with --accelerator NvidiaTeslaT4"

OUT = WORK / "runs_ft"
sys.argv = ["finetune_rp_segm.py",
            "--src", str(SRC),
            "--out", str(OUT),
            "--epochs", os.environ.get("EPOCHS", "6"),
            "--batch-pages", os.environ.get("BATCH_PAGES", "2"),
            "--crops-per-page", os.environ.get("CROPS", "4"),
            "--lr", os.environ.get("LR", "1e-4"),
            "--shrink-r", "0.5"]
import finetune_rp_segm
finetune_rp_segm.main()

# ---- verify the artifact before anyone trusts it -----------------------------------------------
onnx_p = OUT / "segm_ft.onnx"
assert onnx_p.is_file() and onnx_p.stat().st_size > 50e6, f"ONNX missing or truncated: {onnx_p}"
import numpy as np, onnxruntime as ort
s = ort.InferenceSession(str(onnx_p), providers=["CPUExecutionProvider"])
y = s.run(None, {s.get_inputs()[0].name: np.zeros((1, 3, 896, 896), np.float32)})[0]
assert y.shape == (1, 3, 896, 896), y.shape
assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0, "output is not a probability map — sigmoid lost?"
print(f"ONNX verified: {onnx_p} {onnx_p.stat().st_size/1e6:.0f} MB, output {y.shape}, "
      f"range [{float(y.min()):.3f}, {float(y.max()):.3f}]")
print(f"TOTAL {(time.time()-T0)/60:.1f} min")
print("NEXT: pull runs_ft/segm_ft.onnx, then eval/detectors/det_failure.py — merged <= 45 is the gate")
