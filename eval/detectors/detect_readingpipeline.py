"""E4.1 — ai-forever ReadingPipeline (school-notebooks) line detector, zero-shot on the sealed RU exam.

Runs the SEGM-model ONNX (LinkNet, ai-forever/ReadingPipeline-notebooks: segm/segm_model.onnx) exactly as the pipeline
does: BGR image → resize 896×896 → /255 → NCHW → sigmoid outputs [shrinked_text, bordered_text, text_line] →
threshold 0.8, min_area 10 → contours → bbox → rescale → UpscaleBbox [1.4, 2.3] (pipeline_config.json 'shrinked_text').
Writes eval/runs/rp_det.json in the same format as surya_det.json ({file: [[x0,y0,x1,y1,conf], ...]}); score with
eval/detectors/score_detector.py. Needs: onnxruntime, opencv-python-headless, numpy (see /tmp/rp_venv).
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]


def get_upscaled_bbox(b, ux, uy):   # copied from ocrpipeline/predictor.py
    h = b[3] - b[1]; w = b[2] - b[0]
    yc = h * uy - h; xc = w * ux - w
    return max(0, b[0] - int(xc / 2)), max(0, b[1] - int(yc / 2)), b[2] + int(xc / 2), b[3] + int(yc / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="/tmp/rp_weights/segm/segm_model.onnx")
    ap.add_argument("--gt", default=str(ROOT / "eval/testset_v1/ru_pages/ground_truth.json"))
    ap.add_argument("--images", default=str(ROOT / "eval/testset_v1/ru_pages/images"))
    ap.add_argument("--out", default=str(ROOT / "eval/runs/rp_det.json"))
    ap.add_argument("--cls", type=int, default=0, help="output channel: 0 shrinked_text, 1 bordered_text, 2 text_line")
    ap.add_argument("--thr", type=float, default=0.8); ap.add_argument("--min-area", type=int, default=10)
    ap.add_argument("--upscale", type=float, nargs=2, default=[1.4, 2.3])
    a = ap.parse_args()
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]; print("input", inp.name, inp.shape, flush=True)
    gt = json.load(open(a.gt, encoding="utf-8")); out = {}
    for i, fn in enumerate(gt):
        img = cv2.imread(str(Path(a.images) / fn))            # BGR, like the pipeline
        H, W = img.shape[:2]
        x = cv2.resize(img, (896, 896), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        x = np.transpose(x, (2, 0, 1))[None]
        pred = sess.run(None, {inp.name: x})[0][0]             # (C, 896, 896), sigmoid already applied in the graph
        mask = (pred[a.cls] > a.thr).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < a.min_area: continue
            c = c.astype(np.float32); c[:, :, 0] *= W / 896; c[:, :, 1] *= H / 896
            x0, y0, w, h = cv2.boundingRect(c.astype(np.int32)); b = (x0, y0, x0 + w, y0 + h)
            b = get_upscaled_bbox(b, *a.upscale)
            conf = float(pred[a.cls][mask.astype(bool)].mean()) if mask.any() else 1.0
            boxes.append([float(b[0]), float(b[1]), float(min(W, b[2])), float(min(H, b[3])), conf])
        out[fn] = boxes
        print(f"[rp-det {i+1}/{len(gt)}] {fn}: {len(boxes)} boxes (gt {len(gt[fn]['lines'])})", flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1); print("→", a.out)


if __name__ == "__main__":
    main()
