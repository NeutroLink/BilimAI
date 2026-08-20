# E4.2 step 3 on PAI-DSW (free 24 GB GPU) — train the fragment-aware segmenter

One paste into the DSW **Terminal** (GPU environment, the cuda image). Total ≈ 2–2.5 h GPU time.
**Stop the instance when done — the quota counts hours while it runs.**

```bash
set -e
pip install -q segmentation_models_pytorch opencv-python-headless modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
export MS_TOKEN=<your modelscope.ai token>          # modelscope.ai/my/myaccesstoken
# data: raw pages (2.7 GB tar) + the training script from the bootstrap repo
modelscope --endpoint https://www.modelscope.ai --token $MS_TOKEN download Jahongir/bilimai-school-pages-raw --repo-type dataset --local_dir .   # two .part_ files (uplink caps blobs at ~2 GB)
modelscope --endpoint https://www.modelscope.ai --token $MS_TOKEN download Jahongir/bilimai-vast-bootstrap train_segm.py --repo-type dataset --local_dir .
cat school_pages_train.tar.part_* > school_pages_train.tar && tar xf school_pages_train.tar && rm school_pages_train.tar*
nohup python train_segm.py --src train --out runs_segm --epochs 8 --batch 6 > train.out 2>&1 &
tail -f train.out          # Ctrl-C detaches the tail, training keeps running
```

When `DONE best box line F1 …` appears, upload the result and stop the instance:

```bash
modelscope --endpoint https://www.modelscope.ai --token $MS_TOKEN upload Jahongir/bilimai-segm-ft runs_segm --repo-type dataset --commit-message segm_ft
```

Notes
- If `modelscope.ai` is unreachable from DSW (mainland routing), tell the assistant — fallback is uploading the tar to
  the Kaggle-equivalent on modelscope.cn or scp; do not burn quota retrying.
- The script prints per-epoch `box line F1@0.5` on 40 val pages with the production post-processing; the exported
  `segm_ft.onnx` is a drop-in for `RPDetector(onnx=...)` (same input/output convention).
- Acceptance (on the Mac afterwards): exam line F1@0.5 ≥ 0.89 (current) AND fragment recall clearly up AND E2E CER not worse.
