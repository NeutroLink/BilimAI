# eval/runs — result files (all scored with `eval/score.py` on `eval/testset_v1`)

| File | What |
|---|---|
| `glm_zero_lines.json` / `glm_zero_pages.json` | GLM-OCR zero-shot (line CER 0.78) |
| `glm_lora_ru_v1_lines.json` | GLM+LoRA run 1, 20k strips (0.317) |
| `glm_lora_ru_v2_lines.json` | run 2, all 68k strips, epoch 1 (0.263) |
| `glm_lora_ru_v3_lines.json` | round 3, epoch 2 on Vast 4×5090 (0.239) — adapter v3 |
| `glm_train_v4/`, `glm_train_r3/` | training records (trainer_state, results) |
| `deepseek_zero_*` / `olmocr_zero_*` | DeepSeek-OCR-2 (babbles, 18) / olmOCR-2 (0.73) |
| `chandra_zero_*` (HTML-stripped) | Chandra OCR 2 (0.75) |
| `surya_zero_*` | Surya OCR 2 reader (1.35 — reads Cyrillic as Latin) |
| `chukhrovns_zero_*` | Qwen3-VL-4B RU-handwriting fine-tune (0.50) |
| `surya_det.json` | Surya line detector boxes (F1 0.69@0.5) |
| `rp_det.json` / `rp_det_lines.json` | ReadingPipeline word boxes / grouped lines (F1 0.66@0.5, 0.87@0.3) |
| `zeroshot2_v2/`, `zeroshot3_v1/`, `surya_read_v*/`, `limerencii_smoke/`, `hwr200_smoke/`, `limerencii_A/` | raw kernel outputs + logs |
