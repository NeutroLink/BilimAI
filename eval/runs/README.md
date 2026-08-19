# eval/runs — result files (all scored with `eval/score.py` on `eval/testset_v1`)

## Leaderboard (2026-08-18 — new scorer: primary = char-weighted line CER, page-bootstrap 95 % CI; `leaderboard_2026-08-18.json`)

| reader | **cw line CER** [CI] | median (old headline) | word acc | lines perfect | novel-text cw | seen-text cw |
|---|---|---|---|---|---|---|
| GLM zero-shot | 0.817 [0.60, 1.23] | 0.779 | 0.17 | 1 % | 0.610 | 1.13 |
| GLM+LoRA v1 (20 k) | 0.338 [0.30, 0.38] | 0.317 | 0.40 | 15 % | 0.367 | 0.294 |
| GLM+LoRA v2 (68 k) | 0.277 [0.24, 0.32] | 0.263 | 0.47 | 20 % | 0.311 | 0.224 |
| **GLM+LoRA v3** (68 k ×2) | **0.246 [0.21, 0.28]** | 0.239 | 0.50 | 21 % | 0.284 | 0.188 |
| GLM+LoRA r4-20 | 0.274 | 0.263 | 0.50 | 18 % | 0.288 | 0.253 |
| **GLM+LoRA v4** (r4-40) | 0.252 [0.22, 0.29] — paired Δ vs v3 +0.006 [−0.016, +0.031], n.s. | 0.240 | **0.52** | 21 % | **0.270** | 0.225 |
| **GLM+LoRA v5 (R5: from base, school ×3 + HWR200, VISION LoRA + merger, r32)** — production | **0.052 [0.04, 0.06]** on the v1 pages (paired Δ vs v3 −0.19 [−0.22, −0.16]); **0.038 [0.030–0.048] on exam v2 (60 pages)** | 0.056 / 0.034 | **0.83 / 0.87** | 55 % / 58 % | **0.061 / 0.038** | 0.030 |
| GLM zero-shot **E2E** + RP (20 v1 pages, raw boxes / all 60, grown boxes) | 0.655 (all lines 0.668) / 0.703 (0.708) | 0.683 / 0.678 (page) | 0.16 / 0.19 | 0.5 % / 0.4 % | | |
| E2E v5 + RP, raw boxes (20 v1 pages) | 0.061 (all lines 0.096) | page 0.101 | 0.76 | | | |
| **E2E v5 + RP grown boxes** (E4.2, 20 v1 pages / **all 60**) | 0.058 (all lines 0.086) / **0.044 (all lines 0.063)** | page 0.089 / **0.066** | 0.78 / **0.83** | 46 % (60) | 0.045 (60) | line F1@0.5 0.89, recall 0.877 / 0.892 |
| exam v2 baselines (60 pages): v3 / v4 | 0.226 / 0.222 | 0.213 / 0.214 | 0.54 / 0.57 | 19 / 20 % | 0.241 / 0.223 | |
| Qwen3-VL-4B RU (gbull25, as-is) | 0.381 | 0.451 | 0.48 | 15 % | 0.352 | 0.426 |
| chukhrovns Qwen3-VL-4B | 0.430 | 0.500 | 0.41 | 10 % | 0.374 | 0.515 |
| olmOCR-2 zero | 0.529 | 0.732 | 0.36 | 6 % | 0.481 | 0.602 |
| Chandra zero | 0.943 | 0.749 | 0.43 | 6 % | 0.871 | 1.05 |
| **E2E** v3 + ReadingPipeline / + Surya | 0.278 / 0.317 (missed lines counted: **0.305 / 0.451**) | 0.288 / 0.298 | 0.46 / 0.37 | 12 / 11 % | 0.310 / 0.370 | |
| **E2E** v4 + ReadingPipeline / + Surya | 0.278 / 0.323 (all lines: **0.305 / 0.456**) | 0.277 / 0.302 | 0.47 / 0.39 | 11 / 11 % | 0.300 / 0.344 | |
| E2E Qwen + ReadingPipeline | 0.323 (all lines 0.348) | 0.282 | 0.46 | 9 % | 0.357 | |

Reading: v4 ≈ v3 overall (not significant), better on **novel** text and word accuracy, worse on memorised ("seen") texts —
HWR200 traded memorisation for generalisation. Seen/novel split: text twin in the training labels at rapidfuzz ratio ≥ 80
(460 seen / 344 novel of 804 lines; cache `data/derived/strips_ru_v1/seen_lines_strips_ru_v1.json`). E2E "all lines" is
the honest product number today (RP finds 86 % of lines).


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
| `round5b/` | R5b «verbatim» run A (2026-08-19): exam reads under 4 prompts (`glm_lora_lines*.json`), `retention_exam.json` (86: 45 % / keyed 37 % / key-copy 2.3 %), `school_val_score.json` (CER 0.0242 + retention on 362 val pairs 50 %), adapter (not promoted), data_stats/sample, train log |
| `round5/` | R5 = v5: exam v2 predictions, school-val (0.024), held-out (median 0.0), trainer_state, adapter scope (246 vision tensors), baselines_v2 (v3/v4 on exam v2) |
| `dictation/` | key-conditioned verification experiments (keyed_verify_*, tokfeat_*, pmi_* incl. `pmi_r5`), real_misspellings_v2.json (86 strict pairs) |
| `round4/replay20/`, `round4/replay40/` | round 4 (HWR200 221 k + school replay 5.8 % / 10.9 % of mix, from v3): exam 0.263 / **0.240** (v4), HWR200 held-out 0.094 both; trainer_state, holdout preds, gate logs, box logs |
| `qwen3vl_ru_hw/` | gbull25 Qwen3-VL-4B RU fine-tune, fp16, as-is: exam 0.451, held-out 0.283 (dropped) |
| `e2e/e2e_<det>_<reader>.json`, `e2e/summary_2026-08-17.json` | **end-to-end** exam (detector boxes → reader): page CER median v3 0.204 oracle → 0.293 RP / 0.392 Surya; v4 0.225 → **0.299** / 0.401; Qwen 0.343 → 0.342 / 0.477; RP finds 86 % of lines, Surya 73 % |
| `glm_train_r3_regen/` | v3 regenerated 2026-08-17 (original file was truncated); identical exam score 0.239 |
| `rp_det.json` / `rp_det_lines.json` | ReadingPipeline word boxes / grouped lines (F1 0.66@0.5, 0.87@0.3) |
| `zeroshot2_v2/`, `zeroshot3_v1/`, `surya_read_v*/`, `limerencii_smoke/`, `hwr200_smoke/`, `limerencii_A/` | raw kernel outputs + logs |
