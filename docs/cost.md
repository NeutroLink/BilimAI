# Cost & time log (E9.5)

Rule (from 2026-08-19): every rental line in ENGINEERING-LOG records **credit before → after** (Vast `users/current` →
`credit`) so this file can be reconciled; Vast's API exposes deposits but no per-instance ledger.

## GPU spend — Vast.ai (measured)
| item | value |
|---|---|
| Deposits 16–19 Aug 2026 | $6 + $5 + $9 + $10 + $8 + $4 = **$42.00** |
| Credit left 19 Aug 2026 17:15 UTC+5 | **$3.27** |
| **Spent** | **$38.73** |

## Itemised (from ENGINEERING-LOG; the rest is the round-4 day)
| date | run | box | wall | cost |
|---|---|---|---|---|
| 17 Aug | Round 3 (epoch 2 on 68 k, from v2) → v3 | 4× 5090 | 13 min | $0.30 |
| 17 Aug | Round-4 hosts: Oregon 46814884 (IPv6/GCS, 4 MB/s → bailed), Taiwan 42477066 (dead GPU → destroyed), Taiwan 33188966 (worked): v3 regeneration + replay-20/40 + E2E baselines | 4× 5090 | ~1 day incl. waits | ≈ **$21** (not itemised — includes the two failed boxes and the truncated-adapter re-run; "2/3 of the day's compute was wasted") |
| 18 Aug | Round 5 (vision LoRA, 425 k strips, 1 epoch) → v5 + baselines/E2E | Netherlands 4× 5090 | 2 h 02 m + evals | ≈ $6.5 |
| 19 Aug | `pmi_all` — 7,783-word verifier scoring | Belgium 3× 5090 | 7 min (4 provisioning + 2.3 job) | **$0.40** (credit 2.33 → 1.94) |
| 19 Aug | R5b «verbatim» run A (128 px, 1 epoch, 440 k records) + 4-prompt exam eval; 15 min lost to the thread-limit crash | Japan 4× RTX PRO 6000 ($5.87/h) | 1 h 57 m | **$6.64** (credit 9.92 → 3.27) |

## Kaggle (free tier)
Kernels A/B for HWR200 salvage and early rounds: $0 (quota hours only). Kaggle datasets: free.

## Time (rough, founder-facing)
Reader rounds R1–R5: 4 days of assistant work + ~4 GPU-hours. Verifier experiments (PMI, CTC, wide error model, full-exam
measurement): 2 days. Detector (E4.2): ½ day, $0 (measured on the Mac). Landing/README/plans: ~½ day.

## Per-page cost at scale (from 2026-08-17 timing, unchanged)
GLM reader ≈ 15 s per 40-line page unbatched on one 5090 (≈ $0.0017/page at $0.40/GPU-h); batched ≈ 3 s → ≈ $0.0004/page;
detector 0.9 s CPU; CTC judge ~30 ms/word CPU.
