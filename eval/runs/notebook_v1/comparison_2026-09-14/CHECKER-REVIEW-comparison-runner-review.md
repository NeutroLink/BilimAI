# Independent checker re-verification — GLM-vs-CTC notebook runner (repair round 1)

**Verdict: FAIL — one remaining definitional blocker (R1); every previously reported blocker and
minor is verified fixed.** No model was run; no source, label or reference file was edited.

Candidates under review (sha256 at verification time; re-checked after the battery — unchanged):

| artifact | sha256 | size |
|---|---|---|
| `eval/dictation/notebook_reader_comparison.py` | `88035bed7aba227cd17f20f1e51275696a5b0a4df00c039d156f4368d6fceee4` | 873 lines |
| `eval/dictation/notebook_reference.py` | `27656543256d463607f333e287d4754d075bef5dc7597bb93ec992bbfe47e65f` | 456 lines |
| `tests/test_notebook_reader_comparison.py` | `91e12e009bf45792b5a4b72868ee47c7f16e6cc11c9dcb6f6c4bb7524b56bdf9` | 173 lines |
| `eval/runs/notebook_v1/comparison_2026-09-14/protocol.json` | `69bf59ac2bb33f186ebe91eff567907f5b738ff2923fde51002336472bcb92c8` | re-locked |

> Note: the runner was still being written when this round started (sha `89bb42b5…` did not compile —
> dangling fragment at lines 441-444, `build_config` body clobbered). All results below are against
> the settled `88035bed…`, confirmed by hashing immediately before and after the battery.

## Previously reported defects — all verified fixed

| id | defect | probe | result |
|---|---|---|---|
| B1 | judge was `models/readingpipeline/ocr`, not the protocol's `models/cctc_tall` | `build_config(reader='ctc'/'glm', fold='sealed')` | `judge_model.path = models/cctc_tall`; CTC route reuses the reading session as judge; GLM route loads one CTC judge from `--ctc-model` |
| B2a | `--threshold` absent from the config identity | `_config_id` on configs differing only in threshold | differs (test `test_config_identity_binds_threshold_and_frozen_inputs`); config carries `threshold` + `dev_receipt` |
| B2b | `--verify` blind to a changed threshold | synthetic complete dev + sealed runs, then `_verify` | tampered summary (even with a regenerated manifest) now fails: `summary no_key.false_marks: 2621 != recomputed 2620` |
| B2c | `--resume` re-scored a sealed run at a new threshold | `main(['--reader','ctc','--fold','sealed','--out',…,'--resume','--threshold',…,'--dev-receipt',…])` on a complete run | `rc 0`, prints `already complete`, **no file rewritten** |
| B2d | nothing bound the sealed threshold to dev | `_validate_dev_receipt` | accepts the dev dir whose frozen threshold equals `--threshold`; refuses a different threshold and a different reader |
| B3a | protocol model hashes unverifiable | `build_config` protocol lock | data inputs match exactly; `model_hash_match` all `true` for CTC (1 key) and GLM (3 keys) after the re-lock, which now records `model_hash_scheme` = the runner's `_hash_tree` |
| B3b | canonical labels never hashed or verified | config fields + `_verify` | `canonical_aggregate_sha256`, `split_sha256`, `detector_sha256`, runner/helper hashes are stamped and re-checked |
| B3c | sealed CTC run blocked by GLM hashes it never loads | `build_config(reader='ctc', fold='sealed')` | passes (only loaded models gate the run) |
| M1 | dev tie-break chose the most aggressive cutoff | `select_threshold` on a tied mark set | `threshold 5.0`, `runner_up 0.0` — ties now take the higher cutoff; `no_operating_point` freezes a finite above-all cutoff and reports the failure |
| m2 | `_anchor_lines` ignored `line_id` | `_anchor_lines(lines, 999, None, <valid line_id>)` | `[3]` — `line_id` is consulted first, legacy fallback intact |
| m3 | CTC artifacts stamped `device: "mps"` | `build_config(reader='ctc')["device"]` | `None` |
| m4 | discarded `det_line` strip pass, double image reads | source | CTC decodes only detector word crops through `read_pixel_rows`; one image read per page per route |
| m5 | unguarded detector path; no resource stamp | source + `_verify` | frozen paths guarded with `.exists()`; `free_memory_bytes()` before sizing; `peak_rss_bytes()` and `wall_s`/`pages_per_s` stamped per page and per run |

Focused non-model checks (all against `88035bed…`):

```
eval/.venv/bin/python -m pytest tests/test_notebook_reader_comparison.py -q   -> 8 passed in 1.45s
key index-space invariant  len(ref.key_tokens) == len(tokenize(ref.key_text)) elementwise,
                            57 pages x 2 populations                          -> 0 mismatches
populations                no_key 132 / extended 245 / deferred 113, strict subset -> exact
mirror  anchored_errors vs grade_dictation(verifier=None) over 57 perturbed reads -> 0 divergences
verify/plumbing            complete dev and sealed runs verify 0; resume on a complete run rewrites nothing
bootstrap units            degenerate page set -> caught/labels [0.5,0.5]; false/scored x100 -> [1.0,1.0]
```

## R1 — BLOCKER (definitional): false marks are de-duplicated by position, collapsing distinct kinds

`_score_at` and `_population_block` de-duplicate false marks by `mark_position(mark, h2s)`, which is
`["key", key_index]` for every non-extra mark and `["student"|"read", …]` for an extra word. A key
index can carry more than one mark kind (`anchored_errors` emits `spelling` plus one `punctuation_*`
for the same key token), so distinct marks silently become one.

Reproducible, no model:

```python
R._score_at([[{"kind":"spelling","margin":3.0,"status":"false","label_ids":[],"position":["key",7]},
              {"kind":"punctuation_missing","margin":None,"status":"false","label_ids":[],"position":["key",7]}]], 0.0, 100)
# -> {'caught': 0, 'false_marks': 1, 'n_active': 2, 'false_per_100': 1.0}
```

Impact on a synthetic noisy read over all 57 pages (drops/insertions/letter changes/punctuation
stripped, `random.Random(11)`; a simulation, not reader output): **1734 false marks -> 1715 counted
positions, 19 collapsed (1.10 %)**; all 19 positions carried two kinds.

**Expected.** `protocol.json` says "minimize false marks", reports "false marks per 100 scored
words" and separately lists "missing_word, extra_word, capitalization, punctuation and spelling
mark counts". Nothing in the protocol defines a false mark as a position. The unit of the founder's
0.65 budget — and therefore of the frozen dev threshold — must be pinned before the sealed fold,
which is one-shot.

**Fix.** Either (a) add the unit to the protocol (`"false_mark_unit": "distinct mark position
(key_index for non-extra marks, read/student index for extra words)"`) and state it in the summary
caveat, or (b) count per mark — de-duplicate by `(kind, position)`, which cannot co-occur from a
single `anchored_errors` pass and is therefore a no-op. Also align `by_confidence[*]["false"]`
(marks) with `false_marks` (positions); they are currently two units in one block.

## Minor (non-blocking)

* **R2** — `_verify` raises `KeyError: 'canvas'` on a run directory whose config lacks required keys
  (repro: write `run_config.json` = `{}` and call `_verify`). `_validate_dev_receipt` now calls
  `_verify(path)` on a **user-supplied** `--dev-receipt`, so a wrong directory produces a traceback
  instead of a clean refusal. Fix: validate required keys up front and use `config.get("canvas", {})`.
* **R3** (observation) — the false-mark denominator moved from **8938 to 9010** words (+72, +0.8 %)
  because joins now require `joined_verified` (237 of 313 carries verified). The new rule is the
  better-grounded one and nothing has been published against the old one, but the protocol does not
  pin the tokenization/join rule — record it next to `model_hash_scheme`.
* **R4** (observation) — the cross-carry hyphen path is unexercised: 0 of 245 labels quote a fragment
  across a carry, so the `виб-ретто` case is covered by construction only, never by corpus data.
* **R5** (doc nit) — `notebook_reference._tokenize` says "zero-dependency mirror" but delegates to
  `bilimai.dictation.tokenize`.

## Split ruling — satisfied

The extraction landed: 873 lines (runner: readers, marking, config/verify, CLI) + 456 lines
(`notebook_reference.py`: dataclasses, `load_dataset`, stream/token/locate/edits,
`build_page_reference` — inference-free, the module the test imports). No duplicated logic, no
second convention and no compatibility shim found; the runner imports the reference module by name.

## Residual limits

No inference was executed, so reader and judge decode quality, margins, throughput, peak RSS and
stability remain unmeasured and must be stated as unmeasured. My quantitative dedup impact is a
simulation, not reader output. Synthetic figures produced with a stub judge (e.g. the 32.7
false/100 in the tamper probe) are plumbing artefacts and are not results.
