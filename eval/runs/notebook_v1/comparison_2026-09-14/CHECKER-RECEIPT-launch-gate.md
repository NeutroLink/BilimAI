# Independent checker — final launch-gate receipt (repair round 2)

**Verdict: PASS.** Every acceptance item in this packet is verified against the settled files. No
model was loaded or run; no source, label, reference or protocol file was edited by this checker; the
only writes were throwaway probes under `/tmp` and this receipt.

Candidates under review — hashed immediately before and after every battery below; identical
throughout (the runner and tests changed *during* the review — 880→884 and 174→186 lines — so every
number here is bound to these hashes, not to the earlier revisions):

| artifact | sha256 | lines |
|---|---|---|
| `eval/dictation/notebook_reader_comparison.py` | `665a10c3e60950556b50b2e91d179e58d562d23f80dab8332f4b4021436ee16b` | 884 |
| `eval/dictation/notebook_reference.py` | `27656543256d463607f333e287d4754d075bef5dc7597bb93ec992bbfe47e65f` | 456 |
| `tests/test_notebook_reader_comparison.py` | `04d7a0522a9b2c5276e94d64e6292ba2bdf351ff95fd6aca75816b99ed415974` | 186 |
| `eval/runs/notebook_v1/comparison_2026-09-14/protocol.json` | `69bf59ac2bb33f186ebe91eff567907f5b738ff2923fde51002336472bcb92c8` | — |

`notebook_reference.py` and `protocol.json` are byte-identical to the prior review's revisions; the
runner and tests carry the R1/R2 repairs.

**Environment.** `eval/.venv/bin/python` 3.11.14, pytest 9.1.1. Ruff is **not** importable in that
venv (`python -m ruff` → `No module named ruff`); the standalone binary `~/.local/bin/ruff` 0.16.5 was
used. torch 2.13.0 (`mps_available=True`, CUDA absent). ONNX Runtime providers on this machine:
`CoreMLExecutionProvider`, `AzureExecutionProvider`, `CPUExecutionProvider`.

## Measured commands

```
eval/.venv/bin/python -m pytest tests/test_notebook_reader_comparison.py -q   -> 9 passed in 1.41 s
ruff check <runner> <reference> <tests>                                      -> All checks passed (exit 0)
eval/.venv/bin/python -m py_compile <runner> <reference> <tests>             -> exit 0
```

Independent probes (all reproducible without a model; scripts were `/tmp/bilimai_launch_probe{1,3,4,6,7,8}.py`):

* populations and joins — raw-data recount + reference recount + span audit over 57 pages;
* protocol lock — synthetic configs against the real `protocol.json`;
* config identity / resume / refusal — a complete synthetic run directory plus `main([...])` calls;
* dedup + bootstrap — synthetic mark sets and synthetic per-page count vectors;
* association + transcription metrics + atomic write — hand-built `PageReference`/`Tok` objects;
* boundary audit — the 13 unjoined carries against every label index in both populations.

## Acceptance items

| # | item | result | evidence |
|---|---|---|---|
| 1 | common `models/cctc_tall` CoreML judge | PASS | Both branches build the judge through `_make_ctc(Path(args.ctc_model), threads, batch, -inf)` → `CTCWordVerifier(..., coreml=True)`; `args.ctc_model` defaults to `models/cctc_tall`; `_stamp_coreml` raises `SystemExit` unless the **live** session's `get_providers()` contains `CoreMLExecutionProvider`. `execution_providers(avail, True)` → CoreML first here. Model-tree hash matches the protocol (`9e50c19d…`). Not runtime-confirmed: no ONNX session was constructed (packet forbids inference). |
| 2 | CTC reuses the reading session | PASS | `judge = ctc  # one session: the CTC reader IS the judge`; the reader path is `read_pixel_rows(ctc, rows)` → `decode_crops(verifier, …)`, the same object. |
| 3 | GLM MPS guard | PASS | `--device` accepts only `mps`/`cuda` (CPU is not selectable); `_require_glm_accelerator` refuses `cpu` and raises when MPS/CUDA is unavailable; `reader.py::_load` uses `dev = self.device` when set, so there is no silent CPU fallback. Residual: the GLM stamp records the requested device (`"device": args.device`, `providers: None`), not a device read back from the loaded module — see limits. |
| 4 | 132/245 populations | PASS | Raw canonical records: total 251 → counted 245, deferred 113, `no_key` 132, excluded 6. Reference: 132/245/113, `no_key` a strict subset of `extended`, no `no_key` label deferred. Student stream identical across populations (tokens, words, scored flags). |
| 5 | verified-only line joins | PASS | 313 carries; 237 `joined_verified is True`, 76 not verified; 224 merges actually applied. For all 76 unverified carries the line's own span equals its raw tokenization and the continuation loses no token — no unverified carry is ever joined. 13 verified carries are not joined (12 `remainder_mismatch`, 1 `frag_regex_missed`, listed under limits); no label in either population touches a boundary token at any of the 13, so the effect is metric-invisible on this corpus. |
| 6 | threshold + dev receipt in the config identity | PASS | `_config_identity` carries `threshold`, `dev_receipt`, `judge_model`, `canvas`, `models`, runner/helper/split/canonical/detector hashes; changing `threshold` or `dev_receipt` changes `config_id`. Sealed demands both `--threshold` and `--dev-receipt`; `_validate_dev_receipt` re-verifies the dev run, refuses a foreign reader and a non-matching frozen threshold. |
| 7 | route-specific protocol model hashes | PASS | `_protocol_lock` matches the three data inputs exactly (`SystemExit` on split/canonical/detector tamper) and reports `model_hash_match` only for the models the route loads: GLM → `{glm_base, glm_adapter, ctc_model}`; CTC → `{ctc_model}`. A stale CTC hash flags `False` on both routes; a stale GLM hash flags `False` on the GLM route. All six recomputed hashes match `protocol.json`: canonical `6dc2c834…`, split `97d07566…`, detector `00b51a41…`, CTC `9e50c19d…`, GLM base `14f2207e…`, adapter `d5d8c5f8…`. |
| 8 | atomic pre-page `run_config` / resume | PASS | `run_config.json` is written (tmp file + `os.replace`) before the page loop; probe: two successive writes leave exactly one valid JSON file and no `.tmp*`; `--resume` requires a matching `config_id` and re-reads each page's `config_id` + image hash before skipping it. |
| 9 | sealed completed-run is not rewritten | PASS | `--resume` on a complete run prints `already complete` and returns 0 with byte + mtime identical for `run_config.json`, `summary.json`, `manifest.json`, `pages/page-34.json`; a non-resume run into a non-empty `--out` is refused with the files intact; a config-drifted resume is refused. |
| 10 | pupil transcription metrics | PASS | `transcription_metrics` scores `ref.student_words` (pupil) against the read hypothesis and skips unscored student tokens: a deletion of an unscored token is not counted, an insertion is (WER 1.0, n=1 scored word); pupil transcript → accuracy 1.0 while the proxy-key text → < 1.0. |
| 11 | exact read-index association | PASS | Extra words resolve only through `h2s[read_index]`: labelled pupil token → `caught`, unmatched read → `false`, unscored pupil token → `excluded`, out-of-range → `false`; non-extra marks resolve through the key index and exclude unscored key tokens. |
| 12 | false-position dedup | PASS (R1 repair) | The unit is now `(page, kind, position)` in both `_score_at` and `_population_block`: `spelling` + `punctuation_missing` at one key index → **2** false marks (`n_active` 2), while an identical duplicate (same kind+position) still collapses to 1. `by_confidence[*]["false"]` increments only on a new position, so it stays the same unit as `false_marks`. |
| 13 | bootstrap units | PASS | `_bootstrap_ci` resamples **pages** with replacement and sums page counts: one page → `[1.0, 1.0]`; two pages (100/100, 0/100) → `[0.0, 1.0]` (page-clustered — a token-level resample would give ≈`[0.43, 0.57]`); 40 identical pages → `[0.5, 0.5]`; seed `20260914` → deterministic. |

## Previous findings

* **R1 (blocker) fixed** — item 12 above; the test `test_false_marks_deduplicated_by_kind_and_position`
  now pins the kind-aware unit. No protocol change was needed, since de-duplicating by
  `(kind, position)` cannot collapse two marks that one `anchored_errors` pass emits.
* **R2 fixed** — `_verify` on `run_config.json` = `{}` prints
  `VERIFY FAIL: run_config.json missing required keys: [...]` and returns 1, no traceback.
* **R3 (observation, unchanged)** — the scored-word denominator is now **9010** (independently
  reproduced as `sum(n_scored_words)` over the 57 pages); `protocol.json` still does not pin the
  tokenization/join rule next to `model_hash_scheme`.
* **R4 (observation, unchanged)** — 12 counted labels carry a hyphen in `written`, and none of the 245
  (nor of the 132) touches a carry boundary, so the cross-carry hyphen path remains unexercised by
  corpus data.
* **R5 (doc nit, unchanged)** — `notebook_reference._tokenize` says "zero-dependency mirror" but
  delegates to `bilimai.dictation.tokenize`.

## Recurring-defects sweep (`plans/exec/RECURRING-DEFECTS.md`)

*Asserted absence* — every absence claim above names its source and its blind spot: "no label touches
a boundary token" is over the 245/132 labelled indexes of all 57 pages in both populations, not over
the transcript text; "no unverified carry is joined" is a span audit over `_build_stream`, which sees
token spans but not reader output. *Evidence that does not support the claim* — nothing here infers an
event from a populated field; the 9010 denominator is a **reproduction**, not an independent
confirmation of reader behavior. *Numbers that mislead* — 132 and 245 are reported separately and
never summed; no threshold, catch rate or CI is quoted as a result, because no run was executed and
none exists to quote (the sealed fold has not run); the synthetic probe figures are plumbing, not
results. *Identity and naming* — `false_marks` counts distinct `(kind, position)` marks, and the
per-confidence counter uses the same unit (verified above); the key/read index spaces are per
population and each mark is associated against its own population's reference. *Writing and
publishing* — writes are atomic (tmp + `os.replace`), a non-empty target is refused, a completed run
is never rewritten, and no frozen input, baseline or published dump was touched. *Scope* — this
checker wrote only `/tmp` probes and this receipt, and made no git operation; the only callers of
`notebook_reference`/`notebook_reader_comparison` are the runner and the contract tests (no second
copy, no shim, no stale caller).

## Residual limits

1. **No inference or model load was performed**, so reader and judge decode quality, spelling-threshold
   behavior, throughput, peak memory, provider identity at runtime and cross-run reproducibility all
   remain **unmeasured** and must not be quoted from this receipt.
2. **13 of the 237 verified carries are not joined** (the docstring's "then to the transcriber's own
   `joined` word" is unmet there) because `remainder = joined[len(fragment)-1:]` assumes `joined`
   starts at the fragment: p04-l09 `(представительная).`, p17-l05 `-физиологическое`, p21-l20
   `(указательное)`, p36-l44 `(порядковых)`, p50-l03 `северо-западные`, p51-l35 `-конституционная`,
   p52-l06 `Франции-` (fragment not matched by `_FRAG`), p55-l14 `-Республика`, p56-l23 `(Бишкек.`,
   p57-l07 `-западной`, p57-l30 `-президентская`, p57-l31 `-территориальное`, p57-l35
   `-располагается`. The fragments stay split — the conservative direction, never an invented merge —
   and no label, in either population, touches a boundary token at any of the 13, so it is
   metric-invisible on this corpus; recorded, not treated as a launch blocker.
3. **GLM accelerator stamp** is the requested `--device` behind a hard availability guard plus a reader
   with no fallback branch, not a value read back from the loaded module; the CTC side does read the
   live session's providers.
4. The sealed fold has not run: no dev threshold exists yet, so no operating point, catch rate or
   false-mark rate is reported here, and the 0.65 budget is unexercised on real reader output.
