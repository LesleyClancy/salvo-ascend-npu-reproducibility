# Provenance, and what was removed before publication

This package was assembled from a working repository that accumulated twelve weeks of experimental notes, logs and project-management files. Publishing that repository as-is would have exposed material that is not evidence, and some of it would have identified people or machines. This note records exactly what was left out, so that the omissions are auditable rather than invisible.

## Removed categories

| Category | What it was | Why it is not here |
|---|---|---|
| **Project-management notes** | `tasks/` (forty pre-registration notes), `PROJECT_STATE.md`, `DECISIONS.md`, `PROJECT_SPEC.md`, hand-off memos, the kickoff brief | These record how the work was organised and what was planned next. They are not measurements, they change after the fact, and several quote internal working agreements verbatim. None of them is needed to reproduce a number. |
| **Manuscript sources** | `paper/` (section files, bibliography, LaTeX and Word builds, supplementary material) | Published separately. The package points at the paper; it does not duplicate it. |
| **Internal analysis reports** | `reports/` (thirty-nine notes, written in Chinese) | Working interpretation, superseded in places by the paper itself. `docs/DATA_DICTIONARY.md` gives the field-level description that a reader actually needs. |
| **Compiled artefacts and weights** | `artifacts/` (21 GB: ONNX graphs, `.om` files, weight sidecars) | Redistribution permission for `.om` output was not confirmed; weights and ONNX exports can be obtained or rebuilt separately. Hashes are in `MANIFEST.md`. |
| **Profiling captures** | `profiles/` (136 MB of `msprof` output) | Only the derived shares appear in the paper (Section 2.2); the original raw captures are not provided, limiting independent checks. |
| **The full benchmark inputs** | `cases/` except the twelve calibration cases (1.1 GB) | The complete inputs require the in-house prompt definitions, public datasets and input-generation scripts. The omitted prefixes and reference streams limit independent verification. |

## Fields rewritten in what remains

The environment record was rewritten as `configs/environment.json`. Other publication adaptations and the treatment of numerical records are documented below.

| Field | Original | Published |
|---|---|---|
| Environment name and interpreter path | a personal conda environment name and its absolute Windows path, repeated in eight places | removed; the version number (Python 3.13.16) is kept |
| Capture attribution | a named capture tool recorded next to every environment field | removed; `configs/environment.json` records technical facts only |
| Board access | an internal address and SSH alias | removed; `docs/REPRODUCE.md` says "the board" |
| Board home paths | absolute paths under the board's default user | removed from the six calibration JSON `dir` fields and from the calibration wrapper; the default account path remains in unedited harness logs, one verification JSON and this provenance record |
| Workflow annotations | lines recording who authorised a configuration change and when | removed; they describe a workflow, not an environment |

## What was not touched

The calibration measurements retain their recorded numerical values and flags. The curated benchmark and energy files are derived numerical extracts and omit raw prompts, tokens and other fields, as documented in `data/curated/README.md`. Specifically:

- every JSON in `data/calibration/` parses equal to the file the harness wrote on the board, with one field changed: `dir` reads `~/kv_probe/e3` instead of `/home/HwHiAiUser/kv_probe/e3`, and the files were re-serialised after that edit, so they are not byte-identical. No timing, count or exactness flag differs;
- the harness logs in the same directory still contain the failed runs, the watchdog aborts and the `atc` exit faults, because a reader checking our claims should see the failure modes we describe;
- no prompt, token id, KV prefix or reference stream in `cases/calib/` was modified;
- some scripts were adapted for this package; the table below describes the documented changes.

## What was changed for publication

| File | Change | Why |
|---|---|---|
| `tools/export_e3_gamma.py` | added `import pathlib`; case directory defaults to `cases/calib/` with a `KVPROBE_CASE` override; the read of `cases/dev/t07_case.json` was removed (geometry now comes from the manifest); the check builds its input block from `T` rather than always 16; the reference comparison covers `min(T, 16)` rows; a missing reference graph skips instead of aborting; `raw/` is created before the log is written | the shipped package has no `cases/dev/t07_case.json`, and the original ran only with the reference graphs present |
| `tools/export_e2r_graphs.py` | same manifest substitution; absent `B_head.onnx` skips the T=1 cross-check; a skipped comparison no longer counts as a failure in the verdict | `B_head.onnx` belongs to the layout study and is not shipped |
| `tools/board_run_e2o.py` | host pointers go through a `_ptr()` helper that falls back to `bytes_to_ptr` | pyACL's `numpy_to_ptr` is NumPy 1.x ABI; the current board image ships NumPy 2 |
| `tools/board_calib9.sh` | resolve the package root, invoke `tools/board_run_e2o.py`, default to the documented single-graph mode, and return the harness exit status | remove a board-specific path and fix invocation and error reporting; this adapted wrapper has not been re-run on the board |
| `tools/board_atc9_e3.sh` | working directory is `$KVPROBE_E3_DIR`, defaulting to the caller's directory, not a hard-coded board path | the hard-coded path does not exist on a reader's board |
| `tools/compare_calibration.py` | both arms must be present and non-empty; entry count must match `reps`; all six widths must appear; disagreement between the arms is now a failure rather than a printed column | the earlier version could return `yes` on a half-recorded case |
| `tools/make_e2r_inputs.py`, `make_gec_inputs.py`, `make_he_inputs.py`, `make_e2r2_inputs.py`, `export_e2r_graphs.py` | **added** | the documentation referenced them; they were missing from the package |
| every text file | CRLF → LF | Windows checkouts gave `$'\r': command not found` on the board |
| `data/calibration/*.json` | `dir` rewritten to `~/kv_probe/e3`; files re-serialised | the field held the board's absolute path |
| `configs/*.json` | `local_dir` is a relative path | the original held a host absolute path |
| `docs/*.md`, `README.md`, `MANIFEST.md` | rewritten | the originals were internal notes |

**Which version produced the numbers in the paper?** The original scripts, before the edits above. None of these changes touches the arithmetic or the tensor layout. One does touch the timed region: `_ptr()` is called between the two `perf_counter()` readings in `OmModel.call()`, and on NumPy 2 it builds a byte buffer with `tobytes()` before copying. For the small `ids`/`positions` arrays the copy is small, but **its cost inside the timed region has not been measured separately**, and it cannot be inferred from the array sizes alone — the fallback branches on an exception, and call overhead is not proportional to payload. It is a change inside the measured interval; that is all we can say.

The 2026-11-02 run in `data/verification_2026-11-02.md` is the closest check available: it exercised the published versions end to end and recovered the archived timing on one case. That is evidence the adaptation did not disturb the result; it is not evidence that every edited path is equivalent to the original on every case.

## Curated numerical records

The `data/curated/` files preserve selected numeric measurements and document the duplicate `free_1` observation. They are derived records, not full original harness files, and omit prompt text, token streams and absolute paths. Their source inventory contains hashes of source files not shipped here. `REVIEW_HOLD.md` identifies the unresolved rights assessment.

## A note on language

The public candidate uses English filenames and documentation. The omitted working record contains Chinese internal notes.
