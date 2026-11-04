# Data dictionary

Calibration inputs, recorded measurements and curated numerical summaries live under `data/` and `cases/`. This note says what each field means, so that a reader can check a number without reading the harness.

## `cases/calib/<case>/` — one frozen input case

Each directory holds one prompt, frozen at the moment of measurement. Nothing here is regenerated at run time.

| File | dtype / shape | Meaning |
|---|---|---|
| `prefix_ids.bin` | int64 `[L]` | the prompt's token ids, chat template already applied |
| `k_prefix.bin` | fp16 `[24, 2, 256, 64]` | the key cache after the prompt, capacity 256, positions `0 … L-1` |
| `v_prefix.bin` | fp16 `[24, 2, 256, 64]` | the value cache, same layout |
| `prefix_logits.bin` | fp32 `[151936]` | the deployed fp16 T=1 graph's logits after the last prompt token; `argmax` of this is the first generated token |
| `gen_ids.bin` | int64 `[N]` | the reference greedy stream from the deployed T=1 graph, `N` tokens, this is the exactness target |
| `timed_ids.bin` | int64 `[N]` | the same stream, kept separately because an early harness variant timed a different slice |
| `geom.txt` | text `"24 2 256 64"` | layers, KV heads, capacity, head dim |
| `meta.json` | JSON | prompt text, `L`, `N`, `usable`, EOS id, per-file sha256 |

`N` varies by case (16 to 64). It is the number of tokens the deployed baseline generated before EOS or truncation, and both arms are timed over exactly that length.

**The prompts are synthetic.** Every `meta.json` carries the prompt text so that a reader can see what was tokenised, and every one of them is invented for this study: the maintenance log in `copy_1` lists assets `A-1042`/`B-2087`/`C-3391` with pressures `7.4`/`6.1`/`8.2 bar`, none of which refers to a real device, site or recording. `free_*` prompts are open-ended writing requests, `copy_r*` are longer verbatim-copy passages. They were written to exercise a workload property (copy-ability), not to carry data. No personal or operational record was used to generate them, and none can be recovered from the token ids or the KV prefixes.

## `data/calibration/*.json` — one measured run

Written by `tools/board_run_e2o.py`, one file per graph width. The fields a reader needs:

| Field | Meaning |
|---|---|
| `gamma` | graph width `T` under test |
| `single_om` | `true` when only one graph was resident and row 0 served as the baseline (the only mode that fits this board at 1.26 GB per graph) |
| `warmup_full` | `true` when one complete baseline-plus-round warm-up was discarded before measuring |
| `cases.<id>.base_median` | median wall time of the baseline arm over the repetitions, milliseconds |
| `cases.<id>.spec_median` | same for the speculative arm |
| `cases.<id>.base_stage_median_ms` | per-stage medians: `h2d_ms`, `exec_ms`, `d2h_ms`, `commit_ms`. **`exec_ms` is the quantity reported in Table 4**; it excludes host transfer and the KV commit |
| `cases.<id>.base_exact` | per-repetition booleans: did the baseline reproduce `gen_ids.bin` exactly |
| `cases.<id>.spec_exact` | same for the speculative arm |
| `cases.<id>.spec_eq_base` | did the two arms agree with each other |
| `cases.<id>.spec_rounds_rep0` | per-round trace for repetition 0: `kind` (`spec`, `spec_first`, `t1_fallback`, `free_first`), `draft`, `accepted`, `bonus`, `matched_n`, stage times |

### The four divergent prompts of Table S4

Section 6.5 reports four prompts whose speculative stream differs from the deployed baseline: `copy_3`, `free_3`, `he_16`, `he_18`. Their status in this package is mixed, and the distinction matters:

| prompt | input case in `cases/calib/`? | measurement record in `data/calibration/`? |
|---|---|---|
| `copy_3` | **yes** — `e2r_s_inputs_qwen25_05b_copy_3` | **no** |
| `free_3` | **yes** — `e2r_s_inputs_qwen25_05b_free_3` | **no** |
| `he_16` | no — belongs to the HumanEval benchmark set | no |
| `he_18` | no — same | no |

Two of the four inputs are here; none of the four measurement records is in `data/calibration/`. Derived benchmark summaries and recorded flags for all four are in `data/curated/`. The six JSON files in `data/calibration/` cover `copy_1` (all six widths) and `copy_2` (T=1 only); neither `copy_3` nor `free_3` was part of that calibration session. Running `tools/compare_calibration.py` therefore reports `exact = yes` for every width — not because the four cases were checked and matched, but because they are not in the measurements at all.

What is missing for a full check of Table S4:

- **The runs.** The benchmark pass produced the four divergences. `data/curated/` contains numerical summaries and reported equality flags; the original token streams and full measurement JSON files are not included. Rerunning them requires regenerating the benchmark inputs.
- **The margins.** Numerical margin summaries are in `data/curated/`. They were derived by replaying logits, but full logits are not included; independently rerunning that replay needs a logits-returning graph. `tools/export_e2r_graphs.py` builds that graph; `tools/export_e3_gamma.py` builds argmax graphs only.

What the package *does* let a reader check is narrower than "no near-tie implies agreement". The six JSON files record that, for `copy_1` at six widths and `copy_2` at T=1, both arms emitted the same tokens on every repetition. That is an observation about those cases and those configurations, not a general property established across all near-tie-free inputs — and it says nothing about the four divergences, none of which was measured in that session.

## `data/curated/` — derived numerical results

See `data/curated/README.md` for schema, selection and verification. These files contain no prompt text or token streams; the equality flags cannot be independently rechecked from them. The legal status of third-party-derived numerical records is not determined; see `DATA_REUSE.md`.

## `data/calibration/*.log` — the harness transcript

Plain text, as written on the board. It includes the discarded warm-up, the per-repetition totals, and every failure the session hit. Two lines worth knowing about:

- `load single … loaded in 27.0s T=1` — graph load time, excluded from all reported numbers;
- `GUARD: watchdog aborted the build` — a memory watchdog tripping during ATC compilation, not a measurement failure.
