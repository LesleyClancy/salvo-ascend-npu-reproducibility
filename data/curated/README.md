# Curated numerical records

These files are a derived numerical view of selected local records; they are not the original logs, token streams or an independent rerun. They support checking the arithmetic in Table 5, Figure 2, Table S4 and the prefill and energy calculations. They do not allow a reader to independently verify token identity, reconstruct the third-party inputs, rerun the benchmark from scratch, or replay raw logits.

| File | Contents |
|---|---|
| `benchmark_55_summary.csv` | One selected record per case, with the experiment's reported equality flags and full/prefix 4-gram overlap. |
| `benchmark_all_repetitions.csv` | 168 rows of repetition timings covering 56 case-and-width records; a flag marks the 55 used in the analysis. **Rows are aligned by repetition index only, not in time.** The arms ran in blocks — every baseline repetition, then every speculative one — so row *i* does not mean the two values were measured next to each other. The summary medians are computed per arm and are unaffected by this. |
| `margins_55.json` | Per-case margin and exactness summaries; not position-by-position logits. |
| `divergence_margins.json` | Four divergence positions, margins at divergence, and median margins. |
| `prefill_measurements.json` | Four source measurements. The cold `copy_r1` observation is retained but excluded from the warm-prefill fit. |
| `energy_numeric.json` | Power time series, idle summary, phase events and repeated wall times. The power measurements are for the NPU chip, not the full board. |
| `source_inventory.json` | Original source paths relative to the study root, sizes and SHA-256 hashes; those original files are not included here. |

Run `python tools/verify_curated.py` from the repository root to recompute the numerical consistency checks. Benchmark speedup is the ratio of baseline and speculative medians. Copy-ability is the proportion of generated 4-grams found among prompt 4-grams. Prefix values use at most the stated number of generated tokens. Exactness is a recorded flag, not a check rerun by the verifier.

The case `free_1` occurs in both the e5a and e5c source records. The original 55-case analysis used e5c because sorted source files with the same ID were overwritten; the policy analysis retained e5a and reported e5c separately. Both observations and all repetitions are included here. Replacing the e5c record with e5a changes the full-sample Pearson correlation from 0.940069 to 0.939406; both round to 0.94, and the 12-case in-house median is unchanged. These are descriptive checks, not evidence that either observation is the uniquely correct choice.

The JFLEG dataset card labels its dataset CC BY-NC-SA 4.0; HumanEval's upstream repository supplies an MIT license. This package contains neither dataset's text nor token IDs. These facts alone do not establish the applicable rights or license for every derived metric; no additional license grant for those records is asserted. See `DATA_REUSE.md` and `THIRD_PARTY_NOTICES.md`.
