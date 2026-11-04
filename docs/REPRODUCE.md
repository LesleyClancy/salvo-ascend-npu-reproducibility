# Reproducing the measurements

Two machines are involved and it matters which step runs where. Export and input generation happen on the development host; compilation and measurement happen on the board.

## Which machine runs what

| Step | Machine | Needs |
|---|---|---|
| export graphs | host | `torch`, `transformers`, `onnx`, `onnxruntime`, checkpoint |
| regenerate benchmark inputs | host | the same, plus the datasets |
| compile | board | CANN `atc`, the ONNX files and the weight sidecar |
| measure | board | CANN runtime, `numpy`, the compiled `.om` |
| compare | either | `numpy` only (standard library otherwise) |

## 0. Check the environment first

On the board, from the package root:

```bash
bash docs/check_env.sh
```

It prints the board's memory, whether `ASCEND_HOME_PATH` is set, whether `atc` is reachable and which NumPy the TBE would pick up. Fix anything it flags before going further; the two failure modes it catches are silent otherwise.

## 1. Export the graphs (development host)

```bash
python tools/export_e3_gamma.py 1,2,4,8,16,32 --model qwen25
```

Writes, for each width `T`, `g<T>.onnx` plus one shared external-data sidecar of 1.25 GB. The sidecar is shared on purpose: all six graphs read the same weights, so the per-launch weight stream is identical across widths, which is the property Section 6.2 depends on. Record the sha256 of each file. `MANIFEST.md` carries full hashes for `g1.onnx` and `g32.onnx` from their respective recorded runs; for the other widths it lists sizes only. Size is a smoke check, and a reader should record hashes from their own export.

## 2. Copy to the board and compile one width

```bash
# the board needs both the graphs and the scripts; copy them separately so that the
# relative paths inside the package still resolve
scp -r g*.onnx e3.weights.bin <board>:~/kv_probe/e3/qwen25_05b/
scp -r tools <board>:~/kv_probe/
scp -r cases  <board>:~/kv_probe/

ssh <board>
cd ~/kv_probe/e3/qwen25_05b
source /data/Ascend/cann-9.1.1/set_env.sh   # adjust to the installed CANN version
bash ~/kv_probe/tools/board_atc9_e3.sh 1        # ~11-12 min for T=1
```

The script works in the directory it is started from, so run it from `qwen25_05b` (where
the ONNX files are) and give it an absolute path. It deletes only the `kernel_meta`
scratch directory of that working directory.

Repeat with `32`, `16`, `8`, `4`, `2` as needed. Expect:

- the build to consume essentially all 3.4 GB and lean on swap;
- `atc` to print `ATC run success` and then fault during exit (SIGBUS/SIGSEGV). This is the behaviour we saw on this board, not a signal that the build succeeded. Verify by loading the `.om` and running step 3; the exit status tells you nothing.

The script takes a width list, so `bash tools/board_atc9_e3.sh 2 4 8 16` compiles four in sequence.

## 2b. Regenerating the benchmark inputs (only for Table 5 or Table S4)

The twelve calibration cases are shipped, and twelve of them are among the 55 benchmark inputs; the other 43 are not. Regenerating those needs three things, and only the first is in this package.

| Needed | Where it comes from |
|---|---|
| Prompt definitions (15: 9 in-house + 6 long-copy) | **in the package**, `cases/dev/e1_prompts.json` and `e2r2_prompts.json` |
| A **logits-returning** T=1 graph | not shipped; export it first (below) |
| JFLEG `dev.src`, HumanEval `HumanEval.jsonl.gz` | not part of this release; obtain them under their own terms — HumanEval MIT, JFLEG CC BY-NC-SA 4.0 (`THIRD_PARTY_NOTICES.md`) |

```bash
# 1. the logits-returning graph the generators read prefix logits from
#    no arguments: it walks its own model list and exports Qwen2.5-0.5B *and*
#    Qwen3-0.6B. The generators only need the Qwen2.5 g1 output; the rest is
#    extra work you can ignore.
python tools/export_e2r_graphs.py
#    writes artifacts/e2r/qwen25_05b/{g1,g16}.onnx — g1 emits logits, which is what
#    the generators consume. This is a different export from export_e3_gamma.py,
#    which produces the argmax graphs the board runs.

# 2. datasets, placed where the generators look for them
mkdir -p raw/humaneval_2026-10-24
#    JFLEG dev.src        -> raw/jfleg_dev.src.txt
#    HumanEval.jsonl.gz   -> raw/humaneval_2026-10-24/HumanEval.jsonl.gz

# 3. the four generators
python tools/make_e2r_inputs.py     # in-house: copy_1..3, code_1..3, free_1..3
python tools/make_e2r2_inputs.py    # long copy: copy_r1..r6
python tools/make_gec_inputs.py     # JFLEG 20, >=25 source tokens then L <= 180
python tools/make_he_inputs.py      # HumanEval 20, L <= 180
```

The generators read the checkpoint path from `configs/model_qwen25_05b.json`, so the checkpoint must be at `models/Qwen2.5-0.5B-Instruct` or that field must be edited.

## 3. Measure

```bash
# set_env.sh must be sourced in the same shell: it sets LD_LIBRARY_PATH, without
# which `import acl` fails with "libascendcl.so: cannot open shared object file".
source /data/Ascend/cann-9.1.1/set_env.sh

cd <package root>
E3_REPS=5 \
E2O_DIR=~/kv_probe/e3 \
E2R_INPUTS=$PWD/cases/calib \
E3_SINGLE=1 \
E3_WARMUP=1 \
python3 tools/board_run_e2o.py copy 1 copy_1
```

**NumPy version.** pyACL's `numpy_to_ptr` is compiled against the NumPy 1.x C ABI; on a
board whose system Python carries NumPy 2.x, `import acl` succeeds but the first
pointer call raises `numpy.core.multiarray failed to import`. `board_run_e2o.py`
routes every host pointer through a small `_ptr()` helper that falls back to
`acl.util.bytes_to_ptr`, so it runs on either version. If you prefer the original
path, `pip install "numpy<2"` — but the board has no network, so that means carrying
a wheel across.

- `E3_SINGLE=1` is not optional on this board: it loads one graph and uses row 0 as the
  baseline. Without it the harness expects two resident graphs and fails at allocation. At 1.26 GB per graph this is the only mode that fits the board; see `docs/ENVIRONMENT.md`.
- `E3_WARMUP=1` discards one complete round-plus-baseline before measuring. Without it the first repetition carries a 369 ms first-load penalty that survives a single-step warm-up.
- The last two arguments are the workload class and a comma-separated case filter.

Five repetitions is what the paper reports for `T=1`; the other widths used two, and `T=32` used three. The JSON records how many ran, so the record is self-describing.

## 4. Compare against the paper

```bash
python tools/compare_calibration.py data/calibration
```

Prints one line per width: measured execution median, the value in Table 4, the delta, and whether both arms were token-identical. A fresh run dropped into the same directory compares the same way.

## 5. What a pass looks like

```
  T  exec (ms)   paper    delta  cases  exact
  1     105.69   105.5    +0.2%      2  yes
 32     113.60   113.6    +0.0%      1  yes
```

The two anchors are what the paper's absolute numbers rest on. The intermediate widths move by up to 2% between sessions, which is inside the protocol's own resolution (6.8% CV, Section 6.6); do not read a 2% move as a platform difference.

## What an export `PASS` actually certifies

The export scripts print `EXPORT VERDICT: PASS`. That word covers three different checks, and they are not equally strong. **Only one of the first two runs per invocation** — the script takes the reference branch if the reference file exists and the PyTorch branch if it does not, never both:

| Situation | What is compared | How many rows |
|---|---|---|
| reference `g16.onnx` present, T ≤ 16 | the new graph against that **independent** graph | min(T, 16) = T |
| reference present, T = 32 (or wider) | the same reference, which is only 16 tokens wide | **min(T, 16) = 16 of 32** |
| no reference graph (a reader who only has this package) | the new graph against the **PyTorch module in memory**, which shares the weights it was exported from | all T rows |
| `export_e2r_graphs.py`, **T > 1** | compares row 0 against the `g1.onnx` the same script wrote a moment earlier, within tolerance | 1 row |
| `export_e2r_graphs.py`, `--stable-qwen25` | additionally checks against a PyTorch reference for the stable-norm variant | 1 row |
| `export_e2r_graphs.py`, **T = 1** | compares against `artifacts/qwen25_05b/B_head.onnx`, a graph from the layout study that is **not in this package**, so on a reader's machine this check is skipped and only `ran`/`finite` are reported | 0 rows |

Two consequences follow, and the documentation should say them rather than let the word "PASS" imply more:

- **A wide graph is never fully cross-checked.** If the reference is present, the script compares `min(T, 16)` rows and stops: rows 16 onwards are **not checked at all** in that run. If the reference is absent, every row goes to the PyTorch comparison instead. The two never happen together, so no single run covers the whole width independently.
- **The fallback is not independent.** Comparing an ONNX graph against the module it was just exported from catches export bugs (wrong axes, dropped initialisers, dtype slips) but cannot catch a shared numerical misunderstanding.

The scripts print `e2o_g16_rows_compared` alongside the verdict so the coverage is visible in the log rather than inferred from the documentation. When you report a reproduction, quote that number.

## Verification status

| Part | Status |
|---|---|
| graph export (host) | **executed** 2026-11-02 on the development host: T=1 and T=32, `EXPORT VERDICT: PASS`, `g1.onnx` 11,555,568 B and `g32.onnx` 11,561,054 B — both matching `MANIFEST.md` |
| logits export (host) | **executed** 2026-11-02: four graphs, `EXPORT VERDICT: PASS` |
| benchmark inputs (host) | **executed** 2026-11-02: `make_e2r_inputs.py` reported `VERDICT: PASS`, `hf_diff=0` |
| measurement (board), NumPy 2.x | **executed** 2026-11-02: one case, one repetition, `base_exact=True spec_exact=True`, `exec_ms=105.86` against 105.65 archived (0.2%), via the `bytes_to_ptr` fallback |
| measurement (board), NumPy 1.26 | **executed** 2026-11-02 after an offline install: `base_exact=True spec_exact=True`, `base=3238.76ms` against 3294.07 archived (1.7%), via the original `numpy_to_ptr` |
| compilation (board) | **executed** 2026-11-02: `board_atc9_e3.sh 1` recompiled the T=1 graph in 11 m 23 s from the ONNX produced by this package's export script (sha256 `b9dad0e…`, matching the manifest) |

### What the rebuild showed

| | original | rebuild |
|---|---|---|
| `.om` size | 1,264,167,472 B | 1,264,166,074 B (1,398 B smaller) |
| `.om` sha256 | `e59d3028…` | `7724b8cc…` |
| `atc` exit status | SIGBUS/SIGSEGV after `ATC run success` | `ATC_EXIT=0` |
| peak `atc.bin` RSS | 2.24–2.49 GB | 2,084 MB |
| lowest `MemAvailable` | 7–19 MB | 26 MB |
| measurement, same case | 3,294.07 ms | 3,293.10 ms (−0.03%) |

Two conclusions, both bounded by the fact that each condition ran once.

**Compilation is not byte-reproducible** — the artefact differs in size and hash. **On this one case it was functionally reproducible**: same token stream, timing 0.03% below the archive. That is a smoke test, not a bound. It does not show that another width or another case reproduces, and a single pair of measurements says nothing about the variance of the difference. Publish sizes and hashes as a sanity check on the toolchain, not as an identity test on the bytes.

The board half of step 3 was run with `E3_REPS=1`; five repetitions is what the paper
reports and what you should use for a real comparison.

**Line endings.** Every script here is stored with LF endings. A Windows checkout can
silently turn them into CRLF, and the board then fails with `$'\r': command not found`
before a single line runs. If that happens, `sed -i 's/\r$//' tools/*.sh` or set
`core.autocrlf=input` before checking out.

## Failure modes and what they mean

| Symptom | Cause | Fix |
|---|---|---|
| `EC0010 No module named numpy` during ATC | the TBE picked up an interpreter without NumPy, or with NumPy 2.x | put a Python 3.9/3.10 with NumPy < 2 first on PATH |
| `atc: command not found` | `set_env.sh` not sourced, or sourced from a non-login shell | run under `bash -l` |
| build dies at ~4 minutes with no error | a memory watchdog killed it | disable the watchdog; the build needs every MB and swap |
| `ATC run success` then SIGSEGV | the board's ATC faults on exit | ignore the exit code; load the `.om` and run step 3 |
| `load_from_file` returns non-zero | the `.om` was truncated by a failed build | rebuild that width |
| measurement 25% slower than expected | a stray process holds device memory | check `npu-smi info`, reboot if needed |

## Checking curated numbers and limits of reproduction

Run `python tools/verify_curated.py` to recompute the benchmark medians and correlations, prefill fit, and energy calculation from `data/curated/`. These are checks of derived numerical records, not new board measurements or independent token-level verification. The following material remains unavailable in this package:

- **The four divergences of Table S4.** `data/curated/` provides recorded exactness and margin summaries. Two inputs are here (`copy_3`, `free_3`); the full token comparisons, raw logits and remaining inputs are not. Independently replaying the margins needs regenerated benchmark inputs and a **logits-returning** graph at the same width.
- **The benchmark inputs behind Table 5 and Figure 2.** The curated per-case values regenerate the numerical curve. **Twelve of the 55 input cases are included in `cases/calib/`, together with their tokenised prefixes and reference streams.** The remaining 43 must be regenerated: **three from the in-house prompt definitions supplied in the package** (`cases/dev/e1_prompts.json`), and **40 from the separately obtained JFLEG and HumanEval datasets**. It is the prefixes and reference streams for those 43 that are absent. Two of the twelve shipped cases, `copy_3` and `free_3`, are also two of the four divergent prompts of Table S4; that "two of four" counts the divergence set, not the 55. The README describes how the inputs can be regenerated.

The calibration comparison at the end of step 4 reports `exact = yes` for all six widths. Read that as a statement about the six JSON files, not about Table S4.

Two of the four divergent prompts — `copy_3` and `free_3` — have their **input cases** in `cases/calib/`. The six calibration JSON files do not contain their measurements: they record `copy_1` (all widths) and `copy_2` (T=1) only. Derived benchmark summaries for the divergent cases are separate in `data/curated/`. So `exact = yes` cannot mean "the four divergences were tested and passed" — those cases were never run in this calibration session. `DATA_DICTIONARY.md` has the per-prompt table.
