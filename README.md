# Reproduction package: fixed-shape speculative decoding on an Ascend 310B4

This package accompanies the manuscript *Salvo: Speculative Decoding Inside Fixed-Shape Graphs on a Memory-Constrained NPU*. It contains code, frozen calibration inputs and selected measurement records supporting the checks described below.

This package is a subset of the working repository. Its coverage and limitations are stated below. The legal status of the third-party-derived numerical records is not determined; see `DATA_REUSE.md` and `THIRD_PARTY_NOTICES.md`.

## What this package reproduces

| Item in the paper | Reproduced from | Where |
|---|---|---|
| Table 4, graph execution time versus width | recompile six widths, run `tools/board_run_e2o.py` | `tools/`, `cases/calib/` |
| Table S1, two-toolchain platform check | same, with the 9.1.1 toolchain | `configs/environment.json`, `data/calibration/` |
| Table S3, claims mapped to artefacts | path index into this package | `MANIFEST.md` |
| Round-cost identity `1.1495 ≈ 1.1494` | input and selected T=16 times only; the original T=32 record is not included | `cases/calib/e2r_s_inputs_qwen25_05b_copy_2`, `data/curated/` |

## What it does not reproduce

- **The 55-prompt benchmark (Table 5, Figure 2).** The numerical summary and repetition times in `data/curated/` allow the reported medians and correlations to be recalculated with `python tools/verify_curated.py`. Twelve of those 55 inputs are here, as `cases/calib/`, with their tokenised prefixes and reference streams; the other 43 are not. Regenerating them takes the in-house prompt definitions for the three `code` cases and the separately obtained JFLEG and HumanEval sets for the remaining forty. See `THIRD_PARTY_NOTICES.md` and `docs/REPRODUCE.md` §2b.
- **The Qwen3-0.6B boundary result.** Its two 1.5 GB graphs do not fit this board alongside each other; the export path is included (`src/`), the measurements are not.
- **Energy (Section 6.4).** Curated power samples, timing markers and repeated times allow the reported energy calculation to be checked. The original full harness log is not shipped; a fresh measurement needs `npu-smi` on the board.
- **Any compiled `.om` file.** See Licensing below.
- **The four near-tie divergences of Table S4.** Curated per-case equality and margin summaries are included, but the full token comparisons and logits are not. Two input cases are here (`copy_3`, `free_3`); rerunning all cases and reproducing the margins requires the benchmark inputs and a logits-returning graph.

## Quick start

Prerequisites: an Atlas 200I DK A2 board (Ascend 310B4) with CANN and `atc`; a host with Python 3.9-3.11 and the packages below; and the Qwen2.5-0.5B-Instruct checkpoint.

```bash
pip install numpy torch transformers onnx onnxruntime
```

`torch` and `transformers` are needed only to export the graphs and to regenerate
benchmark inputs; the board side needs only the CANN runtime and `numpy`.

```bash
# 1. environment check (on the board)
bash docs/check_env.sh

# 2. export the ONNX graphs and the external weight sidecar (on the development host,
#    from the package root; needs the checkpoint path in configs/model_qwen25_05b.json)
python tools/export_e3_gamma.py 1,2,4,8,16,32 --model qwen25
#    writes artifacts/e3/qwen25_05b/{g*.onnx, e3.weights.bin}

# 3. compile one width (on the board). Copy the ONNX and the sidecar across, cd into
#    that directory, source the toolchain, then compile. set_env.sh must be sourced in
#    the same shell as the build: without it `atc` cannot find its libraries.
cd ~/kv_probe/e3/qwen25_05b          # the directory holding g1.onnx and e3.weights.bin
source /data/Ascend/cann-9.1.1/set_env.sh
bash ~/kv_probe/tools/board_atc9_e3.sh 1   # adjust the path if installed elsewhere

# 4. measure (also on the board, and again with set_env.sh sourced — `import acl` fails
#    without it). Run from the package root so the case path resolves.
cd ~/kv_probe                         # adjust if the package is elsewhere
E3_REPS=5 E3_SINGLE=1 E3_WARMUP=1 E2O_DIR=~/kv_probe/e3 \
E2R_INPUTS=$PWD/cases/calib python3 tools/board_run_e2o.py copy 1 copy_1

# 5. compare against the recorded medians (either machine)
python tools/compare_calibration.py data/calibration
# 6. check curated benchmark, prefill and energy calculations
python tools/verify_curated.py
```

Full detail, including the four environment traps that cost us days, is in `docs/ENVIRONMENT.md` and `docs/REPRODUCE.md`. If a step fails with `No module named 'acl'` or `libascendcl.so`, the cause is almost always a missing `set_env.sh`.

## Layout

```
tools/                 thirteen files: two export drivers, ATC driver, measurement harness,
                       calibration wrapper, comparison check, four input generators,
                       a bandwidth probe, an environment probe, one C probe, a curated-data check
src/                   model adapters, graph contracts and the decoder step
                       (the export driver itself is tools/export_e3_gamma.py)
configs/               machine-readable environment record (sanitised)
cases/calib/           twelve frozen input cases: ids, positions, KV prefixes, reference streams
data/calibration/      measured JSON, one per graph width, plus the harness logs
data/curated/          derived numerical benchmark, margin, prefill and energy records
docs/                  environment, reproduction steps, data dictionary, provenance
```

## Licensing and redistribution

- Code in this package is released under the licence in `LICENSE`.
- **No `.om` file is included.** A compiled graph embeds kernels produced by the vendor's ATC, and those binaries are governed by the CANN licence. We did not redistribute them because we could not confirm that the licence permits it; that is a statement about what we know, not a claim about what the licence says. `docs/REPRODUCE.md` gives the command line, the toolchain version and the resulting sizes and hashes, so a reader with a licensed CANN installation can rebuild them in about twelve minutes per width. A rebuild was performed on 2026-11-02 and is recorded in `data/verification_2026-11-02.md`.
- **No model weights are included.** Fetch the checkpoint from Hugging Face (`Qwen/Qwen2.5-0.5B-Instruct`) and export the sidecar with the script in `src/`; the expected sizes, and the full sha256 of `g1.onnx`, are in `MANIFEST.md`.
- The twelve input cases are derived data: token ids and fp16 KV prefixes computed from the public checkpoint. They are redistributed here under the same terms as the code.

**Third-party material.** The evaluation uses subsets of OpenAI HumanEval and JFLEG as workloads. The original datasets are not included in this release. HumanEval is available under the MIT License; the JFLEG dataset card identifies CC BY-NC-SA 4.0. Please obtain and use each dataset under its applicable terms. The Qwen2.5-0.5B-Instruct and Qwen3-0.6B checkpoints are available separately under Apache-2.0. The project's MIT license applies only to material for which the authors have the right to grant it. `THIRD_PARTY_NOTICES.md` records the upstream link, licence, pinned version and role of each of the four, and what changes if a future release ships model or dataset content.

## Data provenance and what was removed

`docs/PROVENANCE.md` lists every field that was dropped from the working records before publication, and why. In short: identified personal names, host-specific paths and internal network addresses, and project-management notes. Some unchanged device logs and provenance text retain the board-default account path, as explained there. The calibration records retain their numerical measurements and flags. The curated data are derived extracts, and several full raw logs and inputs are not shipped; see the coverage notes above.
