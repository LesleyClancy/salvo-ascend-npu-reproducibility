# Environment

`configs/environment.json` is the machine-readable record. This page explains the parts that are easy to get wrong, because each of them cost real time.

## The board

Atlas 200I DK A2, Ascend 310B4, 3.4 GiB of unified memory shared between host and device, three Cortex-A55 cores. Two facts about that memory matter more than the spec sheet:

- **Three different numbers, routinely confused.** All three are on the same 3513 MB of physical memory, so they are not additive and they do not track each other:

  | What you read | Where | Freshly booted | After a build and a run |
  |---|---|---:|---:|
  | `used` | `free -m` | ~460 MB | ~460 MB (build artefacts are page cache) |
  | `MemAvailable` | `/proc/meminfo` | ~2900 MB | ~2900 MB |
  | Memory-Usage | `npu-smi info` | **~1070-1110 MB** | 2028-2699 MB observed |

  The paper's session gate — "device occupancy below 800 MB" — was set against the idle baseline of the image the measurements were taken on, which a working record from that period gives as **570 MB**. On the CANN 9.1.1 image this release targets, the idle baseline is **1070-1110 MB**, so the same threshold cannot be met here. That is a difference between images, not a different field: an unchanged 800 MB gate would refuse to start on this board. If you re-run on 9.1.1, set the threshold relative to the baseline you measure, and record it.

Only two wrappers implement a gate, and **they check different things** — do not read one as a stand-in for the other.

| Wrapper | Stray processes | Device occupancy |
|---|---|---|
| `tools/board_run_e5.sh` | `llama*` and `python3*` must be zero | parses `npu-smi`, fails above 800 MB |
| `tools/board_run_cn3.sh` | `llama*` must be zero | calls `npu-smi`, **reads a value it never uses** |

The CN3 gate is a process check wearing an occupancy check's clothing. The E5 one does test occupancy, and has two soft spots worth knowing before you reuse it:

```bash
du=$(npu-smi info | grep -oE '[0-9]+ +/ *3514' | head -1 | grep -oE '^[0-9]+')
if [ -n "${du:-}" ] && [ "$du" -gt 800 ]; then
```

The match pattern hard-codes a 3514 MB total. On the 9.1.1 image the total reads 3513, the pattern matches nothing, `du` stays empty, and the second test is skipped — **the gate passes silently**. It also passes when `npu-smi` fails for any other reason. Print `GATE ... dev_used_mb=NA` and check for it, or parse the column by position rather than by total size.
- A compiled graph is **1.26 GB**. Two of them do not fit; loading a pair has driven this board into a kernel panic. Every measurement in the paper that uses a single graph does so because of that, not for convenience.

## Two traps that are not obvious

**1. `set_env.sh` fails silently under a non-login shell.** Sourcing `/usr/local/Ascend/ascend-toolkit/set_env.sh` from a plain `sh -c` or a non-interactive SSH command leaves `ASCEND_HOME_PATH` unset and `atc` off the PATH, with no error. Everything must run under `bash -l`, or the script must source the file itself and check the variable afterwards.

**2. The ATC operator compiler needs a specific Python.** It imports `numpy` from whatever interpreter is first on PATH. On this board the system Python is 3.10 with NumPy 2.x, which the TBE rejects; a Miniconda 3.9.2 with NumPy 1.22.4 works. The symptom is `EC0010 No module named numpy` even though `numpy` imports fine in a normal shell.

**3. NumPy 2 breaks the runtime, not just the compiler.** The board image now ships NumPy 2.2.6, while pyACL's `numpy_to_ptr` was built against the 1.x C ABI: `import acl` succeeds, then the first pointer call raises `numpy.core.multiarray failed to import`. `board_run_e2o.py` works around it with `bytes_to_ptr`; the board has no network, so downgrading means carrying an aarch64 wheel across.

**4. Only about 2.4 GB is actually available for a graph.** Of 3513 MB total, roughly 1.1 GB is held by the driver before you start (see above), which leaves room for one 1.26 GB graph plus the KV cache and the runtime — not two.

If a measurement process dies badly, the next `load_from_file` can fail with `ret=500002` even though the memory looks free. A reboot clears it. Check `npu-smi info` before a session and keep one session per boot when running several configurations.

## Compilation is the slow, fragile step

- One width takes **11–12 minutes** on the board.
- Peak resident set is **2.24–2.49 GB**; available memory drops to **7–19 MB** and the build survives only because of the 8 GB swap file.
- A memory watchdog set above ~20 MB will kill the build mid-way. Ours had to be disabled.
- **The exit status is unreliable in both directions.** In earlier sessions `atc` printed `ATC run success` and then died with SIGBUS or SIGSEGV; in the rebuild on 2026-11-02 it exited cleanly with `ATC_EXIT=0`. So neither a fault nor a zero tells you the artefact is good. The acceptance test is: load it, run the cases, and check the emitted stream against `gen_ids.bin`.

## Toolchain versions

Two of them appear in the paper and they are not interchangeable in the record:

| | CANN | Root | Used for |
|---|---|---|---|
| original | 8.2.RC1 | `/usr/local/Ascend/ascend-toolkit` | all numbers in Tables 4–6 |
| recalibration | 9.1.1 | `/data/Ascend/cann-9.1.1` | the two-anchor platform check, Table S1 |

The `.om` files differ between them (different sizes, different hashes), which is why the package ships build instructions rather than artefacts.
