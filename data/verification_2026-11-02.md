# Verification run, 2026-11-02

Recorded because a reproduction claim is only as good as the run behind it. Everything below was executed on the hardware named, not inferred.

**Raw records:**  holds the compile log and the harness JSON for the rebuilt-graph run. The other two measurement runs overwrote the same filename and survive only as summary lines transcribed from the session; that directory says which is which, so the two kinds of evidence are not confused.

## Environment

| | |
|---|---|
| Board | Atlas 200I DK A2, `davinci-mini`, aarch64, 3 cores, 3513 MB |
| CANN | 9.1.1 (`/data/Ascend/cann-9.1.1`) |
| ATC | from CANN 9.1.1, `must_keep_origin_dtype` |
| Host | Windows, Python 3.13, torch/onnx/onnxruntime/transformers from a local env |
| Board Python | 3.10.6, NumPy 2.2.6 (system) and 1.26.4 (user install, tested separately) |

## 1. Graph export (host)

```
export_e3_gamma.py 1   → EXPORT VERDICT: PASS   g1.onnx   11,555,568 B
export_e3_gamma.py 32  → EXPORT VERDICT: PASS   g32.onnx  11,561,054 B   argmax_shape [32]
```

Both sizes match the sizes recorded when the graphs were first exported.

The two comparisons in `export_e3_gamma.py` are mutually exclusive, and on this host
**the reference graph was absent**, so the T=32 run took the PyTorch branch:

```
T=32: same_as_e2o_g16_rows = null, e2o_g16_rows_compared = 0, same_as_torch = true
```

That means all 32 rows were compared, but against the in-memory PyTorch module the
graph was exported from — a check on the export, not on an independent implementation.
Had the reference been present, the script would have compared only its 16 rows and
left rows 16–31 unverified. No single run covers both.

## 2. Logits export (host)

```
export_e2r_graphs.py → EXPORT VERDICT: PASS
  qwen25 g1  11,460,167 B    qwen25 g16 11,465,141 B
  qwen3  g1     678,719 B    qwen3  g2     683,242 B
```

The `B_head.onnx` cross-check was skipped: that file belongs to the layout study and is not in this package.

## 3. Benchmark inputs (host)

```
make_e2r_inputs.py → VERDICT: PASS (inputs generated; fp16 reference; HF diff recorded)
  qwen3 free_2: L=22 N=64 hf_diff=0
  qwen3 free_3: L=25 N=47 hf_diff=0
```

`hf_diff=0` means the fp16 reference stream and the HuggingFace stream agree token for token on those cases.

## 4. Compilation (board)

```
=== E3 g1 (must_keep_origin_dtype) ===
ONNX_SHA  : b9dad0effd1f116322e0dfdac35290076924ba196d2f0ceae70efbf14fb60596
ATC_EXIT=0
MEM_PEAK  : PEAK atc.bin RSS = 2084 MB | MIN MemAvailable = 26 MB | TRIP=0
ATC run success, welcome to the next use.
duration  : 11 m 23 s
```

| | original | this rebuild |
|---|---|---|
| size | 1,264,167,472 B | 1,264,166,074 B |
| sha256 | `e59d3028034ce0d4…` | `7724b8cc2f7965f3…` |

Not byte-identical. On the single case measured below the rebuild was functionally equivalent; that is one case and one run, not a general claim. The original was restored on the board afterwards.

## 5. Measurement (board)

Case `copy_1` (L=150, N=32), one repetition, full warm-up discarded.

| Condition | base wall | archived |
|---|---:|---:|
| NumPy 2.2.6 + `bytes_to_ptr` fallback, original `.om` | 3377.79 ms | 3294.07 ms (+2.5%) |
| NumPy 1.26.4 + `numpy_to_ptr`, original `.om` | 3238.76 ms | 3294.07 ms (−1.7%) |
| NumPy 1.26.4, **rebuilt** `.om` | 3293.10 ms | 3294.07 ms (−0.03%) |

`base_exact=True` and `spec_exact=True` in every configuration. `exec_ms` on the first run: 105.86 against 105.65 archived.

## What this run supports, and what it does not

Supported: on this case, in this one run, the export → compile → load → measure chain works end to end from the files in this package; the rebuilt artefact emitted the same token stream and a timing 0.03% below the archived value. One repetition per condition is a smoke test, not a bound: it does not establish functional equivalence in general, and the 0.03% carries no statistical weight. Both NumPy ABIs load and run.

Not supported: any statement about the 55-prompt benchmark curve, the four divergences of Table S4, the request-level estimates, or the Qwen3 boundary result. Those need material that is not here, and this run does not change that.
