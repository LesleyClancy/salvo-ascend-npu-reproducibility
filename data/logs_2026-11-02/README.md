# Raw records, 2026-11-02

What is here, and what is not.

| File | What it is | Status |
|---|---|---|
| `compile_g1.log` | The console output of `board_atc9_e3.sh 1`, copied verbatim from the session | **raw** |
| `measure_rebuilt_g1.json` | The harness's own JSON for the run against the rebuilt graph | **raw**, pulled from the board's output directory |

## What is missing, and why

The two measurements against the *original* graph — NumPy 2.2.6 with the `bytes_to_ptr`
fallback, and NumPy 1.26.4 with `numpy_to_ptr` — wrote to the same filename and were
overwritten by the run that followed. Only their summary lines survive:

```
Py2.2.6, original .om : base rep0: total=3377.79ms exact=True
                        spec rep0: total=3378.43ms exact=True
Py1.26.4, original .om : base rep0: total=3238.76ms exact=True
                        spec rep0: total=3241.87ms exact=True
rebuilt .om            : base rep0: total=3293.10ms exact=True   ← the JSON here
```

Those two lines are transcription from the session, not archived files. Treat them as
weaker evidence than `measure_rebuilt_g1.json`, which is the harness's own output.

The first run's `exec_ms` (105.86 against 105.65 archived) also comes from the session
summary and has no JSON behind it in this archive.

## The run_tag is not a date

`measure_rebuilt_g1.json` carries `"run_tag": "2026-10-24"`. That is the hard-coded default in
`board_run_e2o.py` when `E3_RUN_TAG` is unset — the environment variable was not set for this
run, so the tag says nothing about when it happened.

What can be checked inside this package: `compile_g1.log` carries `2026-11-02T22:04:03+08:00`. What cannot
be checked: the JSON's own run time. It contains no timestamp — the `run_tag` is a default — and the board
mtime of `Nov 2 22:04` was seen on the board but is not preserved by copying. That the two files came from
the same session is therefore **our account of it, not a verifiable link**. Set `E3_RUN_TAG` explicitly in future
runs, and keep a timestamped command line alongside the output; a default that happens to look like
a date will be read as one.

## A note on the path field

`measure_rebuilt_g1.json` is unedited: its `dir` field still reads
`/home/HwHiAiUser/kv_probe/e3`. The six files in `data/calibration/` had that field
rewritten to `~/kv_probe/e3`; this one was left alone because it is presented as a raw
record, and editing it would defeat that purpose.

## What this archive does not cover

The export runs on the host (graph export, logits export, input generation) produced console
output that was **not** saved to a file; only their verdict lines are quoted in
`data/verification_2026-11-02.md`. So the only thing here with a genuine raw record is the
compile and the one measurement taken against the rebuilt graph. Everything else in that
verification note is transcription.

## Scope

One case (`copy_1`, L=150, N=32), one repetition per condition, on one board. This is a
smoke test of the chain, not a benchmark re-measurement.
