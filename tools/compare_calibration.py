#!/usr/bin/env python3
"""Compare a calibration run against the medians reported in Table 4.

    python tools/compare_calibration.py [directory-of-json]

Reads the JSON files written by board_run_e2o.py. For each graph width it prints
the median stage time, the value in the paper, and the delta.

Two things this script deliberately does not do:

  * it never treats a missing field as a pass. Exactness is read from the
    `base_exact` / `spec_exact` arrays, which record whether each repetition
    reproduced `gen_ids.bin` -- the deployed fp16 T=1 graph's greedy stream for
    that case. If those arrays are absent the width is reported as UNKNOWN and
    the exit status is non-zero, because a run we cannot check is not a run that
    passed;
  * it does not compare the two arms against each other and call that exactness.
    `spec_eq_base` is printed separately: two arms can agree while both differ
    from the reference.

Exit status: 0 only if every width is within TOLERANCE of the paper and every
recorded repetition was exact.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

PAPER_EXEC = {1: 105.5, 2: 97.6, 4: 96.2, 8: 97.6, 16: 98.9, 32: 113.6}   # Table 4
TOLERANCE = 0.05


def stage_of(case: dict) -> dict:
    return case.get("spec_stage_median_ms") or case.get("base_stage_median_ms") or {}


def as_list(v):
    """Fields are list-or-bool depending on the harness version; normalise."""
    if v is None:
        return None
    if isinstance(v, bool):
        return [v]
    return list(v)


def check_exact(case: dict) -> str:
    """'yes' / 'NO' / 'UNKNOWN' -- never 'yes' by default."""
    base, spec = as_list(case.get("base_exact")), as_list(case.get("spec_exact"))
    # Both arms must be recorded. Checking only the one that exists would call a
    # half-recorded case exact.
    if base is None or spec is None:
        return "UNKNOWN"
    # An empty list is a recorded-but-empty arm, not a passing one.
    if not base or not spec:
        return "UNKNOWN"
    vals = base + spec
    return "yes" if all(vals) else "NO"


def main() -> int:
    directory = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/calibration")
    files = sorted(directory.glob("*.json"),
                   key=lambda f: int(json.loads(f.read_text(encoding="utf-8"))["gamma"]))
    if not files:
        print(f"no calibration JSON under {directory}", file=sys.stderr)
        return 2

    print("reads the saved records in", directory)
    print("NOT a fresh board run; PASS means the archive is self-consistent, not that the device was re-measured")
    print("reference sequence: gen_ids.bin (deployed fp16 T=1 graph, per case)")
    print(f"{'T':>3} {'exec(ms)':>9} {'paper':>7} {'delta':>8} {'ref':>8} {'arms':>6}")
    failures: list[str] = []

    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        gamma = int(d["gamma"])
        cases = d.get("cases") or {}
        if not cases:
            print(f"{gamma:>3}  no cases in {f.name}")
            failures.append(f"T={gamma}: no cases")
            continue

        execs, status, arms = [], [], []
        for cid, case in cases.items():
            st_ms = stage_of(case).get("exec_ms")
            if st_ms is None:
                failures.append(f"T={gamma} {cid}: exec_ms missing")
                continue
            execs.append(st_ms)
            status.append(check_exact(case))
            reps = d.get("reps")
            for arm in ("base_exact", "spec_exact"):
                vals = as_list(case.get(arm))
                if vals is not None and reps and len(vals) != int(reps):
                    failures.append(f"T={gamma} {cid}: {arm} has {len(vals)} entries, reps={reps}")
            eq = as_list(case.get("spec_eq_base"))
            arms.append("UNKNOWN" if eq is None else ("yes" if all(eq) else "NO"))

        if not execs:
            print(f"{gamma:>3}  no exec_ms")
            continue

        med = st.median(execs)
        ref = PAPER_EXEC.get(gamma)
        delta = f"{(med - ref) / ref * 100:+.1f}%" if ref else "—"
        if ref and abs(med - ref) / ref > TOLERANCE:
            failures.append(f"T={gamma}: {delta} outside ±{TOLERANCE*100:.0f}%")

        ref_flag = "UNKNOWN" if "UNKNOWN" in status else ("yes" if set(status) == {"yes"} else "NO")
        arm_flag = "UNKNOWN" if "UNKNOWN" in arms else ("yes" if set(arms) == {"yes"} else "NO")
        if ref_flag == "NO":
            failures.append(f"T={gamma}: stream differs from the reference")
        if ref_flag == "UNKNOWN":
            failures.append(f"T={gamma}: exactness fields missing, cannot verify")
        if arm_flag == "NO":
            failures.append(f"T={gamma}: the two arms disagree with each other")
        if arm_flag == "UNKNOWN":
            failures.append(f"T={gamma}: spec_eq_base missing, arms cannot be compared")

        print(f"{gamma:>3} {med:>9.2f} {ref if ref else 0:>7.1f} {delta:>8} {ref_flag:>8} {arm_flag:>6}")

    present = {int(json.loads(f.read_text(encoding="utf-8"))["gamma"]) for f in files}
    missing = sorted(set(PAPER_EXEC) - present)
    if missing:
        failures.append(f"widths missing from the record: {missing}")

    warm = all(json.loads(f.read_text(encoding='utf-8')).get("warmup_full") for f in files)
    single = all(json.loads(f.read_text(encoding='utf-8')).get("single_om") for f in files)
    print(f"\nwarm-up discarded: {warm}   single graph resident: {single}")

    if failures:
        print("\nRESULT: FAIL")
        for x in failures:
            print("  -", x)
        return 1
    print("\nRESULT: PASS — every width within tolerance, both arms exact and in agreement.")
    print("This certifies the archived JSON, not a reproduction: rerun tools/board_run_e2o.py to measure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
