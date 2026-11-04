#!/bin/bash
# Environment self-check for the reproduction package.
# Run on the board, from the package root: bash docs/check_env.sh
set +u

fail=0
say() { printf '%-34s %s\n' "$1" "$2"; }

echo "== board =="
say "model"            "$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
say "cpu cores"        "$(nproc)"
say "memory total"     "$(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
say "swap total"       "$(awk '/SwapTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
say "memory available" "$(awk '/MemAvailable/{printf "%d MB", $2/1024}' /proc/meminfo)"

echo
echo "== toolchain =="
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    say "ASCEND_HOME_PATH" "UNSET - source set_env.sh from a login shell (bash -l)"
    fail=1
else
    say "ASCEND_HOME_PATH" "$ASCEND_HOME_PATH"
fi
if command -v atc >/dev/null 2>&1; then
    say "atc" "$(command -v atc)"
else
    say "atc" "NOT ON PATH"
    fail=1
fi

echo
echo "== python for the TBE operator compiler =="
python3 - <<'PY' 2>/dev/null || echo "  python3 missing"
import numpy, sys
print(f"  {'interpreter':22s} {sys.executable}")
print(f"  {'version':22s} {sys.version.split()[0]}")
print(f"  {'numpy':22s} {numpy.__version__}")
if int(numpy.__version__.split('.')[0]) >= 2:
    print("  WARNING: NumPy 2.x is rejected by the 9.1.1 TBE; use python 3.9/3.10 with numpy<2")
PY

echo
echo "== inputs =="
if [ -d cases/calib ]; then
    say "calibration cases" "$(ls -1 cases/calib | wc -l) directories"
else
    say "calibration cases" "MISSING (run from the package root)"
    fail=1
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "RESULT: environment looks usable."
else
    echo "RESULT: fix the items marked above before measuring."
fi
exit "$fail"
