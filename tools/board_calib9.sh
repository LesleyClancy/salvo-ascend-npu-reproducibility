#!/bin/bash
# Platform calibration on CANN 9.1.1: re-measure the two anchor points
# (T=1 step and gamma=32 round) with the frozen E3 graphs.
# Usage: bash board_calib9.sh <gamma> <case[,case...]> [reps]
set +u

export PATH="/usr/local/miniconda3/bin:${PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /data/Ascend/cann-9.1.1/set_env.sh

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export E2O_DIR="${E2O_DIR:-${PACKAGE_ROOT}/e3}"
export E2R_INPUTS="${E2R_INPUTS:-${PACKAGE_ROOT}/cases/calib}"
export E3_REPS="${3:-${E3_REPS:-5}}"
export E3_WARMUP="${E3_WARMUP:-1}"
export E3_RUN_TAG="${E3_RUN_TAG:-calib911}"
export E3_SINGLE="${E3_SINGLE:-1}"

GAMMA="${1:?usage: board_calib9.sh <gamma> <case[,case...]> [reps]}"
CASES="${2:?usage: board_calib9.sh <gamma> <case[,case...]> [reps]}"
OUT="calib_g${GAMMA}_$(date +%H%M%S).log"

cd "$PACKAGE_ROOT" || exit 90
echo "=== CALIB gamma=${GAMMA} cases=${CASES} reps=${E3_REPS} single=${E3_SINGLE} warmup=${E3_WARMUP} $(date -Is) ==="
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "MemAvailable=$(awk '/MemAvailable/{printf "%d MB", $2/1024}' /proc/meminfo)"

python3 tools/board_run_e2o.py copy "$GAMMA" "$CASES" 2>&1 | tee "$OUT"
RUN_STATUS=${PIPESTATUS[0]}
echo "EXIT=${RUN_STATUS}"
echo "=== DONE $(date -Is) ==="
exit "$RUN_STATUS"
