#!/bin/bash
# Compile one or more graph widths with the CANN 9.1.1 ATC.
#
# Scope: this script works only inside $KVPROBE_E3_DIR, which defaults to the
# directory you start it from. It cd's there, deletes that directory's
# kernel_meta scratch folder between widths, and touches nothing else.
#
# It also must not kill processes it did not start. The two pkill calls that
# clean up a wedged build are therefore gated behind KVPROBE_KILL_STRAY=1:
# with the default setting a stray atc.bin left by an earlier run is reported
# and ignored, so that a build you have running elsewhere is not touched.
# E3 / Qwen2.5 C=256 stable g1..g32, compiled with CANN 9.1.1 (on /data).
# Serial, one graph at a time, with a host-memory watchdog.
# Usage: bash board_atc9_e3.sh [T ...]     (default: 1 2 4 8 16 32)
set +u

export PATH="/usr/local/miniconda3/bin:${PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /data/Ascend/cann-9.1.1/set_env.sh
export PYTHONPATH="/usr/local/miniconda3/lib/python3.9/site-packages:${PYTHONPATH:-}"
export TE_PARALLEL_COMPILER=1
export MALLOC_ARENA_MAX=1
export OMP_NUM_THREADS=1

DIR="${KVPROBE_E3_DIR:-$(pwd)}"   # the directory holding g*.onnx and e3.weights.bin
PREC="must_keep_origin_dtype"
GUARD_MB=0
export MALLOC_TRIM_THRESHOLD_=65536
export MALLOC_MMAP_THRESHOLD_=131072

cd "$DIR" || { echo "FATAL: cd $DIR"; exit 90; }

LIST="$*"
[ -z "$LIST" ] && LIST="1 2 4 8 16 32"

echo "=== CANN 9.1.1 E3 ATC start: $(date -Is) ==="
echo "HOME=$ASCEND_HOME_PATH atc=$(which atc)"
echo "THERMAL_BEFORE: $(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)"

for T in $LIST; do
    OUT="qwen25_05b_g${T}"
    [ -f "g${T}.onnx" ] || { echo "FATAL: g${T}.onnx missing"; continue; }
    [ -f "e3.weights.bin" ] || { echo "FATAL: weights sidecar missing"; break; }
    rm -rf kernel_meta

    echo
    echo "=== E3 g${T} (${PREC}) ==="
    echo "START     : $(date -Is)"
    echo "MEM_BEFORE: $(awk '/MemAvailable/{printf "%d MB available", $2/1024}' /proc/meminfo)"
    echo "ONNX_SHA  : $(sha256sum g${T}.onnx | cut -d' ' -f1)"

    MEM_LOG="mem_${OUT}_cann911.log"
    : > "$MEM_LOG"
    timeout -k 60 5400 nice -n 19 atc --model="g${T}.onnx" --framework=5 --output="${OUT}" \
        --soc_version=Ascend310B4 --input_format=ND --precision_mode="${PREC}" \
        > "atc_${OUT}_cann911.log" 2>&1 &
    local_pid=$!
    (
        peak=0; mina=999999; trip=0
        while kill -0 "$local_pid" 2>/dev/null; do
            avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
            rss=$(ps -eo rss,cmd | awk '/atc\.bin/ && !/awk/ {s+=$1} END {printf "%d", s/1024}')
            [ -z "$rss" ] && rss=0
            [ "$rss" -gt "$peak" ] && peak=$rss
            [ "$avail" -lt "$mina" ] && mina=$avail
            echo "$(date +%H:%M:%S) avail=${avail}MB atc_rss=${rss}MB" >> "$MEM_LOG"
            if [ "$avail" -lt "$GUARD_MB" ]; then
                echo "$(date +%H:%M:%S) GUARD_TRIP avail=${avail}MB" >> "$MEM_LOG"
                if [ "${KVPROBE_KILL_STRAY:-0}" = "1" ]; then
                    pkill -9 -x atc.bin 2>/dev/null
                fi
                kill -9 "$local_pid" 2>/dev/null
                trip=1
                break
            fi
            sleep 5
        done
        echo "PEAK atc.bin RSS = ${peak} MB | MIN MemAvailable = ${mina} MB | TRIP=${trip}" >> "$MEM_LOG"
    ) &
    wd=$!
    wait "$local_pid"; atc_exit=$?
    for _ in $(seq 1 20); do grep -q '^PEAK' "$MEM_LOG" 2>/dev/null && break; sleep 0.5; done
    kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null

    echo "ATC_EXIT=${atc_exit}"
    echo "MEM_PEAK : $(grep '^PEAK' "$MEM_LOG" 2>/dev/null | tail -1)"
    grep -q GUARD_TRIP "$MEM_LOG" 2>/dev/null && echo "GUARD: watchdog aborted the build"
    tail -3 "atc_${OUT}_cann911.log" 2>/dev/null
    if [ "${KVPROBE_KILL_STRAY:-0}" = "1" ]; then
        pkill -9 -x atc.bin 2>/dev/null
    elif pgrep -x atc.bin >/dev/null 2>&1; then
        echo "NOTE: another atc.bin is running; left alone (set KVPROBE_KILL_STRAY=1 to kill strays)"
    fi
    sleep 1
    rm -rf kernel_meta
    if [ -f "${OUT}.om" ]; then
        ls -l "${OUT}.om"
        sha256sum "${OUT}.om"
    else
        echo "NO_OM_PRODUCED"
    fi
    echo "MEM_AFTER : $(awk '/MemAvailable/{printf "%d MB available", $2/1024}' /proc/meminfo)"
    echo "END       : $(date -Is)"
done

echo
echo "=== ALL DONE: $(date -Is) ==="
ls -l *.om 2>&1
