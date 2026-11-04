#!/bin/bash
# Probe pyACL availability under CANN 9.1.1 for the board harness.
set +u
export PATH="/usr/local/miniconda3/bin:${PATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /data/Ascend/cann-9.1.1/set_env.sh

echo "PYTHONPATH=${PYTHONPATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

for PY in /usr/bin/python3 /usr/local/miniconda3/bin/python3; do
    echo "=== $PY ==="
    $PY -V 2>&1
    $PY -c "import numpy; print('numpy', numpy.__version__)" 2>&1 | tail -2
    $PY -c "import acl; print('acl OK', acl.__file__); print('has rt:', hasattr(acl,'rt'))" 2>&1 | tail -3
done
