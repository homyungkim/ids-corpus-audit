#!/usr/bin/env bash
# The full sequence, in order. Stop at the first failure.
# usage: scripts/run_all.sh [config] [backend]
set -euo pipefail

CFG="${1:-configs/default.yaml}"
BACKEND="${2:-trt}"
PY="${PY:-python}"

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

say "1/9 prepare";      $PY -m clids.cli prepare    -c "$CFG"
say "2/9 train";        $PY -m clids.cli train      -c "$CFG"
say "3/9 baselines";    $PY -m clids.cli baselines  -c "$CFG"
say "4/9 export onnx";  $PY -m clids.cli export     -c "$CFG"

if $PY -c "import tensorrt, pycuda" 2>/dev/null; then
  say "5/9 build engines"; $PY -m clids.cli engines -c "$CFG"
else
  echo "tensorrt/pycuda not importable: falling back to the fake-quant backend."
  echo "Results from it are for development only."
  BACKEND=fakequant
fi

say "6/9 calibration study (Table VII)"
$PY -m clids.cli calibrate -c "$CFG" --backend "$BACKEND"

say "7/9 evaluate (Tables V, VI)"
$PY -m clids.cli evaluate  -c "$CFG" --backend "$BACKEND"

say "8/9 ablation and degraded-link sweep"
$PY -m clids.cli ablation    -c "$CFG"
$PY -m clids.cli packet-loss -c "$CFG"

say "9/9 profiling, figures, tables"
echo "If the constrained operating point is wanted, run"
echo "  sudo scripts/constrain_gpu.sh apply && export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=9"
echo "before this step, and reset afterwards."
$PY -m clids.cli profile -c "$CFG"
$PY -m clids.cli figures -c "$CFG"
$PY -m clids.cli tables  -c "$CFG"

say "done"
