#!/usr/bin/env bash
# Constrain the A6000 toward an embedded operating point (Table IV).
#
# Three controls are applied together. None of them makes the card into an
# embedded module: the memory subsystem, the DLA path and the thermal envelope
# are unchanged, and the manuscript says so. What they do reproduce is the
# shader count and clock, which is what the compute-bound part of the engine
# is sensitive to.
#
# Run as root. Undo with ./constrain_gpu.sh reset
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
TARGET_CLOCK_MHZ="${TARGET_CLOCK_MHZ:-918}"
TARGET_POWER_W="${TARGET_POWER_W:-25}"
SM_PERCENT="${SM_PERCENT:-9}"          # 8 of 84 SMs on an RTX A6000

usage() { echo "usage: $0 [apply|reset|status]"; exit 1; }

apply() {
  echo "== enabling persistence mode"
  nvidia-smi -i "$GPU_ID" -pm 1

  echo "== locking graphics clock to ${TARGET_CLOCK_MHZ} MHz"
  nvidia-smi -i "$GPU_ID" -lgc "${TARGET_CLOCK_MHZ},${TARGET_CLOCK_MHZ}"

  echo "== capping board power to ${TARGET_POWER_W} W"
  # The card refuses values below its floor; the floor is then reported and used.
  if ! nvidia-smi -i "$GPU_ID" -pl "$TARGET_POWER_W"; then
    floor=$(nvidia-smi -i "$GPU_ID" --query-gpu=power.min_limit \
            --format=csv,noheader,nounits)
    echo "   refused; using the reported floor of ${floor} W instead"
    echo "   RECORD THIS in the paper: the 25 W figure was not attainable."
    nvidia-smi -i "$GPU_ID" -pl "$floor"
  fi

  echo "== starting the multi-process service with ${SM_PERCENT}% of the SMs"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$SM_PERCENT"
  nvidia-cuda-mps-control -d || echo "   MPS daemon already running"
  echo
  echo "Export this in the shell that runs the profiling:"
  echo "  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${SM_PERCENT}"
  echo
  status
}

reset() {
  echo "== stopping MPS"
  echo quit | nvidia-cuda-mps-control || true
  echo "== releasing clock and power limits"
  nvidia-smi -i "$GPU_ID" -rgc || true
  default=$(nvidia-smi -i "$GPU_ID" --query-gpu=power.default_limit \
            --format=csv,noheader,nounits)
  nvidia-smi -i "$GPU_ID" -pl "$default" || true
  status
}

status() {
  nvidia-smi -i "$GPU_ID" --query-gpu=name,clocks.gr,clocks.max.gr,\
power.limit,power.min_limit,power.max_limit,memory.total \
    --format=csv
  echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-unset}"
}

case "${1:-}" in
  apply) apply ;;
  reset) reset ;;
  status) status ;;
  *) usage ;;
esac
