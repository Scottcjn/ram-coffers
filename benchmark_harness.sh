#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${ROOT_DIR}/benchmarks/out"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
TOPOLOGY_FILE="$OUT_DIR/topology-${TS}.txt"
REPORT_FILE="$OUT_DIR/benchmark-${TS}.md"
ENV_FILE="$OUT_DIR/env-${TS}.txt"

{
  echo "# RAM Coffers Benchmark Run"
  echo
  echo "- Timestamp: $(date +%Y-%m-%dT%H:%M:%S%z)"
  echo "- Host: $(hostname)"
  echo "- Kernel: $(uname -srmo)"
  echo
  echo "## Tooling"
  command -v gcc >/dev/null 2>&1 && echo "- gcc: $(gcc --version | head -1)" || echo "- gcc: not found"
  command -v clang >/dev/null 2>&1 && echo "- clang: $(clang --version | head -1)" || echo "- clang: not found"
  command -v numactl >/dev/null 2>&1 && echo "- numactl: available" || echo "- numactl: not found"
  command -v lscpu >/dev/null 2>&1 && echo "- lscpu: available" || echo "- lscpu: not found"
} > "$ENV_FILE"

{
  echo "# Machine Topology"
  echo
  if command -v lscpu >/dev/null 2>&1; then
    echo '## lscpu'
    echo '```text'
    lscpu
    echo '```'
    echo
  else
    echo '- lscpu not available'
    echo
  fi

  if command -v numactl >/dev/null 2>&1; then
    echo '## numactl --hardware'
    echo '```text'
    numactl --hardware
    echo '```'
  else
    echo '- numactl not available'
  fi
} > "$TOPOLOGY_FILE"

BUILD_OK=false
BIN="$OUT_DIR/bench_vcipher_collapse"
BUILD_LOG="$OUT_DIR/build-${TS}.log"
RUN_LOG="$OUT_DIR/run-${TS}.log"

if command -v gcc >/dev/null 2>&1; then
  if gcc -mcpu=power8 -mvsx -mcrypto -maltivec -O3 -fopenmp \
      bench_vcipher_collapse.c -o "$BIN" -lm >"$BUILD_LOG" 2>&1; then
    BUILD_OK=true
  fi
fi

{
  echo "# Benchmark Report"
  echo
  echo "- Timestamp: $(date +%Y-%m-%dT%H:%M:%S%z)"
  echo "- Topology snapshot: $(basename "$TOPOLOGY_FILE")"
  echo "- Environment snapshot: $(basename "$ENV_FILE")"
  echo
  if [ "$BUILD_OK" = true ]; then
    echo "## Build"
    echo "- Status: success"
    echo "- Binary: $(basename "$BIN")"
    echo
    echo "## Run Output"
    echo '```text'
    "$BIN" | tee "$RUN_LOG"
    echo '```'
  else
    echo "## Build"
    echo "- Status: skipped or failed"
    echo "- Reason: POWER8-specific benchmark could not be compiled on this host"
    echo
    echo "## Fallback"
    echo "This harness still captured reproducible machine topology and environment details."
    echo "Use the same script on POWER8 hardware to produce benchmark numbers with identical report structure."
    echo
    if [ -s "$BUILD_LOG" ]; then
      echo "## Build Log"
      echo '```text'
      cat "$BUILD_LOG"
      echo '```'
    fi
  fi
} > "$REPORT_FILE"

echo "Generated:"
echo "- $TOPOLOGY_FILE"
echo "- $ENV_FILE"
echo "- $REPORT_FILE"
[ "$BUILD_OK" = true ] && echo "- $RUN_LOG"
