#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="${BENCH_ROOT:-${ROOT_DIR}/.bench/coffers-vs-llamacpp}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/benchmarks/out}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
LLAMA_CPP_URL="${LLAMA_CPP_URL:-https://github.com/ggerganov/llama.cpp.git}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
MODEL_PATH="${MODEL_PATH:-${BENCH_ROOT}/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
THREADS="${THREADS:-}"
JOBS="${JOBS:-}"
RUNS="${RUNS:-3}"
STOCK_NUMA_NODE="${STOCK_NUMA_NODE:-0}"
STOCK_BIN="${STOCK_BIN:-}"
COFFERS_BIN="${COFFERS_BIN:-}"
DRY_RUN=false
SETUP_ONLY=false
ALLOW_SINGLE_NUMA=false

usage() {
  cat <<'EOF'
Usage: ./benchmark_coffers_vs_llamacpp.sh [options]

Build or use llama.cpp, download TinyLlama Q4, run a pp128/tg32 benchmark, and
write a markdown table comparing stock llama.cpp with RAM Coffers NUMA policy.

Options:
  --dry-run              Write a plan report without downloading, building, or running.
  --setup-only           Download/build prerequisites, then stop before benchmarks.
  --allow-single-numa    Permit execution on one-NUMA-node Linux for smoke testing.
  --stock-bin PATH       Use an existing stock llama-bench binary.
  --coffers-bin PATH     Use an existing RAM Coffers llama-bench binary.
  --model PATH           Use an existing GGUF model path.
  --threads N            Override benchmark thread count.
  --help                 Show this help text.

Environment:
  BENCH_ROOT             Workspace for cloned llama.cpp trees and model cache.
  OUT_DIR                Directory for report/log output.
  LLAMA_CPP_REF          llama.cpp git ref to clone, default: master.
  MODEL_URL              TinyLlama Q4 GGUF URL to download.
  RUNS                   llama-bench repetitions, default: 3.
  STOCK_NUMA_NODE        NUMA node for stock run, default: 0.
  JOBS                   Build parallelism, default: detected CPU count.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      ;;
    --setup-only)
      SETUP_ONLY=true
      ;;
    --allow-single-numa)
      ALLOW_SINGLE_NUMA=true
      ;;
    --stock-bin)
      STOCK_BIN="${2:?missing path for --stock-bin}"
      shift
      ;;
    --coffers-bin)
      COFFERS_BIN="${2:?missing path for --coffers-bin}"
      shift
      ;;
    --model)
      MODEL_PATH="${2:?missing path for --model}"
      shift
      ;;
    --threads)
      THREADS="${2:?missing value for --threads}"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

mkdir -p "$BENCH_ROOT" "$OUT_DIR" "$(dirname "$MODEL_PATH")"
TS="$(date +%Y%m%d-%H%M%S)"
REPORT_FILE="${OUT_DIR}/coffers-vs-llamacpp-${TS}.md"
TOPOLOGY_FILE="${OUT_DIR}/coffers-topology-${TS}.txt"
ENV_FILE="${OUT_DIR}/coffers-env-${TS}.txt"
STOCK_LOG="${OUT_DIR}/stock-llamacpp-${TS}.log"
COFFERS_LOG="${OUT_DIR}/ram-coffers-${TS}.log"

cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu
  else
    echo 4
  fi
}

THREADS="${THREADS:-$(cpu_count)}"
JOBS="${JOBS:-$(cpu_count)}"

node_count() {
  if command -v numactl >/dev/null 2>&1; then
    numactl --hardware 2>/dev/null | awk '/available:/ { print $2; exit }'
  else
    echo 0
  fi
}

capture_environment() {
  {
    echo "# Environment"
    echo
    echo "- Timestamp: $(date +%Y-%m-%dT%H:%M:%S%z)"
    echo "- Host: $(hostname)"
    echo "- Kernel: $(uname -srmo 2>/dev/null || uname -a)"
    echo "- Threads: ${THREADS}"
    echo "- Runs: ${RUNS}"
    echo "- llama.cpp ref: ${LLAMA_CPP_REF}"
    echo "- Model: ${MODEL_PATH}"
    echo
    echo "## Tooling"
    for tool in git cmake curl numactl lscpu awk perl; do
      if command -v "$tool" >/dev/null 2>&1; then
        echo "- ${tool}: $(command -v "$tool")"
      else
        echo "- ${tool}: not found"
      fi
    done
  } > "$ENV_FILE"

  {
    echo "# Topology"
    echo
    if command -v lscpu >/dev/null 2>&1; then
      echo "## lscpu"
      echo '```text'
      lscpu
      echo '```'
      echo
    else
      echo "- lscpu not available"
      echo
    fi
    if command -v numactl >/dev/null 2>&1; then
      echo "## numactl --hardware"
      echo '```text'
      numactl --hardware
      echo '```'
    else
      echo "- numactl not available"
    fi
  } > "$TOPOLOGY_FILE"
}

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

download_model() {
  if [ -s "$MODEL_PATH" ]; then
    return
  fi
  need_command curl
  echo "Downloading TinyLlama Q4 model to ${MODEL_PATH}"
  curl -L --fail --continue-at - --output "$MODEL_PATH" "$MODEL_URL"
}

clone_llamacpp() {
  local dest="$1"
  if [ -d "${dest}/.git" ]; then
    git -C "$dest" fetch --depth 1 origin "$LLAMA_CPP_REF"
    git -C "$dest" checkout FETCH_HEAD
  else
    git clone --depth 1 --branch "$LLAMA_CPP_REF" "$LLAMA_CPP_URL" "$dest"
  fi
}

copy_coffers_headers() {
  local dest="$1"
  local arch_dir="${dest}/ggml/src/ggml-cpu/arch/powerpc"
  mkdir -p "$arch_dir"
  cp "${ROOT_DIR}/ggml-ram-coffers.h" "$arch_dir/"
  cp "${ROOT_DIR}/ggml-coffer-mmap.h" "$arch_dir/"
  cp "${ROOT_DIR}/ggml-neuromorphic-coffers.h" "$arch_dir/"
}

build_llamacpp() {
  local src="$1"
  local build_dir="$2"
  shift 2
  need_command cmake
  cmake -S "$src" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP=ON \
    "$@"
  cmake --build "$build_dir" --config Release -j "$JOBS"
}

find_llama_bench() {
  local build_dir="$1"
  for candidate in \
    "${build_dir}/bin/llama-bench" \
    "${build_dir}/bin/llama-bench.exe" \
    "${build_dir}/llama-bench"; do
    if [ -x "$candidate" ]; then
      echo "$candidate"
      return
    fi
  done
  echo "Could not find llama-bench under ${build_dir}" >&2
  exit 1
}

prepare_binaries() {
  local stock_src="${BENCH_ROOT}/llama.cpp-stock"
  local coffers_src="${BENCH_ROOT}/llama.cpp-coffers"
  local stock_build="${stock_src}/build-stock"
  local coffers_build="${coffers_src}/build-coffers"
  local arch
  arch="$(uname -m)"

  if [ -z "$STOCK_BIN" ]; then
    clone_llamacpp "$stock_src"
    build_llamacpp "$stock_src" "$stock_build"
    STOCK_BIN="$(find_llama_bench "$stock_build")"
  fi

  if [ -z "$COFFERS_BIN" ]; then
    clone_llamacpp "$coffers_src"
    copy_coffers_headers "$coffers_src"
    if [[ "$arch" == ppc64* || "$arch" == powerpc* ]]; then
      build_llamacpp "$coffers_src" "$coffers_build" \
        -DCMAKE_C_FLAGS="-mcpu=power8 -mvsx -maltivec -O3" \
        -DCMAKE_CXX_FLAGS="-mcpu=power8 -mvsx -maltivec -O3"
    else
      build_llamacpp "$coffers_src" "$coffers_build"
    fi
    COFFERS_BIN="$(find_llama_bench "$coffers_build")"
  fi
}

run_bench() {
  local label="$1"
  local bin="$2"
  local log_file="$3"
  shift 3

  echo "Running ${label}: $* ${bin} -m ${MODEL_PATH} -p 128 -n 32 -t ${THREADS} -r ${RUNS}"
  {
    echo "# ${label}"
    echo
    echo "Command:"
    printf '%q ' "$@" "$bin" -m "$MODEL_PATH" -p 128 -n 32 -t "$THREADS" -r "$RUNS"
    echo
    echo
    "$@" "$bin" -m "$MODEL_PATH" -p 128 -n 32 -t "$THREADS" -r "$RUNS"
  } > "$log_file" 2>&1
}

extract_metric() {
  local log_file="$1"
  local pattern="$2"
  perl -ne '
    BEGIN { $pat = shift @ARGV; }
    next unless /$pat/i;
    @cells = split /\|/;
    for ($i = $#cells; $i >= 0; $i--) {
      if ($cells[$i] =~ /([0-9]+(?:\.[0-9]+)?)/) {
        print $1;
        exit 0;
      }
    }
  ' "$pattern" "$log_file"
}

write_report() {
  local stock_pp="$1"
  local stock_tg="$2"
  local coffers_pp="$3"
  local coffers_tg="$4"
  local status_note="$5"

  stock_pp="${stock_pp:-unparsed}"
  stock_tg="${stock_tg:-unparsed}"
  coffers_pp="${coffers_pp:-unparsed}"
  coffers_tg="${coffers_tg:-unparsed}"

  {
    echo "# RAM Coffers vs stock llama.cpp benchmark"
    echo
    echo "- Timestamp: $(date +%Y-%m-%dT%H:%M:%S%z)"
    echo "- Wallet: RTC1410e82d545ce0b3ffd21ca83e2465a8f2c3a64e"
    echo "- Model URL: ${MODEL_URL}"
    echo "- Model path: ${MODEL_PATH}"
    echo "- Test shape: pp128 / tg32"
    echo "- Threads: ${THREADS}"
    echo "- Runs: ${RUNS}"
    echo "- Topology: $(basename "$TOPOLOGY_FILE")"
    echo "- Environment: $(basename "$ENV_FILE")"
    echo
    if [ -n "$status_note" ]; then
      echo "> ${status_note}"
      echo
    fi
    echo "| Variant | NUMA policy | pp128 tokens/s | tg32 tokens/s | Raw log |"
    echo "|---|---:|---:|---:|---|"
    echo "| stock llama.cpp | node ${STOCK_NUMA_NODE} bind | ${stock_pp} | ${stock_tg} | $(basename "$STOCK_LOG") |"
    echo "| RAM Coffers | interleave all NUMA nodes | ${coffers_pp} | ${coffers_tg} | $(basename "$COFFERS_LOG") |"
    echo
    echo "## Commands"
    echo
    echo '```bash'
    echo "numactl --cpunodebind=${STOCK_NUMA_NODE} --membind=${STOCK_NUMA_NODE} ${STOCK_BIN:-<stock llama-bench>} -m ${MODEL_PATH} -p 128 -n 32 -t ${THREADS} -r ${RUNS}"
    echo "numactl --interleave=all ${COFFERS_BIN:-<coffers llama-bench>} -m ${MODEL_PATH} -p 128 -n 32 -t ${THREADS} -r ${RUNS}"
    echo '```'
    echo
    echo "## Notes"
    echo
    echo "- The stock run pins CPU and memory allocation to one NUMA node."
    echo "- The RAM Coffers run follows the repository quick-start policy by using the Coffers build and interleaving memory across all NUMA nodes."
    echo "- Existing builds can be supplied with --stock-bin and --coffers-bin when testing a hand-patched POWER8 llama.cpp tree."
  } > "$REPORT_FILE"
}

capture_environment

if [ "$DRY_RUN" = true ]; then
  write_report "" "" "" "" "Dry run only: no model download, build, or benchmark execution was attempted."
  echo "Generated dry-run report: ${REPORT_FILE}"
  exit 0
fi

if [ "$(uname -s)" != "Linux" ]; then
  write_report "" "" "" "" "Unsupported host for execution: this benchmark requires Linux with numactl. Use --dry-run to inspect the planned commands."
  echo "Generated unsupported-host report: ${REPORT_FILE}"
  exit 1
fi

need_command git
need_command numactl
nodes="$(node_count)"
if [ "${nodes:-0}" -lt 2 ] && [ "$ALLOW_SINGLE_NUMA" != true ]; then
  write_report "" "" "" "" "Host has ${nodes:-0} NUMA node(s); pass --allow-single-numa for smoke tests, or run on multi-NUMA Linux for bounty-grade numbers."
  echo "Generated single-NUMA guard report: ${REPORT_FILE}"
  exit 1
fi

download_model
prepare_binaries

if [ "$SETUP_ONLY" = true ]; then
  write_report "" "" "" "" "Setup-only run: model and binaries were prepared, but benchmarks were not executed."
  echo "Generated setup-only report: ${REPORT_FILE}"
  exit 0
fi

run_bench "stock llama.cpp" "$STOCK_BIN" "$STOCK_LOG" \
  numactl "--cpunodebind=${STOCK_NUMA_NODE}" "--membind=${STOCK_NUMA_NODE}"
run_bench "RAM Coffers NUMA policy" "$COFFERS_BIN" "$COFFERS_LOG" \
  numactl --interleave=all

stock_pp="$(extract_metric "$STOCK_LOG" "pp\\s*128|pp128" || true)"
stock_tg="$(extract_metric "$STOCK_LOG" "tg\\s*32|tg32" || true)"
coffers_pp="$(extract_metric "$COFFERS_LOG" "pp\\s*128|pp128" || true)"
coffers_tg="$(extract_metric "$COFFERS_LOG" "tg\\s*32|tg32" || true)"
write_report "$stock_pp" "$stock_tg" "$coffers_pp" "$coffers_tg" ""

echo "Generated benchmark report: ${REPORT_FILE}"
