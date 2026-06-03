#!/bin/bash
# RAM Coffers vs Stock llama.cpp Benchmark
# Reproducible benchmark for multi-NUMA Linux systems
#
# Usage: ./benchmark.sh [--download] [--numa-nodes 2]
#
# Requirements:
# - numactl
# - wget or curl
# - llama.cpp built (./llama.cpp/main)
# - RAM Coffers built (./ram-coffers/main)

set -euo pipefail

MODEL_FILE="TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/${MODEL_FILE}"
MODEL_DIR="./models"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
PP=128
TG=32
PROMPT="Explain the Proof of Antiquity consensus mechanism."
ITERATIONS=3

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[BENCH]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse args
DOWNLOAD=false
NUMA_NODES=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --download) DOWNLOAD=true; shift ;;
        --numa-nodes) NUMA_NODES=$2; shift 2 ;;
        --pp) PP=$2; shift 2 ;;
        --tg) TG=$2; shift 2 ;;
        --iterations) ITERATIONS=$2; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================"
echo "RAM Coffers vs llama.cpp Benchmark"
echo "============================================"
echo ""

# Detect NUMA
if command -v numactl &>/dev/null; then
    NUM_NUMA=$(numactl --hardware | grep "available:" | awk '{print $2}')
    log "NUMA nodes detected: ${NUM_NUMA}"
    for i in $(seq 0 $((NUM_NUMA - 1))); do
        CPUS=$(numactl --hardware | grep "node $i cpus:" | awk '{for(i=4;i<=NF;i++) printf $i" "}')
        log "  Node $i: $CPUS"
    done
else
    warn "numactl not found, assuming single NUMA node"
    NUM_NUMA=1
fi

# Download model
if [ "$DOWNLOAD" = true ] || [ ! -f "$MODEL_PATH" ]; then
    log "Downloading ${MODEL_FILE}..."
    mkdir -p "$MODEL_DIR"
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF "$MODEL_FILE" --local-dir "$MODEL_DIR"
    elif command -v wget &>/dev/null; then
        wget -q --show-progress "$MODEL_URL" -O "$MODEL_PATH"
    else
        curl -L --progress-bar "$MODEL_URL" -o "$MODEL_PATH"
    fi
    log "Model downloaded: $MODEL_PATH"
else
    log "Using existing model: $MODEL_PATH"
fi

echo ""
echo "Benchmark config:"
echo "  Model: $MODEL_FILE"
echo "  PP tokens: $PP"
echo "  TG tokens: $TG"
echo "  Iterations: $ITERATIONS"
echo ""

# Benchmark function
run_benchmark() {
    local BINARY=$1
    local NAME=$2
    local NUMA_NODE=$3
    
    if [ ! -f "$BINARY" ]; then
        warn "Binary not found: $NAME ($BINARY)"
        return 1
    fi
    
    local TOTAL_PP=0
    local TOTAL_TG=0
    
    for i in $(seq 1 $ITERATIONS); do
        log "  $NAME - Iteration $i/$ITERATIONS (NUMA node $NUMA_NODE)"
        
        if [ "$NUMA_NODE" != "none" ] && command -v numactl &>/dev/null; then
            OUTPUT=$(numactl --cpunbind="$NUMA_NODE" --membind="$NUMA_NODE" "$BINARY"                 -m "$MODEL_PATH" -p "$PROMPT" -n "$TG" 2>&1)
        else
            OUTPUT=$("$BINARY" -m "$MODEL_PATH" -p "$PROMPT" -n "$TG" 2>&1)
        fi
        
        # Parse timing from llama.cpp output
        PP_SPEED=$(echo "$OUTPUT" | grep -oP 'pp\s+\d+\.\d+\s+t/s' | grep -oP '[\d.]+' | head -1)
        TG_SPEED=$(echo "$OUTPUT" | grep -oP 'tg\s+\d+\.\d+\s+t/s' | grep -oP '[\d.]+' | head -1)
        
        if [ -n "$PP_SPEED" ]; then
            TOTAL_PP=$(echo "$TOTAL_PP + $PP_SPEED" | bc -l 2>/dev/null || echo "$TOTAL_PP")
            log "    PP: ${PP_SPEED} tok/s"
        fi
        if [ -n "$TG_SPEED" ]; then
            TOTAL_TG=$(echo "$TOTAL_TG + $TG_SPEED" | bc -l 2>/dev/null || echo "$TOTAL_TG")
            log "    TG: ${TG_SPEED} tok/s"
        fi
    done
    
    # Calculate averages
    if [ "$ITERATIONS" -gt 0 ]; then
        AVG_PP=$(echo "scale=2; $TOTAL_PP / $ITERATIONS" | bc -l 2>/dev/null || echo "N/A")
        AVG_TG=$(echo "scale=2; $TOTAL_TG / $ITERATIONS" | bc -l 2>/dev/null || echo "N/A")
        echo "$NAME|$AVG_PP|$AVG_TG"
    fi
}

# Run benchmarks
echo "--- Running Benchmarks ---"
echo ""

STOCK_RESULT=""
COFFERS_RESULT=""

if [ -f "./llama.cpp/main" ]; then
    STOCK_RESULT=$(run_benchmark "./llama.cpp/main" "Stock llama.cpp" "none")
    log "Stock result: $STOCK_RESULT"
else
    warn "Stock llama.cpp not found at ./llama.cpp/main"
fi

if [ -f "./ram-coffers/main" ]; then
    COFFERS_RESULT=$(run_benchmark "./ram-coffers/main" "RAM Coffers" "0")
    log "Coffers result: $COFFERS_RESULT"
else
    warn "RAM Coffers not found at ./ram-coffers/main"
fi

# Generate markdown table
echo ""
echo "============================================"
echo "RESULTS"
echo "============================================"
echo ""
echo "| Metric | Stock llama.cpp | RAM Coffers | Speedup |"
echo "|--------|----------------|-------------|---------|"

if [ -n "$STOCK_RESULT" ] && [ -n "$COFFERS_RESULT" ]; then
    STOCK_PP=$(echo "$STOCK_RESULT" | cut -d'|' -f2)
    STOCK_TG=$(echo "$STOCK_RESULT" | cut -d'|' -f3)
    COFFERS_PP=$(echo "$COFFERS_RESULT" | cut -d'|' -f2)
    COFFERS_TG=$(echo "$COFFERS_RESULT" | cut -d'|' -f3)
    
    PP_SPEEDUP=$(echo "scale=2; $COFFERS_PP / $STOCK_PP" | bc -l 2>/dev/null || echo "N/A")
    TG_SPEEDUP=$(echo "scale=2; $COFFERS_TG / $STOCK_TG" | bc -l 2>/dev/null || echo "N/A")
    
    echo "| Prompt Processing (pp${PP}) | ${STOCK_PP} tok/s | ${COFFERS_PP} tok/s | ${PP_SPEEDUP}x |"
    echo "| Text Generation (tg${TG}) | ${STOCK_TG} tok/s | ${COFFERS_TG} tok/s | ${TG_SPEEDUP}x |"
fi

echo ""
echo "Wallet: jesusmp"
