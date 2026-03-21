#!/bin/bash
# 
# benchmark_coffers.sh
# Reproducible benchmark: RAM Coffers NUMA-aware inference vs stock llama.cpp
#
set -e

# Configuration
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp.git"
LLAMA_CPP_DIR="llama.cpp-coffers-bench"
MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_FILE="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

echo "=========================================================="
echo " RAM Coffers NUMA-Aware Inference Benchmark Script"
echo "=========================================================="

# 1. Check requirements
if ! command -v numactl &> /dev/null; then
    echo "ERROR: numactl is required but not installed."
    echo "Please install it (e.g., sudo apt-get install numactl) and try again."
    exit 1
fi
if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake is required but not installed."
    exit 1
fi

NUM_NODES=$(numactl --hardware | awk '/available:/ {print $2}')
echo "=> Detected $NUM_NODES NUMA node(s)."
if [ "$NUM_NODES" -lt 2 ]; then
    echo "=> WARNING: This system has less than 2 NUMA nodes."
    echo "=> Coffers RAM interleaving scales best on multi-NUMA systems."
fi

# 2. Download Model
if [ ! -f "$MODEL_FILE" ]; then
    echo "=> Downloading TinyLlama Q4 model..."
    curl -L "$MODEL_URL" -o "$MODEL_FILE"
else
    echo "=> Model $MODEL_FILE already exists."
fi

# 3. Setup and build llama.cpp
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "=> Cloning llama.cpp..."
    git clone "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
fi

echo "=> Building llama.cpp llama-bench..."
cd "$LLAMA_CPP_DIR"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)" llama-bench
cd ../..

BENCH_BIN="./$LLAMA_CPP_DIR/build/bin/llama-bench"
if [ ! -f "$BENCH_BIN" ]; then
    # Fallback if cmake outputs to different path
    BENCH_BIN="./$LLAMA_CPP_DIR/build/llama-bench"
fi

if [ ! -f "$BENCH_BIN" ]; then
    echo "ERROR: llama-bench executable not found."
    exit 1
fi

# 4. Benchmarking
echo "=> Running Benchmarks (pp128 / tg32)..."
echo "   This will take a few minutes."

run_bench() {
    local name="$1"
    local cmd_prefix="$2"
    
    echo "   -> Testing: $name"
    # Capture standard llama-bench markdown output
    # Format of llama-bench: | model | size | params | backend | threads | test | t/s |
    local output
    output=$($cmd_prefix "$BENCH_BIN" -m "$MODEL_FILE" -p 128 -n 32 2>/dev/null | grep -E "tg32|pp128|test")
    
    # Simple extraction
    local tg32=$(echo "$output" | grep "tg32" | awk -F '|' '{print $8}' | sed 's/ //g' | head -n 1)
    local pp128=$(echo "$output" | grep "pp128" | awk -F '|' '{print $8}' | sed 's/ //g' | head -n 1)
    
    # Return as format: Name | PP128 | TG32
    echo "| $name | ${pp128:-N/A} | ${tg32:-N/A} |"
}

RES_STOCK=$(run_bench "Stock (OS Default)" "")
RES_SINGLE=$(run_bench "Coffers (Node 0 Bind)" "numactl --cpunodebind=0 --membind=0")
RES_INTER=$(run_bench "Coffers (NUMA Interleave)" "numactl --interleave=all")

# 5. Output Results
echo ""
echo "### Benchmark Results: RAM Coffers NUMA vs Stock"
echo ""
echo "| Configuration | Prompt Processing (pp128) t/s | Text Generation (tg32) t/s |"
echo "|---------------|-------------------------------|----------------------------|"
echo "$RES_STOCK"
echo "$RES_SINGLE"
echo "$RES_INTER"
echo ""
echo "Implementation Details: Tested using TinyLlama Q4 via llama-bench."
echo "OS / Hardware Topology: $(uname -s) $(uname -m) / $NUM_NODES NUMA nodes."

