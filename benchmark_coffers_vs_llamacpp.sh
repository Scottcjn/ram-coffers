#!/bin/bash
# ============================================
# RAM Coffers Benchmark Script
# Compares RAM Coffers NUMA-aware inference vs stock llama.cpp
# Issue: #45 - Bounty: 15 RTC
# ============================================

set -e

echo "=========================================="
echo "RAM Coffers Benchmark"
echo "Coffers vs Stock llama.cpp"
echo "=========================================="
echo ""

# Configuration
MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH="./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
PP_PROMPT="The quick brown fox jumps over the lazy dog. The weather is nice today. I enjoy programming in C and Python. Machine learning is fascinating. Neural networks can learn complex patterns. The sun is shining brightly. Birds are singing in the trees. Water flows downhill. The moon orbits around Earth. Stars twinkle in the night sky. Flowers bloom in spring. Cats like to sleep. Dogs are loyal friends. Books contain knowledge. Music brings joy. Food tastes delicious. Exercise keeps us healthy. Sleep is important. Friends make life better. Laughter is the best medicine."
TG_PROMPT="Once upon a time"

# Parameters
PP_LEN=128
TG_LEN=32
NUMA_NODES=$(ls -d /sys/devices/system/node/node* 2>/dev/null | wc -l)

echo "Configuration:"
echo "  Model: TinyLlama 1.1B Q4_K_M"
echo "  Prefill: $PP_LEN tokens"
echo "  Text Generation: $TG_LEN tokens"
echo "  NUMA Nodes: $NUMA_NODES"
echo ""

# Create models directory
mkdir -p ./models

# Download model if not exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading model..."
    wget -q --show-progress "$MODEL_URL" -O "$MODEL_PATH"
    echo "Model downloaded: $MODEL_PATH"
else
    echo "Model already exists: $MODEL_PATH"
fi

echo ""
echo "=========================================="
echo "Benchmark Results"
echo "=========================================="
echo ""

# Function to run llama.cpp benchmark
run_llamacpp_bench() {
    local binary=$1
    local name=$2
    
    if [ ! -f "$binary" ]; then
        echo "$name: binary not found, skipping"
        return
    fi
    
    echo "Running $name..."
    
    # Prefill benchmark
    local pp_result=$($binary -m "$MODEL_PATH" -p "$PP_PROMPT" -n 0 --timings 2>&1 | grep "eval time" | tail -1)
    local pp_time=$(echo "$pp_result" | grep -oP '\d+\.\d+' | head -1)
    
    # Text generation benchmark
    local tg_result=$($binary -m "$MODEL_PATH" -p "$TG_PROMPT" -n $TG_LEN --timings 2>&1 | grep "eval time" | tail -1)
    local tg_time=$(echo "$tg_result" | grep -oP '\d+\.\d+' | head -1)
    
    echo "$name|$pp_time|$tg_time"
}

# Check for llama.cpp installations
declare -a results=()

# Try system llama.cpp
if command -v llama-cli &> /dev/null; then
    result=$(run_llamacpp_bench "$(which llama-cli)" "llama.cpp (system)")
    results+=("$result")
fi

# Try local build
if [ -f "./llama.cpp/build/bin/llama-cli" ]; then
    result=$(run_llamacpp_bench "./llama.cpp/build/bin/llama-cli" "llama.cpp (local)")
    results+=("$result")
fi

# Try RAM Coffers build
if [ -f "./build/bin/llama-cli" ]; then
    result=$(run_llamacpp_bench "./build/bin/llama-cli" "RAM Coffers")
    results+=("$result")
fi

# Print results table
echo ""
echo "Results Summary:"
echo ""
printf "| %-25s | %-15s | %-15s |\n" "Implementation" "Prefill (s)" "Generate (s)"
printf "|%-27s|%-17s|%-17s|\n" "---------------------------" "-----------------" "-----------------"

for result in "${results[@]}"; do
    name=$(echo "$result" | cut -d'|' -f1)
    pp=$(echo "$result" | cut -d'|' -f2)
    tg=$(echo "$result" | cut -d'|' -f3)
    printf "| %-25s | %-15s | %-15s |\n" "$name" "${pp:-N/A}" "${tg:-N/A}"
done

echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="

# Output markdown table for issue comment
echo ""
echo "## Markdown Table (for GitHub issue)"
echo ""
echo '```markdown'
printf "| Implementation | Prefill (s) | Generate (s) |\n"
printf "|----------------|-------------|--------------|\n"
for result in "${results[@]}"; do
    name=$(echo "$result" | cut -d'|' -f1)
    pp=$(echo "$result" | cut -d'|' -f2)
    tg=$(echo "$result" | cut -d'|' -f3)
    printf "| %s | %s | %s |\n" "$name" "${pp:-N/A}" "${tg:-N/A}"
done
echo '```'

# Wallet info for bounty
echo ""
echo "=========================================="
echo "Bounty Claim"
echo "=========================================="
echo ""
echo "**Wallet**: Dlove123"
echo "**RTC Address**: RTCb72a1accd46b9ba9f22dbd4b5c6aad5a5831572b"
echo "**GitHub**: Dlove123"
