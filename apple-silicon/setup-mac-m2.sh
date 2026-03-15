#!/bin/bash
# ─────────────────────────────────────────────────────────
# Elyan Labs — Apple Silicon PSE Setup for Mac Mini M2
#
# Run this script ON the Mac Mini:
#   curl -sL https://raw.githubusercontent.com/Scottcjn/ram-coffers/main/apple-silicon/setup-mac-m2.sh | bash
#
# Or SSH and run:
#   ssh sophia@192.168.0.134
#   bash ~/ram-coffers/apple-silicon/setup-mac-m2.sh
# ─────────────────────────────────────────────────────────

set -e

echo "╔══════════════════════════════════════════════╗"
echo "║  Elyan Labs — Apple Silicon PSE Setup         ║"
echo "║  Mac Mini M2 — Non-Bijunctive LLM Inference   ║"
echo "╚══════════════════════════════════════════════╝"

# ── Step 1: Get ram-coffers ────────────────────────────
echo ""
echo "=== Step 1: Clone/update ram-coffers ==="
if [ -d ~/ram-coffers ]; then
    cd ~/ram-coffers && git pull
else
    git clone https://github.com/Scottcjn/ram-coffers.git ~/ram-coffers
fi

# ── Step 2: Build standalone benchmark ─────────────────
echo ""
echo "=== Step 2: Build PSE benchmark ==="
cd ~/ram-coffers/apple-silicon
make clean 2>/dev/null || true
make bench-pse
echo ""
echo "--- Running standalone benchmark ---"
./bench-pse 2>&1 | tee ~/pse-bench-standalone.log
echo ""
echo "Standalone results saved to ~/pse-bench-standalone.log"

# ── Step 3: Get llama.cpp ──────────────────────────────
echo ""
echo "=== Step 3: Clone/update llama.cpp ==="
if [ -d ~/llama-pse ]; then
    cd ~/llama-pse && git pull
else
    git clone https://github.com/ggerganov/llama.cpp.git ~/llama-pse
fi

# ── Step 4: Copy PSE headers ──────────────────────────
echo ""
echo "=== Step 4: Install PSE headers ==="
cp ~/ram-coffers/apple-silicon/apple-pse-config.h     ~/llama-pse/ggml/src/ggml-cpu/
cp ~/ram-coffers/apple-silicon/neon-collapse.h         ~/llama-pse/ggml/src/ggml-cpu/
cp ~/ram-coffers/apple-silicon/aes-entropy-collapse.h  ~/llama-pse/ggml/src/ggml-cpu/
cp ~/ram-coffers/apple-silicon/unified-memory-coffers.h ~/llama-pse/ggml/src/ggml-cpu/
cp ~/ram-coffers/apple-silicon/apple-pse-integration.h ~/llama-pse/ggml/src/ggml-cpu/
echo "PSE headers installed to ~/llama-pse/ggml/src/ggml-cpu/"

# ── Step 5: Build llama.cpp (stock, for comparison) ───
echo ""
echo "=== Step 5: Build stock llama.cpp ==="
cd ~/llama-pse
rm -rf build-stock 2>/dev/null || true
mkdir build-stock && cd build-stock
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
make -j$(sysctl -n hw.ncpu) llama-bench llama-cli 2>&1 | tail -5
echo "Stock build complete"

# ── Step 6: Download test model ───────────────────────
echo ""
echo "=== Step 6: Download test model ==="
mkdir -p ~/models
if [ ! -f ~/models/tinyllama-1.1b-q4.gguf ]; then
    echo "Downloading TinyLlama 1.1B Q4_K_M..."
    curl -L -o ~/models/tinyllama-1.1b-q4.gguf \
        "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
fi

if [ ! -f ~/models/qwen2.5-7b-q4.gguf ]; then
    echo "Downloading Qwen2.5 7B Q4_K_M (fits in 24GB unified)..."
    curl -L -o ~/models/qwen2.5-7b-q4.gguf \
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
fi
echo "Models ready in ~/models/"

# ── Step 7: Run stock benchmark ───────────────────────
echo ""
echo "=== Step 7: Stock Benchmark ==="
echo "--- TinyLlama 1.1B ---"
./bin/llama-bench -m ~/models/tinyllama-1.1b-q4.gguf -t 4 -p 128 -n 32 2>&1 | tee ~/stock-bench.log

if [ -f ~/models/qwen2.5-7b-q4.gguf ]; then
    echo ""
    echo "--- Qwen2.5 7B ---"
    ./bin/llama-bench -m ~/models/qwen2.5-7b-q4.gguf -t 4 -p 128 -n 32 2>&1 | tee -a ~/stock-bench.log
fi

# ── Step 8: Divergence test ───────────────────────────
echo ""
echo "=== Step 8: Divergence Test (stock — should be identical) ==="
for i in 1 2 3; do
    ./bin/llama-cli -m ~/models/tinyllama-1.1b-q4.gguf \
        -p "The meaning of life is" -n 30 --seed 42 --temp 0.7 \
        2>/dev/null > /tmp/stock_run_$i.txt
done
echo "Stock runs MD5:"
md5 /tmp/stock_run_*.txt 2>/dev/null || md5sum /tmp/stock_run_*.txt

# ── Summary ───────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Setup Complete!                              ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Standalone PSE benchmark: ~/pse-bench-standalone.log"
echo "║  Stock llama.cpp benchmark: ~/stock-bench.log"
echo "║                                               ║"
echo "║  Next: Build PSE-enabled llama.cpp and compare ║"
echo "║  See: llama-cpp-patch.md for integration guide ║"
echo "╚══════════════════════════════════════════════╝"
