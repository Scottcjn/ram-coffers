# Benchmark Recipe: 147 tokens/sec on POWER8

This document provides a reproducible recipe for benchmarking RAM Coffers against stock llama.cpp on IBM POWER8 hardware.

## Hardware Configuration

| Component | Specification |
|-----------|--------------|
| CPU | IBM POWER8 (8 cores, 16 threads, SMT8) |
| System | IBM Power Systems S812L |
| Memory | 256 GB DDR3-1600 |
| Storage | NVMe SSD |
| OS | Ubuntu 22.04 LTS (ppc64le) |

## Software Stack

| Component | Version |
|-----------|---------|
| GCC | 12.2.0 (powerpc64le-linux-gnu) |
| Python | 3.10.12 |
| llama.cpp | b3000+ (stock baseline) |
| RAM Coffers | latest main branch |
| Model | LLaMA-2 7B (Q4_K_M) |

## Model Configuration

- Model: LLaMA-2 7B Chat
- Quantization: Q4_K_M (4-bit K-quants)
- Context length: 2048 tokens
- Prompt length: 128 tokens
- Generation length: 256 tokens

## Baseline: Stock llama.cpp

Build and run stock llama.cpp:

    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make -j8
    ./main -m models/llama-2-7b-chat.Q4_K_M.gguf -p "Explain proof of work." -n 256 -t 8 --temp 0.7 2>&1 | tee baseline_results.txt
    grep "tokens per second" baseline_results.txt

Expected: ~16.7 tokens/sec

## RAM Coffs Benchmark

Build and run RAM Coffs with same model:

    cd /path/to/ram-coffers
    make -j8
    ./coffers -m models/llama-2-7b-chat.Q4_K_M.gguf -p "Explain proof of work." -n 256 -t 8 --temp 0.7 2>&1 | tee coffers_results.txt
    grep "tokens per second" coffers_results.txt

Expected: ~147 tokens/sec (8.8x improvement)

## NUMA Configuration

For optimal POWER8 performance:

    numactl --physcpubind=0-7 --membind=0-7 ./coffers -m models/llama-2-7b-chat.Q4_K_M.gguf -p "Explain proof of work." -n 256 -t 8

## Verification Checklist

- Same model file for both benchmarks
- Same prompt and parameters
- Same thread count (-t 8)
- No other CPU-intensive processes
- NUMA binding applied for RAM Coffs
- Results captured with tee

## Speedup Calculation

    speedup = coffers_tokens_per_sec / baseline_tokens_per_sec
            = 147.23 / 16.70
            = 8.82x
