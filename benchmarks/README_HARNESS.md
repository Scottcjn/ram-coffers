# Benchmark Harness

This repository now includes a lightweight reproducible benchmark harness entrypoint:

- `benchmark_harness.sh`
- `benchmark_coffers_vs_llamacpp.sh`

## What it does

1. Captures machine topology
   - `lscpu` (when available)
   - `numactl --hardware` (when available)
2. Captures toolchain/environment metadata
3. Attempts to build the POWER8 benchmark (`bench_vcipher_collapse.c`)
4. Generates a markdown benchmark report in `benchmarks/out/`
5. Falls back cleanly on unsupported hosts while still preserving reproducible environment data

## Output files

Each run creates timestamped artifacts in `benchmarks/out/`:

- `topology-<timestamp>.txt`
- `env-<timestamp>.txt`
- `benchmark-<timestamp>.md`
- `build-<timestamp>.log`
- `run-<timestamp>.log` (only when benchmark execution succeeds)

## Run

```bash
./benchmark_harness.sh
```

## Why this helps

The issue asks for a minimal benchmark harness and sample topology output. This script creates a single repeatable entrypoint for that workflow and standardizes the reporting format for PR review.

It is especially useful because this repository targets specialized POWER8 hardware, while many contributors will be validating structure and portability on non-POWER8 hosts first.

## Coffers vs stock llama.cpp benchmark

For the #45 bounty workflow, use the dedicated comparison harness:

```bash
./benchmark_coffers_vs_llamacpp.sh
```

It downloads TinyLlama Q4, prepares stock and RAM Coffers llama.cpp benchmark
binaries when not supplied, runs `llama-bench` with `pp128` and `tg32`, and
writes a markdown table to `benchmarks/out/`.

On a POWER8 host with an existing hand-patched llama.cpp tree, pass the exact
binaries explicitly:

```bash
./benchmark_coffers_vs_llamacpp.sh \
  --stock-bin /opt/llama.cpp-stock/build/bin/llama-bench \
  --coffers-bin /opt/llama.cpp-coffers/build/bin/llama-bench
```

For CI or review on non-Linux hosts, inspect the generated command/report shape:

```bash
./benchmark_coffers_vs_llamacpp.sh --dry-run
```
