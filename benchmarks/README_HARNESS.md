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
./benchmark_coffers_vs_llamacpp.sh \
  --coffers-bin /opt/llama.cpp-coffers/build/bin/llama-bench \
  --coffers-commit "$(git -C /opt/llama.cpp-coffers rev-parse HEAD)"
```

It downloads TinyLlama Q4, prepares the stock llama.cpp benchmark binary when
not supplied, runs `llama-bench` with `pp128` and `tg32`, and writes a markdown
table to `benchmarks/out/`.

## POWER8 result reporting checklist

For a claim-quality reproduction of the README headline, include these fields
with the generated report:

- Model: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` or the exact alternate GGUF path
- Benchmark shape: `pp128` prompt-eval and `tg32` decode reported separately
- Stock binary path and stock llama.cpp commit
- RAM Coffers binary path and RAM Coffers-patched llama.cpp commit
- `RUNS`, `THREADS`, and `STOCK_NUMA_NODE`
- `lscpu` and `numactl --hardware` output from the generated topology file
- Compiler/CMake settings, especially POWER8 flags such as `-mcpu=power8`,
  `-mvsx`, `-maltivec`, and `-mcrypto` when used
- Whether the run used real multi-NUMA POWER8 hardware or `--allow-single-numa`
  for a smoke test

The README's 147.54 t/s figure is a `pp128` prompt-eval throughput number. Do
not compare it to `tg32` decode throughput or to a mixed prompt-plus-generation
average. Non-POWER8 and single-NUMA runs are useful for checking the harness
shape, but they are not POWER8 performance reproductions.

The harness intentionally refuses to build a synthetic RAM Coffers binary from
copied headers alone. Pass `--coffers-bin` for a verified hand-patched
llama.cpp build and `--coffers-commit` for the exact source commit used to build
it.

On a POWER8 host with an existing hand-patched llama.cpp tree, pass the exact
binaries explicitly:

```bash
./benchmark_coffers_vs_llamacpp.sh \
  --stock-bin /opt/llama.cpp-stock/build/bin/llama-bench \
  --stock-commit "$(git -C /opt/llama.cpp-stock rev-parse HEAD)" \
  --coffers-bin /opt/llama.cpp-coffers/build/bin/llama-bench \
  --coffers-commit "$(git -C /opt/llama.cpp-coffers rev-parse HEAD)"
```

For CI or review on non-Linux hosts, inspect the generated command/report shape:

```bash
./benchmark_coffers_vs_llamacpp.sh --dry-run
```
