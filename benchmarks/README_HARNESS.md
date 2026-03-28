# Benchmark Harness

This repository now includes a lightweight reproducible benchmark harness entrypoint:

- `benchmark_harness.sh`

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
