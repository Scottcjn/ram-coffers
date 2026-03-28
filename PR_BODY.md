## Summary

Adds a minimal reproducible benchmark harness and sample topology workflow for RAM Coffers contributors.

## What this PR adds

- `benchmark_harness.sh`
  - captures machine topology (`lscpu`, `numactl --hardware` when available)
  - captures environment/toolchain metadata
  - attempts to build the POWER8 benchmark binary
  - writes a standardized markdown report to `benchmarks/out/`
  - falls back cleanly on unsupported non-POWER8 hosts instead of failing silently
- `benchmarks/README_HARNESS.md`
  - explains purpose, outputs, and usage
- README update
  - documents the one-command harness entrypoint in the contributor benchmark section

## Why

Issue #35 asks for a minimal benchmark harness plus sample topology output. This PR provides a practical contributor entrypoint that standardizes the workflow and reporting format, even for contributors validating structure on non-POWER8 hosts before running on target hardware.

## Validation

Run locally:

```bash
./benchmark_harness.sh
```

Verified on this host that the harness:
- creates topology/environment/report artifacts
- produces a fallback report when POWER8-specific compilation flags are unsupported

## Files changed

- `README.md`
- `benchmark_harness.sh`
- `benchmarks/README_HARNESS.md`
