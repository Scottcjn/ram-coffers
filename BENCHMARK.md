# Benchmark: the 147.54 t/s POWER8 claim

This file exists to answer the question that keeps coming up (issues #667, #662,
#668, and PRs #669, #670, #671): *how do I reproduce the 147.54 t/s POWER8
number in the README, and what exactly was measured?*

Read this before opening a new "please document the benchmark" issue. If
something here is still unclear, that's a real gap, please open an issue
against this file specifically.

## Measured vs. template

This document mixes two kinds of content, marked clearly throughout:

- **MEASURED**: a number or command that is already written down elsewhere in
  this repo (README.md, `benchmarks/BENCHMARK_REPORT.md`, `QUICK_START.md`,
  `CHANGELOG.md`), reported by the author (Scott Boudreaux) from his own
  POWER8 S824 box on the date given. Nobody outside Elyan Labs has
  independently reproduced these numbers yet, and there are no raw
  `llama-bench` logs checked into this repository (`benchmarks/out/` is
  gitignored). Treat "measured" as "the author's self-reported result,"
  not "externally verified."
- **TEMPLATE**: a command you can run yourself, using the harness scripts
  already in this repo, to produce your own comparable numbers. Nobody has
  run these specific invocations for this document; they're the documented
  procedure, not a result.

## Hardware (as documented in this repo, note an unresolved inconsistency)

The README describes the benchmark machine three different ways in three
different places, and this document is not going to silently pick one:

| Location | RAM figure given |
|---|---|
| README.md, line 15 (byline) | "IBM POWER8 S824 (320GB RAM, Dual 8-core)" |
| README.md, line 281 (GPT-OSS 120B section) | "512GB RAM" |
| README.md, Press and References | "768GB IBM POWER8 Server" (dev.to article title) |
| README.md Architecture table (4 coffers) | 193 + 183 + 119 + 62 = 557 GB summed across 4 NUMA nodes |
| `benchmarks/BENCHMARK_REPORT.md` §3.1 | "512 GB DDR3, no GPU" (also says "16 cores / 128 threads") |

What's consistent across every source: **IBM POWER8 S824, dual 8-core POWER8
CPUs (16 cores, SMT8 = 128 hardware threads), CPU-only inference, multi-NUMA
Linux host.** The exact installed/active RAM capacity is documented
inconsistently in this repo and should be confirmed against the actual machine
(`lscpu`, `numactl --hardware`, or `free -g`) rather than trusted from any one
line above. If you're the one running the reproduction, capture and report
your own topology output rather than quoting one of these numbers.

## Model and quantization (MEASURED)

- Model: TinyLlama 1.1B Chat v1.0
- Quantization: Q4_K_M (`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`)
- File size: 638 MB (per `benchmarks/BENCHMARK_REPORT.md` §3.1)
- Default download source used by the harness script: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` on Hugging Face

## Build (MEASURED, general recipe from QUICK_START.md)

`QUICK_START.md` documents this as the build recipe for a RAM Coffers-enabled
`llama.cpp`:

```bash
cd ~/llama.cpp
mkdir build-coffers && cd build-coffers
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP=ON \
    -DCMAKE_C_FLAGS="-mcpu=power8 -mvsx -maltivec -O3" \
    -DCMAKE_CXX_FLAGS="-mcpu=power8 -mvsx -maltivec -O3"
make -j32
```

Prerequisite toolchain per the same file: GCC 10+ with `-mcpu=power8 -mvsx
-maltivec` support.

**Gap:** the exact binary that produced the 147.54 t/s number
(`~/llama.cpp/build-pse-collapse/bin/llama-bench`, per
`benchmarks/BENCHMARK_REPORT.md` §6.1) is a specific pre-built binary on the
author's machine. The repo does not pin the exact CMake invocation, compiler
version, or source commit that produced *that specific* binary, only the
general build recipe above. If you rebuild from the headers in this repo
(`ggml-ram-coffers.h`, `ggml-coffer-mmap.h`, etc., copied into
`llama.cpp/ggml/src/ggml-cpu/arch/powerpc/` per `QUICK_START.md` step 1), you
are reproducing the documented architecture, not byte-for-byte the exact
binary that produced 147.54 t/s.

## Headline numbers (MEASURED, self-reported, Dec 16, 2025)

From README.md "Performance Results" (`pp128` = prompt-eval throughput,
TinyLlama 1.1B Q4_K):

| Configuration | Tokens/sec (pp128) | Speedup vs. stock |
|--------------|-------------------|-------------------|
| Stock llama.cpp | 16.74 | 1.0x |
| + POWER8 VSX | 66.49 | 3.97x |
| + PSE vec_perm Collapse | 84.62 | 5.05x |
| + RAM Coffers + DCBT | **147.54** | **8.81x** |

Per `benchmarks/BENCHMARK_REPORT.md` §3.2, the 8.81x is attributed to three
stacked effects: VSX vectorization (3.97x), thread-count tuning to 64 threads
instead of 128 (1.27x further), and DCBT resident prefetch keeping weight
tensors hot in L2/L3 (1.74x further).

Decode throughput (`tg32`) for the same model, per
`benchmarks/BENCHMARK_REPORT.md` §3.1: **18.88 t/s**. This is reported
separately on purpose: PRs #669 and #671 both flagged that the 147.54
headline is a `pp128` prompt-eval number and should not be quoted as, or mixed
with, decode throughput.

### Thread scaling (MEASURED, from `benchmarks/BENCHMARK_REPORT.md` §3.3)

| Threads | pp128 (t/s) | Per-thread (t/s) |
|---------|-------------|-------------------|
| 16 | 41.55 | 2.60 |
| 32 | 68.06 | 2.13 |
| **64** | **84.62** | **1.32** |
| 96 | 76.54 | 0.80 |
| 128 | 65.83 | 0.51 |

Note this table's 64-thread pp128 figure (84.62) matches the "PSE vec_perm
Collapse" row above, i.e. this table isolates thread scaling before DCBT
resident prefetch is added. The documented reason 64 beats 128: POWER8 SMT8
cache contention past 64 threads.

### DeepSeek-Coder 33B (MEASURED, from `benchmarks/BENCHMARK_REPORT.md` §3.1)

| Model | Quant | Size | pp128 (t/s) | tg32 (t/s) |
|-------|-------|------|-------------|------------|
| DeepSeek-Coder 33B | Q4_K | 18.57 GB | 5.37 | 1.16 |

### QUICK_START.md's own framing (MEASURED, slightly different ratio)

`QUICK_START.md` presents the RAM-Coffers-specific delta only (not the full
stock-to-147.54 chain):

| Model | Without Coffers | With Coffers | Speedup |
|-------|----------------|--------------|---------|
| TinyLlama 1.1B Q4 | 84 t/s | 147 t/s | 1.75x |

This is internally consistent with the 1.74x DCBT contribution above (84.62 to
147.54 is a 1.74x step); the two documents are describing the same step with
independently rounded numbers, not conflicting results.

## The exact reproduction command the author documents running (MEASURED)

From `benchmarks/BENCHMARK_REPORT.md` §6.1, this is the literal command
recorded as producing the headline result, run over SSH against the POWER8
box:

```bash
ssh sophia@100.75.100.89  # POWER8 S824 via Tailscale

export OMP_NUM_THREADS=64
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# PSE + DCBT Resident Prefetch
numactl --interleave=0,1 ~/llama.cpp/build-pse-collapse/bin/llama-bench \
  -m ~/models/tinyllama-1.1b-q4.gguf -t 64 -p 128 -n 32
```

This is an internal, non-public path (`~/llama.cpp/build-pse-collapse/bin/`)
and only works on the author's own machine. It is included here for
transparency about exactly what was run, not as something an outside
contributor can execute directly.

## Reproduce it yourself (TEMPLATE, not yet run for this document)

This repo ships two harness scripts that exist specifically so an outside
POWER8 owner can generate a comparable, falsifiable result instead of taking
the numbers above on faith. Neither was executed as part of writing this
document; what follows is the documented usage, run it and attach your own
output.

### 1. Capture your machine's topology first

```bash
lscpu
numactl --hardware
```

Attach this output to any benchmark report you file. It is the fastest way to
catch whether your box is actually multi-NUMA before spending time on a full
run.

### 2. One-command baseline scaffold

```bash
./benchmark_harness.sh
```

This captures `lscpu` / `numactl --hardware` output, toolchain metadata
(gcc/clang/numactl/lscpu presence), attempts to build
`bench_vcipher_collapse.c`, and writes timestamped files to
`benchmarks/out/`:

- `topology-<timestamp>.txt`
- `env-<timestamp>.txt`
- `benchmark-<timestamp>.md`
- `build-<timestamp>.log`
- `run-<timestamp>.log` (only if the build/run succeeds)

It degrades gracefully on non-POWER8 hosts rather than failing outright, so
you can sanity-check the harness shape on a laptop before touching real
hardware.

### 3. Direct stock-vs-RAM-Coffers comparison

```bash
./benchmark_coffers_vs_llamacpp.sh \
  --threads 64 \
  --stock-bin /opt/llama.cpp-stock/build/bin/llama-bench \
  --stock-commit "$(git -C /opt/llama.cpp-stock rev-parse HEAD)" \
  --coffers-bin /opt/llama.cpp-coffers/build/bin/llama-bench \
  --coffers-commit "$(git -C /opt/llama.cpp-coffers rev-parse HEAD)"
```

What it does, per the script's own `--help` and source:

- Downloads TinyLlama 1.1B Chat Q4_K_M from Hugging Face unless `--model`
  points at an existing local GGUF.
- Builds a stock `llama.cpp` (from `LLAMA_CPP_REF`, default `master`) if
  `--stock-bin` isn't supplied. It deliberately refuses to synthesize a RAM
  Coffers binary from copied headers alone; you must pass `--coffers-bin`
  pointing at a real hand-patched build, plus `--coffers-commit` for the
  source commit.
- Runs `llama-bench` with `pp128` and `tg32`, `RUNS=3` repetitions by default.
- Pins the stock run to `STOCK_NUMA_NODE=0` (single node) by default; the RAM
  Coffers run is expected to use an interleaved/coffer-aware NUMA placement
  from your own build.
- Writes topology snapshot, environment snapshot, raw logs, and a markdown
  comparison table to `benchmarks/out/`.

Relevant environment variables (from the script's `--help`):
`BENCH_ROOT`, `OUT_DIR`, `LLAMA_CPP_REF`, `MODEL_URL`, `RUNS`,
`STOCK_NUMA_NODE`, `STOCK_LLAMA_COMMIT`, `COFFERS_LLAMA_COMMIT`, `JOBS`.

Preview the plan without downloading, building, or running anything:

```bash
./benchmark_coffers_vs_llamacpp.sh --dry-run
```

### 4. What to report back

Matching the checklist requested in issues #667 and #662, a claim-quality
report should include:

- Exact hardware: CPU model, core/thread count, RAM, NUMA node count and
  sizes (`lscpu` + `numactl --hardware` output, not a prose summary)
- Model file name, quantization, and file size
- `llama.cpp` commit for both the stock and RAM Coffers builds
- Compiler and exact flags used for each build (e.g. `-mcpu=power8 -mvsx
  -maltivec -O3`, plus `-mcrypto` if the vcipher path is in use)
- The exact `benchmark_coffers_vs_llamacpp.sh` invocation, including
  `--threads`, any NUMA overrides, and `RUNS`
- `pp128` and `tg32` reported **separately**, do not average or combine them
  into one headline number
- The raw `benchmarks/out/` files (topology, env, and markdown report) attached
  to the PR or issue, not just the final numbers

This keeps performance claims falsifiable, and it's exactly what let this
document distinguish "measured" from "template" in the first place.

## Related issues and PRs

- #667 / #662: requests for a reproducible benchmark recipe (this file)
- #668: requested a `--plan` dry-run NUMA placement mode, implemented in
  `ram_coffers_topology.py --plan`
- #669, #671: README-level reproduction guidance, both merged, and this file
  is now the canonical detail doc they point to
- #670: the `--plan` dry-run implementation for #668
