# Fallback Behavior and Portability Contract

This document is the canonical reference for fallback behavior, platform scope,
and what this repository does **not** promise implicitly.

## Why this exists

`README.md` mentions fallback behavior, Apple Silicon, and smoke-test guidance.
Those statements are easy to over-read as a repository-wide portability claim.
That is not the intent.

Portability must be read **per component/header**, because the repository mixes:
- POWER8-specific acceleration paths
- libnuma-backed host headers
- runtime single-NUMA fallback logic
- a separate Apple Silicon port with unified-memory/cache-tier coffers

## Support levels

- **Supported performance path** — the documented primary target for that component.
- **Correctness-only / degraded path** — code can still run or be smoke-tested, but not as a headline performance configuration.
- **Does not compile without platform-specific changes** — the source makes build-time assumptions the target does not satisfy.

## Per-component matrix

| Component | Compile dependencies | Runtime fallback shape | Linux x86_64 | Linux aarch64 | macOS | Windows |
|---|---|---|---|---|---|---|
| `ggml-ram-coffers.h` | C compiler + `libnuma` headers (`<numa.h>`) | Checks `numa_available()` and falls back to node 0 / non-placement behavior when NUMA is unavailable at runtime | **Correctness-only / degraded** if built with libnuma and run on single-NUMA or no-NUMA Linux; POWER8-specific acceleration remains conditional | **Correctness-only / degraded** under the same Linux+libnuma constraint | **Does not compile without platform-specific changes** (`<numa.h>` required) | **Does not compile without platform-specific changes** (`<numa.h>` required) |
| `ggml-ram-coffer.h` | C compiler + `libnuma` headers (`<numa.h>`) | Checks `numa_available()` before NUMA placement decisions | **Correctness-only / degraded** if built on Linux with libnuma available | **Correctness-only / degraded** if built on Linux with libnuma available | **Does not compile without platform-specific changes** | **Does not compile without platform-specific changes** |
| `ggml-coffer-mmap.h` | C compiler + `libnuma` headers (`<numa.h>`), mmap-capable host | Checks `numa_available()` before NUMA-aware mapping/placement | **Correctness-only / degraded** if built on Linux with libnuma available | **Correctness-only / degraded** if built on Linux with libnuma available | **Does not compile without platform-specific changes** | **Does not compile without platform-specific changes** |
| POWER8 collapse / compatibility headers (`power8-compat.h`, `ggml-topk-collapse-vsx.h`, POWER8 branches inside `ggml-ram-coffers.h`) | PowerPC/POWER toolchain features for the fast path | ISA-specific blocks are compile-guarded; non-POWER builds skip those acceleration paths rather than inheriting equivalent performance | **Correctness-only / degraded** where guarded generic code exists | **Correctness-only / degraded** where guarded generic code exists | **Does not imply support for the POWER8 fast path** | **Does not imply support for the POWER8 fast path** |
| Apple Silicon port (`apple-silicon/`) | Apple/ARM NEON + AES capable environment; see `apple-silicon/README.md` | Separate cache-tier coffer design, not NUMA fallback | N/A | **Separate port** (Linux aarch64 bench/compat mode described in that subtree) | **Supported performance path** for the Apple Silicon subtree | N/A |
| POWER8-only benchmark/harness claims | Verified POWER8 host, real multi-node topology, explicit benchmark inputs | `--allow-single-numa` and similar paths are smoke-test aids, not headline-performance substitutes | **Correctness-only / degraded** for smoke tests | **Correctness-only / degraded** for smoke tests | **Not a supported reproduction target** | **Not a supported reproduction target** |

## What the runtime checks do — and do not mean

Several headers call `numa_available()` at runtime. That is a **runtime fallback**,
not a universal build portability promise.

In particular:
- `ggml-ram-coffers.h`, `ggml-ram-coffer.h`, and `ggml-coffer-mmap.h` all still
  include `<numa.h>` directly.
- That means a target without libnuma headers will fail at compile time unless
  someone adds platform-specific guards or substitutes.
- Therefore, statements like "falls back when NUMA is unavailable" must be read
  as **"after a successful Linux/libnuma build, on a machine whose runtime
  topology lacks usable NUMA"**, not as "builds anywhere."

## Apple Silicon is a separate design track

Apple Silicon support should not be read as proof that the libnuma-backed
headers are portable to macOS unchanged.

The Apple port lives in [`apple-silicon/`](../apple-silicon/README.md) and swaps
NUMA-node coffers for cache-tier/unified-memory coffers. That is a distinct
implementation path.

## Benchmark/reporting rules

When reporting results:
- Identify the exact component/header you built.
- State whether the run used the supported performance path or only a
  correctness/smoke-test path.
- Do not present single-NUMA or non-POWER8 smoke tests as proof of the README's
  POWER8 headline numbers.
- If a run required `--allow-single-numa`, say so explicitly.

## Provenance

This document consolidates and supersedes scattered fallback guidance introduced
around issue #664, while preserving that issue as the provenance for the first
round of fallback documentation work.
