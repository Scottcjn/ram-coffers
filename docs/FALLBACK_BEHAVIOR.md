# Fallback Behavior

RAM Coffers is designed and measured on an IBM POWER8 S824 with four NUMA nodes. This document explains what happens when you run it anywhere else — single-socket x86_64 VMs, Apple Silicon laptops, ARM64 servers, or POWER8 boxes with only one NUMA node. It also gives you a minimal smoke test to verify routing correctness without claiming the POWER8 headline performance numbers.

---

## What works everywhere

The core of RAM Coffers is plain C:

- **Resonance routing** (`route_to_coffer`) uses cosine similarity between a query embedding and coffer domain signatures.
- **Coffer state** (`ram_coffer_t`) tracks which shards are loaded, active, and how many times each was chosen.
- **Prune planning** (`coffer_plan_prune`) is a heuristic block-level skip mask.

These routines do not depend on POWER8 instructions, NUMA libraries, or multi-socket topology. On a single-node or non-POWER machine the system keeps the same semantics: pick the best loaded coffer, activate it, and run inference. Only the performance-oriented optimizations degrade or disappear.

---

## Single-NUMA-node systems

On a host with one NUMA node (most laptops, desktops, cloud VMs, and single-socket servers), the multi-bank topology collapses to a single bank.

### What `coffer_init_numa()` does

```c
int n_nodes = numa_num_configured_nodes();
for (int c = 0; c < MAX_COFFERS && c < n_nodes; c++) { ... }
```

`MAX_COFFERS` is 4, but the loop only initializes as many coffers as there are NUMA nodes. On a single-node system only **Coffer 0** is populated; coffers 1–3 stay zeroed (`is_loaded = 0`).

### Effect on routing

`route_to_coffer()` skips any coffer where `is_loaded == 0`:

```c
for (int c = 0; c < MAX_COFFERS; c++) {
    if (!g_coffers[c].is_loaded) continue;
    ...
}
```

So on a single-NUMA host every query routes to **Coffer 0**. The resonance math still runs, but there is only one candidate.

### Effect on activation and memory placement

- `numa_run_on_node(coffer->numa_node)` is called only under `#ifdef __linux__`. On a single-node system the call is effectively a no-op because the thread is already on node 0.
- `set_mempolicy(MPOL_BIND, ...)` binds to the single-node mask, so allocations land on node 0. This is correct behavior, just without cross-node routing.
- `mbind()`-style page migration is never needed because there is no alternative node.

A warning banner is printed during initialization:

```
║  WARNING: Running without NUMA support                      ║
```

Execution continues normally.

---

## Non-POWER8 architectures

Three POWER8-specific primitives are used in the codebase. Each has a compile-time fallback.

### 1. DCBT prefetch (`ggml-ram-coffers.h` lines 103–111)

```c
#if defined(__powerpc64__) || defined(__powerpc__)
#define DCBT_PREFETCH(addr) __asm__ __volatile__("dcbt 0,%0" : : "r"(addr))
#else
#define DCBT_PREFETCH(addr) (void)(addr)
#endif
```

On non-POWER architectures every `DCBT_PREFETCH` becomes a no-op that evaluates its argument. The `dcbt_resident()` loop still iterates, but it does not actually warm the cache. You keep correctness; you lose the prefetch speedup.

### 2. AltiVec/VSX vector dot product (`ggml-ram-coffers.h` lines 202–226)

```c
#if defined(__powerpc64__) || defined(__powerpc__)
    // AltiVec vector path
#else
    for (int d = 0; d < dim; d++) {
        sum += a[d] * b[d];
    }
#endif
```

On x86_64, ARM64, and other non-POWER ISAs the fallback is a scalar C loop. `cosine_similarity()` and `magnitude()` inherit this fallback, so routing still works but without SIMD acceleration.

**Note:** `ggml-topk-collapse-vsx.h` includes `<altivec.h>` unconditionally and will **not compile** on non-POWER toolchains. That file is only relevant on POWER8 systems.

### 3. PowerPC timebase (`mftb`) entropy

POWER8 uses `mftb` for hardware entropy injection in PSE collapse. The fallback chain is:

| File | POWER8 | x86_64 | aarch64 | Other |
|------|--------|--------|---------|-------|
| `pse-entropy-burst.h` | `mftb` | `rdtsc` | `cntvct_el0` | address of a static variable (weak entropy) |

Entropy-dependent features degrade to deterministic behavior where no hardware counter is available.

---

## Build behavior by platform

| Platform | Compiles | NUMA | DCBT | Vector dot | Notes |
|----------|----------|------|------|------------|-------|
| POWER8 multi-node Linux | ✅ | Full 4-node | Native `dcbt` | AltiVec/VSX | Primary target; 147 t/s claim |
| POWER8 single-node Linux | ✅ | Single node | Native `dcbt` | AltiVec/VSX | Coffer 0 only; correct but no NUMA speedup |
| POWER9/POWER10 Linux | ✅ | Full or single | Native `dcbt` | AltiVec/VSX | Use `-mcpu=power8`; do **not** define `__POWER9_VECTOR__` |
| x86_64 Linux | ✅ with libnuma | If available | No-op | Scalar loop | `ggml-ram-coffers.h` compiles; `ggml-topk-collapse-vsx.h` does not |
| ARM64 Linux | ✅ with libnuma | If available | No-op | Scalar loop | Same caveat as x86_64 |
| Apple Silicon macOS | ⚠️ partial | N/A | No-op | Scalar loop | `ggml-ram-coffer.h` / `ggml-coffer-mmap.h` fail due to unconditional `<numa.h>`; use `apple-silicon/` port |

---

## Expected benchmark collapse

The README headline of **147 tokens/sec** on TinyLlama 1.1B Q4_K is a POWER8 S824 measurement with all optimizations enabled. On other hardware you should expect:

| Optimization | What drops | Approximate impact |
|--------------|-----------|-------------------|
| NUMA-aware shard placement | Single-node allocation | Removes the multi-bank memory-bandwidth win |
| `dcbt` resident prefetch | Cache warming becomes no-op | Removes significant prefetch throughput benefit |
| AltiVec/VSX dot product | Scalar loop | Slower resonance routing and embedding math |
| `mftb` entropy | Deterministic fallback | PSE collapse loses hardware jitter |

**Bottom line:** on x86_64/ARM64 you can verify that routing, loading, and activation behave correctly, but you should not expect to reproduce the 147 t/s number or any POWER8-specific performance claim. Treat off-POWER runs as **correctness validation only**.

---

## Startup detection checklist

At runtime the library already prints several signals that tell you which mode you are in:

1. **NUMA detection**
   - `Coffers: %d NUMA nodes detected` — 1 means single-node fallback.
   - `WARNING: Running without NUMA support` — `numa_available()` returned negative or initialization failed.

2. **Architecture detection**
   - `ggml-ram-coffers.h` does not print the architecture, but you can check at compile time with:
     ```bash
     gcc -dM -E - < /dev/null | grep -E '__x86_64__|__aarch64__|__powerpc64__'
     ```
   - On POWER8/POWER9/POWER10 you should see `__powerpc64__` defined.

3. **Loaded coffers**
   - `Loaded %d coffer shards` — if this is 1 on a multi-coffer config, you are in single-bank mode.

4. **Routing result**
   - `coffer_test_routing()` prints which coffer each sample query chose. In single-node mode every query returns `0`.

---

## Smoke test without performance claims

The repository includes a self-contained routing test in `ggml-ram-coffers.h`:

```c
static void coffer_test_routing(void);
```

You can wrap it in a tiny C program that does not need a GGUF file or a POWER8 machine:

```c
/* smoke_test.c */
#define COFFER_TEST_MAIN
#include "ggml-ram-coffers.h"

int main(void) {
    const char* paths[MAX_COFFERS] = {NULL};
    init_ram_coffers(paths);          /* no shards loaded, but domains init */
    coffer_test_routing();
    shutdown_ram_coffers();
    return 0;
}
```

Build and run on Linux:

```bash
gcc -O2 -std=c11 -lm -lnuma smoke_test.c -o smoke_test
./smoke_test
```

Expected output on a single-NUMA x86_64 machine:

```
╔═══════════════════════════════════════════════════════════════╗
║  RAM Coffers System - POWER8 S824 NUMA Weight Banking        ║
╠═══════════════════════════════════════════════════════════════╣
║  WARNING: Running without NUMA support                      ║
║  Loaded 0 coffer shards                                      ║
╚═══════════════════════════════════════════════════════════════╝

=== Coffer Routing Test ===
General query → Coffer 0
Science query → Coffer 0
Creative query → Coffer 0
=== Test Complete ===
```

The important signal is that routing completes without crashing and consistently returns Coffer 0 when no additional coffers are loaded.

---

## POWER8 compatibility layer (`power8-compat.h`)

`power8-compat.h` provides POWER9-style builtins for POWER8 targets:

| Macro | Fallback |
|-------|----------|
| `vec_xl(offset, ptr)` | `vec_ld` (requires 16-byte alignment) |
| `vec_xst(v, offset, ptr)` | `vec_st` (requires 16-byte alignment) |
| `vec_xl_len(ptr, len)` | `memcpy` into aligned buffer, then `vec_ld` |

It is gated by:

```c
#if defined(__POWER8_VECTOR__) && !defined(__POWER9_VECTOR__)
```

**Do not define `__POWER9_VECTOR__`** when targeting POWER8. Doing so lets the compiler emit POWER9-only opcodes that will SIGILL on POWER8 hardware.

---

## Summary

- RAM Coffers degrades gracefully: routing and correctness logic are portable C.
- Single-NUMA systems run as a one-bank configuration.
- Non-POWER systems lose `dcbt`, AltiVec, and `mftb` optimizations but keep scalar correctness.
- Off-POWER runs are for validation and development, not for reproducing POWER8 performance claims.
- Use the smoke test above to verify that the coffer system initializes, routes, and shuts down cleanly on your machine.