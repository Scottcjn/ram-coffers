# Fallback Behavior on Non-POWER8 and Single-NUMA Systems

RAM Coffers is designed and benchmarked on an **IBM POWER8 S824** with four NUMA nodes. This document describes what happens when you build or run the code on less exotic hardware, so contributors can do a correctness smoke test before touching POWER8-specific paths.

## Quick Summary

| Platform | Build | NUMA routing | DCBT prefetch | Expected performance |
|---|---|---|---|---|
| POWER8 multi-node (S824) | ✅ native | ✅ full 4-node topology | ✅ `dcbt` assembly | 147 tok/s headline |
| POWER8 single-node | ⚠️ compiles | ⚠️ degraded (hardcoded node 3) | ✅ `dcbt` assembly | Unknown / untested |
| x86_64 Linux with libnuma | ✅ compiles | ⚠️ topology mismatch | ❌ no-op | CPU-bound, no NUMA benefit |
| aarch64 Linux with libnuma | ✅ compiles | ⚠️ topology mismatch | ❌ no-op | CPU-bound, no NUMA benefit |
| Any Linux without libnuma-dev | ❌ fails | — | — | Install `libnuma-dev` / `numactl-devel` first |
| Non-Linux | ⚠️ may compile | ❌ skipped entirely (`#ifdef __linux__`) | ❌ skipped | Untested |

## 1. Single-NUMA-Node Systems

The routing tables in `ggml-ram-coffers.h` assume four nodes:

```c
// ggml-ram-coffers.h:50-51
static const int NUMA_TO_COFFER[4] = {2, 1, 3, 0};
static const int COFFER_TO_NUMA[4] = {3, 1, 0, 2};
```

`coffer_init_numa()` only initializes as many coffers as the host reports nodes:

```c
// ggml-ram-coffers.h:274-279
int n_nodes = numa_num_configured_nodes();
for (int c = 0; c < MAX_COFFERS && c < n_nodes; c++) {
    g_coffers[c].coffer_id = c;
    g_coffers[c].numa_node = COFFER_TO_NUMA[c];
    ...
}
```

On a single-node machine, **only Coffer 0 is initialized**, and its `numa_node` is set to `COFFER_TO_NUMA[0] = 3` (the physical node that holds the heavy/general coffer on POWER8).

Consequences:

- `coffer_load_shard(0, ...)` calls `numa_run_on_node(3)` (`ggml-ram-coffers.h:296`). On a single-node host this binds the thread to a non-existent node; the kernel or libnuma may ignore it or return `EINVAL`.
- `coffer_migrate_region()` checks `numa_available() < 0` and returns early only when NUMA is completely unavailable (`ggml-coffer-mmap.h:284-291`). If libnuma *is* present but only one node exists, `mbind()` is still attempted with node mask `1UL << 3`, which will likely fail because node 3 does not exist.
- `activate_coffer()` still calls `numa_run_on_node(3)` (`ggml-ram-coffers.h:426`).

In practice, on a single-node Linux box with libnuma installed, the code compiles and the non-NUMA code paths (mmap, file I/O, routing math) still run, but the NUMA placement calls are best-effort and may silently fail or print warnings depending on kernel/libnuma behavior.

## 2. Non-POWER8 Architectures

The only POWER8-specific instructions currently emitted are the **DCBT prefetch macros**:

```c
// ggml-ram-coffers.h:103-110
#if defined(__powerpc64__) || defined(__powerpc__)
#define DCBT_PREFETCH(addr) __asm__ __volatile__("dcbt 0,%0" : : "r"(addr))
...
#else
#define DCBT_PREFETCH(addr) (void)(addr)
#define DCBT_STREAM_START(addr, id) (void)(addr)
#define DCBT_STREAM_STOP(id) (void)0
#endif
```

On x86_64, ARM64, or Apple Silicon, these macros compile to no-ops, so correctness is preserved but the cache-warming optimization is lost. The headline benchmark (147 tok/s) relies on DCBT + POWER8 cache line sizes and should be expected to collapse to standard CPU-bound throughput.

Other ISA-specific primitives mentioned in earlier write-ups (`mftb`, `vec_perm`) are **not present in the current headers**. The closest portable equivalent is the AES path in `ggml-vcipher-collapse.h`.

## 3. Build Requirements and Matrix

On Linux, `ggml-ram-coffers.h` and `ggml-coffer-mmap.h` unconditionally include `<numa.h>` and `<numaif.h>` inside `#ifdef __linux__` blocks:

```c
// ggml-ram-coffers.h:35-37
#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif
```

Therefore:

| OS / toolchain | Needs libnuma headers | Expected result |
|---|---|---|
| Debian/Ubuntu x86_64 | `sudo apt install libnuma-dev` | Compiles; DCBT no-op |
| Fedora/RHEL x86_64 | `sudo dnf install numactl-devel` | Compiles; DCBT no-op |
| Debian/Ubuntu aarch64 | `sudo apt install libnuma-dev` | Compiles; DCBT no-op |
| macOS | N/A (not `__linux__`) | NUMA code skipped; untested |
| Windows MSVC/MinGW | N/A (not `__linux__`) | NUMA code skipped; untested |

## 4. Minimum Smoke Test for Correctness

There is no dedicated standalone test harness in this repo, but you can verify that the headers compile and the routing tables are initialized with a tiny C program:

```c
// smoke_test.c
#define VITE_TELEMETRY_ENDPOINT ""
#include "ggml-ram-coffers.h"
#include <stdio.h>

int main(void) {
    printf("MAX_COFFERS = %d\n", MAX_COFFERS);
    printf("NUMA_TO_COFFER[0..3] = %d %d %d %d\n",
           NUMA_TO_COFFER[0], NUMA_TO_COFFER[1],
           NUMA_TO_COFFER[2], NUMA_TO_COFFER[3]);
    printf("COFFER_TO_NUMA[0..3] = %d %d %d %d\n",
           COFFER_TO_NUMA[0], COFFER_TO_NUMA[1],
           COFFER_TO_NUMA[2], COFFER_TO_NUMA[3]);
    return 0;
}
```

Build and run on a non-POWER8 Linux box:

```bash
gcc -I. smoke_test.c -o smoke_test -lnuma
./smoke_test
```

Expected output:

```text
MAX_COFFERS = 4
NUMA_TO_COFFER[0..3] = 2 1 3 0
COFFER_TO_NUMA[0..3] = 3 1 0 2
```

This proves the header compiles and the routing tables are loaded, but it does **not** exercise mmap, NUMA placement, or DCBT acceleration.

## 5. Detecting Degraded Mode at Startup

A runtime health check can be added around `coffer_init_numa()`:

1. Call `numa_num_configured_nodes()`.
2. If `n_nodes < 4`, log a warning: "POWER8 S824 4-node topology not detected; NUMA placement will be degraded."
3. If `n_nodes == 1`, additionally warn: "Single-node host detected; numa_run_on_node/mbind may target non-existent nodes."
4. Check `__powerpc64__` / `__powerpc__` at compile time; if undefined, warn that DCBT prefetch is disabled.

Currently the repo does not emit these warnings automatically; contributors adding a startup probe should place it immediately after `coffer_init_numa()`.

## 6. What Is NOT Expected to Work

- **Performance parity**: x86_64/ARM64 smoke tests validate correctness only; the POWER8-specific throughput numbers will not reproduce.
- **4-coffer load on single-node**: Only Coffer 0 is initialized; attempts to load Coffer 1-3 will access uninitialized state.
- **NUMA page migration on mismatched topology**: `mbind()` with node masks based on the POWER8 table is not portable.

## 7. Recommended Contributor Path

1. Build on your local machine (install `libnuma-dev` on Linux).
2. Run the `smoke_test.c` snippet above to confirm header compilation.
3. Make changes inside `#if defined(__powerpc64__) || defined(__powerpc__)` blocks when adding POWER8-only optimizations.
4. Keep non-POWER8 paths as no-ops or scalar fallbacks so smoke tests continue to compile and run.
5. Benchmark on POWER8 S824 hardware before claiming performance improvements.
