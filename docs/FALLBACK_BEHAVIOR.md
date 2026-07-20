# RAM Coffers — Fallback Behavior

## Overview

RAM Coffers is designed and tested on an **IBM POWER8 S824** (4 NUMA nodes, 544 GB free RAM).
This document describes what happens on other architectures and NUMA configurations.

---

## Platform Compatibility Matrix

| Architecture | NUMA Nodes | Status | Details |
|---|---|---|---|
| POWER8 ppc64le | 4 (S824) | ✅ **Primary target** | Full coffer routing, DCBT prefetch, vec_perm acceleration |
| POWER8 ppc64le | 1 (single-node) | ⚠️ Fallback | Coffers collapse to node 0; resonance routing still functions |
| POWER9+ ppc64le | Any | ⚠️ Compile (untested) | `__POWER9_VECTOR__` defines different vec_xl — compile fails unless compat macros used |
| x86_64 | Any | ⚠️ Partial | `numa.h` available on Linux; POWER8 vector intrinsics (`vec_perm`, `vec_ld`) not available — compile error on POWER-specific code |
| aarch64 (Apple Silicon) | 1 (cluster) | ⚠️ Partial | No `numa.h`; POWER8 intrinsics unavailable; scalar paths may apply |
| macOS (any arch) | N/A | ❌ | `numa.h` not available; `mmap` behavior differs; not supported |
| Windows | N/A | ❌ | Not supported (no `numa.h`, no POSIX mmap) |

---

## Fallback Scenarios

### 1. Single NUMA Node (non-POWER8 or single-socket)

When only one NUMA node is available:

- **Coffer routing** collapses: all 4 coffers map to node 0. The `NUMA_TO_COFFER` and `COFFER_TO_NUMA` arrays become identity mappings.
- **Resonance routing** still works — domain embeddings are compared and matched. The physical memory binding is a no-op since all allocations come from the same node.
- **Performance**: No NUMA-local memory benefit. Expect lower throughput than multi-node POWER8.

The relevant guard:
```c
#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif
```

On single-node systems `numa_max_node()` returns 0, and `numa_run_on_node()` becomes a no-op.

### 2. Non-POWER8 Architectures (x86_64, aarch64)

On non-POWER8 systems:

- **POWER8-specific intrinsics** (`vec_perm`, `vec_ld`, `vec_ste`, `__builtin_mftb`, `__builtin_dcbt`) are guarded by `#ifdef __POWER8_VECTOR__`. These code paths will **not compile** on non-PPC targets.
- The `power8-compat.h` header provides `vec_xl` / `vec_xst` / `vec_xl_len` macros for POWER8 only. It is **not** a cross-platform compatibility layer.
- On x86_64, NUMA functions (`numa.h`) are available on Linux, so the coffer topology discovery compiles, but vector paths fail to compile.
- Files with pure-C fallback paths (e.g., scalar GGML operations from `ggml-ram-coffer.h`) may compile if they avoid `#ifdef __POWER8_VECTOR__` blocks.

### 3. macOS

RAM Coffers is not supported on macOS:

- `numa.h` does not exist on macOS.
- `mmap` with `MAP_HUGETLB` (used in coffer allocation) is Linux-specific.
- `sched.h` CPU affinity calls differ.

The build will fail at the first `#include <numa.h>` on macOS.

### 4. POWER9+ (ppc64le)

POWER9 and POWER10 define `__POWER9_VECTOR__` instead of `__POWER8_VECTOR__`:

- The `power8-compat.h` header has an `#if defined(__POWER8_VECTOR__) && !defined(__POWER9_VECTOR__)` guard, so POWER9 will **not** use the compat macros.
- POWER9 has its own `vec_xl`/`vec_xst` builtins, so the code may compile if GCC detects POWER9 at build time.
- Untested — contributions welcome.

---

## Per-File Platform Dependency

| File | POWER8 Only? | Notes |
|---|---|---|
| `ggml-ram-coffers.h` | Partial | `numa.h` is Linux-only; NUMA config arrays used regardless |
| `ggml-ram-coffer.h` | Partial | Memory-mapped allocation with `MAP_HUGETLB` is Linux-only |
| `ggml-coffer-mmap.h` | Partial | `mmap` with NUMA binding |
| `power8-compat.h` | Yes | Only provides macros when `__POWER8_VECTOR__` is defined |
| `ggml-vcipher-collapse.h` | Yes | Uses `vec_perm` and `vec_xor` — POWER8 AltiVec |
| `ggml-topk-collapse-vsx.h` | Yes | VSX (Vector-Scalar Extension) — POWER8/AltiVec |
| `ggml-intelligent-collapse.h` | Partial | Float ops only, may compile on any arch |
| `ggml-neuromorphic-coffers.h` | Partial | Float ops only, may compile on any arch |
| `ggml-pse-integration.h` | Partial | PSE entropy via `mftb` — POWER8-specific timer |
| `ggml-symbolic-neural-bridge.h` | No | Pure float ops — should compile anywhere |
| `pse-entropy-burst.h` | Yes | Uses `__builtin_mftb()` — POWER8-specific |

---

## Building on Non-POWER8

To build on non-POWER8 architectures, you must:

1. Remove or stub files listed as "POWER8 Only" above.
2. Provide stubs for `numa.h` functions if building on non-Linux.
3. Compile with `-DGGML_NO_POWER8` to skip vector paths (not yet implemented — PR welcome).

A full cross-platform port would require:
- Replace `vec_perm` / `vec_xor` with SSE/NEON equivalents.
- Replace `__builtin_mftb()` with `rdtsc` (x86) or `mach_absolute_time` (macOS).
- Replace `MAP_HUGETLB` with transparent hugepages or regular `mmap`.
- Provide stubs for NUMA functions on non-Linux platforms.
