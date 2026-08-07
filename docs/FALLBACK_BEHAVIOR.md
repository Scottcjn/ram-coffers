# Fallback Behavior

This document describes ram-coffers behavior on non-POWER8 and single-NUMA systems.

## Single-NUMA-Node Systems

When only one NUMA node is available, the coffer routing system operates in single-node mode:

1. **NUMA initialization** (`ggml-ram-coffers.h` L267-280): `coffer_init_numa()` uses `numa_available()` and `numa_num_configured_nodes()`. If NUMA is unavailable or only one node detected, it returns gracefully without error.

2. **Coffer allocation** (`ggml-ram-coffers.h` L294-355): Memory allocation uses `mmap()` with NUMA hints. On single-node systems, `numa_run_on_node()` targets Node 0, and `mbind()` with `MPOL_PREFERRED` behaves like standard `mmap()`.

3. **Weight placement** (`ggml-coffer-mmap.h` L284): `numa_available() < 0` triggers early return from `coffer_apply_numa_hints()`, skipping all NUMA-specific placement. Message "NUMA not available, skipping placement" printed to stderr.

4. **Cognitive routing** (`ggml-neuromorphic-coffers.h` L370-434): Cognitive function mapper assigns `target_numa = 0` for all functions with single node. Activation distributes uniformly.

## Non-POWER8 Architectures

### POWER8-Specific Assembly Guards

| Operation | File | Guard | Fallback |
|---|---|---|---|
| DCBT_PREFETCH | ggml-ram-coffers.h L103-110 | __powerpc64__ | (void)(addr) no-op |
| DCBT_STREAM_START | ggml-ram-coffers.h L104-110 | __powerpc64__ | (void)(addr) no-op |
| DCBT_STREAM_STOP | ggml-ram-coffers.h L104-110 | __powerpc64__ | (void)0 |
| dcbt_prefetch_numa | ggml-coffer-mmap.h L410-426 | __powerpc64__ | Calls no-op macro |
| vec_perm collapse | ggml-topk-collapse-vsx.h | POWER VSX | Compile error or scalar fallback |

### Memory Management

- **NUMA API** (`<numa.h>`, `<numaif.h>`): Required on Linux via `#ifdef __linux__` guards. Non-Linux systems skip all NUMA calls silently.
- **Huge pages** (`MAP_HUGETLB`): Guarded by `#ifdef MAP_HUGETLB`. Falls back to 4KB pages.
- **Cache line**: Hardcoded 128 bytes (POWER8). x86_64 (64B) and aarch64 (64/128B) remain functional but suboptimal.

### Build Matrix

| Architecture | NUMA | DCBT | vec_perm | Status |
|---|---|---|---|---|
| POWER8 4-node | Yes | Active | Active | Full build |
| POWER8 single-node | Yes | Active | Active | Single-node fallback |
| POWER9/10 | Yes | Active | Active | Functional, topology differs |
| x86_64 Linux | Via libnuma | No-op | Error/Scalar | Builds, NUMA functional |
| x86_64 macOS | No | No-op | Error/Scalar | Builds, no NUMA |
| aarch64 Linux | Via libnuma | No-op | Error/Scalar | Builds, NUMA functional |
| aarch64 macOS | No | No-op | Error/Scalar | Builds, no NUMA, see apple-silicon/ |

### Key Takeaway

**ram-coffers never crashes on unsupported architectures.** All POWER8-specific operations degrade to no-ops or scalar fallbacks. Worst case (non-Linux, non-NUMA) is standard single-node allocation.

## References

- `ggml-ram-coffers.h` L103-110: DCBT macro guards
- `ggml-ram-coffers.h` L267-280: NUMA init fallback
- `ggml-coffer-mmap.h` L284-287, L370-378: NUMA placement skip
- `ggml-coffer-mmap.h` L410-426: DCBT prefetch guards
- `ggml-neuromorphic-coffers.h` L12-18: NUMA/Brain mapping
