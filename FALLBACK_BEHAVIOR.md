# RAM Coffers: Fallback Behavior on Non-POWER8 / Single-NUMA Systems

This document is the canonical reference describing how RAM Coffers behaves on systems different from the primary target (IBM POWER8 S824 with 4 NUMA nodes and 544GB RAM). Provenance originates from #664 and consolidation tracked in #686.

---

## 1. Per-Header & Component Support Matrix

Different components within RAM Coffers have distinct compilation prerequisites and platform targets:

| Header / Component | Required Dependencies | Supported Platforms | Classification | Notes |
|---|---|---|---|---|
| `ggml-ram-coffers.h` | C99, POSIX (`mmap`) | Linux, macOS, Windows (POSIX) | **Cross-Platform Fallback** | Guarded `numa.h`. DCBT prefetch expands to `(void)`, Altivec SIMD falls back to scalar C loop. |
| `ggml-neuromorphic-coffers.h` | C99, Standard C Math | All POSIX systems | **Universal Pure-C** | Pure C cognitive classification and routing logic. Zero architecture-specific dependencies. |
| `ggml-ram-coffer.h` | Linux, `libnuma` (`numa.h`) | Linux (multi-NUMA) | **Linux NUMA Performance** | Unguarded `numa.h` include. Requires `libnuma-dev`. |
| `ggml-coffer-mmap.h` | Linux, `libnuma` (`numa.h`) | Linux (multi-NUMA) | **Linux NUMA Performance** | Multi-node page migration (`mbind`). Best-effort fallback on single node. |
| `ggml-vcipher-collapse.h` | POWER8 ISA (`vcipher`) | IBM POWER8/POWER9 (ppc64le) | **POWER8-Only Hardware Path** | Hardware AES vector instructions for entropy-guided attention collapse. |
| `ggml-intelligent-collapse.h` | POWER8 ISA (`vec_perm`) | IBM POWER8/POWER9 (ppc64le) | **POWER8-Only Hardware Path** | Vector permutation collapse kernel for non-bijunctive attention. |
| `ggml-topk-collapse-vsx.h` | POWER8 VSX | IBM POWER8/POWER9 (ppc64le) | **POWER8-Only Hardware Path** | Vector scalar extensions for top-k pruning. |
| `apple-silicon/unified-memory-coffers.h` | macOS, ARM NEON, ARM Crypto | Apple Silicon (M1–M4) | **Apple Silicon Specialized Port** | Cache-tier banking (L1/L2/RAM/Swap) with `vqtbl1q_u8` and `vaeseq_u8`. |

---

## 2. Single-NUMA-Node Systems (x86 Desktops, Laptops, Cloud VMs)

When running on hardware where `numactl` or `/sys/devices/system/node/` reports only 1 node:

1. **Initialization (`coffer_init_numa()` in `ggml-ram-coffers.h`):**
   - Calls `numa_num_configured_nodes()` and initialises `min(MAX_COFFERS, n_nodes)`.
   - On a single-node host, only **Coffer 0** is populated; coffers 1–3 remain in their zeroed, unloaded state (`is_loaded = 0`).
   - If NUMA is completely unavailable, `init_ram_coffers()` outputs a diagnostic warning and continues without crashing.
2. **Coffer Routing (`route_to_coffer()`):**
   - Skips unloaded coffers (`is_loaded == 0`), gracefully routing queries through Coffer 0 without affinity faults.
3. **Memory Allocation & Page Migration:**
   - `coffer_load_shard()` applies single-node policy or falls back to standard `mmap(MAP_PRIVATE | MAP_ANONYMOUS)`.
   - `coffer_migrate_region()` returns gracefully without errors since there are no foreign NUMA nodes to migrate across.
4. **Performance Characteristics:**
   - Single-NUMA and non-POWER8 runs serve as **correctness and shape validation**, operating identically to standard single-node memory allocations without multi-node throughput advantages.

---

## 3. Non-POWER8 Architectures (x86_64, aarch64)

All POWER8-specific hardware intrinsics are guarded by preprocessor macros (`#if defined(__powerpc64__) || defined(__powerpc__)`):

- **DCBT Prefetch:** `DCBT_PREFETCH`, `DCBT_STREAM_START`, and `DCBT_STREAM_STOP` expand to no-op expressions `(void)(addr)`.
- **AltiVec Vector Math:** Vector dot products (`dot_product`) fall back to clean scalar C loops.
- **Compatibility Builtins:** `power8-compat.h` provides safe compilation wrappers for vector builtins (`vec_xl`, `vec_xst`, `vec_xl_len`).

---

## 4. Verification & Testing

Verify fallback behavior across local configurations:

```bash
# 1. Run local test suite
python3 tests/test_coffers.py

# 2. Benchmark comparison (supports --allow-single-numa for dry-run smoke checks)
./benchmark_coffers_vs_llamacpp.sh

# 3. Check system NUMA topology
numactl --hardware 2>/dev/null || echo "Single-node / Non-NUMA"
```
