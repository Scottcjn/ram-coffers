# RAM Coffers: Fallback Behavior on Non-POWER8 / Single-NUMA Systems

This document describes how RAM Coffers behaves on systems different from the primary target (IBM POWER8 S824 with 4 NUMA nodes). It answers the question: "Will this build on my machine?"

## Quick Reference: Build Matrix

| Platform | Compiles? | NUMA Routing | DCBT Prefetch | Vec_perm | Status |
|----------|-----------|--------------|---------------|----------|--------|
| POWER8 multi-node (S824) | ✅ Yes | ✅ Full 4-coffer routing | ✅ Hardware DCBT | ✅ Altivec vec_perm | **Primary target** |
| POWER8 single-node | ✅ Yes | ⚠️ Single node only | ✅ Hardware DCBT | ✅ Altivec vec_perm | Functional, no multi-node benefit |
| x86_64 | ✅ Yes | ⚠️ NUMA-aware (if available) | ❌ No-op (compiled out) | ❌ Scalar fallback | Functional, reduced performance |
| aarch64 (ARM64) | ✅ Yes | ⚠️ NUMA-aware (if available) | ❌ No-op (compiled out) | ❌ Scalar fallback | Functional, reduced performance |
| Apple Silicon | ✅ Yes | ❌ No NUMA API | ❌ No-op (compiled out) | ❌ Scalar fallback | Functional, NUMA hints skipped |

## Single-NUMA-Node Systems

### What Happens

When running on a system with only one NUMA node (or where NUMA is not available):

1. **Initialization** (`coffer_init_numa()` — `ggml-ram-coffers.h`):
   - `numa_available()` returns < 0 → prints `"Coffers: NUMA not available"` to stderr
   - Returns -1, but `init_ram_coffers()` prints `"WARNING: Running without NUMA support"` and continues
   - All 4 coffers are still initialized, but mapped to node 0

2. **Memory Allocation** (`coffer_load_shard()` — `ggml-ram-coffers.h`):
   - `set_mempolicy(MPOL_BIND, ...)` binds allocation to node 0
   - `mmap()` still works — memory is allocated on the single available node
   - No page migration occurs (nothing to migrate to)

3. **Activation** (`activate_coffer_ex()` — `ggml-ram-coffers.h`):
   - `numa_run_on_node(coffer->numa_node)` — on single-node systems, this binds to node 0 regardless of which coffer is activated
   - Prefetch still works (DCBT is architecture-dependent, not NUMA-dependent)

4. **Routing** (`route_to_coffer()` — `ggml-ram-coffers.h`):
   - All 4 coffers are still loaded and routable
   - Resonance routing still selects the best coffer based on query embedding
   - **But**: all coffers share the same physical memory — no NUMA locality benefit

5. **Page Migration** (`coffer_migrate_region()` — `ggml-coffer-mmap.h`):
   - `numa_available()` returns < 0 → returns -1
   - Pages stay where they were allocated (node 0)
   - No error — migration is best-effort

### Performance Impact

- **No NUMA locality benefit**: All memory accesses go to the same node
- **Still functional**: Routing, prefetch, and pruning all work correctly
- **Expected performance**: Similar to standard llama.cpp with mmap

## Non-POWER8 Architectures

### POWER8-Specific Code

The following code is POWER8-specific and has fallbacks:

| Feature | Location | POWER8 Behavior | Non-POWER8 Fallback |
|---------|----------|-----------------|---------------------|
| **DCBT Prefetch** | `ggml-ram-coffers.h:127-132` | `dcbt` instruction for L2/L3 residency | No-op: `(void)(addr)` — compiled out |
| **DCBT Stream** | `ggml-ram-coffers.h:128-129` | `dcbt 0,reg,stream_id` for streaming | No-op: `(void)(addr)` and `(void)0` |
| **Vec_perm Attention** | `ggml-topk-collapse-vsx.h` | `vec_perm` for non-bijunctive collapse | Scalar fallback (not in these headers) |
| **Dot Product** | `ggml-ram-coffers.h:191-204` | Altivec `vec_ld` + `vec_madd` SIMD | Scalar loop: `sum += a[d] * b[d]` |

### Compilation Guards

All POWER8-specific code is guarded by preprocessor macros:

```c
#if defined(__powerpc64__) || defined(__powerpc__)
    // POWER8-specific: DCBT, Altivec, vec_perm
#else
    // Fallback: no-ops, scalar code
#endif
```

**Result**: The code compiles on any architecture — POWER8 features are simply disabled on non-PowerPC systems.

### What Still Works on x86/ARM

- ✅ **Resonance routing**: Cosine similarity works on all architectures (scalar dot product)
- ✅ **Coffer loading**: mmap works everywhere
- ✅ **Layer-ahead prefetch**: The pipeline structure works, but prefetch is a no-op
- ✅ **Non-bijunctive pruning**: The mask-based pruning logic is architecture-independent
- ✅ **Neuromorphic routing**: Cognitive classification and routing work on all platforms
- ✅ **Domain signatures**: All routing logic is pure C

### What's Missing on x86/ARM

- ❌ **DCBT hardware prefetch**: No L2/L3 residency hints → higher cache miss rate
- ❌ **Altivec SIMD**: Dot product uses scalar code → 4x slower (no vectorization)
- ❌ **Vec_perm attention**: Non-bijunctive collapse uses scalar fallback
- ❌ **NUMA locality**: Memory is allocated on default node, no interleave

## Code References

### ggml-ram-coffers.h

- **Lines 127-132**: DCBT prefetch macros (POWER8-specific with fallback)
- **Lines 135-146**: `dcbt_resident()` — prefetches entire region using DCBT
- **Lines 191-204**: `dot_product()` — Altivec SIMD with scalar fallback
- **Lines 210-215**: `coffer_init_numa()` — NUMA initialization with graceful degradation
- **Lines 283-295**: `activate_coffer_ex()` — NUMA binding with fallback

### ggml-coffer-mmap.h

- **Lines 13-15**: NUMA includes (guarded by `__linux__`)
- **Lines 167-195**: `coffer_migrate_region()` — page migration with NUMA fallback
- **Lines 237-250**: `coffer_apply_numa_hints()` — NUMA placement with availability check
- **Lines 263-290**: `coffer_prefetch_layer_weights()` — DCBT prefetch (POWER8 only)

### ggml-neuromorphic-coffers.h

- **Lines 1-100**: Cognitive classification — pure C, architecture-independent
- **Lines 100-200**: Routing logic — no architecture-specific code
- **Lines 200-300**: Sensor integration — pure C, architecture-independent

## Recommendations

### For x86_64 Users

1. **It will compile and run** — all features degrade gracefully
2. **Performance will be lower** — expect ~50-70% of POWER8 throughput due to:
   - No DCBT prefetch (cache misses increase)
   - Scalar dot product (no SIMD)
3. **Consider using llama.cpp directly** — RAM Coffers' advantage is primarily on POWER8
4. **Testing is welcome** — run `tests/test_coffers.py` to verify functionality

### For Apple Silicon Users

1. **It will compile and run** — macOS has no NUMA API, so NUMA features are skipped
2. **Apple Silicon has its own advantages** — unified memory architecture makes NUMA less relevant
3. **No DCBT** — Apple Silicon uses different prefetch mechanisms
4. **Consider Metal acceleration** — RAM Coffers doesn't use GPU; llama.cpp with Metal is faster

### For POWER8 Single-Node Users

1. **Full functionality** — all POWER8 features (DCBT, Altivec) work
2. **No multi-node benefit** — all coffers share one node
3. **Still faster than stock llama.cpp** — DCBT prefetch and vec_perm attention help
4. **Test with**: `./benchmark_coffers_vs_llamacpp.sh` to measure improvement

## Testing

Run the test suite to verify your platform:

```bash
# Basic functionality test
python3 tests/test_coffers.py

# Benchmark comparison
./benchmark_coffers_vs_llamacpp.sh

# Check NUMA topology (Linux)
numactl --hardware

# Check compilation flags
gcc -dM -E - < /dev/null | grep -E "__powerpc__|__x86_64__|__aarch64__"
```

## Summary

RAM Coffers is designed to degrade gracefully:

- **POWER8 multi-node**: Full functionality, maximum performance
- **POWER8 single-node**: Full functionality, reduced NUMA benefit
- **x86_64 / ARM64**: Functional, but loses hardware prefetch and SIMD advantages
- **Apple Silicon**: Functional, but no NUMA or DCBT features

The core routing and pruning logic is architecture-independent C code. POWER8-specific features (DCBT, Altivec, vec_perm) are optional optimizations that are compiled out on other platforms.
