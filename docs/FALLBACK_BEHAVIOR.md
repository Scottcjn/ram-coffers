# Fallback Behavior Documentation

This document describes the fallback mechanisms implemented in `ram-coffers` for non-POWER8/PPC64 architectures and environments without NUMA support.

## Architecture Guards

### POWER8/PPC64 Detection (`ggml-ram-coffers.h`)

**Lines 103-108**: Prefetch intrinsics are conditionally compiled based on architecture.

```c
#if defined(__powerpc64__) || defined(__powerpc__)
#define DCBT_PREFETCH(addr) __asm__ __volatile__("dcbt 0,%0" : : "r"(addr))
#else
#define DCBT_PREFETCH(addr) (void)(addr)
```

On non-POWER architectures, prefetch operations compile to no-ops `(void)(addr)` to maintain API compatibility without generating invalid instructions.

### Vectorized Dot Product (`ggml-ram-coffers.h`)

**Lines 202-209**: The `dot_product` function uses AltiVec SIMD on POWER, with a scalar fallback path. When `__powerpc64__` or `__powerpc__` is not defined, compilation falls through to a standard C loop.

## NUMA Fallbacks

### Initialization Guard (`ggml-ram-coffers.h`)

**Lines 268-275**: NUMA initialization explicitly checks availability before use.

```c
#ifdef __linux__
    if (numa_available() < 0) {
        fprintf(stderr, "Coffers: NUMA not available\n");
        return -1;
    }
```

If `libnuma` is absent or the kernel lacks NUMA support, the function logs to stderr and returns `-1`. Callers must handle this error code gracefully.

### Memory Placement Skip (`ggml-coffer-mmap.h`)

**Lines 370-374**: Tensor placement hints are optional; missing NUMA does not abort mapping.

```c
static int coffer_apply_numa_hints(coffer_model_hint_t* hint) {
    if (numa_available() < 0) {
        fprintf(stderr, "Coffer: NUMA not available, skipping placement\n");
        return 0;
    }
```

Unlike initialization, this returns `0` (success) to allow mmap to proceed with default kernel page placement.

## POWER8 Compatibility Layer (`power8-compat.h`)

**Lines 1-35**: Provides source-level compatibility for code written against POWER9 vector builtins while targeting POWER8 hardware.

| Macro | Line | Fallback Strategy |
|-------|------|-------------------|
| `vec_xl(offset, ptr)` | 16-17 | Maps to `vec_ld` (requires 16-byte alignment) |
| `vec_xst(v, offset, ptr)` | 21-22 | Maps to `vec_st` (requires 16-byte alignment) |
| `vec_xl_len(ptr, len)` | 29-35 | Copies `min(len,16)` bytes into aligned buffer via `memcpy`, then loads |

**Critical constraint** (line 4): Do NOT define `__POWER9_VECTOR__`. Defining it enables GCC to emit POWER9-only opcodes that will SIGILL on POWER8.

## Build Matrix

| Target | NUMA | Vector Path | Notes |
|--------|------|-------------|-------|
| POWER9 + Linux | ✅ | Native POWER9 | Optimal path |
| POWER8 + Linux | ✅ | `power8-compat.h` shims | Requires aligned buffers |
| POWER8 + Linux (no libnuma) | ❌ | `power8-compat.h` shims | Placement skipped, functional |
| x86_64 + Linux | ✅/❌ | Scalar / AVX (separate) | DCBT becomes no-op |
| ARM64 + Linux | ✅/❌ | Scalar / NEON (separate) | DCBT becomes no-op |
| macOS (any arch) | N/A | Scalar / NEON | `#ifdef __linux__` guards skip NUMA entirely |
