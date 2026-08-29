# NUMA Node Detection Fallback on Single-Socket Systems

This document addresses the behavior of RAM Coffers when `numactl` or `/sys/devices/system/node/` reports only one NUMA node, as raised in [#691](https://github.com/Scottcjn/ram-coffers/issues/691).

## Detection Behavior

When the system has a single NUMA node:

1. **`numa_available()`** returns `0` (success) — NUMA API is present and functional
2. **`numa_max_node()`** returns `0` — only node 0 exists
3. **`coffer_init_numa()`** succeeds without warnings (unlike missing NUMA entirely)
4. All coffers are allocated on node 0 via `set_mempolicy(MPOL_BIND, {0})`

The system does **not** error out. It operates correctly but without multi-node locality benefits.

## Explicit Single-Node Mode

Operators can force single-node behavior even on multi-node systems for testing:

```bash
export RAM_COFFERS_FORCE_SINGLE_NODE=1
```

When set, `coffer_init_numa()` will:
- Log `"Coffers: Forced single-node mode via RAM_COFFERS_FORCE_SINGLE_NODE"` to stderr
- Restrict all allocations to node 0 regardless of detected topology
- Skip inter-node migration attempts

This is useful for reproducing single-socket behavior on development machines.

## Performance Characteristics

| Metric | Multi-Node (4 nodes) | Single-Node | Impact |
|--------|---------------------|-------------|--------|
| Memory bandwidth | Aggregated across nodes | Single node bandwidth | ~4x lower peak BW |
| Cache locality | Per-coffer affinity | Shared L3 cache | Higher contention |
| Allocation latency | Parallel across nodes | Serial on node 0 | Slightly higher init time |
| Routing correctness | ✅ Full | ✅ Full | No impact |
| DCBT prefetch | ✅ Hardware | ✅ Hardware (POWER8) | No impact |

Expected throughput on single-socket POWER8: ~60-70% of 4-node baseline due to memory bandwidth saturation, not routing overhead.

## Verification Commands

Confirm your system's NUMA topology before deployment:

```bash
# Check node count
numactl --hardware | grep "^available:"

# Verify coffer allocation at runtime
RAM_COFFERS_LOG_LEVEL=debug ./benchmark_coffers 2>&1 | grep "node"

# Force single-node for comparison
RAM_COFFERS_FORCE_SINGLE_NODE=1 ./benchmark_coffers
```

## Relationship to FALLBACK_BEHAVIOR.md

This document covers **single-node detection** specifically. For architecture-level fallbacks (non-POWER8, missing libnuma, Apple Silicon), see [FALLBACK_BEHAVIOR.md](FALLBACK_BEHAVIOR.md). The two are orthogonal: a system can be POWER8 + single-node (covered here) or x86_64 + multi-node (covered in FALLBACK_BEHAVIOR.md).
