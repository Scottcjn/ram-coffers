# Fallback Behavior

This document describes how `ggml-ram-coffers` behaves on
non-POWER8 / single-NUMA-node systems.

## Single-NUMA-Node Systems

When the system has only NUMA node 0, the coffer routing map
`NUMA_TO_COFFER` is a no-op transformation (node 0 鈫?coffer 2,
but all coffers remain on node 0). The `numa_run_on_node()` call
(line ~210 in `ggml-ram-coffers.h`) stays on the single node.

## Non-POWER8 Architectures

The following parts are POWER8-specific and guarded by `#ifdef __linux__`:

| Feature | Guard | Reference |
|---------|-------|-----------|
| `numa.h` / `numaif.h` includes | `#ifdef __linux__` | `ggml-ram-coffers.h:50-52` |
| `numa_run_on_node()` calls | `#ifdef __linux__` | `ggml-ram-coffers.h:210` |
| `numa_available()` check | `#ifdef __linux__` | `ggml-ram-coffers.h:160` |
| `numa_num_configured_nodes()` | `#ifdef __linux__` | `ggml-ram-coffers.h:165` |

On non-Linux systems (macOS, Windows), none of the NUMA bindings compile.
The coffer data is allocated in heap memory without node pinning.

## Build Matrix

| Platform | NUMA Support | Notes |
|----------|-------------|-------|
| Linux + POWER8 (multi-node) | Full | Designed topology |
| Linux + POWER8 (single-node) | Degraded | No node spread |
| Linux + x86_64 | Degraded | NUMA may be available; topology differs |
| macOS | None | Heap-only |
| Windows | None | Heap-only |
| aarch64 | None | Heap-only (no NUMA in headers) |
