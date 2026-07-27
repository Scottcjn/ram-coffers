# Fallback Behavior: non-POWER8 and single-NUMA-node systems

The README documents the headline path: a 4-node POWER8 S824 with 544GB
spread across NUMA nodes. This document answers the question from issue #664:
**what happens everywhere else?** Every claim below cites the header and line
range it comes from (line numbers as of this commit).

## TL;DR

- `ggml-ram-coffers.h` and `ggml-neuromorphic-coffers.h` **compile and run
  correctly on any POSIX system** (x86_64, aarch64, macOS). POWER8 paths are
  compile-time guarded and degrade to no-ops or scalar loops.
- `ggml-coffer-mmap.h` and `ggml-ram-coffer.h` **require Linux with libnuma
  development headers** (`numa.h`, `numaif.h`) to compile at all; NUMA
  *runtime* absence is handled gracefully, the *headers* are not optional.
- `ggml-vcipher-collapse.h`, `ggml-intelligent-collapse.h`, and
  `ggml-topk-collapse-vsx.h` are **POWER8-only** (unguarded `<altivec.h>`
  include) and will not compile off-POWER. Use the `apple-silicon/` port for
  M-series, or simply don't include them.

## 1. Single-NUMA-node systems

What the coffer routing does when there is only node 0:

- `coffer_init_numa()` caps the initialized coffers at
  `min(MAX_COFFERS, numa_num_configured_nodes())`
  (`ggml-ram-coffers.h:277`). On a single-node machine only Coffer-0 gets a
  NUMA assignment; coffers 1-3 keep `numa_node = 0` from the zeroed global
  array (`ggml-ram-coffers.h:96`) and are simply never routed to, because
  `route_to_coffer()` skips coffers whose `is_loaded` flag is false
  (`ggml-ram-coffers.h:245-246`).
- **Known rough edge:** the single initialized coffer is still assigned
  `numa_node = COFFER_TO_NUMA[0] = 3` (`ggml-ram-coffers.h:279,51`). On a
  machine that genuinely has only node 0, the subsequent
  `numa_run_on_node(3)` / `set_mempolicy(MPOL_BIND, 1<<3)` calls
  (`ggml-ram-coffers.h:296,320-321`) will fail at runtime. The failure is
  non-fatal — `numa_run_on_node` has no error check in the code path and the
  `mmap` that follows (`ggml-ram-coffers.h:326-335`) succeeds anyway — but
  memory is allocated with default policy, i.e. all "coffer" behavior
  collapses to ordinary local allocation.
- If NUMA is entirely unavailable (`numa_available() < 0`),
  `coffer_init_numa()` prints `Coffers: NUMA not available` and returns -1
  (`ggml-ram-coffers.h:269-272`); `init_ram_coffers()` downgrades this to a
  warning (`Running without NUMA support`) and continues
  (`ggml-ram-coffers.h:531-533`).
- The mmap page-placement helpers degrade silently: `coffer_migrate_region()`
  returns -1 immediately when NUMA is unavailable
  (`ggml-coffer-mmap.h:283-286`), and retries with `MPOL_PREFERRED` if the
  `MPOL_BIND | MPOL_MF_MOVE` mbind fails (`ggml-coffer-mmap.h:298-305`).
  `coffer_apply_numa_hints()` prints `NUMA not available, skipping placement`
  and returns success (`ggml-coffer-mmap.h:370-374`) — a deliberate no-op.
- Routing itself is **not** NUMA-dependent: `route_to_coffer()` is pure
  cosine similarity over loaded coffers' domain signatures
  (`ggml-ram-coffers.h:241-261`) and works identically on one node.

**Net effect on single-node:** the system runs, all weights land in node-0
memory by default policy, and coffer routing still selects shards by domain —
you get the software routing logic with zero NUMA locality benefit.

## 2. Non-POWER8 architectures

What is POWER8-specific and what happens elsewhere:

| Feature | Where | Off-POWER8 behavior |
|---|---|---|
| `dcbt` prefetch | `ggml-ram-coffers.h:103-111` | Macros become `(void)` no-ops; `dcbt_resident()` (`:114-129`) still runs its loop but does nothing. Correct, just no prefetch. |
| AltiVec dot product | `ggml-ram-coffers.h:202-226` | `#else` scalar loop (`:220-224`). Bit-exact same math, slower. |
| Layer-weight prefetch | `ggml-coffer-mmap.h:406-431` | Entire function body is inside `#if defined(__powerpc64__)`; compiles to an empty function elsewhere. |
| `mftb` timebase | `pse-entropy-burst.h:65-80` | Portable fallbacks: `rdtsc` on x86_64, `cntvct_el0` on aarch64, address-of-counter as last resort. |
| `vec_xl`/`vec_xst` | `power8-compat.h:10-42` | Only defined when `__POWER8_VECTOR__ && !__POWER9_VECTOR__`; maps to aligned `vec_ld`/`vec_st`, never active off-POWER. |
| `vec_perm` collapse kernels | `ggml-vcipher-collapse.h:31`, `ggml-intelligent-collapse.h:26`, `ggml-topk-collapse-vsx.h:19` | **Refuse to compile** — `<altivec.h>` is included unconditionally. These headers are POWER8-only by design (`-mcpu=power8 -mcrypto`). |

So: guarded POWER8 code takes a scalar/no-op path and stays correct; the
three collapse kernels are compile-time POWER8-only. Nothing silently
misbehaves — it either works scalar or fails the build loudly.

For Apple Silicon specifically, there is a dedicated port under
`apple-silicon/` (NEON collapse headers, Metal shaders, own Makefile and
`setup-mac-m2.sh`) — that is the supported path on aarch64/macOS, not the
POWER8 headers.

## 3. Will this build on my machine?

| Platform | `ggml-ram-coffers.h` + `ggml-neuromorphic-coffers.h` | `ggml-coffer-mmap.h` / `ggml-ram-coffer.h` | collapse kernels (`vcipher`, `intelligent`, `topk-vsx`) |
|---|---|---|---|
| POWER8 S824, 4 NUMA nodes (target) | ✅ full path | ✅ full NUMA placement | ✅ `-mcpu=power8 -mcrypto` |
| POWER8, single node | ✅ compiles, runs; NUMA calls fail harmlessly (see §1 rough edge) | ✅ compiles; `mbind` degrades/fails per-region | ✅ |
| x86_64 Linux (+ libnuma-dev) | ✅ scalar/no-op fallbacks | ✅ compiles; graceful skip if NUMA unavailable | ❌ compile error (`<altivec.h>`) |
| aarch64 Linux | ✅ | ✅ same as x86_64 | ❌ |
| macOS (any) | ✅ (all NUMA code is `#ifdef __linux__`, `ggml-ram-coffers.h:35-39`) | ❌ unconditional `#include <numa.h>` (`ggml-coffer-mmap.h:27-28`, `ggml-ram-coffer.h:25-26`) | ❌ (use `apple-silicon/` instead) |

Build requirements summary:

- **Anywhere:** a C99 compiler and POSIX `mmap`/`open` for the base headers.
- **Linux only:** `ggml-coffer-mmap.h` / `ggml-ram-coffer.h` need
  `libnuma-dev` (Debian) / `numactl-devel` (RPM) installed even to compile.
- **POWER8 only:** the three collapse kernels, with `-mcpu=power8 -mcrypto`.

## 4. Smoke-testing the logic without POWER hardware

There is a portable correctness check that exercises the routing/placement
logic with no NUMA and no AltiVec — the pure-Python topology planner:

```sh
python3 -m unittest tests.test_ram_coffers_topology -v     # unit tests
python3 ram_coffers_topology.py --help                     # CLI planner
```

For the C side, a minimal smoke test on any POSIX machine is to include
`ggml-ram-coffers.h`, call `init_ram_coffers()` with paths to any files
(they are mmap'd read-only), and call `coffer_test_routing()`
(`ggml-ram-coffers.h:619-638`), which routes three synthetic query embeddings
and prints the chosen coffer. Expect the banner line
`WARNING: Running without NUMA support` on non-NUMA hosts — that is the
startup signal that you are in degraded (correctness-only) mode.

## 5. Which benchmark numbers collapse off-POWER8

All headline numbers in the README (147.54 t/s etc.) depend on the DCBT
prefetch pipeline (`ggml-ram-coffers.h:139-193`) and the `vec_perm`/vcipher
collapse kernels. Off-POWER8 the prefetch macros are no-ops (§2), so expect
memory-bound throughput to drop substantially; the collapse kernels don't
build at all. Treat off-POWER runs as **correctness validation only**, never
as performance evidence for or against the POWER8 claims.

## 6. Startup detection checklist

To tell which mode you are in, watch stderr at startup:

- `Coffers: NUMA not available` → no NUMA, software-routing-only mode
  (`ggml-ram-coffers.h:270`).
- `Coffers: N NUMA nodes detected` → N < 4 means partial topology
  (`ggml-ram-coffers.h:274-275`).
- `Coffer: NUMA not available, skipping placement` → mmap hint layer in
  no-op mode (`ggml-coffer-mmap.h:372`).
- Compile failure on `<altivec.h>` → you included a POWER8-only collapse
  kernel off-POWER (§2 table).
