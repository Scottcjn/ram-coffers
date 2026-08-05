# Portability

RAM Coffers was written for the IBM POWER8 S824 (4 NUMA nodes, AltiVec/VSX,
`dcbt` prefetch, `mftb` timebase entropy). It now also **builds and runs** on
machines with none of those things: x86-64 laptops, single-socket servers,
ARM boxes, and any Linux host without `libnuma-dev` installed.

This document describes the capability macros, the uniform-memory mode, and
the verified build matrix.

## The problem this fixes

Before this change the headers unconditionally did:

```c
#include <numa.h>       /* ggml-ram-coffer.h, ggml-coffer-mmap.h, ggml-ram-coffers.h */
#include <numaif.h>
#include <altivec.h>    /* the three collapse headers */
```

and `coffer_init()` did:

```c
if (numa_available() < 0) {
    fprintf(stderr, "NUMA not available!\n");
    return -1;                    /* hard fail on an ordinary laptop */
}
```

On a machine without `libnuma-dev`, or on anything that is not POWER, this
did not degrade gracefully — **it failed to compile**. That is the hardest
possible failure for someone evaluating the project. Everything below exists
to make the degradation graceful instead.

## Capability macros

All detection lives in [`coffers-portability.h`](coffers-portability.h),
which every other header now includes first.

| Macro | Meaning | Auto-detected when |
|---|---|---|
| `GGML_COFFERS_HAVE_NUMA` | libnuma is usable | Linux **and** `<numa.h>` + `<numaif.h>` present |
| `GGML_COFFERS_HAVE_ALTIVEC` | POWER vector intrinsics usable | POWER **and** `__ALTIVEC__`/`__VEC__` **and** `<altivec.h>` present |
| `GGML_COFFERS_HAVE_VCIPHER` | ISA 2.07 hardware AES usable | AltiVec **and** `__CRYPTO__` |
| `GGML_COFFERS_IS_POWER` | Target is PowerPC/POWER | `__powerpc__` / `__powerpc64__` / `__PPC__` / `__PPC64__` |
| `COFFERS_CACHE_LINE` | Cache line size | 128 on POWER, 64 elsewhere |

Detection uses `__has_include`, so a missing `libnuma-dev` is detected at
**compile time** rather than exploding.

### Overriding detection

| Build flag | Effect |
|---|---|
| `-DGGML_COFFERS_NO_NUMA` | Force uniform-memory mode; do not link `-lnuma` |
| `-DGGML_COFFERS_HAVE_NUMA=0` | Same, by setting the macro directly |
| `-DGGML_COFFERS_NO_ALTIVEC` | Force the scalar collapse path |
| `-DGGML_COFFERS_HAVE_VCIPHER=0` | Force the scalar (non-AES) vcipher path |

**NUMA is ON by default whenever `<numa.h>` is present.** It is deliberately
*not* opt-in: making it opt-in would silently disable NUMA on the POWER8
machines this project was built for, which is a performance regression
disguised as a portability fix.

## Uniform-memory mode

When NUMA is unavailable **or** `numa_num_configured_nodes() == 1`, the
system initialises **one coffer covering all memory** and continues
successfully. It is a supported configuration, not an error.

Announced once, on stderr:

```
RAM Coffers: NUMA unavailable - running in uniform-memory mode (1 coffer)
RAM Coffers: vector path = scalar C, prefetch = __builtin_prefetch, entropy = CLOCK_MONOTONIC
```

In this mode:

- Routing, prune planning, prefetch, shard loading and stats all work.
- Node-affinity calls (`numa_run_on_node`, `mbind`, `set_mempolicy`,
  thread pinning) become **successful no-ops** — with one memory region,
  "already local" is the correct answer, not a failure.
- `coffers_alloc_on_node()` falls back to `malloc()`.
- `coffer_init()` returns `0`. It no longer returns `-1`.

## Topology-adaptive coffer mapping

The old mapping was the literal `COFFER_TO_NUMA[4] = {3, 1, 0, 2}` — correct
only on the POWER8 S824, and an out-of-range node id on anything smaller.
The mapping is now derived from the node count actually detected:

| Nodes detected | Coffers | Mapping |
|---|---|---|
| 1 (or no NUMA) | 1 | uniform, node 0 |
| 2 | 2 | `{0, 1}` |
| 4 | 4 | `{3, 1, 0, 2}` — original POWER8 ordering preserved |
| N | `min(N, 8)` | `{0 .. N-1}` |

Use `coffers_node_for(coffer_id)`; it never returns a node id past the real
node count. `coffer_plan_layer_placement()` now sorts nodes by *measured*
free space instead of assuming the S824 layout.

## Scalar fallbacks for POWER intrinsics

| POWER feature | Fallback | Bit-identical? |
|---|---|---|
| `dcbt` prefetch | `__builtin_prefetch`, else no-op | n/a (advisory hint) |
| `dcbt` stream start/stop | dropped (no equivalent) | n/a (advisory hint) |
| `mftb` timebase entropy | `clock_gettime(CLOCK_MONOTONIC)` | no — different counter, same role |
| `vec_perm` collapse | plain C byte permute (`coffers_perm_bytes`) | **yes** — see below |
| `vec_madd`/`vec_sel`/`vec_cmpgt` | scalar compare + multiply | yes |
| `__builtin_crypto_vcipher` (AES) | xorshift-multiply avalanche mixer | **no** — see below |

### vec_perm

AltiVec `vec_perm(a, b, c)` is a byte permute: `result[i] = (a||b)[c[i] & 0x1F]`
over the 32-byte concatenation. `coffers_perm_bytes()` reproduces exactly that
in plain C, so `intelligent_collapse_scores()` yields the same logical result
on both paths. The shared `coffers_perm_t` type is `vector unsigned char` on
AltiVec and a 16-byte struct otherwise, so call sites need no `#ifdef`.

> On little-endian POWER (`ppc64le`) the hardware intrinsic indexes the
> concatenation in the opposite byte order. `coffers_perm_bytes()` defines the
> **big-endian reference semantics** and is used only on non-POWER builds,
> where there is no hardware result to diverge from.

### vcipher

`ggml-vcipher-collapse.h` uses POWER8 ISA 2.07 hardware AES as its entropy and
diffusion source. The scalar fallback keeps the same API and the same
*intent* — entropy-seeded pattern generation, top-K selection, winner
amplification, loser zeroing — but substitutes a software mixer for the AES
round. It is **behaviourally equivalent in intent, not bit-identical**, and
`vcipher_collapse_banner()` says so at runtime.

## Verified build matrix

Verified on x86-64 (Linux 6.17, GCC), with `libnuma` present and a single
NUMA node. `tests/build_matrix.sh` runs all of it.

| # | Configuration | Build | Run |
|---|---|---|---|
| 1 | auto-detect (libnuma present, 1 node) | pass | pass |
| 2 | `-DGGML_COFFERS_NO_NUMA`, not linked against `-lnuma` | pass | pass |
| 3 | `-DGGML_COFFERS_HAVE_NUMA=0` | pass | pass |
| 4 | `-DGGML_COFFERS_NO_ALTIVEC` | pass | pass |
| 5 | both forced off (uniform + scalar) | pass | pass |
| 6 | compiled as C++17 | pass | n/a |

Cross-compiled to `powerpc64le-linux-gnu` with `-mcpu=power8 -maltivec -mvsx`
(compile + disassemble only; no POWER hardware was available to execute it):

- `GGML_COFFERS_HAVE_ALTIVEC == 1`, `GGML_COFFERS_IS_POWER == 1`,
  `COFFERS_CACHE_LINE == 128` — asserted via `_Static_assert`.
- Objects contain real `vperm`, `mftb` and `dcbt` instructions, confirming the
  POWER branch is selected and unchanged.

### Per-header status

| Header | C (x86) | C++ (x86) | C (ppc64le + AltiVec) |
|---|---|---|---|
| `coffers-portability.h` | ok | ok | ok |
| `ggml-ram-coffer.h` | ok | ok | ok |
| `ggml-ram-coffers.h` | ok | ok | ok |
| `ggml-coffer-mmap.h` | ok | ok | ok |
| `ggml-intelligent-collapse.h` | ok | ok | ok |
| `ggml-topk-collapse-vsx.h` | ok | ok | ok |
| `ggml-vcipher-collapse.h` | ok | ok | ok |
| `ggml-symbolic-neural-bridge.h` | ok | ok | ok |
| `power8-compat.h` | ok | ok | ok |
| `ggml-neuromorphic-coffers.h` | ok | **fail** | ok |
| `pse-entropy-burst.h` | **fail** | ok | **fail** |
| `ggml-pse-integration.h` | **fail** | **fail** | **fail** |

## Known limitations (pre-existing, not portability-related)

These were already broken on `main` before this work and are **not** caused by
NUMA or architecture. They are listed so nobody re-investigates them as
portability bugs.

1. **`ggml-pse-integration.h` cannot compile from a clean checkout.** It
   `#include`s `ggml-dcbt-resident.h`, `ggml-sparse-softmax.h` and
   `ggml-pse-symbolic-gate.h`, none of which are published in this
   repository, and it calls `pse_should_collapse()` / `pse_gate_report()`
   from the last of those. The three includes are now wrapped in
   `__has_include` so the failure is legible rather than a bare
   "file not found", but the header still needs the private integration tree.
2. **`pse-entropy-burst.h` is C++-only** by design — it includes `<cstdio>`
   and `<algorithm>`.
3. **`ggml-neuromorphic-coffers.h` does not compile as C++** because of
   `static symbolic_neural_state_t g_sn_state = {0};`, where C++ will not
   convert `int` to the leading `tetra_t` member. It is fine as C and fine
   on POWER.
4. **`ggml-ram-coffer.h` and `ggml-ram-coffers.h` cannot be included in the
   same translation unit** — both define `ram_coffer_t` and
   `coffer_print_stats`. The tests keep them in separate files.

## Running the tests

```bash
cd tests
./build_matrix.sh            # full matrix, all modes
```

Or a single configuration:

```bash
cd tests
gcc -std=c11 -I.. -Wall test_portability.c   -o tp -lm -lnuma && ./tp
gcc -std=c11 -I.. -Wall test_coffer_headers.c -o tc -lm -lnuma && ./tc

# force uniform-memory mode, no libnuma needed
gcc -std=c11 -I.. -Wall -DGGML_COFFERS_NO_NUMA test_portability.c -o tp -lm && ./tp
```
