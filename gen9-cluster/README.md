# gen9-cluster — DeepSeek V4 Pro across ninth-generation consoles

A splitting and serving stack that runs a frontier MoE model on a fleet of
PlayStation 5 and Xbox Series X/S consoles, including the variant boards that
have components disabled and the salvage desktop parts built from the same
silicon (AMD 4700S, 4800S, and the BC-250 blade).

It is RAM Coffers' idea applied one level up. Upstream, a coffer is a NUMA bank
inside one POWER8 box holding a known slice of the model. Here a coffer is a
memory tier inside a console — a Series X is 10 GB at 560 GB/s plus 3.5 GB at
336 GB/s plus a 2.4 GB/s NVMe — and the fleet is a coffer hierarchy several
hundred banks wide. The routing that upstream does across NUMA nodes, this does
across consoles, and DeepSeek's own MoE router decides which ones wake up.

```
                shelf 0                          shelf 1
   ┌───────────────────────────────┐   ┌───────────────────────────────┐
   │ host: Series X   layers 0-7   │   │ host: Series X   layers  8-15 │
   │   attention + its KV cache    │   │                               │
   │   router + shared expert      │──▶│   ... 62 blocks over 8 shelves│
   │ 21 more consoles: routed      │   │                               │
   │   experts, ~6 lit per token   │   │                               │
   └───────────────────────────────┘   └───────────────────────────────┘
```

## Why this is possible at all

DeepSeek V4 Pro is ~803 GiB of weights and no console has more than 16 GB.
Three properties make the fleet work anyway:

1. **Only ~3% of the model runs per token.** 50 of 1600 billion parameters
   activate: attention, the router, one shared expert, and 6 of 384 routed
   experts per layer. The other 97% only has to be *stored*, and storage is the
   one thing a pile of consoles has.
2. **Decode is bandwidth-bound, not compute-bound.** A PS5's 448 GB/s is what
   matters, and it is a genuinely good number — a 4700S carries the same DRAM
   on the same 256-bit bus, but reaching it from the CPU instead of the GPU
   measures 92.9 GB/s, a fifth as much, and the planner knows the difference.
3. **The checkpoint ships quantised.** V4 stores its experts — which are nearly
   all of it — in MXFP4, and everything read every token in FP8, so 1.6 T
   parameters occupy 0.54 bytes each. That is the single biggest term in the
   console count: V4 Pro has 2.4x V3's parameters and takes only 1.26x its
   space. `kernels/fp8.c` and `gen9_cluster/fp8.py` implement E4M3FN and are
   tested against each other over all 256 codes.

The result is a plan, not a benchmark: **~11.1 tok/s estimated** for V4 Pro at
8k context on 170 consoles (100 PS5 + 40 Series X + 30 BC-250), and a **73-PS5
floor** to hold it at all. V4 Flash is a fifth of the size and fits on **20**.
Those figures are arithmetic over memory bandwidth, network hops, and NVMe
reads. Nothing in this repository has run on a console yet.

## The profiles

| profile | params | activated | on disk | PS5s to hold it |
|---|---|---|---|---|
| `deepseek-v4-pro` | 1599 B | 49 B | 802 GiB | 73 |
| `deepseek-v4-flash` | 291 B | 13 B | 148 GiB | 20 |
| `deepseek-v3` | 683 B | 37 B | 638 GiB | 58 |
| `deepseek-tiny` | — | — | 0.5 GiB | 1 (CI only) |

Both V4 profiles are the published configurations, not extrapolations, and the
tests check them against the published parameter counts. Two of their
properties change how the planner thinks:

- **Hybrid attention.** V4 alternates CSA (every 4 tokens compress to one
  entry, then attend sparsely to the best 1024 of them) with HCA (every 128
  tokens compress to one, attended densely), plus a sliding-window branch on
  every layer. Neighbouring layers therefore differ by 32x in cache size, and a
  CSA layer *reads* a small fraction of what it *holds* — so the planner sizes
  residency and decode bandwidth separately, per layer, instead of multiplying
  one KV figure by the layer count. The whole 1 M-token cache is 4.6 GiB, about
  an eighth of V3's.
- **Hyper-connections.** The residual stream is 4x hidden, so a shelf boundary
  ships 56 KiB per token rather than 14. At 250 µs a hop that is still small
  against a 90 ms token, but it is 4x what a V3-shaped model would cost.

## Quick start

```bash
cd gen9-cluster
python3 -m gen9_cluster model --model deepseek-v4-pro    # what it costs
python3 -m gen9_cluster model --model deepseek-v4-flash  # the small one
python3 -m gen9_cluster size  --model deepseek-v4-pro --ps5 100 \
        --xbox-series-x 40 --bc-250 30                   # how many consoles
python3 -m gen9_cluster plan  fleet.json --config cluster.json
python3 -m gen9_cluster probe                            # measure this box
python3 -m gen9_cluster serve --unit-id ps5-01           # run a node
python3 -m gen9_cluster health cluster.json
```

An inventory records what each *specific* unit actually has:

```json
{"fleet": [
  {"unit_id": "ps5-01", "sku": "ps5", "runtime": "ps5-linux",
   "host": "10.0.0.11", "backend": "vulkan"},
  {"unit_id": "ps5-07", "sku": "ps5", "runtime": "ps5-linux",
   "host": "10.0.0.17",
   "downbin": {"cu_disabled": 4, "gpu_ghz_cap": 1.8,
               "tier_losses": {"gddr6": 2147483648},
               "reasons": ["dead GDDR6 package", "fan replaced"]}},
  {"unit_id": "bc-01", "sku": "bc-250", "runtime": "salvage-linux",
   "host": "10.0.0.31", "cu_enabled_override": 40, "backend": "rocm",
   "measured_gemv_gflops": 940.0}
]}
```

## Hardware, and how much of it is really there

| SKU | CUs (phys/enabled) | Fast coffer | Slow coffer | NVMe |
|---|---|---|---|---|
| PS5 / Slim | 40 / 36 | 16 GB @ 448 GB/s | — | 5.5 GB/s |
| PS5 Pro | 64 / 60 | 16 GB @ 576 GB/s | 2 GB DDR5 (OS) | 5.5 GB/s |
| Xbox Series X | 56 / 52 | 10 GB @ 560 GB/s | 6 GB @ 336 GB/s | 2.4 GB/s |
| Xbox Series S | 24 / 20 | 8 GB @ 224 GB/s | 2 GB @ 56 GB/s | 2.4 GB/s |
| AMD 4700S | 40 / **0** | — | 15 GB @ 92 GB/s | SATA 0.55 GB/s |
| AMD 4800S | 56 / **0** | — | 15 GB @ 92 GB/s | 3.5 GB/s |
| BC-250 | 40 / 24 (40 with override) | 16 GB @ 448 GB/s | — | host NVMe |

Every console in this table is *already* a downbinned part, and a real fleet
downbins further in ways the SKU name cannot tell you: a dev-mode sandbox that
hands out 5 GB, a dead memory package, a depopulated channel, a unit clocked
down by a failed fan. `Downbin` records what one unit lost, `ConsoleUnit.
effective()` folds it into the nominal SKU, and the planner reads only the
effective numbers — so the fleet may be arbitrarily heterogeneous and the
planner rebalances instead of refusing.

The salvage boards are worth their own note. The 4700S (a PS5 Ariel die) and the
4800S (a Series X one) have the GPU fused off entirely: CPU-only nodes, and the
GDDR6 that was sized for a GPU becomes a large slow coffer for cold experts. It
is bandwidth-rich for system memory and still short of the 200 GB/s a shelf host
needs, so a fleet of nothing but kits cannot host a shelf at all. A card in the
slot fixes that, but only on the 4800S: the 4700S's slot is PCIe 2.0 x4, and
~2 GB/s cannot feed a GPU from board memory. The BC-250 is the opposite — a PS5
Oberon part with 6 of 8 cores and 24 of 40 CUs enabled, running ordinary Linux
with amdgpu, which makes it the member of this family whose GPU compute path
gets exercised most routinely.

## Backends, and what each can be trusted with

| Backend | Where | Status |
|---|---|---|
| `cpu-avx2` | everywhere (Zen 2, no AVX-512) | built and tested here |
| `vulkan` | PS5 under ps5-linux, BC-250 | shader compiles and validates; not run on a device |
| `rocm` | opt-in, per node | see below |
| `d3d12` | Xbox GDK/devkit only | not implemented |

ROCm on gfx1013 is *supported but not assumed*. Debian ROCm builds do run on
this target, but the library stack is uneven: some libraries do not ship the
target at all, and others fall back to a lower capability level because their
build scripts assume a higher gfx tier. So a node must declare `"backend":
"rocm"` explicitly and supply `measured_gemv_gflops`; the planner will not
invent a ROCm figure for a console. `kernels/expert_hip.hip` is written to build
with plain `hipcc --offload-arch=gfx1013` and deliberately depends on no
rocBLAS, Tensile, or hipBLASLt kernel — the tuned-library layer is exactly the
part that is unreliable here. Vulkan/RADV remains the default GPU path.

## How the model is split

Four decisions, in order:

1. **Shelves.** The fleet is cut into groups of ~22 consoles, dealt
   round-robin in capability order so no shelf inherits all the fast units. A
   shelf is the expert fan-out group: a token's top-6 should be answerable
   without leaving it.
2. **Layer ranges.** Each shelf gets a contiguous run of blocks, sized by its
   capacity. One console per shelf is the *host*: it holds attention, the KV
   cache, the router and the shared expert for those layers, and it must have
   both the room and ≥200 GB/s to do it.
3. **Routed experts** are spread across that shelf's members, weighted by
   measured or nominal throughput — an equal split would make the slowest
   console the tail latency of every token.
4. **Overflow** goes to the slow coffer, then to NVMe as a memory-mapped tier.
   The page cache then keeps whichever experts actually get routed to, which
   converges on the popular ones without anybody profiling them.

A slow plan is still a plan: the planner warns about cross-shelf hops and NVMe
streaming and reports an estimate, and only refuses when the weights genuinely
do not fit in the tiers it is allowed to use.

## Runtime

`G9XC` is the wire protocol: a fixed 32-byte header, request-id multiplexing
over persistent TCP with Nagle off, and one message per *console* per layer
rather than one per expert. See [docs/G9XC.md](docs/G9XC.md).

- `node.py` — the console-side worker: shard store (RAM or mmap'd NVMe),
  expert execution, block forwarding.
- `dispatch.py` — groups a token's experts by console, one batched request each,
  reduces the per-expert replies in the router's top-k order so the same prompt
  gives the same token *whatever console holds which expert*, and fails over to
  a replica when one is available.
- `dedup.py` — a bounded per-node cache keyed by batch id, so a retry after a
  timeout is replayed rather than re-run.
- `coordinator.py` — chains shelves in layer order. An expert-only console can
  be routed around; a shelf *host* cannot, because it owns the KV cache.

## What is measured, what is estimated, what is assumed

**Measured** (on this x86-64 build host, not a console): the CPU kernel at
67 GFLOP/s and 134 GB/s effective; the SPIR-V shader compiles and passes
`spirv-val`; 159 Python tests pass, including the protocol, transport, dispatch
and coordinator paths over real loopback sockets.

**Estimated**: every throughput figure for a console. They come from datasheet
bandwidth, the fan-out and hop structure of the plan, and NVMe read rates.

**Read off the model cards**: both V4 configurations. `deepseek-v4-pro` and
`deepseek-v4-flash` are the published `config.json` files, checked in the tests
against the published parameter counts (1.6 T / 49 B activated and 284 B / 13 B).
An earlier revision of this stack extrapolated V4 Pro from V3 and was wrong in
almost every field; the `assumed` flag and its `ASSUMED CONFIGURATION` stamp
remain in the code for the next unpublished model, but no shipped profile sets
it.

**Interpreted, not published**: how those fields turn into bytes. The paper
gives the architecture, not a storage layout, so the per-layer weight counts in
`HybridAttentionConfig.weight_params` are this repository's reading of it —
validated against the published totals, which is a check on the sum and not on
every term. The split between what a CSA layer *holds* and what it *reads* is
likewise the paper's sparsity claim carried into the planner, not a measurement.

**Not done**: nothing here has run on a PS5, an Xbox, or a BC-250. There is no
D3D12 backend. The Vulkan and HIP kernels have never executed on a device.

See [docs/GEN9_SPLITTING.md](docs/GEN9_SPLITTING.md) for the arithmetic and
[docs/REFERENCES.md](docs/REFERENCES.md) for the prior work this borrows from.

## Tests

```bash
python3 -m unittest discover -s tests -t .   # 159 tests
cd kernels && make && make test              # CPU kernel + FP8 conformance
make vulkan                                  # needs glslang-tools
make hip                                     # needs hipcc; skipped otherwise
```
