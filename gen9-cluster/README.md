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
   │ host: Series X   layers 0-9   │   │ host: Series X   layers 10-19 │
   │   attention + MLA KV cache    │   │                               │
   │   router + shared expert      │──▶│   ... 74 blocks over 8 shelves│
   │ 21 more consoles: routed      │   │                               │
   │   experts, ~8 lit per token   │   │                               │
   └───────────────────────────────┘   └───────────────────────────────┘
```

## Why this is possible at all

DeepSeek V4 Pro (assumed configuration — see below) is ~1345 GiB of FP8 weights
and no console has more than 16 GB. Three properties make the fleet work
anyway:

1. **Only ~4% of the model runs per token.** 52 of 1359 billion parameters
   activate: attention, the router, one shared expert, and 8 of 384 routed
   experts per layer. The other 96% only has to be *stored*, and storage is the
   one thing a pile of consoles has.
2. **Decode is bandwidth-bound, not compute-bound.** A PS5's 448 GB/s is what
   matters, and it is a genuinely good number — a 4700S carries the same DRAM
   on the same 256-bit bus, but reaching it from the CPU instead of the GPU
   measures 92.9 GB/s, a fifth as much, and the planner knows the difference.
3. **FP8 is the native format.** One byte per parameter, with a scale per
   128-element block, halves the console count against BF16 and roughly doubles
   the token rate. `kernels/fp8.c` and `gen9_cluster/fp8.py` implement E4M3FN
   and are tested against each other over all 256 codes.

The result is a plan, not a benchmark: **~9.4 tok/s estimated** for V4 Pro at
8k context on 170 consoles (100 PS5 + 40 Series X + 30 BC-250). That figure is
arithmetic over memory bandwidth, network hops, and NVMe reads. Nothing in this
repository has run on a console yet.

## Quick start

```bash
cd gen9-cluster
python3 -m gen9_cluster model --model deepseek-v4-pro    # what it costs
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
   shelf is the expert fan-out group: a token's top-8 should be answerable
   without leaving it.
2. **Layer ranges.** Each shelf gets a contiguous run of blocks, sized by its
   capacity. One console per shelf is the *host*: it holds attention, the MLA KV
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

**Assumed**: DeepSeek V4 Pro's configuration. No V4 Pro config is public. The
profile here extrapolates the V2→V3 progression — 72 layers, hidden 8192, 384
routed experts, top-8, MLA rank 512, 2 MTP heads, FP8 — and every plan it
produces is stamped `ASSUMED CONFIGURATION`. `deepseek-v3` is a real
configuration and is the honest thing to plan against today.

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
