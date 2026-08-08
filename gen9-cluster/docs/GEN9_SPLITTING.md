# Splitting DeepSeek V4 Pro across a console fleet

The arithmetic behind `gen9_cluster.planner`, and what it does and does not
claim.

## The problem

DeepSeek V4 Pro is **803 GiB** — 1.6 T parameters, but MXFP4 experts and FP8
everywhere else come to 0.54 bytes each. The largest console in this family has
16 GB of RAM, of which about 13 GB is reachable, so the model needs ~62
consoles just to be *stored* and 73 once each shelf host also has to hold its
layers' hot blocks and cache.

What makes this tractable rather than absurd is that MoE decode touches almost
none of it:

| per token, per MoE layer | size |
|---|---|
| attention (CSA layer, incl. indexer and grouped output) | ~353 MiB |
| attention (HCA layer) | ~290 MiB |
| router + shared expert | ~36 MiB |
| 6 routed experts of 384 | ~201 MiB |
| **total read** | **~530-590 MiB** |
| stored but untouched | ~12.4 GiB |

50 of 1600 billion parameters activate. The other 97% has to be held somewhere
with an address, and that is precisely what a stack of consoles is good for.

V4 Flash is the same architecture at a fifth of the scale — 148 GiB, 43 layers,
256 experts — and its floor is **20 PS5s**, which is the first point in this
family where the fleet is a shelf rather than a rack.

## The four decisions

### 1. Shelves

The fleet is cut into groups of ~22 (`DEFAULT_SHELF_SIZE`), dealt round-robin
in descending capability order. Round-robin rather than contiguous because a
shelf's slowest member sets the floor for every layer it owns: one shelf
inheriting all the Series X units and another all the Series S ones would make
the second shelf's layers permanently the slow ones.

A shelf is the *expert fan-out group*. A token's top-6 experts for a layer
should be answerable inside one shelf, so a layer costs one fan-out and one
gather, not a tour of the fleet.

### 2. Layer ranges

Each shelf takes a contiguous run of blocks, sized in proportion to its
capacity, so a shelf of Series S consoles takes fewer layers rather than the
same number and overflowing to NVMe.

The block count is `n_layers + n_mtp_heads` — the MTP heads are full decoder
blocks and have to be placed like any other.

One member of each shelf is the **host**. It holds, for that shelf's layers:

- attention weights and that layer's KV cache,
- the router,
- the shared expert (always on, so it must never be a network hop),
- the layer norms.

A host must have both the room and ≥200 GB/s (`HOT_BANDWIDTH_FLOOR_GBPS`). That
floor is why a 4700S — a PS5 die with the GPU fused off, its GDDR6 measuring
92.9 GB/s from the CPU — can hold cold experts but is never chosen as a host.

Compressed attention is what makes long context affordable, and in V4 it is
compressed along the *sequence*: a CSA layer folds every 4 tokens into one
512-wide shared-KV entry and an HCA layer every 128, so the whole cache at a
million tokens is 8.2 GiB rather than the 65 GiB V3's MLA would need — and at
8k it is 74 MiB, small enough that context is not what decides the fleet size.

The two kinds are interleaved, so **neighbouring layers differ by 32x in cache
size** and the planner sizes each block separately (`block_kv_bytes`). Sizing
from an average would under-provision every CSA host and waste room on every
HCA one.

### 3. Routed experts

Spread across the shelf's members, weighted by throughput — measured where the
inventory gives a figure, nominal otherwise. An equal split would make the
slowest console in the shelf the tail latency of every token that routes to it,
and with top-6 across 22 consoles, most tokens touch a slow one.

### 4. Overflow

In order: fast coffer → slow coffer → NVMe as a memory-mapped tier.

Mapping rather than reading is what makes NVMe usable. The page cache then
retains whichever experts actually get routed to, which converges on the
popular ones without anybody having to profile them — the same effect
KTransformers gets by explicitly pinning hot experts to the GPU, arrived at by
letting the kernel do it.

## Throughput estimate

Per token, the estimate sums:

- per block: `hot_bytes / (host bandwidth × efficiency)`,
- per block: the KV the block has to *read*, which under CSA is not the KV it
  holds — the FP4 indexer scans every compressed entry but attention then reads
  only the 1024 it selected, so a million-token CSA layer reads ~17 MiB of a
  260 MiB cache. Treating residency as the read cost would overstate
  long-context decode by an order of magnitude,
- per block: the slowest expert console's `expert_bytes / bandwidth`, since the
  fan-out is concurrent and the gather waits for the last reply,
- NVMe reads where experts live there,
- `(n_shelves − 1) × hop_seconds` for crossing between shelves.

`BANDWIDTH_EFFICIENCY` derates each backend from datasheet peak. Hop cost
defaults to 250 µs, which is a plausible switched-gigabit round trip — for V4
that activation is 56 KiB rather than 14, because hyper-connections make the
residual stream four times hidden-wide and a pipeline boundary has to carry all
of it. The hop cost is still the single most consequential guess in the model.

Worked example, 100 PS5 + 40 Series X + 30 BC-250:

```
deepseek-v4-pro on 170 consoles in 8 shelves
  context               8192 tokens
  decode estimate       10.84 tok/s (92 ms/token)
  consoles per layer    ~6 lit by one token
  streamed from NVMe    3.0 GiB of experts
```

Seven inter-shelf hops cost ~1.8 ms of the 92 ms, so this configuration is
memory-bandwidth-bound rather than network-bound. That flips if the fleet is
built from many small consoles: 300 Series S units would be 14 shelves and the
hops start to matter. It is also the reason the eventual PS6 answer is *fewer,
larger* nodes rather than more of these.

## Why the planner warns instead of refusing

A slow plan is still a plan. Warnings cover: assumed model configuration,
inter-shelf hop count, NVMe streaming volume, cross-shelf expert spill, and
shelves left with no layers. A `PlanningError` is raised only when the weights
genuinely do not fit in the tiers the planner was allowed to use, and the
message says what would fix it (more fast-coffer consoles, or a shorter
context).

## Failure behaviour

- **An expert-only console dies**: its batch is retried on a replica that holds
  *all* of the failed batch's experts, since a partial replica would silently
  drop terms from the sum. Where no such replica exists, the token fails
  visibly rather than quietly losing experts.
- **A shelf host dies**: no failover. The KV cache for those layers exists
  only there, so continuing would produce a fluent, wrong continuation. The
  request fails and the session must be replayed elsewhere.

## Status of every number here

**Measured on the x86-64 build host** (not a console): CPU kernel 67 GFLOP/s /
134 GB/s effective; SPIR-V compiles and passes `spirv-val`; 159 Python tests
pass.

**Estimated**: all console throughput. Datasheet bandwidth, derated, plus hop
and NVMe terms.

**Read from the model cards**: both V4 configurations, checked in the tests
against their published parameter counts. How those fields become bytes — the
per-layer weight terms, and the split between cache held and cache read — is
this repository's reading of the paper, validated on the totals rather than
term by term.

**Not attempted**: execution on a PS5, an Xbox, or a BC-250.
