# G9XC — the wire protocol, version 2

G9XC carries activations, expert requests and shard loads between a shelf
coordinator and its consoles. It is a descendant of the PS3 port's P3XC, with
the changes ninth-generation hardware forces. Normative implementation:
[`gen9_cluster/protocol.py`](../gen9_cluster/protocol.py).

## What changed from P3XC, and why

**A console holds many experts, not one.** A PS3 had 256 MB and held a single
expert, so P3XC could put the expert id in the header. A PS5 holds dozens, and
a token whose top-6 routes into three experts on one console must not cost three
round trips. `EXPERT_BATCH` therefore names a *set* of experts with their gate
weights and gets one `EXPERT_RESULT` back. At 62 blocks and a
quarter-millisecond hop this is the difference between ~5 ms and ~20 ms of pure
latency per token.

**The reply is one row per expert, not a sum.** This is P3XC's rule *kept*. A
node returns `gate_i * expert_i(x)` for each expert it was asked for, tagged
with the expert id, and the coordinator adds them in the router's top-k order.
See [Reproducibility](#reproducibility) for why, and `FAST` for the way out.

**Weights are FP8 with block scales.** The dtype field names
`fp8-e4m3-b128` and a `LOAD_SHARD` carrying it also carries its scales, so a
shard ships in the format the checkpoint already uses and is never materialised
at fp32 on the way.

**Little-endian.** P3XC was big-endian because the Cell's PPE was. Every console
here is x86-64, so wire order is host order and encoding is a memcpy.

## Framing

A fixed 32-byte header, then exactly `payload_len` bytes. Fixed-size headers
mean a reader can read a header, learn the length, and read exactly that: no
delimiter scanning, no partial-parse states.

```
offset size field
  0     4   magic        "G9XC"
  4     1   version      2
  5     1   type         MsgType
  6     2   flags        Flags bitfield
  8     4   request_id   echoed by the reply
 12     2   layer
 14     2   expert       first expert, where a single one is meant
 16     4   token        position, for KV-cache correlation
 20     1   dtype        DType
 21     1   rank         tensor rank hint
 22     2   reserved     zero
 24     4   payload_len  bytes following the header
 28     4   padding      zero, to align the payload to 32
```

`struct` format: `<4sBBHIHHIBBHI4x`.

A header is rejected unless it is exactly 32 bytes, starts with `G9XC`, has
version 2, names a known `MsgType`, and declares `payload_len <= 64 MiB`. The
size cap exists so a corrupt length field cannot ask for all of memory; a
legitimate frame is an activation (tens of KiB) or a shard chunk.

## Multiplexing

Every frame carries `request_id`; a reply echoes it. The transport matches
replies to requests by that id alone, so several requests may be in flight on
one connection. Connections are persistent with `TCP_NODELAY` set — at 62
blocks per token, connection setup or Nagle delay would dominate.

Request ids are per-connection, wrap at 2³², and skip 0.

## Message types

| # | Type | Direction | Payload |
|---|---|---|---|
| 1 | `HELLO` | node → coord | `HelloPayload` |
| 2 | `HELLO_ACK` | coord → node | empty |
| 3 | `EXPERT_BATCH` | coord → node | `ExpertBatchPayload` |
| 4 | `EXPERT_RESULT` | node → coord | `ExpertRowsPayload`, or a vector under `FAST` |
| 5 | `BLOCK_FWD` | coord → node | vector (hidden state) |
| 6 | `BLOCK_RESULT` | node → coord | vector |
| 7 | `LOAD_SHARD` | coord → node | `ShardHeader` + body |
| 8 | `LOAD_ACK` | node → coord | empty |
| 9 | `PING` / 10 `PONG` | either | empty |
| 11 | `STATUS` / 12 `STATUS_REPLY` | coord → node | JSON |
| 13 | `ERROR` | either | UTF-8 message, ≤4096 bytes |
| 14 | `SHUTDOWN` | coord → node | empty |

### Flags

| bit | name | direction | meaning |
|---|---|---|---|
| 0 | `PARTIAL` | reply | a collapsed partial sum; set on a `FAST` reply and only there |
| 1 | `FROM_STORAGE` | reply | served from NVMe, not RAM — why this layer was slow |
| 2 | `BACKPRESSURE` | reply | the node wants less of this |
| 3 | `FAST` | request | collapse this batch into one sum; opt-in, see below |
| 4 | `PER_EXPERT` | reply | the payload is `ExpertRowsPayload`. The default |
| 5 | `REPLAYED` | reply | served from the dedup cache; the experts did not run again |

`FROM_STORAGE` is the one worth setting carefully. It is how a coordinator
explains a slow layer instead of guessing at it.

A coordinator **must reject** an `EXPERT_RESULT` carrying `PARTIAL` for a batch
that did not set `FAST`. Such a reply is a node that re-associated the sum
without being asked, and its number cannot be placed in top-k order; accepting
it would produce a plausible wrong answer, which is the exact failure this
design exists to prevent.

### `ExpertBatchPayload`

```
u16  n_experts
u32  n_activation        (elements, not bytes)
u8   has_batch_id        0 or 1; any other value is malformed
u64  batch_id            present only if has_batch_id == 1
u16  expert_id           x n_experts
f32  gate                x n_experts
f32  activation          x n_activation
```

The activation length is *stated*, not inferred from the remaining buffer. A
truncated tail would otherwise be a perfectly well-formed batch with a shorter
hidden state, and the node would compute a confidently wrong answer from it.
A decoder must require the payload to be exactly the implied size, and must
reject a batch naming zero experts.

`batch_id` names the *logical* batch, as distinct from the header's
`request_id`, which is per-connection and is reallocated on reconnect. A
coordinator reuses one `batch_id` across every attempt at the same batch; see
[Errors and retries](#errors-and-retries).

### `ExpertRowsPayload`

The default `EXPERT_RESULT`, flagged `PER_EXPERT`.

```
u16  n_rows
u32  width               elements per row
u16  expert_id           x n_rows
f32  row                 x n_rows * width   (row-major)
```

`row[i]` is `gate_i * expert_i(activation)`, with no cross-expert addition
performed on the node. Rows may be returned in any order; the tag is what
matters. A decoder must require the exact implied length and reject zero rows.

A coordinator must check that the set of returned ids equals the set it asked
for. A missing row is a missing term in the layer's sum, and silently dropping
one costs an expert's contribution without any error being raised.

### `ShardHeader`

```
u16  layer
u16  first_expert
u16  n_experts
u32  hidden_size
u32  intermediate_size
u8   dtype
u16  tier_len
     tier                ascii: "fast" | "slow" | "ssd"
```

followed by the body. For `FP32`, the body is
`n_experts × 3 × hidden × intermediate` little-endian floats, in gate, up, down
order per expert.

For `FP8_E4M3_B128` the body is **all codes, then all block scales**:

```
u8   code    x n_experts * 3 * hidden * intermediate
f32  scale   x n_experts * 3 * ceil(hidden * intermediate / 128)
```

Separated rather than interleaved, so the codes land contiguous and
page-aligned in the coffer: a GPU backend uploads them as one buffer, and a
memory-mapped shard reads them without a gather. The scales are three orders of
magnitude smaller and can afford to be a second, small transfer. A body of the
wrong length is refused with `ERROR` — a shard whose scales are missing would
otherwise decode to plausible-looking noise.

### `HelloPayload`

```
u64  weight_bytes
u64  fast_bytes
f64  gemv_gflops
u8   protocol_version
     then 4 length-prefixed utf-8 strings:
     unit_id, sku, backend, runtime
```

A node announces what it *is*, not what the plan believes it to be. A console
back from a repair with fewer CUs, a smaller sandbox or a different backend is
then caught at connect time rather than by mystery slowness three hours later.

## Errors and retries

A node turns a malformed request into an `ERROR` frame with the same
`request_id`; it does not drop the connection. The transport raises the error
to the caller attributed to the unit that produced it, so a fleet-wide symptom
names one console.

Retry safety is per operation, not per error:

- `EXPERT_BATCH` is **stateless** — the same experts on the same activation
  give the same rows — so it may be retried, and may be re-sent to a replica
  console that holds *all* of the failed batch's experts.
- `BLOCK_FWD` is **stateful**: it appends to the host's MLA KV cache. It is
  never retried automatically, and a shelf host is never failed over, because
  the KV cache for those layers exists only there.

### Exactly-once, and where it stops

A coordinator that times out cannot tell whether the console ran the experts
before the socket died or after, so a plain retry is *at-least-once*. Every
attempt at one batch therefore carries the same `batch_id`, and a node keeps a
bounded cache of what it answered
([`gen9_cluster/dedup.py`](../gen9_cluster/dedup.py)):

- Same `batch_id`, same content: the first attempt's bytes are replayed,
  flagged `REPLAYED`, and the weights are not touched again.
- Same `batch_id`, still running: the retry **waits** for the first attempt
  rather than starting a second pass. This is the common case on a console that
  is merely slow, and it is where the saving actually is.
- Same `batch_id`, different content: rejected with `ERROR`. Replaying the
  wrong activation's output would be a wrong token that no log explains.
- No `batch_id`: no deduplication. Legal, and only costs a console's time.

The cache is bounded three ways — entries, total cached bytes, and a TTL — and
never evicts a batch that is still running; when every tracked batch is in
flight a new one is refused with `BACKPRESSURE` rather than something in flight
being dropped. A failed batch is not cached as an answer, and a reply too large
for the byte budget is returned but not retained.

**This is per node process.** A retry sent to a *different* console (the
replica path) cannot be deduplicated without shared state, which G9XC does not
have and does not want. That case is safe for a different reason: the
coordinator uses one reply and discards the other, so the duplicated execution
cannot reach the sum. A node restart also clears the cache, and a retry across
it re-executes.

## Reproducibility

Floating-point addition is not associative, so *the order a layer's experts are
summed in is part of the answer*. G9XC fixes that order at the only place that
knows it: the router's top-k ranking.

- A node never adds two experts together unless asked. It returns rows.
- The coordinator folds those rows left to right in top-k order, whatever order
  the replies arrived in and whatever console each came from.
- Every summation in the stack — coordinator reduction, a node collapsing a
  `FAST` batch, a single-process reference run — goes through one `accumulate`
  helper that folds left to right. Notably *not* `np.sum`, which sums pairwise:
  that is more accurate, and it is a different number.

What this buys: the same prompt gives the same bits regardless of how the
planner spread the experts over the shelf. On a fleet of second-hand consoles
that is not a theoretical property — a unit dies, the planner reshuffles, and
without this the model would quietly start emitting different tokens, with no
error anywhere and nothing in the logs pointing at the console that failed.

### What it does not buy

Reduction order is fixed; **kernel arithmetic is not**. A shelf may mix
`cpu-avx2`, `vulkan` and `rocm` nodes, and those three do not accumulate an
expert's FMAs in the same order internally. Two consoles running the same
expert on different backends may therefore return rows differing in the last
bits, and the layer sum differs with them.

So, precisely: **for a fixed assignment of experts to backends, the output is
reproducible and placement-independent.** Bit-identity across a heterogeneous
fleet would additionally require pinning one reference kernel, which this stack
deliberately does not do — its whole point is to use whatever silicon is in the
rack. A deployment that needs fleet-wide bit-identity must run one backend.

### `FAST`

Setting `FAST` on an `EXPERT_BATCH` asks the node to collapse its experts into
one gate-weighted sum, replied with `PARTIAL`. The coordinator then has no
per-expert structure to reduce and falls back to summing by unit id: still
deterministic run to run, no longer invariant under a replan.

The trade is reply bandwidth. Exact mode sends `k` rows per layer where `FAST`
sends one per console touched. At the published V4-Pro profile (hidden 7168,
k=6, 61 MoE layers) that is ~5.2 MB/token exact against ~4.7 MB/token
collapsed, because a shelf wide enough to be worth building already scatters
the top-6 across ~5.4 of its 22 consoles — the sum being collapsed is usually
one or two terms long. ~12% of reply bandwidth is a poor price for an answer
that depends on the current plan, which is why exact is the default and `FAST`
is a flag.

## Versioning

`version` is a hard match: a frame with a different version is rejected at the
header. Adding a message type is backwards compatible (unknown types are
rejected as malformed, which is the intended outcome for a peer that should not
be sending them); changing a payload layout is not, and requires the version
byte to move.

**v1 → v2.** `EXPERT_RESULT` became `ExpertRowsPayload` by default, and
`ExpertBatchPayload` gained `has_batch_id`/`batch_id`. A v1 node and a v2
coordinator disagree about the shape of every expert reply, so the version
check catches it at the first frame rather than as wrong logits an hour later.
