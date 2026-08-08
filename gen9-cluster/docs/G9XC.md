# G9XC — the wire protocol, version 1

G9XC carries activations, expert requests and shard loads between a shelf
coordinator and its consoles. It is a descendant of the PS3 port's P3XC, with
the changes ninth-generation hardware forces. Normative implementation:
[`gen9_cluster/protocol.py`](../gen9_cluster/protocol.py).

## What changed from P3XC, and why

**A console holds many experts, not one.** A PS3 had 256 MB and held a single
expert, so P3XC could put the expert id in the header. A PS5 holds dozens, and
a token whose top-8 routes into three experts on one console must not cost three
round trips. `EXPERT_BATCH` therefore names a *set* of experts with their gate
weights, and the node replies with their gate-weighted sum as one
`EXPERT_RESULT`. At 74 blocks and a quarter-millisecond hop this is the
difference between ~5 ms and ~20 ms of pure latency per token.

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
  4     1   version      1
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
version 1, names a known `MsgType`, and declares `payload_len <= 64 MiB`. The
size cap exists so a corrupt length field cannot ask for all of memory; a
legitimate frame is an activation (tens of KiB) or a shard chunk.

## Multiplexing

Every frame carries `request_id`; a reply echoes it. The transport matches
replies to requests by that id alone, so several requests may be in flight on
one connection. Connections are persistent with `TCP_NODELAY` set — at 74
blocks per token, connection setup or Nagle delay would dominate.

Request ids are per-connection, wrap at 2³², and skip 0.

## Message types

| # | Type | Direction | Payload |
|---|---|---|---|
| 1 | `HELLO` | node → coord | `HelloPayload` |
| 2 | `HELLO_ACK` | coord → node | empty |
| 3 | `EXPERT_BATCH` | coord → node | `ExpertBatchPayload` |
| 4 | `EXPERT_RESULT` | node → coord | vector (gate-weighted sum) |
| 5 | `BLOCK_FWD` | coord → node | vector (hidden state) |
| 6 | `BLOCK_RESULT` | node → coord | vector |
| 7 | `LOAD_SHARD` | coord → node | `ShardHeader` + body |
| 8 | `LOAD_ACK` | node → coord | empty |
| 9 | `PING` / 10 `PONG` | either | empty |
| 11 | `STATUS` / 12 `STATUS_REPLY` | coord → node | JSON |
| 13 | `ERROR` | either | UTF-8 message, ≤4096 bytes |
| 14 | `SHUTDOWN` | coord → node | empty |

### Flags

| bit | name | meaning |
|---|---|---|
| 0 | `PARTIAL` | a partial sum the coordinator must add to others |
| 1 | `FROM_STORAGE` | served from NVMe, not RAM — why this layer was slow |
| 2 | `SHEDDING` | the node wants less of this |

`FROM_STORAGE` is the one worth setting carefully. It is how a coordinator
explains a slow layer instead of guessing at it.

### `ExpertBatchPayload`

```
u16  n_experts
u32  n_activation        (elements, not bytes)
u16  expert_id           x n_experts
f32  gate                x n_experts
f32  activation          x n_activation
```

The activation length is *stated*, not inferred from the remaining buffer. A
truncated tail would otherwise be a perfectly well-formed batch with a shorter
hidden state, and the node would compute a confidently wrong answer from it.
A decoder must require the payload to be exactly the implied size, and must
reject a batch naming zero experts.

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
  give the same sum — so it may be retried, and may be re-sent to a replica
  console that holds *all* of the failed batch's experts.
- `BLOCK_FWD` is **stateful**: it appends to the host's MLA KV cache. It is
  never retried automatically, and a shelf host is never failed over, because
  the KV cache for those layers exists only there.

## Versioning

`version` is a hard match in v1: a frame with a different version is rejected
at the header. Adding a message type is backwards compatible (unknown types are
rejected as malformed, which is the intended outcome for a peer that should not
be sending them); changing a payload layout is not, and requires the version
byte to move.
