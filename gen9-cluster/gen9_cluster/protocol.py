"""G9XC: the wire format between a shelf coordinator and its consoles.

Adapted from the PS3 port's P3XC, with the changes ninth-generation hardware
forces:

* **A console holds many experts, not one.** P3XC could put the expert id in the
  header because a PS3 held exactly one. Here a node holds dozens, and a token
  routing into three of them must not cost three round trips — so
  :class:`ExpertBatchPayload` names a *set* of experts and their gate weights in
  one frame. That single change is what keeps the per-layer latency at one round
  trip per console instead of one per expert.
* **The reply is per-expert rows, not a partial sum.** This is P3XC's rule kept,
  not dropped: a node returns one weighted row per expert, tagged with the
  expert it came from (:class:`ExpertRowsPayload`), and the coordinator adds
  them in strict top-k order. Collapsing a console's experts into one fp32 sum
  re-associates the addition, so the same token would produce different logits
  depending on which console happened to hold which expert — a replan would
  silently change the output. ``Flags.FAST`` asks for the collapsed sum anyway,
  for a caller that has decided it wants the bandwidth back; it is opt-in
  precisely because it is the answer-changing choice.
* **Weights are FP8 with block scales.** The dtype field therefore has to name
  ``fp8-e4m3-b128``, and a frame carrying it also carries its scale block, so a
  shard can be shipped in the format the checkpoint already uses.
* **Little-endian.** P3XC was big-endian because the Cell's PPE was. Every
  console here is x86-64, so the wire order is the host order and encoding is a
  memcpy on both ends.

The framing is deliberately dull: a fixed 32-byte header, then the payload.
Fixed-size headers mean a node can read a header, learn the payload length, and
read exactly that — no delimiter scanning, no partial-parse states to get wrong
at three in the morning when a shelf is wedged.

Every frame carries ``request_id``, and a reply must echo it. The transport
matches replies to requests by that id alone, which is what allows several
requests to be in flight on one connection: at 92 blocks per token and a quarter
of a millisecond per hop, a strictly serial connection would spend most of a
token's time waiting.

``request_id`` is a *transport* id, scoped to one connection and reallocated on
reconnect, so it cannot identify a retried batch. A batch that wants
exactly-once execution carries its own 64-bit ``batch_id`` in the payload; the
node runs it under that id and replays the first attempt's bytes to a retry
(:mod:`gen9_cluster.dedup`).
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

MAGIC = b"G9XC"
#: 2 since the reply to an ``EXPERT_BATCH`` became per-expert rows by default
#: and the batch gained an optional ``batch_id``. Version 1 nodes and version 2
#: coordinators disagree about the shape of every expert reply, which a version
#: check catches at the first frame rather than as wrong logits.
VERSION = 2

#: ``magic(4) version(1) type(1) flags(2) request_id(4) layer(2) expert(2)
#: token(4) dtype(1) rank(1) reserved(2) payload_len(4)`` = 28, padded to 32.
HEADER = struct.Struct("<4sBBHIHHIBBHI4x")
HEADER_SIZE = HEADER.size
assert HEADER_SIZE == 32

#: Refuse anything larger rather than allocating it. A legitimate frame is an
#: activation (tens of KiB) or a shard chunk; 64 MiB is far above both and far
#: below "a malformed length field just asked for all of memory".
MAX_PAYLOAD = 64 * 1024 * 1024


class MsgType(enum.IntEnum):
    HELLO = 1
    HELLO_ACK = 2
    #: Coordinator -> node: run these experts on this activation.
    EXPERT_BATCH = 3
    #: Node -> coordinator: one weighted row per expert, or their sum under
    #: ``Flags.FAST``.
    EXPERT_RESULT = 4
    #: Coordinator -> node: run a hot block (attention + router + shared expert).
    BLOCK_FWD = 5
    BLOCK_RESULT = 6
    #: Residency management.
    LOAD_SHARD = 7
    LOAD_ACK = 8
    #: Liveness and telemetry.
    PING = 9
    PONG = 10
    STATUS = 11
    STATUS_REPLY = 12
    ERROR = 13
    SHUTDOWN = 14


class DType(enum.IntEnum):
    FP32 = 0
    FP16 = 1
    BF16 = 2
    #: FP8 E4M3 with one fp32 scale per 128 elements, as DeepSeek ships it.
    FP8_E4M3_B128 = 3

    @property
    def itemsize(self) -> float:
        return {DType.FP32: 4.0, DType.FP16: 2.0, DType.BF16: 2.0,
                DType.FP8_E4M3_B128: 1.0}[self]

    @property
    def numpy(self) -> np.dtype:
        """The numpy view of the *payload* bytes.

        BF16 and FP8 have no portable numpy dtype, so they travel as raw bytes
        and are converted by the kernel; the tests exercise the fp32 path, which
        is what the reference worker uses.
        """
        names: Dict["DType", str] = {DType.FP32: "<f4", DType.FP16: "<f2",
                                     DType.BF16: "<u2",
                                     DType.FP8_E4M3_B128: "u1"}
        return np.dtype(names[self])


class Flags(enum.IntFlag):
    NONE = 0
    #: The payload is a partial sum that the coordinator must add to others.
    #: Set on a ``FAST`` reply, and only on one.
    PARTIAL = 1 << 0
    #: The sender read this expert from NVMe rather than RAM; the coordinator
    #: uses it to explain a slow layer without guessing.
    FROM_STORAGE = 1 << 1
    #: The node is shedding load and would like fewer of these.
    BACKPRESSURE = 1 << 2
    #: Request: collapse this batch into one gate-weighted sum instead of
    #: returning a row per expert. Opt-in, because it re-associates the layer's
    #: fp32 reduction and can therefore change the token that gets emitted.
    FAST = 1 << 3
    #: Reply: the payload is :class:`ExpertRowsPayload` — one weighted row per
    #: expert, tagged with its expert id. The default shape.
    PER_EXPERT = 1 << 4
    #: Reply: served from the node's dedup cache, i.e. this is a retry of a
    #: batch that already ran and the experts were *not* run a second time.
    REPLAYED = 1 << 5


@dataclass
class Frame:
    """A decoded G9XC frame: a header and an undecoded payload."""

    msg_type: MsgType
    request_id: int
    payload: bytes = b""
    layer: int = 0
    expert: int = 0
    token: int = 0
    dtype: DType = DType.FP32
    rank: int = 0
    flags: Flags = Flags.NONE
    version: int = VERSION

    def encode(self) -> bytes:
        if len(self.payload) > MAX_PAYLOAD:
            raise ValueError(f"payload of {len(self.payload)} bytes exceeds "
                             f"the {MAX_PAYLOAD} byte limit")
        head = HEADER.pack(MAGIC, self.version, int(self.msg_type),
                           int(self.flags), self.request_id, self.layer,
                           self.expert, self.token, int(self.dtype), self.rank,
                           0, len(self.payload))
        return head + self.payload

    @classmethod
    def decode_header(cls, raw: bytes) -> Tuple["Frame", int]:
        """Parse a header, returning the frame and its payload length."""
        if len(raw) != HEADER_SIZE:
            raise ValueError(f"header must be {HEADER_SIZE} bytes, "
                             f"got {len(raw)}")
        (magic, version, msg_type, flags, request_id, layer, expert, token,
         dtype, rank, _reserved, payload_len) = HEADER.unpack(raw)
        if magic != MAGIC:
            raise ValueError(f"bad magic {magic!r}; not a G9XC stream")
        if version != VERSION:
            raise ValueError(f"unsupported G9XC version {version} "
                             f"(this build speaks {VERSION})")
        if payload_len > MAX_PAYLOAD:
            raise ValueError(f"declared payload of {payload_len} bytes exceeds "
                             f"the {MAX_PAYLOAD} byte limit")
        try:
            typed = MsgType(msg_type)
        except ValueError as exc:
            raise ValueError(f"unknown message type {msg_type}") from exc
        frame = cls(msg_type=typed, request_id=request_id, layer=layer,
                    expert=expert, token=token, dtype=DType(dtype), rank=rank,
                    flags=Flags(flags), version=version)
        return frame, payload_len


# -- payload helpers -------------------------------------------------------
#
# Payload layouts are declared here rather than inline at call sites so both
# ends of the wire read from the same description.

_U32 = struct.Struct("<I")
_U16 = struct.Struct("<H")
_U64 = struct.Struct("<Q")


def encode_vector(vec: np.ndarray) -> bytes:
    """A dense fp32 vector: the activation format between consoles.

    Activations stay fp32 on the wire even though the weights are FP8. One
    hidden state is 32 KiB at V4-Pro width, which a gigabit link moves in a
    quarter of a millisecond, and halving it would buy less than the accuracy it
    costs when 8 partial sums from 8 consoles are added together.
    """
    return np.ascontiguousarray(vec, dtype="<f4").tobytes()


def decode_vector(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="<f4")


@dataclass
class ExpertBatchPayload:
    """The experts one console must run for one token, and their gate weights.

    ``expert_ids`` are global ids within the layer, so a node that has been
    handed a different shard range still interprets them the same way.

    ``batch_id`` names this *logical* batch, as distinct from the header's
    per-connection ``request_id``. A coordinator reuses it when it re-sends the
    same batch after a timeout or a dropped socket, which is what lets the node
    recognise the retry and replay its first answer instead of running the
    experts twice. It is optional: a caller that does not care sends ``None``
    and gets at-least-once execution, which for a pure function of
    (activation, weights) is only a waste of a console's time.
    """

    expert_ids: Tuple[int, ...]
    gates: Tuple[float, ...]
    activation: np.ndarray
    batch_id: Optional[int] = None

    def encode(self) -> bytes:
        if len(self.expert_ids) != len(self.gates):
            raise ValueError("expert_ids and gates must be the same length")
        if self.batch_id is not None and not 0 <= self.batch_id < 1 << 64:
            raise ValueError(f"batch_id {self.batch_id} is not a u64")
        activation = encode_vector(self.activation)
        out = [_U16.pack(len(self.expert_ids)),
               # The activation length is stated rather than inferred from
               # what is left in the buffer: a truncated tail is otherwise a
               # perfectly well-formed batch with a shorter hidden state, and
               # the node would happily compute a wrong answer from it.
               _U32.pack(len(activation) // 4),
               # Presence byte rather than a header flag: the payload then
               # parses without reference to the header, so there is exactly
               # one place that can be wrong about whether an id is present.
               b"\x01" if self.batch_id is not None else b"\x00"]
        if self.batch_id is not None:
            out.append(_U64.pack(self.batch_id))
        for eid in self.expert_ids:
            out.append(_U16.pack(eid))
        out.append(np.asarray(self.gates, dtype="<f4").tobytes())
        out.append(activation)
        return b"".join(out)

    @classmethod
    def decode(cls, payload: bytes) -> "ExpertBatchPayload":
        """Parse a batch, refusing anything short.

        Every length is checked before it is used. A truncated frame is either
        a bad peer or a bad network, and the difference between a ValueError
        the node turns into an ERROR reply and a struct.error that escapes as
        an unhandled exception is the difference between a logged incident and
        a dropped connection.
        """
        if len(payload) < _U16.size + _U32.size + 1:
            raise ValueError("expert batch is too short to hold its counts")
        (count,) = _U16.unpack_from(payload, 0)
        (n_activation,) = _U32.unpack_from(payload, _U16.size)
        offset = _U16.size + _U32.size
        has_id = payload[offset]
        offset += 1
        if has_id not in (0, 1):
            raise ValueError(f"expert batch batch_id presence byte is "
                             f"{has_id}, which is neither 0 nor 1")
        batch_id: Optional[int] = None
        if has_id:
            if len(payload) < offset + _U64.size:
                raise ValueError("expert batch claims a batch_id but is "
                                 "truncated before it")
            (batch_id,) = _U64.unpack_from(payload, offset)
            offset += _U64.size
        needed = offset + count * (_U16.size + 4) + n_activation * 4
        if len(payload) != needed:
            raise ValueError(f"expert batch declares {count} experts and a "
                             f"{n_activation}-wide activation, which needs "
                             f"{needed} bytes; got {len(payload)}")
        if count == 0:
            raise ValueError("expert batch names no experts")
        ids = []
        for _ in range(count):
            (eid,) = _U16.unpack_from(payload, offset)
            ids.append(eid)
            offset += _U16.size
        gates = np.frombuffer(payload, dtype="<f4", count=count, offset=offset)
        offset += 4 * count
        activation = np.frombuffer(payload, dtype="<f4", count=n_activation,
                                   offset=offset)
        return cls(tuple(ids), tuple(float(g) for g in gates), activation,
                   batch_id)


@dataclass
class ExpertRowsPayload:
    """One weighted contribution per expert, each tagged with its expert id.

    This is the default reply to an :class:`ExpertBatchPayload`, and the reason
    the whole stack is reproducible. ``rows[i]`` is ``gate_i * expert_i(x)``
    with no cross-expert addition performed anywhere on the node, so the
    coordinator can add every expert of a layer in strict top-k order no matter
    which console each one came from. Move an expert to a different console,
    replan the fleet, lose a node to a replica — the arithmetic is unchanged,
    because the additions never happened in a placement-dependent order in the
    first place.

    The cost is upstream bandwidth, and it is smaller than it looks: a shelf
    wide enough to be worth building already scatters a token's top-k across
    almost as many consoles as there are experts in it, so the sum a node would
    have collapsed is usually one or two terms long. ``Flags.FAST`` buys those
    back and gives up the guarantee.

    Layout::

        n_rows  : uint16
        width   : uint32          elements per row, stated not inferred
        experts : n_rows * uint16
        rows    : n_rows * width * float32
    """

    expert_ids: Tuple[int, ...]
    rows: np.ndarray                    # (n_rows, width) float32

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows, dtype=np.float32)
        if rows.ndim != 2:
            raise ValueError(f"rows must be 2-D (n_rows, width), got shape "
                             f"{rows.shape}")
        if rows.shape[0] != len(self.expert_ids):
            raise ValueError(f"{len(self.expert_ids)} expert ids against "
                             f"{rows.shape[0]} rows")
        self.rows = rows

    def encode(self) -> bytes:
        n_rows, width = self.rows.shape
        out = [_U16.pack(n_rows), _U32.pack(width)]
        for eid in self.expert_ids:
            out.append(_U16.pack(eid))
        out.append(np.ascontiguousarray(self.rows, dtype="<f4").tobytes())
        return b"".join(out)

    @classmethod
    def decode(cls, payload: bytes) -> "ExpertRowsPayload":
        if len(payload) < _U16.size + _U32.size:
            raise ValueError("expert rows payload is too short to hold its "
                             "counts")
        (n_rows,) = _U16.unpack_from(payload, 0)
        (width,) = _U32.unpack_from(payload, _U16.size)
        offset = _U16.size + _U32.size
        needed = offset + n_rows * _U16.size + n_rows * width * 4
        if len(payload) != needed:
            raise ValueError(f"expert rows payload declares {n_rows} rows of "
                             f"width {width}, which needs {needed} bytes; got "
                             f"{len(payload)}")
        if n_rows == 0:
            raise ValueError("expert rows payload carries no rows")
        ids = []
        for _ in range(n_rows):
            (eid,) = _U16.unpack_from(payload, offset)
            ids.append(eid)
            offset += _U16.size
        rows = np.frombuffer(payload, dtype="<f4", count=n_rows * width,
                             offset=offset).reshape(n_rows, width)
        return cls(tuple(ids), rows)


def accumulate(rows: Sequence[np.ndarray]) -> np.ndarray:
    """Add contributions left to right, in the order given.

    The one place in the stack where a sum of expert outputs is formed. It is a
    function rather than ``np.sum(rows, axis=0)`` because numpy sums pairwise:
    that is more accurate, and it is *a different number*. Reproducibility here
    means every summation — on a node collapsing a FAST batch, on the
    coordinator reducing a layer, in a single-process reference run — folds in
    exactly the same order, so a plan change cannot move a bit.

    The caller is responsible for the order being the meaningful one (top-k
    position, not arrival, and not unit id).
    """
    if not rows:
        raise ValueError("nothing to accumulate")
    total = np.array(rows[0], dtype=np.float32, copy=True)
    for row in rows[1:]:
        total += np.asarray(row, dtype=np.float32)
    return total


@dataclass
class ShardHeader:
    """Describes a shard being loaded onto a node."""

    layer: int
    first_expert: int
    n_experts: int
    hidden_size: int
    intermediate_size: int
    dtype: DType = DType.FP32
    #: Which coffer the receiving node should hold it in.
    tier: str = "fast"

    _FIXED = struct.Struct("<HHHIIB")

    def encode(self) -> bytes:
        tier = self.tier.encode("ascii")
        return (self._FIXED.pack(self.layer, self.first_expert, self.n_experts,
                                 self.hidden_size, self.intermediate_size,
                                 int(self.dtype))
                + _U16.pack(len(tier)) + tier)

    @classmethod
    def decode(cls, payload: bytes) -> Tuple["ShardHeader", int]:
        if len(payload) < cls._FIXED.size + _U16.size:
            raise ValueError("shard header is truncated")
        (layer, first, count, hidden, inter,
         dtype) = cls._FIXED.unpack_from(payload, 0)
        offset = cls._FIXED.size
        (length,) = _U16.unpack_from(payload, offset)
        offset += _U16.size
        if len(payload) < offset + length:
            raise ValueError("shard header tier name is truncated")
        tier = payload[offset:offset + length].decode("ascii")
        offset += length
        return cls(layer, first, count, hidden, inter, DType(dtype), tier), offset


@dataclass
class HelloPayload:
    """What a node tells the coordinator when it connects.

    A node announces what it *is*, not what the plan believes it to be, so a
    console that came back from a repair with fewer CUs, a smaller sandbox, or a
    different backend is caught at connect time rather than by mystery
    slowness three hours later.
    """

    unit_id: str
    sku: str
    backend: str
    runtime: str
    weight_bytes: int
    fast_bytes: int
    gemv_gflops: float
    protocol_version: int = VERSION

    def encode(self) -> bytes:
        parts = [self.unit_id, self.sku, self.backend, self.runtime]
        out = [struct.pack("<QQdB", self.weight_bytes, self.fast_bytes,
                           self.gemv_gflops, self.protocol_version)]
        for text in parts:
            raw = text.encode("utf-8")
            out.append(_U16.pack(len(raw)))
            out.append(raw)
        return b"".join(out)

    @classmethod
    def decode(cls, payload: bytes) -> "HelloPayload":
        weight, fast, gflops, version = struct.unpack_from("<QQdB", payload, 0)
        offset = struct.calcsize("<QQdB")
        texts = []
        for _ in range(4):
            (length,) = _U16.unpack_from(payload, offset)
            offset += _U16.size
            texts.append(payload[offset:offset + length].decode("utf-8"))
            offset += length
        return cls(texts[0], texts[1], texts[2], texts[3], weight, fast,
                   gflops, version)


def encode_error(message: str) -> bytes:
    return message.encode("utf-8")[:4096]


def decode_error(payload: bytes) -> str:
    return payload.decode("utf-8", errors="replace")
