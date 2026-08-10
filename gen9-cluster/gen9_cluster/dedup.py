"""Bounded batch-id dedup, so a retried expert batch runs once.

A coordinator whose socket dies mid-batch cannot tell whether the node ran the
experts before the connection went away or after. Re-sending is therefore
*at-least-once execution*. For routed experts that is merely wasteful — the
computation is a pure function of (activation, weights) and mutates nothing —
but "merely wasteful" is a console-second on hardware that has single-digit
tokens per second to give, and it is spent at exactly the moment the fleet is
already in trouble.

This cache turns it back into exactly-once. A batch carries a 64-bit
``batch_id`` (:class:`~gen9_cluster.protocol.ExpertBatchPayload`); the node runs
it under that id and remembers the encoded reply, so a retry of the same logical
batch gets the first attempt's bytes rather than a second pass over the weights.
A retry that arrives while the first attempt is still running *waits* for it
instead of racing it, which also collapses the duplicate-work case that a
coordinator with an over-eager timeout creates on a slow console.

The cache binds each id to a fingerprint of the request content. A reused id
carrying a different activation, layer, expert list or gate vector is a
coordinator bug, and it is rejected rather than answered with a stale reply:
replaying the wrong bytes would be a wrong token that no log would explain.

Three bounds keep this from becoming unbounded per-token state on a console with
10 GB of usable memory and a model to hold in it:

* ``max_entries`` — at most this many logical batches tracked. When every
  tracked batch is still running, a new one is refused with
  :class:`DedupCapacityError` rather than evicting work in flight; completed
  batches are evicted oldest-first to make room.
* ``max_bytes`` — completed replies are counted precisely and evicted
  least-recently-finished until a new one fits. A reply larger than the whole
  budget is returned but not retained, so a later retry re-executes it.
* ``ttl`` — a completed reply older than this is dropped on the next touch. A
  retry after that re-executes, and is honestly at-least-once again.

A retry that lands on a *different console* (the dispatcher's replica path)
cannot be deduplicated without shared state, which this deliberately does not
introduce. That case is safe for a different reason: the coordinator accepts one
reply and discards the other, so the duplicate execution cannot reach the sum.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Retained replies per node. A shelf coordinator has at most a few batches in
#: flight per node at a time; this is generous enough to cover a burst of
#: retries without being a memory decision anyone has to think about.
DEFAULT_DEDUP_ENTRIES = 128

#: How long a completed reply stays replayable, in seconds. Longer than any
#: sane request timeout, short enough that a wedged fleet drains.
DEFAULT_DEDUP_TTL = 60.0

#: Total completed-reply byte budget. At V4-Pro width one exact reply is
#: ``k`` rows of 32 KiB, so 32 MiB is a few hundred cached replies — noise
#: against a coffer, and bounded whatever the traffic does.
DEFAULT_DEDUP_BYTES = 32 * 1024 * 1024


class MismatchedBatchError(ValueError):
    """The same batch id arrived with different content."""


class DedupCapacityError(RuntimeError):
    """Every tracked batch is in flight; there is no room to start another."""


def batch_fingerprint(layer: int, expert_ids: Sequence[int],
                      gates: Sequence[float], activation: np.ndarray,
                      fast: bool) -> bytes:
    """A stable digest of everything that changes a batch's correct answer.

    Deliberately includes ``fast``: the collapsed sum and the per-expert rows
    are different bytes and different arithmetic, so serving one where the
    other was asked for is not a cache hit.
    """
    digest = hashlib.sha256()
    digest.update(b"g9xc-batch\x00")
    digest.update(int(layer).to_bytes(4, "little"))
    digest.update(b"\x01" if fast else b"\x00")
    digest.update(len(expert_ids).to_bytes(4, "little"))
    for expert in expert_ids:
        digest.update(int(expert).to_bytes(4, "little"))
    digest.update(np.ascontiguousarray(gates, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(activation, dtype="<f4").tobytes())
    return digest.digest()


class _Slot:
    """One logical batch: in flight, then its reply bytes."""

    __slots__ = ("done", "reply", "error", "started", "finished",
                 "fingerprint", "cached", "size")

    def __init__(self, fingerprint: bytes) -> None:
        self.done = threading.Event()
        self.reply: Optional[bytes] = None
        self.error: Optional[BaseException] = None
        self.started = time.monotonic()
        self.finished: Optional[float] = None
        self.fingerprint = fingerprint
        self.cached = True          # False for errors and oversized replies
        self.size = 0


class DedupCache:
    """Runs a batch at most once per ``(batch_id, fingerprint)``, within bounds."""

    def __init__(self, max_entries: int = DEFAULT_DEDUP_ENTRIES,
                 ttl: float = DEFAULT_DEDUP_TTL,
                 max_bytes: int = DEFAULT_DEDUP_BYTES) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if ttl <= 0:
            raise ValueError("ttl must be > 0")
        if max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")
        self.max_entries = max_entries
        self.ttl = ttl
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._slots: Dict[int, _Slot] = {}
        self._bytes = 0
        #: Observability. The tests assert on these, and so should an operator
        #: wondering why a shelf is doing more work than the plan says.
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expiries = 0
        self.rejected = 0
        self.oversized = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._slots)

    @property
    def bytes_used(self) -> int:
        with self._lock:
            return self._bytes

    def run(self, batch_id: Optional[int], fingerprint: bytes,
            compute: Callable[[], bytes],
            timeout: Optional[float] = None) -> Tuple[bytes, bool]:
        """Return ``(reply_bytes, replayed)``, computing the reply at most once.

        ``batch_id`` of ``None`` always computes: an unnamed batch has nothing
        to deduplicate on.
        """
        if batch_id is None:
            return compute(), False
        with self._lock:
            self._expire_locked()
            existing = self._slots.get(batch_id)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    raise MismatchedBatchError(
                        f"batch id {batch_id} was reused with different "
                        f"content; refusing to replay the first answer")
                self.hits += 1
            else:
                if not self._reserve_locked():
                    self.rejected += 1
                    raise DedupCapacityError(
                        f"all {self.max_entries} tracked batches are in flight")
                self.misses += 1
                slot = self._slots[batch_id] = _Slot(fingerprint)
        # Deliberately outside the lock: waiting for someone else's batch must
        # not stall every other batch on this console for the length of a
        # kernel run.
        if existing is not None:
            return self._await(batch_id, existing, timeout), True
        try:
            reply = compute()
        except BaseException as exc:            # noqa: BLE001 - re-raised
            slot.error = exc
            slot.finished = time.monotonic()
            slot.cached = False
            slot.done.set()
            with self._lock:
                self._slots.pop(batch_id, None)
            raise
        self._finish(batch_id, slot, reply)
        return reply, False

    def replay(self, batch_id: int) -> Optional[bytes]:
        """The remembered reply for ``batch_id``, if it is still cached."""
        with self._lock:
            self._expire_locked()
            slot = self._slots.get(batch_id)
        if slot is None or not slot.done.is_set() or not slot.cached:
            return None
        return slot.reply

    def clear(self) -> None:
        with self._lock:
            self._slots.clear()
            self._bytes = 0

    # -- internals ---------------------------------------------------------
    def _await(self, batch_id: int, slot: _Slot,
               timeout: Optional[float]) -> bytes:
        """Wait for the attempt already running under this id."""
        if not slot.done.wait(timeout):
            raise TimeoutError(f"batch {batch_id} is still running from an "
                               f"earlier attempt")
        if slot.error is not None:
            raise slot.error
        if slot.reply is None:
            raise RuntimeError(f"batch {batch_id} completed without a reply")
        return slot.reply

    def _reserve_locked(self) -> bool:
        """Make room for one new batch without evicting anything in flight."""
        while len(self._slots) >= self.max_entries:
            victim = self._oldest_completed_locked()
            if victim is None:
                return False
            self._drop_locked(victim)
            self.evictions += 1
        return True

    def _finish(self, batch_id: int, slot: _Slot, reply: bytes) -> None:
        slot.reply = reply
        slot.finished = time.monotonic()
        slot.size = len(reply)
        slot.done.set()
        with self._lock:
            if slot.size > self.max_bytes:
                # Too large to retain. Answer it, then forget it, so a later
                # retry re-executes rather than the budget being blown by one
                # outlier. Waiters already woken hold this reply.
                slot.cached = False
                self.oversized += 1
                self._slots.pop(batch_id, None)
                return
            self._make_room_locked(slot.size)
            self._bytes += slot.size

    def _make_room_locked(self, needed: int) -> None:
        while self._bytes + needed > self.max_bytes:
            victim = self._oldest_completed_locked()
            if victim is None:
                break
            self._drop_locked(victim)
            self.evictions += 1

    def _oldest_completed_locked(self) -> Optional[int]:
        candidates: List[Tuple[float, int]] = [
            (slot.finished, bid) for bid, slot in self._slots.items()
            if slot.done.is_set() and slot.cached and slot.finished is not None]
        return min(candidates)[1] if candidates else None

    def _expire_locked(self) -> None:
        cutoff = time.monotonic() - self.ttl
        stale = [bid for bid, slot in self._slots.items()
                 if slot.done.is_set() and slot.cached
                 and slot.finished is not None and slot.finished < cutoff]
        for bid in stale:
            self._drop_locked(bid)
            self.expiries += 1

    def _drop_locked(self, batch_id: int) -> None:
        slot = self._slots.pop(batch_id, None)
        if slot is not None and slot.done.is_set() and slot.cached:
            self._bytes = max(0, self._bytes - slot.size)
