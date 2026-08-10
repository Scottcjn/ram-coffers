"""Fan a token's top-k out to the consoles that hold those experts.

This is the hot path: it runs once per MoE layer per token, so ~69 times per
token on the V4-Pro profile. Three decisions define it.

**Group by console before sending.** A token's eight experts may live on three
consoles. Naively that is eight requests; grouped, it is three, and each console
answers for every expert it holds in one reply. Round trips, not FLOPs, are what
a console fleet is short of.

**Send concurrently, reduce in top-k order.** The requests go out together and
land in whatever order the network decides, but the reduction does not depend on
the network *or on the placement*. Each console returns one row per expert,
tagged with the expert it came from, and this module adds them strictly in the
order the router ranked them — the same order, and therefore bit-for-bit the
same sum, that a single machine holding all the experts would produce.

Reducing by unit id instead would be reproducible run to run but not across
plans: floating-point addition is not associative, so moving one expert to
another console would silently change the logits and eventually the token. On a
fleet of second-hand consoles, where a dead unit means a replan, that is a
model whose output quietly depends on which console last failed.
``fast=True`` accepts exactly that in exchange for the bandwidth.

**Retry only what is safe to retry.** An expert evaluation mutates nothing, so a
timeout or a dropped connection can be re-dispatched to a replica; the error
types in :mod:`gen9_cluster.errors` carry that decision so this module never has
to infer it from an errno. If no replica holds the expert, the dispatcher fails
the token rather than silently dropping an expert from the sum — a quietly wrong
answer is worse than a loud failure. Every attempt at a given batch reuses one
``batch_id``, so a retry that reaches the *same* console (the common case: a
timeout on a console that is merely slow) is answered from its dedup cache
instead of running the experts twice.
"""

from __future__ import annotations

import itertools
import secrets
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import Gen9Error, ProtocolError, ShardMissing
from .protocol import (ExpertBatchPayload, ExpertRowsPayload, Flags, Frame,
                       MsgType, accumulate, decode_vector)
from .transport import ConnectionPool, NodeConnection

#: (layer, expert) -> the unit ids that hold it, best first.
Placement = Dict[Tuple[int, int], List[str]]

#: Key a FAST reply is filed under: it is one console's whole batch collapsed,
#: so it belongs to no single expert. Negative, so it can never collide with a
#: real expert id.
_WHOLE_BATCH = -1

#: Batch ids are drawn from a per-process random base and then counted up.
#: Random so two coordinators (a restart, a standby taking over) cannot collide
#: on a node's dedup cache and be served each other's answers; counted so the
#: ids inside one process are cheap and unique.
_BATCH_SEQUENCE = itertools.count(secrets.randbits(48) << 16)


def _new_batch_id() -> int:
    return next(_BATCH_SEQUENCE) % (1 << 64)


@dataclass
class NodeAddress:
    unit_id: str
    host: str
    port: int


@dataclass
class DispatchStats:
    """What the last layer actually cost, for the operator and the logs."""

    consoles_contacted: int = 0
    experts_run: int = 0
    retries: int = 0
    storage_hits: int = 0
    #: Replies a console served from its dedup cache instead of recomputing.
    #: Non-zero means retries are landing back on the original console, which
    #: is cheap; it is the *retries* count that costs a round trip.
    replays: int = 0
    failed_units: List[str] = field(default_factory=list)


class ExpertDispatcher:
    """Routes expert work for one shelf."""

    def __init__(self, placement: Placement,
                 addresses: Dict[str, NodeAddress], *,
                 pool: Optional[ConnectionPool] = None,
                 max_retries: int = 1,
                 request_timeout: float = 10.0,
                 max_workers: int = 32):
        self.placement = placement
        self.addresses = addresses
        self.pool = pool or ConnectionPool()
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix="g9-dispatch")
        self._down: Dict[str, bool] = {}
        self._lock = threading.Lock()

    def close(self) -> None:
        self._executor.shutdown(wait=False)
        self.pool.close()

    def __enter__(self) -> "ExpertDispatcher":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- routing ----------------------------------------------------------
    def group_by_unit(self, layer: int, expert_ids: Sequence[int],
                      gates: Sequence[float]
                      ) -> Dict[str, Tuple[List[int], List[float]]]:
        """Split a token's top-k into one batch per console that holds them."""
        grouped: Dict[str, Tuple[List[int], List[float]]] = defaultdict(
            lambda: ([], []))
        for expert, gate in zip(expert_ids, gates):
            unit = self._owner(layer, expert)
            ids, weights = grouped[unit]
            ids.append(int(expert))
            weights.append(float(gate))
        return dict(grouped)

    def _owner(self, layer: int, expert: int, *,
               skip: Sequence[str] = ()) -> str:
        holders = self.placement.get((layer, int(expert)))
        if not holders:
            raise ShardMissing("no console in the plan holds this expert",
                               layer=layer, expert=int(expert))
        for unit in holders:
            if unit in skip:
                continue
            with self._lock:
                if self._down.get(unit):
                    continue
            return unit
        # Every holder is marked down; try them anyway rather than failing the
        # token on stale liveness state.
        for unit in holders:
            if unit not in skip:
                return unit
        raise ShardMissing("every console holding this expert has failed",
                           layer=layer, expert=int(expert))

    # -- the hot path -----------------------------------------------------
    def run_layer(self, layer: int, activation: np.ndarray,
                  expert_ids: Sequence[int], gates: Sequence[float], *,
                  token: int = 0,
                  fast: bool = False) -> Tuple[np.ndarray, DispatchStats]:
        """Run one MoE layer's routed experts and return their summed output.

        ``expert_ids`` is the router's top-k *in rank order*, and that order is
        the reduction order. With ``fast=False`` (the default) the result is
        the same bits regardless of how the experts are spread over the shelf.
        With ``fast=True`` each console collapses its own experts first, which
        is fewer bytes on the wire and an answer that depends on the plan.
        """
        stats = DispatchStats()
        batches = self.group_by_unit(layer, expert_ids, gates)
        stats.consoles_contacted = len(batches)
        stats.experts_run = len(expert_ids)
        batch_id = _new_batch_id()

        futures = {
            unit: self._executor.submit(self._call_unit, unit, layer,
                                        activation, ids, weights, token, stats,
                                        batch_id, fast)
            for unit, (ids, weights) in batches.items()
        }
        replies: Dict[str, Dict[int, np.ndarray]] = {}
        errors: List[BaseException] = []
        for unit, future in futures.items():
            try:
                replies[unit] = future.result()
            except BaseException as exc:                    # noqa: BLE001
                errors.append(exc)
                stats.failed_units.append(unit)
        if errors:
            raise errors[0]

        if fast:
            # One partial per console, so top-k order is not recoverable and
            # unit id is the only stable order left. Reproducible per plan,
            # not across plans — which is what asking for FAST means.
            return accumulate([replies[unit][_WHOLE_BATCH]
                               for unit in sorted(replies)]), stats

        by_expert: Dict[int, np.ndarray] = {}
        for rows in replies.values():
            by_expert.update(rows)
        ordered = []
        for expert in expert_ids:
            row = by_expert.get(int(expert))
            if row is None:
                raise ProtocolError(
                    f"no console returned a row for expert {int(expert)}; "
                    f"the layer's sum would be missing a term", layer=layer,
                    expert=int(expert))
            ordered.append(row)
        return accumulate(ordered), stats

    def _call_unit(self, unit: str, layer: int, activation: np.ndarray,
                   expert_ids: Sequence[int], gates: Sequence[float],
                   token: int, stats: DispatchStats, batch_id: int,
                   fast: bool) -> Dict[int, np.ndarray]:
        """One console's contribution, keyed by expert id.

        A ``FAST`` reply has no per-expert structure, so it is returned under
        :data:`_WHOLE_BATCH` and the caller reduces by unit id instead.
        """
        attempted: List[str] = []
        target: Optional[str] = unit
        last: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            if target is None:
                break
            try:
                conn = self._connect(target)
                payload = ExpertBatchPayload(tuple(int(e) for e in expert_ids),
                                             tuple(float(g) for g in gates),
                                             activation, batch_id).encode()
                reply = conn.request(
                    Frame(MsgType.EXPERT_BATCH, 0, payload, layer=layer,
                          token=token,
                          flags=Flags.FAST if fast else Flags.NONE),
                    timeout=self.request_timeout)
                if reply.flags & Flags.FROM_STORAGE:
                    stats.storage_hits += 1
                if reply.flags & Flags.REPLAYED:
                    stats.replays += 1
                self._mark_up(target)
                return self._decode_reply(reply, target, layer, expert_ids,
                                          fast)
            except Gen9Error as exc:
                last = exc
                self._mark_down(target)
                if not exc.retry_safe:
                    raise
                attempted.append(target)
                stats.retries += 1
                # A replica for *all* of these experts, or nothing: splitting a
                # failed batch across two replicas would double-count any expert
                # both of them hold.
                target = self._replica_for(layer, expert_ids, attempted)
                if target is None:
                    raise
        assert last is not None
        raise last

    def _decode_reply(self, reply: Frame, unit: str, layer: int,
                      expert_ids: Sequence[int],
                      fast: bool) -> Dict[int, np.ndarray]:
        """Turn a console's reply into rows, checking it answered what we asked.

        A node that returns a partial sum to a batch that did not set ``FAST``
        is a version or configuration mismatch, and accepting it would produce
        a plausible-looking number with the wrong arithmetic behind it. That is
        precisely the failure this design exists to prevent, so it is refused.
        """
        per_expert = bool(reply.flags & Flags.PER_EXPERT)
        if fast:
            if per_expert:
                raise ProtocolError(
                    "console answered a FAST batch with per-expert rows",
                    unit_id=unit, layer=layer)
            return {_WHOLE_BATCH: decode_vector(reply.payload).astype(
                np.float32, copy=False)}
        if not per_expert:
            raise ProtocolError(
                "console collapsed a batch that did not ask for FAST; its "
                "reduction order is unknown and its answer cannot be placed "
                "in top-k order", unit_id=unit, layer=layer)
        rows = ExpertRowsPayload.decode(reply.payload)
        expected = {int(e) for e in expert_ids}
        got = set(rows.expert_ids)
        if got != expected:
            raise ProtocolError(
                f"console answered for experts {sorted(got)}, was asked for "
                f"{sorted(expected)}", unit_id=unit, layer=layer)
        return {int(eid): rows.rows[index]
                for index, eid in enumerate(rows.expert_ids)}

    def _replica_for(self, layer: int, expert_ids: Sequence[int],
                     attempted: Sequence[str]) -> Optional[str]:
        candidates: Optional[set] = None
        for expert in expert_ids:
            holders = {unit for unit in self.placement.get((layer, int(expert)),
                                                           [])
                       if unit not in attempted}
            candidates = holders if candidates is None else candidates & holders
            if not candidates:
                return None
        return sorted(candidates)[0] if candidates else None

    def _connect(self, unit: str) -> NodeConnection:
        address = self.addresses.get(unit)
        if address is None:
            raise ShardMissing(f"no address known for console {unit!r}",
                               unit_id=unit)
        return self.pool.get(unit, address.host, address.port)

    def _mark_down(self, unit: str) -> None:
        with self._lock:
            self._down[unit] = True
        self.pool.drop(unit)

    def _mark_up(self, unit: str) -> None:
        with self._lock:
            if self._down.get(unit):
                self._down[unit] = False


def placement_from_plan(plan) -> Placement:
    """Build the (layer, expert) -> consoles index from a :class:`SplitPlan`."""
    placement: Placement = defaultdict(list)
    for unit in plan.units.values():
        for shard in unit.shards:
            for expert in shard.expert_ids:
                placement[(shard.layer, expert)].append(unit.unit_id)
    return dict(placement)
