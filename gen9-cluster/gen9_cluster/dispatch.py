"""Fan a token's top-k out to the consoles that hold those experts.

This is the hot path: it runs once per MoE layer per token, so ~69 times per
token on the V4-Pro profile. Three decisions define it.

**Group by console before sending.** A token's eight experts may live on three
consoles. Naively that is eight requests; grouped, it is three, and each console
returns the gate-weighted sum of the experts it holds. Round trips, not FLOPs,
are what a console fleet is short of.

**Send concurrently, reduce deterministically.** The requests go out together
and land in whatever order the network decides, but the partial sums are added
back in a fixed order (by unit id). Floating-point addition is not associative,
so reducing in arrival order would make the same prompt produce different
tokens on different runs — a bug that is invisible until it is infuriating.

**Retry only what is safe to retry.** An expert evaluation mutates nothing, so a
timeout or a dropped connection can be re-dispatched to a replica; the error
types in :mod:`gen9_cluster.errors` carry that decision so this module never has
to infer it from an errno. If no replica holds the expert, the dispatcher fails
the token rather than silently dropping an expert from the sum — a quietly wrong
answer is worse than a loud failure.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import Gen9Error, ShardMissing
from .protocol import (ExpertBatchPayload, Flags, Frame, MsgType,
                       decode_vector)
from .transport import ConnectionPool, NodeConnection

#: (layer, expert) -> the unit ids that hold it, best first.
Placement = Dict[Tuple[int, int], List[str]]


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
                  token: int = 0) -> Tuple[np.ndarray, DispatchStats]:
        """Run one MoE layer's routed experts and return their summed output."""
        stats = DispatchStats()
        batches = self.group_by_unit(layer, expert_ids, gates)
        stats.consoles_contacted = len(batches)
        stats.experts_run = len(expert_ids)

        futures = {
            unit: self._executor.submit(self._call_unit, unit, layer,
                                        activation, ids, weights, token, stats)
            for unit, (ids, weights) in batches.items()
        }
        partials: Dict[str, np.ndarray] = {}
        errors: List[BaseException] = []
        for unit, future in futures.items():
            try:
                partials[unit] = future.result()
            except BaseException as exc:                    # noqa: BLE001
                errors.append(exc)
                stats.failed_units.append(unit)
        if errors:
            raise errors[0]

        # Deterministic reduction: sorted by unit id, never by arrival.
        out = np.zeros_like(activation, dtype=np.float32)
        for unit in sorted(partials):
            out += partials[unit]
        return out, stats

    def _call_unit(self, unit: str, layer: int, activation: np.ndarray,
                   expert_ids: Sequence[int], gates: Sequence[float],
                   token: int, stats: DispatchStats) -> np.ndarray:
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
                                             activation).encode()
                reply = conn.request(
                    Frame(MsgType.EXPERT_BATCH, 0, payload, layer=layer,
                          token=token),
                    timeout=self.request_timeout)
                if reply.flags & Flags.FROM_STORAGE:
                    stats.storage_hits += 1
                self._mark_up(target)
                return decode_vector(reply.payload).astype(np.float32,
                                                           copy=False)
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
