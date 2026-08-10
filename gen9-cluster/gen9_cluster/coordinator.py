"""Shelf and fleet coordinators: driving a token through the plan.

Two levels, matching the two levels of the split:

:class:`ShelfCoordinator` owns one shelf — a contiguous range of blocks, a host
console holding their hot weights, and the members holding their routed experts.
It runs its range end to end without talking to any other shelf.

:class:`FleetCoordinator` chains the shelves. It hands the activation to shelf 0,
takes what comes out, hands it to shelf 1, and so on. That is the *entire*
inter-shelf protocol, and it is that small on purpose: one 32 KiB activation per
boundary, no weights, no KV, no coordination barrier. The hierarchy exists for
the same reason Condor's PS3 cluster had head nodes over subclusters — one
coordinator cannot hold hundreds of sockets open and still make its latency
budget, but it can hold eight.

Fault behaviour is deliberately blunt and visible. A shelf whose host console
dies cannot continue: the KV cache for its layers lived there, and inventing a
replacement would mean silently answering with a different conversation. So the
coordinator raises, names the console, and lets the operator decide — restart
the shelf from a checkpointed KV cache, or re-plan without that console. Expert
consoles, by contrast, are replaceable mid-token, because they hold nothing but
weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dispatch import (DispatchStats, ExpertDispatcher, NodeAddress, Placement,
                       placement_from_plan)
from .errors import Gen9Error
from .protocol import Frame, MsgType, decode_vector, encode_vector
from .transport import ConnectionPool

#: Given a layer index and a hidden state, return (expert_ids, gate_weights).
Router = Callable[[int, np.ndarray], Tuple[Sequence[int], Sequence[float]]]


@dataclass
class LayerTrace:
    """Per-layer timing, which is the only way to find the slow console."""

    layer: int
    seconds: float
    consoles: int
    storage_hits: int
    retries: int


@dataclass
class TokenTrace:
    seconds: float = 0.0
    layers: List[LayerTrace] = field(default_factory=list)

    @property
    def slowest(self) -> Optional[LayerTrace]:
        return max(self.layers, key=lambda trace: trace.seconds) if self.layers else None

    def summary(self) -> str:
        worst = self.slowest
        tail = (f", slowest layer {worst.layer} at {worst.seconds * 1000:.1f} ms"
                if worst else "")
        storage = sum(t.storage_hits for t in self.layers)
        retries = sum(t.retries for t in self.layers)
        return (f"{self.seconds * 1000:.0f} ms/token over {len(self.layers)} "
                f"blocks{tail}"
                + (f", {storage} NVMe hits" if storage else "")
                + (f", {retries} retries" if retries else ""))


class ShelfCoordinator:
    """Drives the blocks assigned to one shelf."""

    def __init__(self, stage, addresses: Dict[str, NodeAddress], *,
                 placement: Placement, router: Router,
                 pool: Optional[ConnectionPool] = None,
                 request_timeout: float = 10.0,
                 dense_layers: int = 0, fast: bool = False):
        self.stage = stage
        self.addresses = addresses
        self.router = router
        self.dense_layers = dense_layers
        #: Off by default: see :mod:`gen9_cluster.dispatch` for what turning it
        #: on costs. It saves reply bandwidth and makes the output depend on
        #: the current plan.
        self.fast = fast
        self.pool = pool or ConnectionPool()
        self.dispatcher = ExpertDispatcher(placement, addresses, pool=self.pool,
                                           request_timeout=request_timeout)
        self.request_timeout = request_timeout
        self._last_stats = DispatchStats()

    def close(self) -> None:
        self.dispatcher.close()

    @property
    def host_unit(self) -> str:
        return self.stage.host_unit

    def run(self, activation: np.ndarray, *, token: int = 0,
            trace: Optional[TokenTrace] = None) -> np.ndarray:
        """Run every block of this shelf's range over one activation."""
        state = activation
        for layer in self.stage.layers:
            started = time.perf_counter()
            state = self._run_block(layer, state, token)
            stats = self._last_stats
            if trace is not None:
                trace.layers.append(LayerTrace(
                    layer=layer, seconds=time.perf_counter() - started,
                    consoles=stats.consoles_contacted,
                    storage_hits=stats.storage_hits, retries=stats.retries))
        return state

    def _run_block(self, layer: int, activation: np.ndarray,
                   token: int) -> np.ndarray:
        # 1. The hot block runs on the shelf's host: attention against the KV
        #    cache that lives there, the router, the always-on shared expert.
        address = self.addresses[self.host_unit]
        conn = self.pool.get(self.host_unit, address.host, address.port)
        reply = conn.request(
            Frame(MsgType.BLOCK_FWD, 0, encode_vector(activation), layer=layer,
                  token=token), timeout=self.request_timeout)
        state = decode_vector(reply.payload).astype(np.float32, copy=False)

        # 2. Dense layers have no routed experts; the block is the whole layer.
        if layer < self.dense_layers:
            self._last_stats = DispatchStats(consoles_contacted=1)
            return state

        # 3. The router's choice decides which consoles are woken. Everything
        #    else in the fleet stays idle for this token, which is the entire
        #    economic argument for MoE on hardware like this.
        expert_ids, gates = self.router(layer, state)
        routed, stats = self.dispatcher.run_layer(layer, state, expert_ids,
                                                  gates, token=token,
                                                  fast=self.fast)
        self._last_stats = stats
        return state + routed


class FleetCoordinator:
    """Chains shelves in layer order; the front door for a whole fleet."""

    def __init__(self, plan, addresses: Dict[str, NodeAddress], *,
                 router: Router, request_timeout: float = 10.0,
                 fast: bool = False):
        self.plan = plan
        self.addresses = addresses
        self.pool = ConnectionPool()
        placement = placement_from_plan(plan)
        dense = _dense_layer_count(plan)
        self.shelves = [
            ShelfCoordinator(stage, addresses, placement=placement,
                             router=router, pool=self.pool,
                             request_timeout=request_timeout,
                             dense_layers=dense, fast=fast)
            for stage in plan.stages
        ]

    def close(self) -> None:
        for shelf in self.shelves:
            shelf.close()
        self.pool.close()

    def __enter__(self) -> "FleetCoordinator":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def forward(self, activation: np.ndarray, *, token: int = 0
                ) -> Tuple[np.ndarray, TokenTrace]:
        """One token through every shelf, in order."""
        trace = TokenTrace()
        started = time.perf_counter()
        state = np.asarray(activation, dtype=np.float32)
        for shelf in self.shelves:
            try:
                state = shelf.run(state, token=token, trace=trace)
            except Gen9Error as exc:
                if exc.unit_id == shelf.host_unit:
                    raise Gen9Error(
                        f"shelf {shelf.stage.stage_id} lost its host console, "
                        f"which held the KV cache for layers "
                        f"{shelf.stage.first_layer}-"
                        f"{shelf.stage.first_layer + shelf.stage.n_layers - 1}; "
                        f"the shelf must be restarted, not failed over",
                        unit_id=shelf.host_unit) from exc
                raise
        trace.seconds = time.perf_counter() - started
        return state, trace

    def health(self) -> Dict[str, str]:
        """Ping every console named in the plan; report what answers."""
        status: Dict[str, str] = {}
        for unit_id, address in sorted(self.addresses.items()):
            try:
                conn = self.pool.get(unit_id, address.host, address.port)
                reply = conn.request(Frame(MsgType.STATUS, 0), timeout=2.0)
                status[unit_id] = reply.payload.decode("utf-8", "replace")
            except Gen9Error as exc:
                status[unit_id] = f"DOWN: {exc}"
        return status


def _dense_layer_count(plan) -> int:
    """Recover the dense-layer count from the plan's own placement.

    The first layer that has routed experts anywhere is the first MoE layer;
    reading it off the plan avoids threading the model profile through the
    runtime, which would only be used for this one number.
    """
    layers_with_experts = {shard.layer for unit in plan.units.values()
                           for shard in unit.shards}
    return min(layers_with_experts) if layers_with_experts else 0
