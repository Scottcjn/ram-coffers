"""Splitting planner: a DeepSeek MoE checkpoint across a heterogeneous fleet.

The PS3 port of this idea had one placement axis, because a 256 MB console can
hold exactly one thing: **one expert per console**. A ninth-generation console
holds 8-13 GiB, which is enough for a whole layer's hot weights *and* a few
dozen routed experts, so the interesting question stops being "does it fit" and
becomes "what belongs where". This module answers it along four axes at once:

1. **Pipeline, by contiguous layer range.** A stage owns consecutive decoder
   layers so a token crosses the network once per stage, carrying one residual
   state (56 KiB at V4-Pro width in bf16, four times hidden because of its
   hyper-connections) rather than anything proportional to the weights.
   Gigabit ethernet is the fleet's scarcest resource and this is the axis that
   respects it.

2. **Hot/cold, by coffer.** Everything an MoE layer reads for *every* token —
   attention, the router, the shared expert, the norms — is small
   (~330 MiB at V4-Pro width) and must sit in the fastest coffer the owning
   console has. The routed experts are ~12.9 GiB per layer and are read a handful
   at a time, so they belong in slow coffers and on NVMe. This is the RAM
   Coffers thesis applied to a memory hierarchy the console vendors built for
   their own reasons: the Series X's 560/336 GB/s split is a hot/cold boundary
   already drawn in silicon.

3. **Expert-parallel, within a stage.** A layer's routed experts are spread over
   the stage's consoles in proportion to what each can hold *and* how fast it
   can read, so a Series S and a Series X in the same stage finish their share
   of a token at the same time instead of the small one stalling the group.

4. **SSD streaming, as the overflow tier.** These consoles have 2.4-5.5 GB/s
   NVMe, which is the one respect in which they beat a commodity desktop. An
   expert that does not fit in RAM is not a planning failure; it is a resident
   of the cold tier, streamed on a routing hit, and the plan reports what that
   costs.

The load-balancing rule (axis 3) and the hot/cold rule (axis 2) are borrowed
directly from what works for DeepSeek on consumer hardware: KTransformers and
Fiddler both keep attention and the popular experts on the fast device and push
the long tail to the slow one, partitioning by arithmetic intensity rather than
by parameter count; llama.cpp's ``-ot`` expert-offload does the same thing by
hand. The difference here is that the "fast device" and the "slow device" are two
coffers of the same console, and there are dozens of consoles.

The planner is pure arithmetic. It never touches a console, so a fleet can be
sized, rebalanced, and argued about from a laptop, and the same plan is what the
deployment tools turn into a ``cluster.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .hardware import (BANDWIDTH_EFFICIENCY, GB, MB, ConsoleUnit,
                       EffectiveCapability)
from .model import EngramConfig, ModelProfile, QuantSpec

#: A weight read every token belongs in a coffer at least this fast. Below it,
#: the hot block alone would dominate the layer's time budget. Chosen just under
#: the Series S's 224 GB/s tier so a Series S can still host attention.
HOT_BANDWIDTH_FLOOR_GBPS = 200.0

#: Headroom kept free on every unit for activations, the KV cache's growth, the
#: runtime, and fragmentation. Fractional, applied to the unit's usable bytes.
RESIDENCY_HEADROOM = 0.08

#: Consoles per shelf. A shelf is the unit of expert fan-out: one switch, one
#: coordinator, one contiguous range of layers. 22 is the size Condor's PS3
#: cluster used per subcluster and it remains a good number — large enough that
#: a layer's experts are spread thin, small enough that a gigabit switch and one
#: coordinator can serve the fan-out for every layer of every token.
DEFAULT_SHELF_SIZE = 22

#: One-way cost of a stage hop on the fleet's network, in seconds. Gigabit
#: ethernet plus a switch; the payload is one activation, so this is latency,
#: not bandwidth. Overridable per plan.
DEFAULT_HOP_SECONDS = 0.00025


class PlanningError(ValueError):
    """The fleet cannot host the model even with every fallback enabled."""


@dataclass
class ExpertShard:
    """A contiguous run of one layer's routed experts, on one unit."""

    layer: int
    first_expert: int
    n_experts: int
    unit_id: str
    #: "fast" | "slow" | "ssd" — which coffer holds them.
    tier: str

    @property
    def expert_ids(self) -> range:
        return range(self.first_expert, self.first_expert + self.n_experts)

    def to_dict(self) -> Dict[str, object]:
        return {"layer": self.layer, "first_expert": self.first_expert,
                "n_experts": self.n_experts, "unit_id": self.unit_id,
                "tier": self.tier}


@dataclass
class EngramRowShard:
    """One unit's slice of an Engram table's row store.

    Rows are hash-addressed, so the routing map only has to say which row ids
    each unit holds: a lookup hashes its n-grams to row ids, finds the units
    owning those ranges, and gathers. ``tier`` is ``"ram"`` when the slice
    sits in the unit's memory (the ``--no-ssd`` path) or ``"ssd"`` when the
    whole table is one unit's NVMe resident — the record is the same either
    way so a coordinator can route by row id without caring which tier the
    store landed on.
    """

    layer: int
    first_row: int
    n_rows: int
    byte_count: int
    #: "ram" | "ssd" — which tier holds the rows.
    tier: str

    def to_dict(self) -> Dict[str, object]:
        return {"layer": self.layer, "first_row": self.first_row,
                "n_rows": self.n_rows, "bytes": self.byte_count,
                "tier": self.tier}


@dataclass
class UnitPlan:
    """Everything one console is asked to hold."""

    unit_id: str
    sku: str
    backend: str
    #: Layers whose hot block (attention + router + shared expert) lives here.
    hot_layers: List[int] = field(default_factory=list)
    #: Dense (non-MoE) layers hosted whole.
    dense_layers: List[int] = field(default_factory=list)
    #: "embedding" / "lm_head" / "mtp" pieces hosted here.
    io_pieces: List[str] = field(default_factory=list)
    shards: List[ExpertShard] = field(default_factory=list)
    hot_bytes: int = 0
    fast_expert_bytes: int = 0
    slow_expert_bytes: int = 0
    ssd_expert_bytes: int = 0
    #: Engram row bytes held in RAM (layer -> bytes) — the --no-ssd fallback,
    #: where a table shards across the fleet like a read-mostly expert.
    engram_rows: Dict[int, int] = field(default_factory=dict)
    #: The row-range records behind ``engram_rows`` (plus whole-table records
    #: for NVMe-resident stores): what a deployment loads and routes by.
    engram_shards: List[EngramRowShard] = field(default_factory=list)
    kv_bytes: int = 0
    capacity_bytes: int = 0
    fast_capacity_bytes: int = 0
    gemv_gflops: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def resident_bytes(self) -> int:
        """Bytes held in RAM (the SSD tier is not resident)."""
        return (self.hot_bytes + self.fast_expert_bytes + self.slow_expert_bytes
                + sum(self.engram_rows.values()) + self.kv_bytes)

    @property
    def n_experts(self) -> int:
        return sum(s.n_experts for s in self.shards)

    @property
    def headroom_bytes(self) -> int:
        return self.capacity_bytes - self.resident_bytes

    def to_dict(self) -> Dict[str, object]:
        return {
            "unit_id": self.unit_id, "sku": self.sku, "backend": self.backend,
            "hot_layers": list(self.hot_layers),
            "dense_layers": list(self.dense_layers),
            "io_pieces": list(self.io_pieces),
            "shards": [s.to_dict() for s in self.shards],
            "hot_bytes": self.hot_bytes,
            "fast_expert_bytes": self.fast_expert_bytes,
            "slow_expert_bytes": self.slow_expert_bytes,
            "ssd_expert_bytes": self.ssd_expert_bytes,
            "engram_ram_bytes": sum(self.engram_rows.values()),
            "engram_shards": [s.to_dict() for s in self.engram_shards],
            "kv_bytes": self.kv_bytes,
            "resident_bytes": self.resident_bytes,
            "capacity_bytes": self.capacity_bytes,
            "n_experts": self.n_experts,
            "gemv_gflops": self.gemv_gflops,
            "warnings": list(self.warnings),
        }


@dataclass
class StagePlan:
    """One pipeline stage: a contiguous layer range and the units serving it."""

    stage_id: str
    first_layer: int
    n_layers: int
    #: Unit holding the hot blocks (and the KV cache) for these layers.
    host_unit: str
    #: Units holding this stage's routed experts, including the host.
    expert_units: List[str] = field(default_factory=list)

    @property
    def layers(self) -> range:
        return range(self.first_layer, self.first_layer + self.n_layers)

    def to_dict(self) -> Dict[str, object]:
        return {"stage_id": self.stage_id, "first_layer": self.first_layer,
                "n_layers": self.n_layers, "host_unit": self.host_unit,
                "expert_units": list(self.expert_units)}


@dataclass
class SplitPlan:
    """The whole placement, plus what it is expected to cost."""

    model: str
    model_assumed: bool
    context_tokens: int
    stages: List[StagePlan]
    units: Dict[str, UnitPlan]
    feasible: bool
    shortfall_bytes: int = 0
    tokens_per_second: float = 0.0
    seconds_per_token: float = 0.0
    ssd_expert_bytes: int = 0
    active_units_per_token: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def n_units(self) -> int:
        return len(self.units)

    def unit(self, unit_id: str) -> UnitPlan:
        return self.units[unit_id]

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": self.model, "model_assumed": self.model_assumed,
            "context_tokens": self.context_tokens,
            "stages": [s.to_dict() for s in self.stages],
            "units": {k: v.to_dict() for k, v in self.units.items()},
            "feasible": self.feasible, "shortfall_bytes": self.shortfall_bytes,
            "tokens_per_second": self.tokens_per_second,
            "seconds_per_token": self.seconds_per_token,
            "ssd_expert_bytes": self.ssd_expert_bytes,
            "active_units_per_token": self.active_units_per_token,
            "warnings": list(self.warnings),
        }


def _block_bytes(profile: ModelProfile, index: int) -> int:
    """Per-token weights of block ``index``, counting MTP heads as blocks."""
    return profile.block_bytes(index)


def _block_need(profile: ModelProfile, index: int, context_tokens: int) -> int:
    """What a hot host has to hold for block ``index``: weights plus its cache.

    V4 interleaves attention kinds whose caches differ by a factor of 32, so
    this is per block rather than one figure reused for every layer.
    """
    return (_block_bytes(profile, index)
            + profile.block_kv_bytes(index, context_tokens))


def _has_routed_experts(profile: ModelProfile, index: int) -> bool:
    return index >= profile.moe.n_dense_layers


def _usable(cap: EffectiveCapability) -> int:
    return int(cap.weight_bytes * (1.0 - RESIDENCY_HEADROOM))


def _usable_fast(cap: EffectiveCapability) -> int:
    return int(cap.weight_bytes_at_least(HOT_BANDWIDTH_FLOOR_GBPS)
               * (1.0 - RESIDENCY_HEADROOM))


def plan_split(profile: ModelProfile, fleet: Sequence[ConsoleUnit], *,
               context_tokens: int = 8192,
               shelf_size: int = DEFAULT_SHELF_SIZE,
               allow_ssd_tier: bool = True,
               hop_seconds: float = DEFAULT_HOP_SECONDS) -> SplitPlan:
    """Place ``profile`` on ``fleet``.

    The fleet is first cut into **shelves** of at most ``shelf_size`` consoles,
    and each shelf is given a contiguous range of layers in proportion to how
    much it can hold. Everything a layer needs then lives inside one shelf: its
    hot block and KV cache on the shelf's fastest console, its routed experts
    spread over the shelf's members. That is what bounds the fan-out — a token's
    top-k crosses one switch inside one shelf per layer, instead of scattering
    over the whole fleet — and it is the same reason Condor's PS3 cluster was
    wired as subclusters of 22 behind a head node rather than as one flat farm.

    Raises :class:`PlanningError` only when the fleet cannot host the model
    in the tiers it was allowed to use: a fleet that is merely *slow* gets a
    plan and a warning, because "this works, at 0.4 tokens/s" is a useful
    answer and refusing to plan is not.
    """
    if not fleet:
        raise PlanningError("empty fleet")
    if shelf_size < 1:
        raise PlanningError("shelf_size must be >= 1")
    caps = {u.unit_id: u.effective() for u in fleet}
    if len(caps) != len(fleet):
        raise PlanningError("duplicate unit_id in fleet")

    units: Dict[str, UnitPlan] = {
        uid: UnitPlan(unit_id=uid, sku=cap.sku, backend=cap.backend.value,
                      capacity_bytes=_usable(cap),
                      fast_capacity_bytes=_usable_fast(cap),
                      gemv_gflops=cap.gemv_gflops,
                      warnings=list(cap.warnings))
        for uid, cap in caps.items()
    }
    warnings: List[str] = []
    if profile.assumed:
        warnings.append(
            f"{profile.name} is an assumed configuration; every size in this "
            f"plan moves when the real config is published")

    shelves = _build_shelves(caps, units, shelf_size)
    stages = _assign_layer_ranges(profile, caps, units, shelves,
                                  context_tokens, warnings)
    shortfall = _assign_experts(profile, caps, units, stages, allow_ssd_tier,
                                warnings)
    shortfall += _assign_io(profile, caps, units, stages, allow_ssd_tier,
                            warnings)

    if shortfall > 0:
        raise PlanningError(
            f"fleet is short {shortfall / GB:.1f} GiB for {profile.name} at "
            f"{context_tokens} tokens of context: add units, enable the SSD "
            f"tier, or shrink the context")

    sec, active = _estimate_decode(profile, caps, units, stages, hop_seconds,
                                   context_tokens)
    ssd_bytes = sum(u.ssd_expert_bytes for u in units.values())
    if ssd_bytes:
        warnings.append(
            f"{ssd_bytes / GB:.1f} GiB of weights live on NVMe — routed "
            f"experts streamed on a routing hit, Engram tables and any spilled "
            f"io pieces read on demand; where that happens the block is bound "
            f"by SSD reads, not by memory bandwidth")
    return SplitPlan(
        model=profile.name, model_assumed=profile.assumed,
        context_tokens=context_tokens, stages=stages, units=units,
        feasible=True, shortfall_bytes=0,
        tokens_per_second=(1.0 / sec if sec > 0 else 0.0),
        seconds_per_token=sec, ssd_expert_bytes=ssd_bytes,
        active_units_per_token=active, warnings=warnings)


# -- shelves ---------------------------------------------------------------

def _build_shelves(caps: Dict[str, EffectiveCapability],
                   units: Dict[str, UnitPlan],
                   shelf_size: int) -> List[List[str]]:
    """Cut the fleet into shelves, each a balanced mix of what is available.

    Units are dealt round-robin in descending capability order, so every shelf
    gets some of the fast consoles rather than one shelf inheriting all the
    Series X units and another all the Series S ones. A shelf's slowest member
    sets its per-layer floor, so an unbalanced cut would make one shelf's layers
    permanently the slow ones.
    """
    ordered = sorted(units, key=lambda uid: (-units[uid].capacity_bytes,
                                             -units[uid].gemv_gflops, uid))
    n_shelves = max(1, math.ceil(len(ordered) / shelf_size))
    shelves: List[List[str]] = [[] for _ in range(n_shelves)]
    for index, uid in enumerate(ordered):
        shelves[index % n_shelves].append(uid)
    return shelves


def _shelf_capacity(shelf: Sequence[str], units: Dict[str, UnitPlan]) -> int:
    return sum(units[uid].capacity_bytes for uid in shelf)


def _shelf_host(shelf: Sequence[str], units: Dict[str, UnitPlan],
                caps: Dict[str, EffectiveCapability], need: int) -> Optional[str]:
    """The shelf member that should hold hot blocks: fastest coffer first."""
    eligible = [uid for uid in shelf
                if units[uid].fast_capacity_bytes >= need
                and caps[uid].fast_bandwidth_gbps >= HOT_BANDWIDTH_FLOOR_GBPS]
    if not eligible:
        return None
    return max(eligible, key=lambda uid: (caps[uid].fast_bandwidth_gbps,
                                          units[uid].fast_capacity_bytes, uid))


# -- axis 1 + 2: layer ranges per shelf, hot residency on its host ----------

def _assign_layer_ranges(profile: ModelProfile,
                         caps: Dict[str, EffectiveCapability],
                         units: Dict[str, UnitPlan],
                         shelves: Sequence[Sequence[str]],
                         context_tokens: int,
                         warnings: List[str]) -> List[StagePlan]:
    """Give each shelf a contiguous run of layers, sized by what it can hold.

    A shelf's share is proportional to its capacity, so a shelf of Series S
    consoles takes fewer layers than a shelf of PS5s instead of taking the same
    number and overflowing to NVMe.
    """
    n_blocks = profile.planning_layers

    total_capacity = sum(_shelf_capacity(s, units) for s in shelves)
    if total_capacity <= 0:
        raise PlanningError("no unit in the fleet has usable memory")

    # Layers per shelf, proportional to capacity, at least one each until the
    # layers run out (a shelf with no layers is dead weight the operator should
    # see as such, so it keeps its slot and gets a warning instead).
    quotas: List[int] = []
    for shelf in shelves:
        share = _shelf_capacity(shelf, units) / total_capacity
        quotas.append(int(math.floor(n_blocks * share)))
    while sum(quotas) < n_blocks:
        # Hand the remainder to the roomiest shelves, largest first.
        index = max(range(len(shelves)),
                    key=lambda i: (_shelf_capacity(shelves[i], units)
                                   / max(quotas[i] + 1, 1)))
        quotas[index] += 1
    while sum(quotas) > n_blocks:
        index = max(range(len(shelves)), key=lambda i: quotas[i])
        quotas[index] -= 1

    stages: List[StagePlan] = []
    layer = 0
    for shelf_index, shelf in enumerate(shelves):
        count = quotas[shelf_index]
        if count <= 0:
            warnings.append(
                f"shelf {shelf_index} ({len(shelf)} units, "
                f"{_shelf_capacity(shelf, units) / GB:.1f} GiB) holds no layers; "
                f"it is too small to be worth a layer of this model")
            continue
        first = layer
        block_need = sum(_block_need(profile, first + offset, context_tokens)
                         for offset in range(count))
        host = _shelf_host(shelf, units, caps, block_need)
        if host is None:
            # No member can hold every hot block for this many layers. Shrink
            # the shelf's range until one can, and push the rest onward.
            count, host = _shrink_until_hosted(profile, shelf, units, caps,
                                               first, count, context_tokens)
            if host is None:
                raise PlanningError(
                    f"shelf {shelf_index} has no member able to hold even one "
                    f"layer's hot block plus its KV cache at {context_tokens} "
                    f"tokens of context in a coffer at or above "
                    f"{HOT_BANDWIDTH_FLOOR_GBPS:.0f} GB/s")
            spilled = quotas[shelf_index] - count
            for later in range(shelf_index + 1, len(shelves)):
                quotas[later] += spilled // max(1, len(shelves) - shelf_index - 1)
            warnings.append(
                f"shelf {shelf_index} could only host {count} of "
                f"{quotas[shelf_index]} layers' hot blocks; the rest moved to "
                f"later shelves")
        plan = units[host]
        for offset in range(count):
            index = first + offset
            if index < profile.moe.n_dense_layers:
                plan.dense_layers.append(index)
            else:
                plan.hot_layers.append(index)
            if index >= profile.n_layers:
                plan.io_pieces.append(f"mtp-head-{index - profile.n_layers}")
            plan.hot_bytes += _block_bytes(profile, index)
            plan.kv_bytes += profile.block_kv_bytes(index, context_tokens)
        stages.append(StagePlan(stage_id=f"st-{len(stages):04d}",
                                first_layer=first, n_layers=count,
                                host_unit=host, expert_units=list(shelf)))
        layer += count
        if layer >= n_blocks:
            break

    if layer < n_blocks:
        raise PlanningError(
            f"only {layer} of {n_blocks} layers could be given a hot "
            f"host; the fleet needs more consoles with a fast coffer, or a "
            f"shorter context "
            f"({profile.kv_cache_bytes(context_tokens) / profile.n_layers / MB:.0f}"
            f" MiB of KV per layer on average at {context_tokens} tokens)")
    if len(stages) > 1:
        hops = len(stages) - 1
        warnings.append(
            f"{len(stages)} shelves: a token crosses the network {hops} "
            f"time{'s' if hops != 1 else ''} between shelves per forward pass, "
            f"carrying one {profile.activation_bytes() / 1024:.0f} KiB "
            f"activation each time")
    return stages


def _shrink_until_hosted(profile: ModelProfile, shelf: Sequence[str],
                         units: Dict[str, UnitPlan],
                         caps: Dict[str, EffectiveCapability],
                         first: int, count: int, context_tokens: int
                         ) -> Tuple[int, Optional[str]]:
    """Largest prefix of a shelf's range that one member can host, and who."""
    while count > 0:
        need = sum(_block_need(profile, first + o, context_tokens)
                   for o in range(count))
        host = _shelf_host(shelf, units, caps, need)
        if host is not None:
            return count, host
        count -= 1
    return 0, None


# -- axis 3 + 4: expert placement inside a shelf, with an NVMe overflow -----

def _assign_experts(profile: ModelProfile,
                    caps: Dict[str, EffectiveCapability],
                    units: Dict[str, UnitPlan], stages: Sequence[StagePlan],
                    allow_ssd_tier: bool, warnings: List[str]) -> int:
    """Spread each MoE layer's routed experts over its own shelf.

    Within a shelf, a unit's share is proportional to its throughput, not to its
    capacity: what matters at decode time is that every member finishes reading
    its slice of the token's top-k at the same moment. A Series S given a PS5's
    share would hold the whole shelf up on every token that routes to it.

    Anything the shelf cannot hold in RAM goes to NVMe on the shelf's roomiest
    member, and only if the shelf's storage is full does the planner reach
    outside the shelf — recorded as a warning, because a cross-shelf expert
    costs a round trip the shelf-local ones do not.

    Returns bytes that could not be placed anywhere (0 when the plan fits).
    """
    all_units = sorted(units)
    shortfall = 0
    cross_shelf = 0

    for stage in stages:
        shelf = list(stage.expert_units)
        for layer in stage.layers:
            if not _has_routed_experts(profile, layer):
                continue
            moe = profile.moe_for_block(layer)
            expert = profile.expert_bytes(layer)
            remaining = moe.n_routed_experts
            first = 0
            shares = _shelf_shares(profile, shelf, units, moe)
            for uid in sorted(shelf, key=lambda u: (-units[u].headroom_bytes, u)):
                if remaining <= 0:
                    break
                plan = units[uid]
                free = plan.headroom_bytes
                if free < expert:
                    continue
                take = int(max(1, min(remaining, free // expert, shares[uid])))
                tier = _tier_for(caps[uid], plan, take * expert)
                plan.shards.append(ExpertShard(layer, first, take, uid, tier))
                if tier == "fast":
                    plan.fast_expert_bytes += take * expert
                else:
                    plan.slow_expert_bytes += take * expert
                first += take
                remaining -= take
            if remaining <= 0:
                continue
            if allow_ssd_tier:
                placed = _place_on_ssd(layer, first, remaining, expert, shelf,
                                       stage, caps, units)
                first += placed
                remaining -= placed
            if remaining <= 0:
                continue
            # Last resort: outside the shelf.
            outside = [uid for uid in all_units if uid not in shelf]
            for uid in sorted(outside,
                              key=lambda u: (-units[u].headroom_bytes, u)):
                if remaining <= 0:
                    break
                plan = units[uid]
                take = int(min(remaining, plan.headroom_bytes // expert))
                if take <= 0:
                    continue
                tier = _tier_for(caps[uid], plan, take * expert)
                plan.shards.append(ExpertShard(layer, first, take, uid, tier))
                if tier == "fast":
                    plan.fast_expert_bytes += take * expert
                else:
                    plan.slow_expert_bytes += take * expert
                if uid not in stage.expert_units:
                    stage.expert_units.append(uid)
                first += take
                remaining -= take
                cross_shelf += take
            shortfall += remaining * expert

    if cross_shelf:
        warnings.append(
            f"{cross_shelf} expert placements spilled outside their shelf; "
            f"those routing hits cost a cross-shelf round trip, so add "
            f"consoles to the shelves that overflowed rather than anywhere")
    return shortfall


def _shelf_shares(profile: ModelProfile, shelf: Sequence[str],
                  units: Dict[str, UnitPlan], moe) -> Dict[str, int]:
    """How many of a layer's experts each shelf member should take."""
    total = sum(max(units[uid].gemv_gflops, 1.0) for uid in shelf)
    shares: Dict[str, int] = {}
    for uid in shelf:
        fraction = max(units[uid].gemv_gflops, 1.0) / total
        shares[uid] = max(1, int(round(moe.n_routed_experts * fraction)))
    return shares


def _tier_for(cap: EffectiveCapability, plan: UnitPlan, want: int) -> str:
    """Which coffer a new shard lands in: the fast one while it has room.

    On a Series X this is the 560/336 GB/s boundary doing real work — the first
    experts a unit takes sit in the fast coffer beside the hot block, and the
    rest fall into the slow one, which is exactly the tiering the console was
    built with and the reason a Series X is worth more than its 16 GB suggests.
    """
    fast_used = plan.hot_bytes + plan.kv_bytes + plan.fast_expert_bytes
    return "fast" if fast_used + want <= plan.fast_capacity_bytes else "slow"


def _place_on_ssd(layer: int, first: int, count: int, expert_bytes: int,
                  shelf: Sequence[str], stage: StagePlan,
                  caps: Dict[str, EffectiveCapability],
                  units: Dict[str, UnitPlan]) -> int:
    """Put the remainder of a layer's experts on shelf-local NVMe.

    Streaming is the one respect in which these consoles beat a commodity
    desktop — 2.4 GB/s on an Xbox, 5.5 GB/s on a PS5, against a spinning fleet's
    nothing — so an expert that misses RAM costs a read, not a refusal. Half of
    each unit's drive is left alone for the OS, the checkpoint's own copy, and
    the fact that a console's drive is also a console's drive.
    """
    placed = 0
    for uid in sorted(shelf, key=lambda u: (-caps[u].storage.capacity_bytes, u)):
        if placed >= count:
            break
        cap = caps[uid]
        plan = units[uid]
        room = cap.storage.capacity_bytes // 2 - plan.ssd_expert_bytes
        take = min(count - placed, max(0, room // expert_bytes))
        if take <= 0:
            continue
        plan.shards.append(ExpertShard(layer, first + placed, take, uid, "ssd"))
        plan.ssd_expert_bytes += take * expert_bytes
        placed += take
    return placed


def _assign_io(profile: ModelProfile, caps: Dict[str, EffectiveCapability],
               units: Dict[str, UnitPlan], stages: Sequence[StagePlan],
               allow_ssd_tier: bool, warnings: List[str]) -> int:
    """Place the embedding, the LM head, the vision tower, and Engram tables.

    Both ends of the model are read once per token and are large (~2 GiB each at
    V4-Pro width in bf16, since the vocabulary tables are not FP8), so they go
    wherever there is room. The vision tower joins them — on the first stage's
    host when it fits, since that is where the embeddings an image produces
    first land.

    Engram tables are different: tens of GiB each, hash-addressed, and read a
    few kilobytes per token, so the row store is NVMe-resident *by design* —
    the Engram paper's whole point is deterministic addressing that tolerates
    host memory. Each row store goes to the SSD of the shelf host that owns
    its layer, so the lookup never crosses a shelf boundary ahead of the layer
    that needs it. The small fusion projection does the opposite: read every
    token, so it sits in the stage host's RAM.

    Returns bytes that fit in no unit's allowed storage (0 on success). When
    ``allow_ssd_tier`` is false, Engram row stores shard across fleet RAM
    instead (hash addressing means any unit can hold any slice) and a
    RAM-overflowing io piece counts toward the shortfall: the flag is a
    storage constraint, not a preference, and a plan that quietly used NVMe
    anyway would lie about being RAM-only.
    """
    unplaced = 0
    preferred: Dict[str, str] = {}
    if stages and profile.vision is not None:
        preferred["vision-encoder"] = stages[0].host_unit

    pieces = [("embedding", profile.embedding_bytes()),
              ("lm_head", profile.lm_head_bytes())]
    if profile.vision is not None:
        pieces.append(("vision-encoder", profile.vision_bytes()))
    for name, size in pieces:
        first_choice = preferred.get(name)
        target = units[first_choice] if first_choice is not None else None
        if target is None or target.headroom_bytes < size:
            target = max(units.values(),
                         key=lambda p: (p.headroom_bytes, p.unit_id))
        if target.headroom_bytes >= size:
            target.io_pieces.append(name)
            target.hot_bytes += size
            continue
        if not allow_ssd_tier:
            warnings.append(
                f"{name} ({size / GB:.2f} GiB) fits in no unit's remaining "
                f"RAM, and the SSD tier is disabled")
            unplaced += size
            continue
        cadence = ("once per image" if name == "vision-encoder"
                   else "once per token")
        warnings.append(
            f"{name} ({size / GB:.2f} GiB) does not fit in any unit's "
            f"remaining RAM; it is placed on {target.unit_id}'s NVMe, which "
            f"costs a streamed read {cadence}")
        target.ssd_expert_bytes += size
        target.io_pieces.append(f"{name}@ssd")

    if profile.engram is not None:
        host_by_layer: Dict[int, str] = {}
        for stage in stages:
            for layer in stage.layers:
                host_by_layer[layer] = stage.host_unit

        def ssd_room(uid: str) -> int:
            return (caps[uid].storage.capacity_bytes // 2
                    - units[uid].ssd_expert_bytes)

        shelf_by_layer: Dict[int, StagePlan] = {}
        for stage in stages:
            for layer in stage.layers:
                shelf_by_layer[layer] = stage

        for table, layer_id in enumerate(profile.engram.layer_ids):
            host = host_by_layer.get(layer_id)
            fuse = profile.engram.fusion_bytes(profile.hidden_size,
                                               profile.weights)
            fuse_target = units.get(host) if host is not None else None
            if fuse_target is None or fuse_target.headroom_bytes < fuse:
                fallback = max(units.values(),
                               key=lambda p: (p.headroom_bytes, p.unit_id))
                if fallback.headroom_bytes >= fuse:
                    warnings.append(
                        f"engram fusion for layer {layer_id} does not fit on "
                        f"its stage host's RAM; it rides on "
                        f"{fallback.unit_id}, so every token pays a cross-shelf "
                        f"hop for it")
                    fuse_target = fallback
                else:
                    fuse_target = None
            if fuse_target is None:
                unplaced += fuse
            else:
                fuse_target.hot_bytes += fuse
                fuse_target.io_pieces.append(f"engram-{layer_id}-fuse")

            rows = profile.engram.table_row_bytes(table, profile.weights)
            if not allow_ssd_tier:
                # No NVMe to lean on: the row store shards across fleet RAM
                # instead — deterministic addressing means a lookup can route
                # to whichever units hold the rows it needs.
                unplaced += _place_engram_ram(
                    table, layer_id, profile.engram, profile.weights,
                    shelf_by_layer[layer_id].expert_units, units, warnings)
                continue
            target_id = host
            if target_id is not None and ssd_room(target_id) < rows:
                warnings.append(
                    f"engram table at layer {layer_id} ({rows / GB:.1f} GiB) "
                    f"does not fit on its stage host's NVMe; looking across "
                    f"the fleet, so lookups pay a cross-shelf hop")
                target_id = None
            if target_id is None:
                fits = [uid for uid in units if ssd_room(uid) >= rows]
                if not fits:
                    unplaced += rows
                    continue
                target_id = max(fits, key=lambda uid: (ssd_room(uid), uid))
            target = units[target_id]
            target.ssd_expert_bytes += rows
            target.io_pieces.append(f"engram-{layer_id}@ssd")
            target.engram_shards.append(EngramRowShard(
                layer=layer_id, first_row=0,
                n_rows=profile.engram.num_embeddings[table],
                byte_count=rows, tier="ssd"))
    return unplaced


def _place_engram_ram(table: int, layer_id: int, engram: EngramConfig,
                      quant: QuantSpec, shelf: Sequence[str],
                      units: Dict[str, UnitPlan],
                      warnings: List[str]) -> int:
    """Shard a table's row store across RAM — the ``--no-ssd`` path.

    Engram addressing is a hash of the token's n-grams, so the rows a lookup
    needs are scattered uniformly no matter how the store is cut; sharding is
    therefore free in exactly one direction that matters — any unit can hold
    any slice — and costly in the other: every token's lookup gathers from
    each holder over the network. Rows fill the table's own shelf first (one
    switch away from the fusion projection), then the roomiest fleet members.
    The cut is in whole rows: each unit gets a contiguous ``[first_row,
    first_row + n_rows)`` range, recorded as an ``EngramRowShard`` so the
    deployment config can name exactly which rows to load where and route
    lookups by row id.
    """
    n_rows = engram.num_embeddings[table]
    row_unit = engram.row_unit_bytes(quant)
    total_bytes = engram.table_row_bytes(table, quant)
    remaining = n_rows
    first_row = 0
    placed_bytes = 0

    def take_into(uid: str) -> None:
        nonlocal remaining, first_row, placed_bytes
        if remaining <= 0:
            return
        take = min(remaining, int(units[uid].headroom_bytes // row_unit))
        if take <= 0:
            return
        # Rounding each slice can drift a byte or two off the table total;
        # give the last slice the remainder so the records sum exactly.
        byte_count = (total_bytes - placed_bytes if take == remaining
                      else int(round(take * row_unit)))
        unit = units[uid]
        unit.engram_shards.append(EngramRowShard(
            layer=layer_id, first_row=first_row, n_rows=take,
            byte_count=byte_count, tier="ram"))
        unit.engram_rows[layer_id] = (unit.engram_rows.get(layer_id, 0)
                                      + byte_count)
        unit.io_pieces.append(f"engram-{layer_id}-rows")
        first_row += take
        placed_bytes += byte_count
        remaining -= take

    for uid in sorted(shelf, key=lambda u: (-units[u].headroom_bytes, u)):
        take_into(uid)
    if remaining > 0:
        outside = sorted((u for u in units if u not in shelf),
                         key=lambda u: (-units[u].headroom_bytes, u))
        for uid in outside:
            take_into(uid)
        if remaining < n_rows:
            warnings.append(
                f"engram table at layer {layer_id} is sharded beyond its own "
                f"shelf; every lookup gathers from those units over the "
                f"network")
    if remaining > 0:
        warnings.append(
            f"engram table at layer {layer_id} needs "
            f"{total_bytes / GB:.1f} GiB of RAM with no NVMe to use; "
            f"{remaining * row_unit / GB:.1f} GiB found nowhere")
    return int(round(remaining * row_unit))


# -- cost model ------------------------------------------------------------

def _estimate_decode(profile: ModelProfile,
                     caps: Dict[str, EffectiveCapability],
                     units: Dict[str, UnitPlan], stages: Sequence[StagePlan],
                     hop_seconds: float,
                     context_tokens: int) -> Tuple[float, int]:
    """Seconds per decoded token, and how many consoles a token lights up.

    Decode at batch 1 is memory-bound everywhere, so a layer costs what it takes
    to *read* the weights that token needs. The shelf's members read in
    parallel, so the layer costs the slowest of them, plus one intra-shelf
    round trip to fan the activation out to the expert holders and gather their
    partial sums back. That round trip, times the layer count, is usually a
    larger term than anyone expects: it is why shelves are bounded and why the
    experts of a layer are not scattered across the fleet.

    Expert reads are prorated by the chance a unit holds one of the top-k, which
    is what makes a wider shelf faster: the same top-k work spread over more
    readers.
    """
    total = 0.0
    active_per_layer: List[float] = []

    shards_by_layer: Dict[int, List[Tuple[str, ExpertShard]]] = {}
    for plan in units.values():
        for shard in plan.shards:
            shards_by_layer.setdefault(shard.layer, []).append(
                (plan.unit_id, shard))

    # Which unit holds each Engram table's row store — the lookup reads rows
    # off that unit's NVMe, plus a cross-shelf hop when it left its stage.
    engram_holder: Dict[int, str] = {}
    for plan in units.values():
        for piece in plan.io_pieces:
            if piece.startswith("engram-") and piece.endswith("@ssd"):
                engram_holder[int(piece[len("engram-"):-len("@ssd")])] = (
                    plan.unit_id)

    for stage in stages:
        host_cap = caps[stage.host_unit]
        for layer in stage.layers:
            moe = profile.moe_for_block(layer)
            expert_bytes = profile.expert_bytes(layer)
            top_k = moe.top_k
            n_routed = moe.n_routed_experts
            # Under CSA the cache a layer *reads* is a small selection of the
            # cache it holds, so the read is what decode costs, not residency.
            kv_read = profile.block_kv_read_bytes(layer, context_tokens)
            worst = _read_seconds(host_cap,
                                  _block_bytes(profile, layer) + kv_read,
                                  "fast")
            # Units a token lights up, deduplicated: a console holding both
            # experts and Engram rows still counts once, at its largest
            # participation probability.
            touched_by: Dict[str, float] = {stage.host_unit: 1.0}
            remote = False
            for uid, shard in shards_by_layer.get(layer, ()):
                expected = shard.n_experts * top_k / n_routed
                if expected <= 0:
                    continue
                worst = max(worst, _read_seconds(caps[uid],
                                                 expected * expert_bytes,
                                                 shard.tier))
                # P(this unit holds at least one of the token's top-k).
                touched_by[uid] = max(
                    touched_by.get(uid, 0.0),
                    1.0 - (1.0 - shard.n_experts / n_routed) ** top_k)
                remote = remote or uid != stage.host_unit
            if remote:
                worst += 2 * hop_seconds  # fan out, gather back
            if (profile.engram is not None
                    and layer in profile.engram.layer_ids):
                # The lookup is serial with the layer that consumes it: row
                # reads off wherever the store landed, then the fusion
                # projection off the stage host's RAM.
                holder = engram_holder.get(layer)
                worst = max(worst, _read_seconds(
                    host_cap,
                    profile.engram.fusion_bytes(profile.hidden_size,
                                                profile.weights),
                    "fast"))
                if holder is not None:
                    worst += _read_seconds(
                        caps[holder],
                        profile.engram.read_bytes_per_token(profile.weights),
                        "ssd")
                    touched_by[holder] = 1.0
                    if holder != stage.host_unit:
                        worst += 2 * hop_seconds
                else:
                    # RAM-sharded store: the token's rows are spread uniformly
                    # over the holders, each answering its share in parallel,
                    # and any holder outside the stage adds the gather hop.
                    # Hash addressing means a lookup touches every holder, so
                    # each counts fully toward the layer's fan-out.
                    shares = [(uid, units[uid].engram_rows[layer])
                              for uid in units
                              if layer in units[uid].engram_rows]
                    held = sum(share for _, share in shares)
                    if held:
                        remote_rows = False
                        for uid, share in shares:
                            part = (profile.engram.read_bytes_per_token(
                                profile.weights) * share / held)
                            worst = max(worst,
                                        _read_seconds(caps[uid], part, "slow"))
                            touched_by[uid] = 1.0
                            remote_rows = (remote_rows
                                           or uid != stage.host_unit)
                        if remote_rows:
                            worst += 2 * hop_seconds
            total += worst
            active_per_layer.append(sum(touched_by.values()))

    # io pieces that overflowed onto NVMe are part of the pipeline too: the
    # embedding is read before the first stage and the lm_head after the last,
    # so each costs its holder a streamed read plus one hop when it has to
    # reach a different shelf. The vision tower is absent here on purpose —
    # it is read once per *image*, not once per decoded token.
    io_sizes = {"embedding": profile.embedding_bytes(),
                "lm_head": profile.lm_head_bytes()}
    consumer = {"embedding": stages[0].host_unit,
                "lm_head": stages[-1].host_unit}
    for plan in units.values():
        for piece in plan.io_pieces:
            if not piece.endswith("@ssd") or piece.startswith("engram-"):
                continue
            size = io_sizes.get(piece[:-len("@ssd")])
            if size is None:
                continue
            total += _read_seconds(caps[plan.unit_id], size, "ssd")
            if plan.unit_id != consumer[piece[:-len("@ssd")]]:
                total += hop_seconds

    total += hop_seconds * max(0, len(stages) - 1)
    peak = max(active_per_layer) if active_per_layer else 0.0
    return total, int(round(peak))


def _read_seconds(cap: EffectiveCapability, nbytes: float, tier: str) -> float:
    """Time to read ``nbytes`` from one tier of a unit, at real efficiency."""
    if nbytes <= 0:
        return 0.0
    if tier == "ssd":
        return nbytes / (cap.storage.effective_read_gbps * 1e9)
    if tier == "fast" or len(cap.coffers) == 1:
        bandwidth = cap.fast_bandwidth_gbps
    else:
        bandwidth = min(c.bandwidth_gbps for c in cap.coffers)
    return nbytes / (bandwidth * 1e9 * BANDWIDTH_EFFICIENCY[cap.backend])


def describe_plan(plan: SplitPlan, *, per_console: bool = True) -> List[str]:
    """Readable summary, shared by the CLIs.

    ``per_console=False`` keeps the headline numbers and the shelf layout but
    drops the line-per-console listing, which is what ``g9 size`` wants: for a
    170-console fleet that listing is the answer to a different question.
    """
    lines = [
        f"{plan.model}"
        + (" (ASSUMED CONFIGURATION)" if plan.model_assumed else "")
        + f" on {plan.n_units} consoles in {len(plan.stages)} shelves",
        f"  context               {plan.context_tokens} tokens",
        f"  decode estimate       {plan.tokens_per_second:.2f} tok/s "
        f"({plan.seconds_per_token * 1000:.0f} ms/token)",
        f"  consoles per layer    ~{plan.active_units_per_token} lit by one token",
    ]
    if plan.ssd_expert_bytes:
        lines.append(f"  streamed from NVMe    "
                     f"{plan.ssd_expert_bytes / GB:.1f} GiB of experts")
    lines.append("  shelves:")
    for stage in plan.stages:
        last = stage.first_layer + stage.n_layers - 1
        lines.append(
            f"    {stage.stage_id}  layers {stage.first_layer:>3}-{last:<3} "
            f"host {stage.host_unit:<14} "
            f"experts on {len(stage.expert_units)} consoles")
    if per_console:
        lines.append("  consoles:")
        for uid in sorted(plan.units):
            u = plan.units[uid]
            lines.append(
                f"    {uid:<14} {u.sku:<15} {u.backend:<9} "
                f"{u.resident_bytes / GB:6.2f}/{u.capacity_bytes / GB:5.2f} GiB "
                f"{u.n_experts:>6} experts "
                f"{len(u.hot_layers) + len(u.dense_layers):>3} hot layers"
                + (f" +{u.ssd_expert_bytes / GB:.1f} GiB NVMe"
                   if u.ssd_expert_bytes else ""))
    for warning in plan.warnings:
        lines.append(f"  ! {warning}")
    return lines
