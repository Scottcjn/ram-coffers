"""Splitting planner: a DeepSeek MoE checkpoint across a heterogeneous fleet.

The PS3 port of this idea had one placement axis, because a 256 MB console can
hold exactly one thing: **one expert per console**. A ninth-generation console
holds 8-13 GiB, which is enough for a whole layer's hot weights *and* a few
dozen routed experts, so the interesting question stops being "does it fit" and
becomes "what belongs where". This module answers it along four axes at once:

1. **Pipeline, by contiguous layer range.** A stage owns consecutive decoder
   layers so a token crosses the network once per stage, carrying one
   ``hidden_size`` activation (16 KiB at V4-Pro width in bf16) rather than
   anything proportional to the weights. Gigabit ethernet is the fleet's
   scarcest resource and this is the axis that respects it.

2. **Hot/cold, by coffer.** Everything an MoE layer reads for *every* token —
   MLA attention, the router, the shared expert, the norms — is small
   (~300 MiB at V4-Pro width) and must sit in the fastest coffer the owning
   console has. The routed experts are ~19 GiB per layer and are read a handful
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
from .model import ModelProfile

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
    kv_bytes: int = 0
    capacity_bytes: int = 0
    fast_capacity_bytes: int = 0
    gemv_gflops: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def resident_bytes(self) -> int:
        """Bytes held in RAM (the SSD tier is not resident)."""
        return (self.hot_bytes + self.fast_expert_bytes + self.slow_expert_bytes
                + self.kv_bytes)

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
    if index < profile.moe.n_dense_layers:
        return profile.dense_layer_bytes()
    if index < profile.n_layers:
        return profile.hot_bytes_per_moe_layer()
    return profile.mtp_hot_bytes()


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

    Raises :class:`PlanningError` only when the fleet cannot host the model even
    with the SSD tier enabled: a fleet that is merely *slow* gets a plan and a
    warning, because "this works, at 0.4 tokens/s" is a useful answer and
    refusing to plan is not.
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
    _assign_io(profile, caps, units, warnings)

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
            f"{ssd_bytes / GB:.1f} GiB of routed experts live on NVMe and are "
            f"streamed on a routing hit; where that happens the layer is bound "
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
    kv_per_layer = context_tokens * profile.mla.kv_cache_bytes_per_token("bf16")

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
        block_need = sum(_block_bytes(profile, first + offset) + kv_per_layer
                         for offset in range(count))
        host = _shelf_host(shelf, units, caps, block_need)
        if host is None:
            # No member can hold every hot block for this many layers. Shrink
            # the shelf's range until one can, and push the rest onward.
            count, host = _shrink_until_hosted(profile, shelf, units, caps,
                                               first, count, kv_per_layer)
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
            plan.kv_bytes += kv_per_layer
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
            f"shorter context ({kv_per_layer / MB:.0f} MiB of KV per layer at "
            f"{context_tokens} tokens)")
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
                         first: int, count: int, kv_per_layer: int
                         ) -> Tuple[int, Optional[str]]:
    """Largest prefix of a shelf's range that one member can host, and who."""
    while count > 0:
        need = sum(_block_bytes(profile, first + o) + kv_per_layer
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
    expert = profile.expert_bytes()
    all_units = sorted(units)
    shortfall = 0
    cross_shelf = 0

    for stage in stages:
        shelf = list(stage.expert_units)
        for layer in stage.layers:
            if not _has_routed_experts(profile, layer):
                continue
            remaining = profile.moe.n_routed_experts
            first = 0
            shares = _shelf_shares(profile, shelf, units)
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
                  units: Dict[str, UnitPlan]) -> Dict[str, int]:
    """How many of a layer's experts each shelf member should take."""
    total = sum(max(units[uid].gemv_gflops, 1.0) for uid in shelf)
    shares: Dict[str, int] = {}
    for uid in shelf:
        fraction = max(units[uid].gemv_gflops, 1.0) / total
        shares[uid] = max(1, int(round(profile.moe.n_routed_experts * fraction)))
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
               units: Dict[str, UnitPlan], warnings: List[str]) -> None:
    """Place the embedding, the LM head, and any MTP heads.

    Both ends of the model are read once per token and are large (~2 GiB each at
    V4-Pro width in bf16, since the vocabulary tables are not FP8), so they go
    wherever there is room, preferring the units that already host the first and
    last layers so the token starts and ends where it is embedded.
    """
    pieces = [("embedding", profile.embedding_bytes()),
              ("lm_head", profile.lm_head_bytes())]
    for name, size in pieces:
        target = max(units.values(),
                     key=lambda p: (p.headroom_bytes, p.unit_id))
        if target.headroom_bytes < size:
            warnings.append(
                f"{name} ({size / GB:.2f} GiB) does not fit in any unit's "
                f"remaining RAM; it is placed on {target.unit_id}'s NVMe, which "
                f"costs a streamed read once per token")
            target.ssd_expert_bytes += size
            target.io_pieces.append(f"{name}@ssd")
            continue
        target.io_pieces.append(name)
        target.hot_bytes += size


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
    expert_bytes = profile.expert_bytes()
    top_k = profile.moe.top_k
    n_routed = profile.moe.n_routed_experts
    kv_read = context_tokens * profile.mla.kv_cache_bytes_per_token("bf16")
    total = 0.0
    active_per_layer: List[float] = []

    shards_by_layer: Dict[int, List[Tuple[str, ExpertShard]]] = {}
    for plan in units.values():
        for shard in plan.shards:
            shards_by_layer.setdefault(shard.layer, []).append(
                (plan.unit_id, shard))

    for stage in stages:
        host_cap = caps[stage.host_unit]
        for layer in stage.layers:
            worst = _read_seconds(host_cap,
                                  _block_bytes(profile, layer) + kv_read,
                                  "fast")
            touched = 1.0
            remote = False
            for uid, shard in shards_by_layer.get(layer, ()):
                expected = shard.n_experts * top_k / n_routed
                if expected <= 0:
                    continue
                worst = max(worst, _read_seconds(caps[uid],
                                                 expected * expert_bytes,
                                                 shard.tier))
                # P(this unit holds at least one of the token's top-k).
                touched += 1.0 - (1.0 - shard.n_experts / n_routed) ** top_k
                remote = remote or uid != stage.host_unit
            if remote:
                worst += 2 * hop_seconds  # fan out, gather back
            total += worst
            active_per_layer.append(touched)

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
