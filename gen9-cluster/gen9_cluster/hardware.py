"""Ninth-generation console hardware model: SKUs, coffers, and downbinning.

RAM Coffers (upstream) partitions a model across NUMA banks inside one POWER8
box, each bank a "coffer" holding a known slice of knowledge. A ninth-generation
console has the same structure in miniature and for free: its unified GDDR6 is
*already* split into bandwidth tiers by the hardware designers, and its NVMe is a
third tier fast enough to stream weights. So a console is not one coffer, it is a
small coffer hierarchy:

    Xbox Series X   10 GB @ 560 GB/s  (fast coffer)  + 6 GB @ 336 GB/s (3.5 GB
                    usable after the OS reservation, slow coffer) + 2.4 GB/s NVMe
    Xbox Series S    8 GB @ 224 GB/s  + 2 GB @ 56 GB/s + 2.4 GB/s NVMe
    PS5 / PS5 Slim  16 GB @ 448 GB/s  (single tier) + 5.5 GB/s NVMe
    PS5 Pro         16 GB @ 576 GB/s  + 2 GB DDR5 OS-only + 5.5 GB/s NVMe

The same silicon is also sold *outside* a console shell, harvested harder: the
AMD 4700S desktop kit is a PS5 "Ariel" SoC and the 4800S a Series X one, both
with the GPU fused off entirely and 16 GB of GDDR6 soldered to the board as
system memory, and the BC-250 blade is a PS5 Oberon/Cyan Skillfish part with 6
of 8 CPU cores and 24 of 40 CUs enabled by the driver. Those boards are the
cheapest way to add capacity to a fleet, they run a normal Linux, and the BC-250
in particular is the only member of the family with a *routinely exercised* GPU
compute path — so they are first-class SKUs here, not a footnote.

Two things this module exists to get right:

**Variant boards with components disabled.** Every one of these consoles is a
downbinned part *by design* — the Series X ships 56 physical CUs and enables 52,
the PS5 ships 40 and enables 36 — and the fleet a real deployment gets hold of
adds its own reductions on top: a dev-mode sandbox that only hands out ~5 GB, a
board with a dead GDDR6 package, a Series S whose 10th memory channel is
depopulated, a console throttled to a lower clock by a failed fan. None of that
may be inferred from the SKU name. :class:`Downbin` records what a *specific*
unit lost, :class:`ConsoleUnit.effective` folds it into the nominal SKU, and the
planner in ``planner.py`` only ever reads the effective numbers. A fleet is
therefore allowed to be wildly heterogeneous; the planner rebalances rather than
refusing.

**Backends differ in what they can be trusted to do.** A PS5 running
``ps5-linux`` gets a GFX1013 amdgpu, which Vulkan/RADV drives well; ROCm builds
for gfx1013 exist and can be pointed at, but parts of the ROCm library stack
either do not ship that target or fall back to a lower capability level because
their build scripts assume a higher gfx tier, so ROCm throughput must be
*measured per node* rather than assumed. An Xbox in retail Dev Mode has no
compute-shader path we can rely on and a hard memory sandbox. Hence
:class:`ComputeBackend` and the per-unit ``measured_gemv_gflops`` override.

Nothing here touches hardware; it is a datasheet plus arithmetic, so a fleet can
be planned from a laptop. Numbers marked ``UNVERIFIED`` in comments come from
vendor disclosures rather than measurement on the fleet.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

MB = 1024 * 1024
GB = 1024 * MB


class ComputeBackend(enum.Enum):
    """How a console actually runs the GEMV kernel.

    The distinction matters to the planner because a console's usable throughput
    is a property of its *software* path as much as its silicon.
    """

    #: Zen 2 cores, AVX2 (no AVX-512 on Zen 2), 8 cores / 16 threads.
    CPU_AVX2 = "cpu-avx2"
    #: Vulkan compute via RADV on the console's amdgpu (PS5 under ps5-linux).
    #: The default GPU path: no tuned-library assumptions, works on GFX1013.
    VULKAN = "vulkan"
    #: ROCm/HIP. Usable where a working gfx1013 ROCm stack is installed (Debian
    #: builds do run), but library coverage is uneven: parts of the userspace
    #: stack do not ship the target at all and others fall back to a lower
    #: capability level because their build scripts assume a higher gfx tier.
    #: Declare it per node and let the planner use ``measured_gemv_gflops``
    #: rather than a nominal figure.
    ROCM = "rocm"
    #: Direct3D 12 compute, the only GPU path an Xbox GDK title has. Reachable
    #: with a real GDK/devkit; NOT reachable from a retail Dev Mode UWP app with
    #: any throughput worth planning around.
    D3D12 = "d3d12"

    @property
    def is_gpu(self) -> bool:
        return self is not ComputeBackend.CPU_AVX2

    @property
    def throughput_is_assumed(self) -> bool:
        """True when the nominal figure must not be trusted without measuring.

        ROCm on a console GPU hits libraries that may silently drop to a lower
        capability path; Vulkan compute we write ourselves, so its ceiling is the
        hardware's.
        """
        return self is ComputeBackend.ROCM


class Runtime(enum.Enum):
    """The software environment a console node runs under."""

    #: ps5-linux (patched-hypervisor Linux, PS5 phat/slim, FW 3.00-7.61):
    #: full 8c/16t at 3.5 GHz and the GPU at 2.23 GHz via amdgpu/GFX1013.
    PS5_LINUX = "ps5-linux"
    #: PS5 homebrew ELF under HEN: CPU only, no GPU compute, reduced allocator.
    PS5_HEN = "ps5-hen"
    #: Retail Xbox Series Dev Mode. Hard memory sandbox (app ~1 GB, Creators
    #: game ~5 GB), partial CPU cores, no dependable compute path.
    XBOX_DEVMODE = "xbox-devmode"
    #: Xbox GDK title on a devkit / activated console: full memory and D3D12.
    XBOX_GDK = "xbox-gdk"
    #: A stock Linux distribution on a desktop/blade board built from console
    #: silicon (4700S, 4800S, BC-250). No sandbox, no hypervisor, ordinary
    #: amdgpu where a GPU survives the harvest.
    SALVAGE_LINUX = "salvage-linux"
    #: Not a console: an x86-64 host standing in for one in tests and bring-up.
    HOST_SIM = "host-sim"

    @property
    def console_family(self) -> str:
        if self in (Runtime.PS5_LINUX, Runtime.PS5_HEN):
            return "playstation"
        if self in (Runtime.XBOX_DEVMODE, Runtime.XBOX_GDK):
            return "xbox"
        if self is Runtime.SALVAGE_LINUX:
            return "salvage"
        return "host"


@dataclass(frozen=True)
class MemoryTier:
    """One bandwidth tier of a console's unified memory: a coffer.

    ``os_reserved_bytes`` is memory the platform keeps for itself and is
    subtracted before anything is placed. ``bandwidth_gbps`` is the vendor's
    figure for the tier, in GB/s.
    """

    name: str
    total_bytes: int
    bandwidth_gbps: float
    os_reserved_bytes: int = 0
    #: False for a tier that exists but cannot hold weights read every token
    #: (e.g. the PS5 Pro's 2 GB DDR5, which is OS-only).
    usable_for_weights: bool = True

    @property
    def usable_bytes(self) -> int:
        if not self.usable_for_weights:
            return 0
        return max(0, self.total_bytes - self.os_reserved_bytes)


@dataclass(frozen=True)
class StorageSpec:
    """The console's NVMe, used as the cold coffer for streamed experts."""

    capacity_bytes: int
    read_gbps: float                 # raw sequential
    compressed_read_gbps: float = 0.0  # with the console's hardware decompressor

    @property
    def effective_read_gbps(self) -> float:
        return max(self.read_gbps, self.compressed_read_gbps)


@dataclass(frozen=True)
class ConsoleSKU:
    """Nominal datasheet for one console model.

    ``cu_physical`` vs ``cu_enabled`` is the vendor's own downbin: these consoles
    ship with compute units fused off for yield (Series X 56 -> 52, PS5 40 -> 36,
    Series S 24 -> 20). Per-unit losses beyond that live in :class:`Downbin`.
    """

    sku: str
    family: str                     # "playstation" | "xbox"
    marketing_name: str
    cpu_cores: int
    cpu_threads: int
    cpu_ghz: float
    cu_physical: int
    cu_enabled: int
    gpu_ghz: float
    gpu_tflops_fp32: float
    tiers: Tuple[MemoryTier, ...]
    storage: StorageSpec
    #: Backends the model can reach at all, most preferred first.
    backends: Tuple[ComputeBackend, ...] = (ComputeBackend.CPU_AVX2,)
    #: Board variants that are electrically the same compute part (a digital
    #: edition is the same SoC minus a Blu-ray drive), recorded so an inventory
    #: can name the board it has without inventing a new capability row.
    board_revisions: Tuple[str, ...] = ()
    notes: str = ""

    @property
    def total_memory_bytes(self) -> int:
        return sum(t.total_bytes for t in self.tiers)

    @property
    def weight_memory_bytes(self) -> int:
        return sum(t.usable_bytes for t in self.tiers)

    @property
    def fast_tier(self) -> MemoryTier:
        """The highest-bandwidth tier that can hold weights."""
        usable = [t for t in self.tiers if t.usable_bytes > 0]
        if not usable:
            raise ValueError(f"{self.sku} has no tier usable for weights")
        return max(usable, key=lambda t: t.bandwidth_gbps)


@dataclass(frozen=True)
class Downbin:
    """What one physical unit lost relative to its SKU.

    Every field is a *reduction*, so an all-zero ``Downbin`` is a pristine
    console and the planner needs no special case for one. Reasons are free text
    and end up in the plan's audit trail, because "why is this node holding four
    fewer experts" is the first question a fleet operator asks.

    ``memory_bytes_lost`` is applied to the tiers named in ``tier_losses`` when
    those are given, and to the slow tier first otherwise — a dead GDDR6 package
    takes capacity off one tier, not off the console uniformly.
    """

    cu_disabled: int = 0
    cpu_cores_disabled: int = 0
    cpu_ghz_cap: Optional[float] = None
    gpu_ghz_cap: Optional[float] = None
    #: tier name -> bytes lost on that tier.
    tier_losses: Dict[str, int] = field(default_factory=dict)
    #: tier name -> fraction of the tier's bandwidth still available (0..1).
    tier_bandwidth_scale: Dict[str, float] = field(default_factory=dict)
    #: Hard ceiling on what the process may allocate at all (Dev Mode sandbox).
    memory_budget_bytes: Optional[int] = None
    reasons: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cu_disabled < 0 or self.cpu_cores_disabled < 0:
            raise ValueError("downbin counts are reductions and cannot be negative")
        for tier, scale in self.tier_bandwidth_scale.items():
            if not 0.0 < scale <= 1.0:
                raise ValueError(
                    f"bandwidth scale for tier {tier!r} must be in (0, 1], got {scale}")

    @property
    def is_pristine(self) -> bool:
        return (self.cu_disabled == 0 and self.cpu_cores_disabled == 0
                and self.cpu_ghz_cap is None and self.gpu_ghz_cap is None
                and not self.tier_losses and not self.tier_bandwidth_scale
                and self.memory_budget_bytes is None)

    def merge(self, other: "Downbin") -> "Downbin":
        """Combine two reductions (a sandbox plus a hardware fault)."""
        tier_losses = dict(self.tier_losses)
        for name, lost in other.tier_losses.items():
            tier_losses[name] = tier_losses.get(name, 0) + lost
        scales = dict(self.tier_bandwidth_scale)
        for name, scale in other.tier_bandwidth_scale.items():
            scales[name] = scales.get(name, 1.0) * scale
        budgets = [b for b in (self.memory_budget_bytes,
                               other.memory_budget_bytes) if b is not None]
        caps = [c for c in (self.cpu_ghz_cap, other.cpu_ghz_cap) if c is not None]
        gpu_caps = [c for c in (self.gpu_ghz_cap, other.gpu_ghz_cap)
                    if c is not None]
        return Downbin(
            cu_disabled=self.cu_disabled + other.cu_disabled,
            cpu_cores_disabled=self.cpu_cores_disabled + other.cpu_cores_disabled,
            cpu_ghz_cap=min(caps) if caps else None,
            gpu_ghz_cap=min(gpu_caps) if gpu_caps else None,
            tier_losses=tier_losses,
            tier_bandwidth_scale=scales,
            memory_budget_bytes=min(budgets) if budgets else None,
            reasons=self.reasons + other.reasons)


@dataclass(frozen=True)
class EffectiveCapability:
    """What a specific unit can actually be asked to do.

    This is the only view the planner consumes. ``weight_bytes`` is what may hold
    weights after OS reservations, per-unit memory loss, and any sandbox ceiling;
    ``coffers`` is that capacity split by bandwidth tier, fastest first.
    """

    unit_id: str
    sku: str
    runtime: Runtime
    backend: ComputeBackend
    cu_active: int
    cpu_cores_active: int
    cpu_ghz: float
    gpu_ghz: float
    gemv_gflops: float
    coffers: Tuple[MemoryTier, ...]
    storage: StorageSpec
    throughput_measured: bool
    warnings: Tuple[str, ...] = ()

    @property
    def weight_bytes(self) -> int:
        return sum(t.usable_bytes for t in self.coffers)

    @property
    def fast_bytes(self) -> int:
        return self.coffers[0].usable_bytes if self.coffers else 0

    @property
    def fast_bandwidth_gbps(self) -> float:
        return self.coffers[0].bandwidth_gbps if self.coffers else 0.0

    def weight_bytes_at_least(self, bandwidth_gbps: float) -> int:
        """Capacity in tiers at or above a bandwidth floor.

        Used to place weights that are read every token (attention, shared
        expert, router) only where reading them is affordable.
        """
        return sum(t.usable_bytes for t in self.coffers
                   if t.bandwidth_gbps >= bandwidth_gbps)


# --------------------------------------------------------------------------
# SKU table
# --------------------------------------------------------------------------
# Vendor-disclosed figures. Memory reservations are the publicly stated
# game-available splits; where a platform never stated one, the value is marked
# UNVERIFIED and chosen conservatively (reserving more, not less).

_PS5_STORAGE = StorageSpec(capacity_bytes=825 * GB, read_gbps=5.5,
                           compressed_read_gbps=8.0)
_PS5_SLIM_STORAGE = StorageSpec(capacity_bytes=1024 * GB, read_gbps=5.5,
                                compressed_read_gbps=8.0)
_PS5_PRO_STORAGE = StorageSpec(capacity_bytes=2048 * GB, read_gbps=5.5,
                               compressed_read_gbps=8.0)
_XSX_STORAGE = StorageSpec(capacity_bytes=1024 * GB, read_gbps=2.4,
                           compressed_read_gbps=4.8)
_XSS_STORAGE = StorageSpec(capacity_bytes=512 * GB, read_gbps=2.4,
                           compressed_read_gbps=4.8)

#: PS5 keeps ~3.5 GB of its 16 GB for the OS on the pre-Pro boards (UNVERIFIED:
#: Sony never published the split; 13.5 GB game-available is the widely
#: reproduced figure and is what we budget).
_PS5_OS_RESERVE = 16 * GB - 13 * GB

PS5 = ConsoleSKU(
    sku="ps5",
    family="playstation",
    marketing_name="PlayStation 5 (CFI-10xx/11xx/12xx)",
    cpu_cores=8, cpu_threads=16, cpu_ghz=3.5,
    cu_physical=40, cu_enabled=36, gpu_ghz=2.23, gpu_tflops_fp32=10.28,
    tiers=(MemoryTier("gddr6", 16 * GB, 448.0, _PS5_OS_RESERVE),),
    storage=_PS5_STORAGE,
    backends=(ComputeBackend.VULKAN, ComputeBackend.ROCM,
              ComputeBackend.CPU_AVX2),
    board_revisions=("CFI-1000A", "CFI-1000B", "CFI-1100A", "CFI-1100B",
                     "CFI-1200A", "CFI-1200B"),
    notes="Oberon; 4 of 40 CUs fused off for yield. B boards are the digital "
          "edition: identical SoC, no Blu-ray drive.",
)

PS5_SLIM = replace(
    PS5, sku="ps5-slim",
    marketing_name="PlayStation 5 Slim (CFI-20xx)",
    storage=_PS5_SLIM_STORAGE,
    board_revisions=("CFI-2000A", "CFI-2000B", "CFI-2100A", "CFI-2100B"),
    notes="Oberon Plus (6 nm shrink); same 36-of-40 CU configuration and same "
          "448 GB/s. B boards are digital: same SoC, no drive.",
)

PS5_PRO = ConsoleSKU(
    sku="ps5-pro",
    family="playstation",
    marketing_name="PlayStation 5 Pro (CFI-70xx)",
    cpu_cores=8, cpu_threads=16, cpu_ghz=3.85,
    # 60 enabled CUs / 30 WGPs. Physical count on the Trinity die is not
    # disclosed; assume the same 4-CU harvest as Oberon (UNVERIFIED).
    cu_physical=64, cu_enabled=60, gpu_ghz=2.17, gpu_tflops_fp32=16.7,
    tiers=(MemoryTier("gddr6", 16 * GB, 576.0, 0),
           MemoryTier("ddr5-os", 2 * GB, 0.0, 2 * GB,
                      usable_for_weights=False)),
    storage=_PS5_PRO_STORAGE,
    backends=(ComputeBackend.VULKAN, ComputeBackend.ROCM,
              ComputeBackend.CPU_AVX2),
    board_revisions=("CFI-7000", "CFI-7020"),
    notes="A separate 2 GB DDR5 die holds the OS, so the whole 16 GB GDDR6 is "
          "application-available. No public ps5-linux support for this board "
          "yet (UNVERIFIED): treat Pro units as CPU-only until proven.",
)

XBOX_SERIES_X = ConsoleSKU(
    sku="xbox-series-x",
    family="xbox",
    marketing_name="Xbox Series X (1 TB)",
    cpu_cores=8, cpu_threads=16, cpu_ghz=3.8,
    cu_physical=56, cu_enabled=52, gpu_ghz=1.825, gpu_tflops_fp32=12.15,
    # The published split: 10 GB "GPU optimal" at 560 GB/s and 6 GB at
    # 336 GB/s, of which 2.5 GB is the system reservation.
    tiers=(MemoryTier("gddr6-fast", 10 * GB, 560.0, 0),
           MemoryTier("gddr6-slow", 6 * GB, 336.0, 2560 * MB)),
    storage=_XSX_STORAGE,
    backends=(ComputeBackend.D3D12, ComputeBackend.CPU_AVX2),
    board_revisions=("1882", "1883-digital-1tb"),
    notes="Scarlett; 4 of 56 CUs fused off for yield. The 2024 all-digital "
          "1 TB board is the same compute part without the drive.",
)

XBOX_SERIES_S = ConsoleSKU(
    sku="xbox-series-s",
    family="xbox",
    marketing_name="Xbox Series S (512 GB / 1 TB)",
    cpu_cores=8, cpu_threads=16, cpu_ghz=3.6,
    cu_physical=24, cu_enabled=20, gpu_ghz=1.565, gpu_tflops_fp32=4.0,
    tiers=(MemoryTier("gddr6-fast", 8 * GB, 224.0, 0),
           MemoryTier("gddr6-slow", 2 * GB, 56.0, 2 * GB)),
    storage=_XSS_STORAGE,
    backends=(ComputeBackend.D3D12, ComputeBackend.CPU_AVX2),
    board_revisions=("1901", "1901-1tb-carbon"),
    notes="Lockhart; 20 of 24 CUs enabled. The 2 GB slow tier is entirely the "
          "system reservation, so only the 8 GB tier can hold weights.",
)

# -- salvage silicon: the same SoCs sold as cheap boards, harvested harder ----

AMD_4700S = ConsoleSKU(
    sku="amd-4700s",
    family="salvage",
    marketing_name="AMD 4700S Desktop Kit (PS5 'Ariel' SoC, GPU fused off)",
    cpu_cores=8, cpu_threads=16, cpu_ghz=3.2,
    cu_physical=40, cu_enabled=0, gpu_ghz=0.0, gpu_tflops_fp32=0.0,
    # 16 GB GDDR6 (8 SK hynix packages, 256-bit, 14 Gbps) soldered to the board
    # and used as system memory. The DRAM is the console's, but the CPU reaches
    # it through a fabric never meant to be its only client: 92.9 GB/s AIDA64
    # copy, against the PS5 GPU's 448 GB/s, at 145 ns rather than DDR4's 74 ns
    # (MEASURED by Tom's Hardware on a retail kit, not by us). Latency that high
    # is survivable here because expert GEMV reads are sequential.
    tiers=(MemoryTier("gddr6", 16 * GB, 92.0, 1 * GB),),
    # SATA only: the board has no M.2 and the sole PCIe slot may not be spent on
    # a carrier card, so the cold tier is a 6 Gb/s SSD, not an NVMe.
    storage=StorageSpec(capacity_bytes=512 * GB, read_gbps=0.55),
    backends=(ComputeBackend.CPU_AVX2,),
    board_revisions=("4700s-itx",),
    notes="GPU permanently disabled, so this is a pure CPU node with an unusual "
          "amount of bandwidth-rich memory: the natural host for cold routed "
          "experts. The slot is x16 mechanically but PCIe 2.0 x4 electrically "
          "(~2 GB/s), too narrow to feed a card's VRAM from board memory; a "
          "card added here is a compute unit for what already fits in it.",
)

AMD_4800S = replace(
    AMD_4700S, sku="amd-4800s",
    marketing_name="AMD 4800S Desktop Kit (Xbox Series X SoC, GPU fused off)",
    cpu_ghz=4.0, cu_physical=56,
    # Same memory arrangement, but the board is mATX with an M.2 for storage and
    # a PCIe 4.0 x4 slot (~7.9 GB/s) instead of the 4700S's Gen2 x4.
    storage=StorageSpec(capacity_bytes=512 * GB, read_gbps=3.5),
    board_revisions=("4800s-matx",),
    notes="The later kit is a different harvest, not a revision: Series X "
          "silicon rather than the 4700S's PS5 Ariel, clocked to 4.0 GHz, with "
          "an M.2 and a Gen4 x4 slot. Same CPU-only capability model, but the "
          "wider slot is what makes a discrete card worth considering.",
)

BC_250 = ConsoleSKU(
    sku="bc-250",
    family="salvage",
    marketing_name="AMD BC-250 blade (PS5 Oberon / Cyan Skillfish, gfx1013)",
    # 6 of 8 Zen 2 cores and 24 of the die's 40 CUs enabled by the driver
    # baseline; the 40-CU unlock is community reverse-engineering, opted into
    # per unit via ``ConsoleUnit.cu_enabled_override``.
    cpu_cores=6, cpu_threads=12, cpu_ghz=3.5,
    cu_physical=40, cu_enabled=24, gpu_ghz=2.0, gpu_tflops_fp32=6.8,
    # 16 GB unified GDDR6, but what a compute job can actually map is bounded by
    # the GTT cap rather than by capacity: ~13 GiB is the figure the community
    # reaches after raising the kernel's TTM limits.
    tiers=(MemoryTier("gddr6", 16 * GB, 360.0, 3 * GB),),
    storage=StorageSpec(capacity_bytes=256 * GB, read_gbps=3.5),
    backends=(ComputeBackend.VULKAN, ComputeBackend.ROCM,
              ComputeBackend.CPU_AVX2),
    board_revisions=("bc-250", "bc250"),
    notes="Ex-mining blade, Linux-only, and the best-exercised GPU compute path "
          "in this family: Vulkan/RADV works, while ROCm's userspace does not "
          "ship gfx1013. Bandwidth is the measured ~360 GB/s, not the PS5's "
          "448 GB/s figure.",
)

HOST_SIM_SKU = ConsoleSKU(
    sku="host-sim",
    family="host",
    marketing_name="x86-64 host standing in for a console",
    cpu_cores=8, cpu_threads=16, cpu_ghz=3.0,
    cu_physical=0, cu_enabled=0, gpu_ghz=0.0, gpu_tflops_fp32=0.0,
    tiers=(MemoryTier("host-ram", 8 * GB, 50.0, 0),),
    storage=StorageSpec(capacity_bytes=64 * GB, read_gbps=1.0),
    backends=(ComputeBackend.CPU_AVX2,),
    notes="Bring-up and CI only; never part of a capability claim.",
)

SKUS: Dict[str, ConsoleSKU] = {
    s.sku: s for s in (PS5, PS5_SLIM, PS5_PRO, XBOX_SERIES_X, XBOX_SERIES_S,
                       AMD_4700S, AMD_4800S, BC_250, HOST_SIM_SKU)
}

#: Board revision string -> SKU, so an inventory may name the board it has.
BOARD_TO_SKU: Dict[str, str] = {
    rev: sku.sku for sku in SKUS.values() for rev in sku.board_revisions
}


def sku_for(name: str) -> ConsoleSKU:
    """Look a SKU up by SKU id or by board revision."""
    if name in SKUS:
        return SKUS[name]
    if name in BOARD_TO_SKU:
        return SKUS[BOARD_TO_SKU[name]]
    raise KeyError(f"unknown console SKU or board revision {name!r}; known: "
                   f"{sorted(SKUS)} / {sorted(BOARD_TO_SKU)}")


# --------------------------------------------------------------------------
# Runtime sandboxes
# --------------------------------------------------------------------------
# A runtime is itself a downbin: it takes memory and cores away from an
# otherwise healthy console. Modelling it this way means a Dev Mode Xbox and a
# console with a dead memory package flow through exactly the same arithmetic.

#: Retail Dev Mode hands a UWP *app* ~1 GB and a Creators-program *game* ~5 GB,
#: with 4 exclusive + 2 shared CPU cores for games. We plan for the game budget
#: and take the CPU reduction; there is no compute path worth planning around.
XBOX_DEVMODE_GAME_BUDGET = 5 * GB
XBOX_DEVMODE_APP_BUDGET = 1 * GB


def runtime_downbin(runtime: Runtime, *, sku: ConsoleSKU,
                    devmode_app: bool = False) -> Downbin:
    """The reduction a runtime imposes on an otherwise pristine console."""
    if runtime is Runtime.XBOX_DEVMODE:
        budget = (XBOX_DEVMODE_APP_BUDGET if devmode_app
                  else XBOX_DEVMODE_GAME_BUDGET)
        return Downbin(
            cpu_cores_disabled=2,
            memory_budget_bytes=budget,
            reasons=(f"retail Dev Mode sandbox: "
                     f"{budget // MB} MB process budget, 6 of 8 CPU cores, no "
                     f"dependable compute-shader path",),
        )
    if runtime is Runtime.PS5_HEN:
        # Homebrew under HEN runs in a userland process with the GPU owned by
        # the system; no compute path, and the allocator is well short of the
        # game budget (UNVERIFIED: conservative).
        return Downbin(
            memory_budget_bytes=8 * GB,
            reasons=("PS5 HEN homebrew: CPU only, conservative 8 GB allocator "
                     "budget",),
        )
    if runtime is Runtime.PS5_LINUX:
        # A real Linux userland, so the kernel and page cache cost something.
        return Downbin(
            tier_losses={sku.fast_tier.name: 1 * GB},
            reasons=("ps5-linux kernel/userland reservation",),
        )
    if runtime is Runtime.SALVAGE_LINUX:
        return Downbin(
            tier_losses={sku.fast_tier.name: 1 * GB},
            reasons=("Linux kernel/userland reservation on a salvage board",),
        )
    return Downbin()


def pick_backend(sku: ConsoleSKU, runtime: Runtime,
                 requested: Optional[ComputeBackend] = None
                 ) -> Tuple[ComputeBackend, Tuple[str, ...]]:
    """Choose a compute backend for a unit, with the reasons it was not better.

    ``requested`` is honoured when the SKU can reach it and the runtime allows
    it; otherwise the best allowed backend is returned along with a warning
    naming what was refused. GPU compute is only offered where the runtime
    actually exposes it: ``ps5-linux`` (amdgpu/GFX1013) and an Xbox GDK title.
    """
    warnings: List[str] = []
    allowed: Tuple[ComputeBackend, ...]
    if runtime is Runtime.SALVAGE_LINUX:
        allowed = tuple(b for b in sku.backends
                        if b is not ComputeBackend.D3D12)
    elif runtime in (Runtime.XBOX_DEVMODE, Runtime.PS5_HEN, Runtime.HOST_SIM):
        allowed = (ComputeBackend.CPU_AVX2,)
        if requested is not None and requested.is_gpu:
            warnings.append(
                f"{requested.value} refused under {runtime.value}: no usable "
                f"compute path, falling back to CPU")
    else:
        allowed = tuple(b for b in sku.backends
                        if b is not ComputeBackend.D3D12
                        or runtime is Runtime.XBOX_GDK)
    if not allowed:
        allowed = (ComputeBackend.CPU_AVX2,)
        warnings.append(f"{sku.sku} exposes no compute backend under "
                        f"{runtime.value}; using the CPU path")
    if requested is not None and requested in allowed:
        chosen = requested
    else:
        if requested is not None and requested not in allowed:
            warnings.append(
                f"{requested.value} not available on {sku.sku} under "
                f"{runtime.value}; using {allowed[0].value}")
        chosen = allowed[0]
    if chosen is ComputeBackend.ROCM:
        warnings.append(
            "ROCm on a console GPU (gfx1013) is only partially supported: some "
            "libraries do not ship the target and others fall back to a lower "
            "capability level, so throughput must come from "
            "measured_gemv_gflops rather than the datasheet")
    return chosen, tuple(warnings)


# --------------------------------------------------------------------------
# Throughput model
# --------------------------------------------------------------------------
# MoE decode is memory-bound, not FLOP-bound: one token reads each active
# expert's weights once and does two passes of arithmetic over them. So the
# figure the planner needs is not peak TFLOPS but "how fast can this unit stream
# weights through its ALUs", which is why the estimate is anchored on bandwidth
# and only clipped by a fraction of peak compute.

#: Fraction of a tier's peak bandwidth a well-written GEMV kernel sustains.
#: Conservative; console GPUs are easier to feed than discrete ones because
#: there is no PCIe hop, but a strided FP8 read with per-block scales is not a
#: pure streaming load.
BANDWIDTH_EFFICIENCY = {
    ComputeBackend.CPU_AVX2: 0.35,
    ComputeBackend.VULKAN: 0.70,
    ComputeBackend.ROCM: 0.70,
    ComputeBackend.D3D12: 0.70,
}

#: Fraction of peak FP32 a GEMV (arithmetic intensity ~2 flop/byte) can reach.
COMPUTE_EFFICIENCY = 0.25


def _cpu_gflops(cores: int, ghz: float) -> float:
    """Zen 2 AVX2 peak: 2 FMA units x 8 fp32 lanes x 2 flop, per core."""
    return cores * ghz * 2 * 8 * 2


def estimate_gemv_gflops(backend: ComputeBackend, *, cu_active: int,
                         gpu_ghz: float, cpu_cores: int, cpu_ghz: float,
                         bandwidth_gbps: float) -> float:
    """Sustainable GEMV rate for a unit, in GFLOP/s.

    Memory-bound: bandwidth x 2 flop/byte, clipped by a fraction of peak
    arithmetic so a hypothetical unit with enormous bandwidth and four CUs is
    not credited with the bandwidth.
    """
    if backend is ComputeBackend.CPU_AVX2:
        peak = _cpu_gflops(cpu_cores, cpu_ghz)
    else:
        # RDNA2: 64 lanes/CU x 2 flop x clock.
        peak = cu_active * 64 * 2 * gpu_ghz
    memory_bound = bandwidth_gbps * BANDWIDTH_EFFICIENCY[backend] * 2.0
    return min(memory_bound, peak * COMPUTE_EFFICIENCY)


@dataclass
class ConsoleUnit:
    """One physical console in the fleet.

    The inventory describes units, not models: ``sku`` says what it was born as,
    ``downbin`` says what it is now, and ``measured_gemv_gflops`` lets an
    operator who has actually benchmarked the box override the estimate (the
    only honest option for a ROCm node).
    """

    unit_id: str
    sku: str
    runtime: Runtime = Runtime.HOST_SIM
    backend: Optional[ComputeBackend] = None
    downbin: Downbin = field(default_factory=Downbin)
    measured_gemv_gflops: Optional[float] = None
    devmode_app: bool = False
    #: Compute units enabled beyond the SKU's driver baseline. The BC-250's
    #: 40-CU unlock is the only case: the die has 40, the driver lights 24, and
    #: a community tool raises it at runtime. Capped at ``cu_physical``.
    cu_enabled_override: Optional[int] = None
    #: Free-form: rack position, MAC, serial. Carried into the plan untouched.
    labels: Dict[str, str] = field(default_factory=dict)

    def spec(self) -> ConsoleSKU:
        return sku_for(self.sku)

    def effective(self) -> EffectiveCapability:
        """Fold SKU, runtime sandbox, and per-unit losses into one capability."""
        spec = self.spec()
        total = runtime_downbin(self.runtime, sku=spec,
                                devmode_app=self.devmode_app).merge(self.downbin)
        backend, warnings = pick_backend(spec, self.runtime, self.backend)

        baseline = spec.cu_enabled
        if self.cu_enabled_override is not None:
            baseline = min(int(self.cu_enabled_override), spec.cu_physical)
        cu_active = max(0, baseline - total.cu_disabled)
        cores = max(1, spec.cpu_cores - total.cpu_cores_disabled)
        cpu_ghz = min(spec.cpu_ghz, total.cpu_ghz_cap or spec.cpu_ghz)
        gpu_ghz = min(spec.gpu_ghz, total.gpu_ghz_cap or spec.gpu_ghz)
        if backend.is_gpu and cu_active == 0:
            backend = ComputeBackend.CPU_AVX2
            warnings = warnings + (
                "every compute unit is disabled on this unit; forced to the CPU "
                "backend",)

        coffers = _apply_memory_downbin(spec, total)
        if not coffers:
            warnings = warnings + (
                "no memory tier survives this unit's reductions; the planner "
                "will place nothing here",)

        bandwidth = coffers[0].bandwidth_gbps if coffers else 0.0
        estimated = estimate_gemv_gflops(
            backend, cu_active=cu_active, gpu_ghz=gpu_ghz, cpu_cores=cores,
            cpu_ghz=cpu_ghz, bandwidth_gbps=bandwidth)
        if self.measured_gemv_gflops is not None:
            gflops = float(self.measured_gemv_gflops)
            measured = True
        else:
            gflops = estimated
            measured = False
            if backend.throughput_is_assumed:
                warnings = warnings + (
                    f"{backend.value} throughput is an estimate "
                    f"({estimated:.0f} GFLOP/s) on a target with incomplete "
                    f"library support; measure it and set "
                    f"measured_gemv_gflops",)
        if (self.cu_enabled_override is not None
                and self.cu_enabled_override > spec.cu_enabled):
            warnings = warnings + (
                f"{cu_active} CUs active via an unlock above the {spec.cu_enabled}"
                f"-CU driver baseline; this is community reverse-engineering, "
                f"so verify stability and set measured_gemv_gflops",)
        for reason in total.reasons:
            warnings = warnings + (reason,)
        return EffectiveCapability(
            unit_id=self.unit_id, sku=spec.sku, runtime=self.runtime,
            backend=backend, cu_active=cu_active, cpu_cores_active=cores,
            cpu_ghz=cpu_ghz, gpu_ghz=gpu_ghz, gemv_gflops=gflops,
            coffers=coffers, storage=spec.storage,
            throughput_measured=measured, warnings=warnings)


def _apply_memory_downbin(spec: ConsoleSKU,
                          down: Downbin) -> Tuple[MemoryTier, ...]:
    """Tiers left after OS reservation, per-tier loss, and any sandbox cap.

    Per-tier losses are named explicitly when the operator knows which package
    or channel died. A sandbox ceiling is different in kind: it does not remove
    a tier, it caps the sum, so it is applied fastest-tier-first — a Dev Mode
    process gets its 5 GB out of the 560 GB/s pool before the 336 GB/s one.
    """
    tiers: List[MemoryTier] = []
    for tier in spec.tiers:
        lost = down.tier_losses.get(tier.name, 0)
        scale = down.tier_bandwidth_scale.get(tier.name, 1.0)
        total = max(0, tier.total_bytes - lost)
        tiers.append(MemoryTier(tier.name, total,
                                tier.bandwidth_gbps * scale,
                                min(tier.os_reserved_bytes, total),
                                tier.usable_for_weights))
    tiers.sort(key=lambda t: t.bandwidth_gbps, reverse=True)

    if down.memory_budget_bytes is not None:
        budget = down.memory_budget_bytes
        capped: List[MemoryTier] = []
        for tier in tiers:
            if budget <= 0:
                break
            take = min(tier.usable_bytes, budget)
            budget -= take
            if take > 0:
                capped.append(MemoryTier(tier.name, take, tier.bandwidth_gbps,
                                         0, tier.usable_for_weights))
        tiers = capped
    return tuple(t for t in tiers if t.usable_bytes > 0)


@dataclass(frozen=True)
class FleetSummary:
    """What a fleet adds up to, once every unit's downbin is applied.

    A dataclass rather than a dict because callers ask it arithmetic questions
    and a typo in a string key should not silently become a KeyError three
    layers down in a CLI.
    """

    units: int
    by_sku: Dict[str, int]
    weight_bytes: int
    fast_bytes: int
    gemv_gflops: float
    #: Units carrying at least one recorded per-unit loss.
    downbinned_units: int
    #: Units whose throughput is a nominal figure rather than a measurement.
    #: On a ROCm fleet this number being large is the thing to fix first.
    estimated_throughput_units: int
    warnings: List[str]


def fleet_summary(units: Sequence[ConsoleUnit]) -> FleetSummary:
    """Aggregate a fleet's effective capability."""
    caps = [u.effective() for u in units]
    by_sku: Dict[str, int] = {}
    for cap in caps:
        by_sku[cap.sku] = by_sku.get(cap.sku, 0) + 1
    return FleetSummary(
        units=len(caps),
        by_sku=by_sku,
        weight_bytes=sum(c.weight_bytes for c in caps),
        fast_bytes=sum(c.fast_bytes for c in caps),
        gemv_gflops=sum(c.gemv_gflops for c in caps),
        downbinned_units=sum(1 for u in units if not u.downbin.is_pristine),
        estimated_throughput_units=sum(1 for c in caps
                                       if not c.throughput_measured),
        warnings=sorted({w for c in caps for w in c.warnings}))
