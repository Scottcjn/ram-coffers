"""Fleet inventory: describing consoles that are not what their labels say.

An inventory file is the operator's description of the hardware they actually
have. It is JSON because it gets edited by hand at three in the morning next to
a shelf of consoles, and every field beyond ``unit_id`` and ``sku`` is optional
with a sane default.

.. code-block:: json

    {
      "fleet": [
        {"unit_id": "ps5-01", "sku": "ps5", "runtime": "ps5-linux",
         "host": "10.0.0.11"},

        {"unit_id": "ps5-02", "sku": "ps5", "runtime": "ps5-linux",
         "host": "10.0.0.12",
         "downbin": {"cu_disabled": 4, "gpu_ghz_cap": 1.8,
                     "reasons": ["dead shader array", "fan replaced, clocked down"]}},

        {"unit_id": "xsx-01", "sku": "xbox-series-x", "runtime": "xbox-devmode",
         "host": "10.0.0.21", "devmode_app": false},

        {"unit_id": "bc-01", "sku": "bc-250", "runtime": "salvage-linux",
         "host": "10.0.0.31", "cu_enabled_override": 40,
         "backend": "rocm", "measured_gemv_gflops": 940.0},

        {"unit_id": "kit-01", "sku": "amd-4700s", "runtime": "salvage-linux",
         "host": "10.0.0.41",
         "downbin": {"tier_losses": {"gddr6": 2147483648},
                     "reasons": ["one GDDR6 package failed memtest"]}}
      ]
    }

The ``downbin`` block is the important one and the reason this file exists: it
is where a unit's *individual* damage goes. Nothing about a second-hand console
can be inferred from its model name, so anything that makes this unit different
from a catalogue one — fused-off CUs, a dead memory package, a clock cap, a
sandbox budget — is recorded per unit, with a human-readable reason that shows
up in the plan.

``measured_gemv_gflops`` is the other one. Any node whose throughput the planner
cannot predict — every ROCm node, and anything unusual — should carry a measured
figure from ``g9-probe``. Without it the planner falls back to an estimate and
says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .dispatch import NodeAddress
from .hardware import ComputeBackend, ConsoleUnit, Downbin, Runtime

DEFAULT_PORT = 9713


@dataclass
class FleetEntry:
    """One console in an inventory file: what it is, and where it is."""

    unit: ConsoleUnit
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT

    @property
    def address(self) -> NodeAddress:
        return NodeAddress(self.unit.unit_id, self.host, self.port)


def _downbin_from_dict(raw: Dict) -> Downbin:
    return Downbin(
        cu_disabled=int(raw.get("cu_disabled", 0)),
        cpu_cores_disabled=int(raw.get("cpu_cores_disabled", 0)),
        cpu_ghz_cap=raw.get("cpu_ghz_cap"),
        gpu_ghz_cap=raw.get("gpu_ghz_cap"),
        tier_losses={str(k): int(v)
                     for k, v in (raw.get("tier_losses") or {}).items()},
        tier_bandwidth_scale={str(k): float(v) for k, v in
                              (raw.get("tier_bandwidth_scale") or {}).items()},
        memory_budget_bytes=raw.get("memory_budget_bytes"),
        reasons=tuple(raw.get("reasons") or ()),
    )


def _downbin_to_dict(downbin: Downbin) -> Dict:
    out: Dict[str, object] = {}
    if downbin.cu_disabled:
        out["cu_disabled"] = downbin.cu_disabled
    if downbin.cpu_cores_disabled:
        out["cpu_cores_disabled"] = downbin.cpu_cores_disabled
    if downbin.cpu_ghz_cap is not None:
        out["cpu_ghz_cap"] = downbin.cpu_ghz_cap
    if downbin.gpu_ghz_cap is not None:
        out["gpu_ghz_cap"] = downbin.gpu_ghz_cap
    if downbin.tier_losses:
        out["tier_losses"] = dict(downbin.tier_losses)
    if downbin.tier_bandwidth_scale:
        out["tier_bandwidth_scale"] = dict(downbin.tier_bandwidth_scale)
    if downbin.memory_budget_bytes is not None:
        out["memory_budget_bytes"] = downbin.memory_budget_bytes
    if downbin.reasons:
        out["reasons"] = list(downbin.reasons)
    return out


def entry_from_dict(raw: Dict) -> FleetEntry:
    try:
        unit_id = raw["unit_id"]
        sku = raw["sku"]
    except KeyError as exc:
        raise ValueError(f"fleet entry is missing {exc}") from exc
    backend = raw.get("backend")
    unit = ConsoleUnit(
        unit_id=str(unit_id),
        sku=str(sku),
        runtime=Runtime(raw.get("runtime", "host-sim")),
        backend=ComputeBackend(backend) if backend else None,
        downbin=_downbin_from_dict(raw.get("downbin") or {}),
        measured_gemv_gflops=raw.get("measured_gemv_gflops"),
        devmode_app=bool(raw.get("devmode_app", False)),
        cu_enabled_override=raw.get("cu_enabled_override"),
        labels={str(k): str(v) for k, v in (raw.get("labels") or {}).items()},
    )
    return FleetEntry(unit=unit, host=str(raw.get("host", "127.0.0.1")),
                      port=int(raw.get("port", DEFAULT_PORT)))


def entry_to_dict(entry: FleetEntry) -> Dict:
    unit = entry.unit
    out: Dict[str, object] = {"unit_id": unit.unit_id, "sku": unit.sku,
                              "runtime": unit.runtime.value,
                              "host": entry.host, "port": entry.port}
    if unit.backend is not None:
        out["backend"] = unit.backend.value
    if unit.measured_gemv_gflops is not None:
        out["measured_gemv_gflops"] = unit.measured_gemv_gflops
    if unit.devmode_app:
        out["devmode_app"] = True
    if unit.cu_enabled_override is not None:
        out["cu_enabled_override"] = unit.cu_enabled_override
    downbin = _downbin_to_dict(unit.downbin)
    if downbin:
        out["downbin"] = downbin
    if unit.labels:
        out["labels"] = dict(unit.labels)
    return out


def load_fleet(path: Path) -> List[FleetEntry]:
    """Read an inventory file."""
    raw = json.loads(Path(path).read_text())
    entries = raw["fleet"] if isinstance(raw, dict) else raw
    fleet = [entry_from_dict(item) for item in entries]
    seen = set()
    for entry in fleet:
        if entry.unit.unit_id in seen:
            raise ValueError(f"duplicate unit_id {entry.unit.unit_id!r} in "
                             f"{path}")
        seen.add(entry.unit.unit_id)
    return fleet


def save_fleet(path: Path, fleet: Sequence[FleetEntry]) -> None:
    payload = {"fleet": [entry_to_dict(e) for e in fleet]}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def units(fleet: Sequence[FleetEntry]) -> List[ConsoleUnit]:
    return [entry.unit for entry in fleet]


def addresses(fleet: Sequence[FleetEntry]) -> Dict[str, NodeAddress]:
    return {entry.unit.unit_id: entry.address for entry in fleet}


def synthetic_fleet(counts: Dict[str, int], *,
                    runtime: Optional[Dict[str, str]] = None,
                    base_port: int = DEFAULT_PORT) -> List[FleetEntry]:
    """Build a fleet of nominal consoles, for sizing exercises and tests.

    Useful for the question operators actually ask first — "how many consoles
    would I need?" — before anyone has bought anything.
    """
    defaults = {"ps5": "ps5-linux", "ps5-slim": "ps5-linux",
                "ps5-pro": "ps5-linux", "xbox-series-x": "xbox-gdk",
                "xbox-series-s": "xbox-gdk", "amd-4700s": "salvage-linux",
                "amd-4800s": "salvage-linux", "bc-250": "salvage-linux",
                "host-sim": "host-sim"}
    defaults.update(runtime or {})
    fleet: List[FleetEntry] = []
    port = base_port
    for sku, count in counts.items():
        for index in range(count):
            unit = ConsoleUnit(unit_id=f"{sku}-{index:03d}", sku=sku,
                               runtime=Runtime(defaults.get(sku, "host-sim")))
            fleet.append(FleetEntry(unit=unit, host="127.0.0.1", port=port))
            port += 1
    return fleet


def deployment_config(plan, fleet: Sequence[FleetEntry]) -> Dict:
    """Turn a plan plus an inventory into what each console must be told.

    One section per console: the shards to load, whether it hosts hot blocks,
    and which coffer each shard belongs in. This is the artefact that gets
    copied to the consoles, and it is deliberately a plain dict so it can be
    diffed between plans — the usual question after a console dies is "what
    changed", and that should be answerable with ``diff``.
    """
    by_id = {entry.unit.unit_id: entry for entry in fleet}
    config: Dict[str, object] = {
        "model": plan.model,
        "model_assumed": plan.model_assumed,
        "context_tokens": plan.context_tokens,
        "estimated_tokens_per_second": round(plan.tokens_per_second, 3),
        "shelves": [stage.to_dict() for stage in plan.stages],
        "nodes": {},
    }
    nodes: Dict[str, object] = {}
    for unit_id, unit_plan in sorted(plan.units.items()):
        entry = by_id.get(unit_id)
        nodes[unit_id] = {
            "host": entry.host if entry else None,
            "port": entry.port if entry else None,
            "sku": unit_plan.sku,
            "backend": unit_plan.backend,
            "hot_layers": unit_plan.hot_layers,
            "dense_layers": unit_plan.dense_layers,
            "io_pieces": unit_plan.io_pieces,
            "resident_bytes": unit_plan.resident_bytes,
            "capacity_bytes": unit_plan.capacity_bytes,
            "ssd_expert_bytes": unit_plan.ssd_expert_bytes,
            "shards": [shard.to_dict() for shard in unit_plan.shards],
            "warnings": unit_plan.warnings,
        }
    config["nodes"] = nodes
    return config
