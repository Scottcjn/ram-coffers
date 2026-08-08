"""Command line: ``python -m gen9_cluster <command>``.

Five commands, matching the five things an operator does:

``model``   what a checkpoint costs, before any hardware exists
``size``    how many consoles of each kind would be needed
``plan``    place a model on a real inventory, and say what it should run at
``probe``   measure what *this* console can do, and write it back
``serve``   run the node worker on a console
``health``  ask a planned fleet what it thinks it is
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from . import __version__
from .backends import describe_backends, select_runner
from .hardware import GB, fleet_summary, SKUS
from .inventory import (addresses, deployment_config, load_fleet,
                        save_fleet, synthetic_fleet, units)
from .model import PROFILES, describe, profile_for
from .planner import PlanningError, describe_plan, plan_split


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="deepseek-v4-pro",
                        choices=sorted(PROFILES),
                        help="model profile to place")
    parser.add_argument("--context", type=int, default=8192,
                        help="context length the KV cache must hold")
    parser.add_argument("--shelf-size", type=int, default=22,
                        help="consoles per shelf (the expert fan-out group)")
    parser.add_argument("--no-ssd", action="store_true",
                        help="refuse to place experts on NVMe")
    parser.add_argument("--hop-ms", type=float, default=0.25,
                        help="one-way network hop, in milliseconds")


def cmd_model(args: argparse.Namespace) -> int:
    for name in ([args.model] if args.model else sorted(PROFILES)):
        print("\n".join(describe(profile_for(name))))
        print()
    return 0


def cmd_size(args: argparse.Namespace) -> int:
    """Answer "how many consoles?" before anyone has bought any."""
    profile = profile_for(args.model)
    given = vars(args)
    counts = {sku: given.get(sku.replace("-", "_")) or 0 for sku in SKUS}
    counts = {sku: n for sku, n in counts.items() if n}
    if not counts:
        print("give at least one --<sku> count, e.g. --ps5 40 --xbox-series-x 20",
              file=sys.stderr)
        return 2
    fleet = synthetic_fleet(counts)
    print(f"total weights: {profile.total_bytes() / GB:.1f} GiB "
          f"({profile.name}"
          + (", ASSUMED CONFIGURATION" if profile.assumed else "") + ")")
    print("\n".join(_fleet_capacity_lines(fleet)))
    try:
        plan = plan_split(profile, units(fleet), context_tokens=args.context,
                          shelf_size=args.shelf_size,
                          allow_ssd_tier=not args.no_ssd,
                          hop_seconds=args.hop_ms / 1000.0)
    except PlanningError as exc:
        print(f"\nDOES NOT FIT: {exc}")
        return 1
    print()
    print("\n".join(describe_plan(plan, per_console=False)))
    return 0


def _fleet_capacity_lines(fleet) -> List[str]:
    summary = fleet_summary(units(fleet))
    lines = [f"fleet: {summary.units} consoles, "
             f"{summary.weight_bytes / GB:.1f} GiB usable for weights "
             f"({summary.fast_bytes / GB:.1f} GiB of it in fast coffers), "
             f"{summary.gemv_gflops:.0f} GFLOP/s nominal"]
    for sku, count in sorted(summary.by_sku.items()):
        lines.append(f"  {count:>4} x {sku}")
    return lines


def cmd_plan(args: argparse.Namespace) -> int:
    fleet = load_fleet(Path(args.fleet))
    profile = profile_for(args.model)
    try:
        plan = plan_split(profile, units(fleet), context_tokens=args.context,
                          shelf_size=args.shelf_size,
                          allow_ssd_tier=not args.no_ssd,
                          hop_seconds=args.hop_ms / 1000.0)
    except PlanningError as exc:
        print(f"cannot place {profile.name} on this fleet: {exc}",
              file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        print("\n".join(describe_plan(plan)))
    if args.config:
        Path(args.config).write_text(
            json.dumps(deployment_config(plan, fleet), indent=2) + "\n")
        print(f"\ndeployment config written to {args.config}")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Measure this console, rather than trusting the datasheet.

    Any node the planner cannot predict — every ROCm node, anything with an
    unusual downbin — should be probed and its number written into the
    inventory. The plan is only as honest as its slowest assumption.
    """
    print(f"gen9-cluster {__version__} probe")
    print("\n".join(describe_backends()))

    runner = select_runner(args.backend)
    hidden, intermediate = args.hidden, args.intermediate
    rng = np.random.default_rng(0)
    from .node import ExpertWeights
    weights = [ExpertWeights(
        gate=rng.standard_normal((intermediate, hidden), dtype=np.float32),
        up=rng.standard_normal((intermediate, hidden), dtype=np.float32),
        down=rng.standard_normal((hidden, intermediate), dtype=np.float32))]
    x = rng.standard_normal(hidden, dtype=np.float32)

    runner(x, weights, [1.0])                       # warm the caches
    started = time.perf_counter()
    for _ in range(args.iterations):
        runner(x, weights, [1.0])
    elapsed = time.perf_counter() - started

    flops = 2.0 * 3.0 * hidden * intermediate * args.iterations
    gflops = flops / elapsed / 1e9
    bytes_read = 3.0 * hidden * intermediate * 4.0 * args.iterations
    print(f"\nrunner            {runner.name}")
    print(f"expert shape      {hidden} x {intermediate}")
    print(f"measured          {gflops:.1f} GFLOP/s, "
          f"{bytes_read / elapsed / 1e9:.1f} GB/s, "
          f"{elapsed / args.iterations * 1e3:.2f} ms/expert")

    if args.write_fleet and args.unit_id:
        path = Path(args.write_fleet)
        fleet = load_fleet(path)
        for entry in fleet:
            if entry.unit.unit_id == args.unit_id:
                entry.unit.measured_gemv_gflops = round(gflops, 1)
                break
        else:
            print(f"unit {args.unit_id!r} is not in {path}", file=sys.stderr)
            return 1
        save_fleet(path, fleet)
        print(f"wrote measured_gemv_gflops={gflops:.1f} for {args.unit_id} "
              f"into {path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the node worker, loading whatever the deployment config assigns."""
    from .node import NodeServer, ShardStore

    config = json.loads(Path(args.config).read_text())
    node_config = config["nodes"].get(args.unit_id)
    if node_config is None:
        print(f"{args.unit_id!r} has no section in {args.config}",
              file=sys.stderr)
        return 1
    store = ShardStore(capacity_bytes=node_config.get("capacity_bytes", 0))
    runner = select_runner(node_config.get("backend"))
    server = NodeServer(store, unit_id=args.unit_id, host=args.host,
                        port=args.port or node_config.get("port", 9713),
                        runner=runner, sku=node_config.get("sku", "unknown"),
                        backend=node_config.get("backend", "cpu-avx2"))
    shards = node_config.get("shards", [])
    experts = sum(s["n_experts"] for s in shards)
    print(f"{args.unit_id}: serving on {args.host}:{server.port}, "
          f"{len(shards)} shards / {experts} experts assigned, "
          f"{len(node_config.get('hot_layers', []))} hot layers, "
          f"runner={runner.name}")
    print("weights are loaded over G9XC by the coordinator (LOAD_SHARD)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    from .coordinator import FleetCoordinator

    fleet = load_fleet(Path(args.fleet))
    profile = profile_for(args.model)
    plan = plan_split(profile, units(fleet), context_tokens=args.context,
                      shelf_size=args.shelf_size)

    def router(layer, state):
        k = profile.moe.top_k
        return list(range(k)), [1.0 / k] * k

    with FleetCoordinator(plan, addresses(fleet), router=router) as fc:
        for unit_id, status in fc.health().items():
            print(f"{unit_id:<16} {status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen9_cluster",
        description="DeepSeek inference across PS5 / Xbox Series / salvage "
                    "console hardware")
    parser.add_argument("--version", action="version",
                        version=f"gen9-cluster {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_model = sub.add_parser("model", help="size a model profile")
    p_model.add_argument("--model", default=None, choices=sorted(PROFILES))
    p_model.set_defaults(func=cmd_model)

    p_size = sub.add_parser("size", help="how many consoles would be needed")
    _add_plan_arguments(p_size)
    for sku in sorted(SKUS):
        p_size.add_argument(f"--{sku}", type=int, default=0,
                            dest=sku.replace("-", "_"),
                            help=f"number of {sku} consoles")
    p_size.set_defaults(func=cmd_size)

    p_plan = sub.add_parser("plan", help="place a model on an inventory")
    p_plan.add_argument("fleet", help="inventory JSON")
    _add_plan_arguments(p_plan)
    p_plan.add_argument("--json", action="store_true", help="emit the raw plan")
    p_plan.add_argument("--config", help="write a deployment config here")
    p_plan.set_defaults(func=cmd_plan)

    p_probe = sub.add_parser("probe", help="measure this console")
    p_probe.add_argument("--backend", default=None)
    p_probe.add_argument("--hidden", type=int, default=4096)
    p_probe.add_argument("--intermediate", type=int, default=1024)
    p_probe.add_argument("--iterations", type=int, default=5)
    p_probe.add_argument("--write-fleet", help="inventory to update in place")
    p_probe.add_argument("--unit-id", help="which entry to update")
    p_probe.set_defaults(func=cmd_probe)

    p_serve = sub.add_parser("serve", help="run the node worker")
    p_serve.add_argument("config", help="deployment config from `plan --config`")
    p_serve.add_argument("unit_id")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=0)
    p_serve.set_defaults(func=cmd_serve)

    p_health = sub.add_parser("health", help="ask a fleet what it thinks it is")
    p_health.add_argument("fleet")
    _add_plan_arguments(p_health)
    p_health.set_defaults(func=cmd_health)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
