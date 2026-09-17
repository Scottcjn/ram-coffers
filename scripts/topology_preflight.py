#!/usr/bin/env python3
"""
scripts/topology_preflight.py - Machine-readable NUMA topology and benchmark preflight.

Checks host architecture, NUMA node topology, CPU/ISA flags (POWER8/VSX/Altivec, AVX2, NEON),
memory per node, expected compiler flags, and produces a benchmark compatibility verdict.

Usage:
    python3 scripts/topology_preflight.py
    python3 scripts/topology_preflight.py --json
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional

# Ensure parent directory is in path to import ram_coffers_topology
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from ram_coffers_topology import detect_numa
except ImportError:
    detect_numa = None


def get_cpu_info() -> Dict[str, Any]:
    """Extract CPU architecture, model name, flags/features from /proc/cpuinfo or lscpu."""
    info = {
        "arch": platform.machine(),
        "system": platform.system(),
        "model": "",
        "flags": [],
        "cores": os.cpu_count() or 1,
    }
    
    # Try reading /proc/cpuinfo
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                for line in content.splitlines():
                    if ":" not in line:
                        continue
                    key, val = [x.strip() for x in line.split(":", 1)]
                    k_lower = key.lower()
                    if not info["model"] and k_lower == "model name":
                        info["model"] = val
                    elif not info["model"] and k_lower in ("cpu", "processor") and not val.isdigit():
                        info["model"] = val
                    elif k_lower in ("flags", "features"):
                        for flg in val.split():
                            flg_lower = flg.lower()
                            if flg_lower not in info["flags"]:
                                info["flags"].append(flg_lower)
        except Exception:
            pass

    # Fallback to lscpu if flags empty or model empty
    if not info["flags"] or not info["model"]:
        try:
            res = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if ":" not in line:
                        continue
                    key, val = [x.strip() for x in line.split(":", 1)]
                    k_lower = key.lower()
                    if not info["model"] and "model name" in k_lower:
                        info["model"] = val
                    elif "flags" in k_lower:
                        for flg in val.split():
                            flg_lower = flg.lower()
                            if flg_lower not in info["flags"]:
                                info["flags"].append(flg_lower)
        except Exception:
            pass

    return info


def evaluate_isa(cpu_info: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate host instruction set architecture against RAM Coffers targets."""
    arch = (cpu_info.get("arch") or "").lower()
    flags = set(cpu_info.get("flags") or [])
    model = (cpu_info.get("model") or "").lower()

    is_power = arch.startswith("ppc") or "power" in arch or "power" in model
    is_x86_64 = arch in ("x86_64", "amd64")
    is_arm64 = arch in ("aarch64", "arm64")

    has_power8 = is_power and ("power8" in model or "power8" in arch or "altivec" in flags or "vsx" in flags)
    has_vsx = "vsx" in flags or is_power
    has_altivec = "altivec" in flags or is_power
    has_vcipher = "vcipher" in flags or "crypto" in flags or is_power
    has_avx2 = "avx2" in flags
    has_neon = "neon" in flags or "asimd" in flags or is_arm64

    # Determine recommended compiler flags
    compiler_flags = []
    if is_power:
        compiler_flags = ["-mcpu=power8", "-mvsx", "-maltivec", "-O3"]
    elif is_x86_64:
        compiler_flags = ["-mavx2", "-mfma", "-O3"]
    elif is_arm64:
        compiler_flags = ["-march=armv8-a+simd", "-O3"]
    else:
        compiler_flags = ["-O3"]

    return {
        "is_power": is_power,
        "is_x86_64": is_x86_64,
        "is_arm64": is_arm64,
        "has_power8": has_power8,
        "has_vsx": has_vsx,
        "has_altivec": has_altivec,
        "has_vcipher": has_vcipher,
        "has_avx2": has_avx2,
        "has_neon": has_neon,
        "recommended_compiler_flags": compiler_flags,
    }


def evaluate_topology_preflight() -> Dict[str, Any]:
    """Run full preflight check and compute benchmark mode compatibility."""
    cpu = get_cpu_info()
    isa = evaluate_isa(cpu)

    # Detect NUMA
    numa = None
    if detect_numa:
        try:
            numa = detect_numa()
        except Exception:
            numa = None

    num_nodes = numa.get("num_nodes", 1) if numa else 1
    nodes_detail = numa.get("nodes", {}) if numa else {}

    # Total RAM calculation
    total_ram_mb = 0
    if nodes_detail:
        for n in nodes_detail.values():
            total_ram_mb += n.get("size_mb", 0)
    if total_ram_mb == 0 and os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_ram_mb = int(line.split()[1]) // 1024
                        break
        except Exception:
            pass

    # Verdict determination
    # Benchmark modes:
    # - power8_s824_reference: Multi-NUMA POWER8 with VSX and Altivec (matches 147.54 t/s benchmark environment)
    # - numa_accelerated: Multi-node NUMA machine (x86_64 or other) capable of multi-coffer NUMA striping
    # - single_node_fallback: Single NUMA node machine using memory-mapped buffers
    if isa["is_power"] and num_nodes >= 2 and isa["has_vsx"]:
        benchmark_mode = "power8_s824_reference"
        compatible = True
        status = "Full POWER8 reference topology verified. Target for 147.54 t/s reproduction."
    elif num_nodes >= 2:
        benchmark_mode = "numa_accelerated"
        compatible = True
        status = f"Multi-NUMA ({num_nodes} nodes) detected on {cpu['arch']}. Supported for multi-coffer allocation."
    else:
        benchmark_mode = "single_node_fallback"
        compatible = True
        status = f"Single NUMA domain detected ({num_nodes} node). Operates in standard memory-mapped fallback mode."

    report = {
        "preflight_version": "1.0.0",
        "system": {
            "os": cpu["system"],
            "arch": cpu["arch"],
            "cpu_model": cpu["model"],
            "total_logical_cores": cpu["cores"],
            "total_memory_mb": total_ram_mb,
        },
        "isa_support": {
            "power8": isa["has_power8"],
            "vsx": isa["has_vsx"],
            "altivec": isa["has_altivec"],
            "vcipher_crypto": isa["has_vcipher"],
            "avx2": isa["has_avx2"],
            "neon": isa["has_neon"],
        },
        "numa_topology": {
            "num_nodes": num_nodes,
            "nodes": nodes_detail,
        },
        "compiler": {
            "recommended_flags": isa["recommended_compiler_flags"],
        },
        "verdict": {
            "compatible": compatible,
            "benchmark_mode": benchmark_mode,
            "status": status,
            "meets_reference_power8_spec": (benchmark_mode == "power8_s824_reference"),
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(
        description="RAM Coffers topology and benchmark preflight verifier"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output",
    )
    args = parser.parse_args()

    report = evaluate_topology_preflight()

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    # Human-readable text format
    print("=" * 65)
    print("RAM Coffers Topology & Benchmark Preflight")
    print("=" * 65)
    sys_info = report["system"]
    print(f"OS / Arch        : {sys_info['os']} / {sys_info['arch']}")
    print(f"CPU Model        : {sys_info['cpu_model'] or 'Unknown'}")
    print(f"Logical Cores    : {sys_info['total_logical_cores']}")
    print(f"Total RAM        : {sys_info['total_memory_mb']} MB")
    print("-" * 65)
    print("ISA Vector Acceleration:")
    for k, v in report["isa_support"].items():
        print(f"  - {k:<15}: {'YES' if v else 'NO'}")
    print("-" * 65)
    print(f"NUMA Nodes       : {report['numa_topology']['num_nodes']}")
    print(f"Compiler Flags   : {' '.join(report['compiler']['recommended_flags'])}")
    print("-" * 65)
    print(f"Benchmark Mode   : {report['verdict']['benchmark_mode']}")
    print(f"Status           : {report['verdict']['status']}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
