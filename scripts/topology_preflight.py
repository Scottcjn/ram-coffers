#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
scripts/topology_preflight.py - Machine-readable topology & benchmark preflight

Inspects system architecture, NUMA node topology, CPU/ISA feature flags,
and compiler flags expected by ram-coffers. Evaluates benchmark reproducibility
and compatibility verdict.

Usage:
    python3 scripts/topology_preflight.py --json
    python3 scripts/topology_preflight.py
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional


def get_architecture_info() -> Dict[str, Any]:
    """Detect system architecture and OS details."""
    machine = platform.machine().lower()
    system = platform.system().lower()
    is_power = any(p in machine for p in ("ppc", "powerpc"))
    return {
        "machine": machine,
        "system": system,
        "is_powerpc": is_power,
        "python_version": platform.python_version(),
    }


def detect_cpu_info() -> Dict[str, Any]:
    """Detect CPU model, core counts, and ISA / SIMD features."""
    info: Dict[str, Any] = {
        "model": "unknown",
        "cores": os.cpu_count() or 1,
        "threads_per_core": 1,
        "isa_flags": [],
        "has_altivec": False,
        "has_vsx": False,
        "has_crypto": False,
    }

    system = platform.system().lower()

    if system == "linux":
        # Check /proc/cpuinfo
        if os.path.exists("/proc/cpuinfo"):
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                flags = set()
                for line in content.splitlines():
                    if ":" not in line:
                        continue
                    key, val = [p.strip() for p in line.split(":", 1)]
                    k_lower = key.lower()
                    if k_lower in ("cpu", "model name"):
                        if info["model"] == "unknown":
                            info["model"] = val
                    elif k_lower in ("flags", "features"):
                        flags.update(val.lower().split())

                info["isa_flags"] = sorted(list(flags))
                info["has_altivec"] = any("altivec" in f for f in flags)
                info["has_vsx"] = any("vsx" in f for f in flags)
                info["has_crypto"] = any(f in flags for f in ("vcipher", "crypto", "aes", "sha"))
            except Exception:
                pass

        # Try lscpu for threads per core and model
        try:
            res = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if "Thread(s) per core:" in line:
                        info["threads_per_core"] = int(line.split(":")[-1].strip())
                    elif "Model name:" in line and info["model"] == "unknown":
                        info["model"] = line.split(":")[-1].strip()
        except Exception:
            pass

    elif system == "darwin":
        try:
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                info["model"] = res.stdout.strip()
            res_feat = subprocess.run(["sysctl", "-n", "machdep.cpu.features"], capture_output=True, text=True, timeout=5)
            if res_feat.returncode == 0:
                feats = res_feat.stdout.strip().lower().split()
                info["isa_flags"] = sorted(feats)
        except Exception:
            pass

    elif system == "windows":
        info["model"] = os.environ.get("PROCESSOR_IDENTIFIER", "x86/x64 Windows Host")
        arch = os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64").lower()
        if "arm" in arch:
            info["isa_flags"] = ["neon"]
        else:
            info["isa_flags"] = ["sse", "sse2", "avx"]

    return info


def detect_numa_nodes() -> Dict[str, Any]:
    """Detect NUMA topology and per-node memory."""
    nodes: Dict[int, Dict[str, Any]] = {}
    system = platform.system().lower()

    if system == "linux":
        # 1. Try numactl --hardware
        try:
            res = subprocess.run(["numactl", "--hardware"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[0] == "node" and parts[1].isdigit():
                        nid = int(parts[1])
                        field = parts[2].rstrip(":")
                        nodes.setdefault(nid, {"cpus": [], "size_mb": 0, "free_mb": 0})
                        if field == "cpus":
                            cpus = []
                            for p in parts[3:]:
                                if "-" in p:
                                    s, e = p.split("-")
                                    cpus.extend(range(int(s), int(e) + 1))
                                else:
                                    cpus.append(int(p))
                            nodes[nid]["cpus"] = cpus
                        elif field == "size" and len(parts) >= 4:
                            nodes[nid]["size_mb"] = int(parts[3])
                        elif field == "free" and len(parts) >= 4:
                            nodes[nid]["free_mb"] = int(parts[3])
        except Exception:
            pass

        # 2. Fallback to /sys/devices/system/node/
        if not nodes and os.path.exists("/sys/devices/system/node"):
            try:
                for item in sorted(os.listdir("/sys/devices/system/node")):
                    if item.startswith("node") and item[4:].isdigit():
                        nid = int(item[4:])
                        npath = os.path.join("/sys/devices/system/node", item)
                        cpulist = []
                        cpupath = os.path.join(npath, "cpulist")
                        if os.path.exists(cpupath):
                            with open(cpupath, "r") as cf:
                                cstr = cf.read().strip()
                            if cstr:
                                for part in cstr.split(","):
                                    if "-" in part:
                                        s, e = part.split("-")
                                        cpulist.extend(range(int(s), int(e) + 1))
                                    elif part.isdigit():
                                        cpulist.append(int(part))

                        size_mb = 0
                        mempath = os.path.join(npath, "meminfo")
                        if os.path.exists(mempath):
                            with open(mempath, "r") as mf:
                                for mline in mf:
                                    if "MemTotal:" in mline:
                                        mparts = mline.split()
                                        size_mb = int(mparts[3]) // 1024
                                        break
                        nodes[nid] = {"cpus": cpulist, "size_mb": size_mb, "free_mb": 0}
            except Exception:
                pass

    if not nodes:
        # Single-node fallback
        total_mem_mb = 0
        if system == "linux" and os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if "MemTotal:" in line:
                            total_mem_mb = int(line.split()[1]) // 1024
                            break
            except Exception:
                pass
        nodes[0] = {
            "cpus": list(range(os.cpu_count() or 1)),
            "size_mb": total_mem_mb,
            "free_mb": 0,
        }

    return {
        "num_nodes": len(nodes),
        "nodes": [
            {
                "node_id": nid,
                "cpus": data["cpus"],
                "cpu_count": len(data["cpus"]),
                "size_mb": data["size_mb"],
                "free_mb": data["free_mb"],
            }
            for nid, data in sorted(nodes.items())
        ],
    }


def get_expected_compiler_flags() -> Dict[str, str]:
    """Return the expected compiler and preprocessor flags for builds."""
    return {
        "power8_release": "-mcpu=power8 -mvsx -maltivec -O3 -funroll-loops",
        "power8_mass": "-DGGML_USE_MASS=1 -I/opt/ibm/mass/include -L/opt/ibm/mass/lib -lmassvp8 -lmass",
        "power9_compat": "-mcpu=power8 -mvsx -maltivec -O3 (uses power8-compat.h shim)",
        "x86_64_fallback": "-mavx2 -mfma -O3",
        "apple_silicon": "-mcpu=apple-m1 -O3",
    }


def evaluate_compatibility(
    arch_info: Dict[str, Any],
    cpu_info: Dict[str, Any],
    numa_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate host compatibility verdict for reproducing RAM Coffers benchmarks."""
    is_power = arch_info["is_powerpc"]
    has_vsx = cpu_info["has_vsx"]
    has_altivec = cpu_info["has_altivec"]
    has_crypto = cpu_info["has_crypto"]
    num_nodes = numa_info["num_nodes"]

    if is_power and (has_vsx or has_altivec) and num_nodes >= 4:
        mode = "native_power8_4coffer"
        status = "OPTIMAL"
        can_reproduce = True
        summary = (
            "Host meets all prerequisites for reproducing canonical 147.54 t/s benchmarks "
            "(POWER8/POWER9 with AltiVec/VSX and >= 4 NUMA nodes)."
        )
        recommendations = [
            "Compile with: -mcpu=power8 -mvsx -maltivec -O3 -funroll-loops",
            "Ensure 64 threads optimal configuration is selected (-t 64)",
            "Run benchmark_coffers_vs_llamacpp.sh or benchmark_harness.sh",
        ]
    elif is_power and (has_vsx or has_altivec):
        mode = "native_power8_sub_topology"
        status = "COMPATIBLE"
        can_reproduce = False
        summary = (
            f"Host is a POWER system with vector acceleration but only {num_nodes} NUMA node(s) "
            "detected (expected 4 nodes for full coffer sharding)."
        )
        recommendations = [
            "Compile with: -mcpu=power8 -mvsx -maltivec -O3",
            "Multi-coffer sharding will fold weights onto available NUMA node(s)",
            "Benchmark measurements will reflect single/reduced socket throughput",
        ]
    else:
        mode = "foreign_architecture_emulation"
        status = "FALLBACK_ONLY"
        can_reproduce = False
        summary = (
            f"Host architecture '{arch_info['machine']}' is non-POWER8. "
            "RAM Coffers will operate in scalar/AVX2/NEON emulation fallback mode. "
            "Cannot reproduce hardware-specific 147.54 t/s POWER8 AltiVec/VSX claims."
        )
        recommendations = [
            "Use x86-64/ or apple-silicon/ fallback headers",
            "Useful for functional validation, API testing, and logic verification",
            "Do not submit performance benchmark reports from non-POWER8 hosts",
        ]

    return {
        "status": status,
        "benchmark_mode": mode,
        "can_reproduce_canonical_147_ts": can_reproduce,
        "summary": summary,
        "recommendations": recommendations,
    }


def run_preflight() -> Dict[str, Any]:
    """Execute complete topology preflight inspection and return data dict."""
    arch = get_architecture_info()
    cpu = detect_cpu_info()
    numa = detect_numa_nodes()
    compiler_flags = get_expected_compiler_flags()
    verdict = evaluate_compatibility(arch, cpu, numa)

    return {
        "preflight_version": "1.0.0",
        "architecture": arch,
        "cpu_info": cpu,
        "numa_topology": numa,
        "compiler_flags": compiler_flags,
        "verdict": verdict,
    }


def format_text_report(data: Dict[str, Any]) -> str:
    """Format human-readable CLI report."""
    lines = []
    lines.append("=" * 72)
    lines.append("  RAM COFFERS - TOPOLOGY & BENCHMARK PREFLIGHT REPORT")
    lines.append("=" * 72)

    arch = data["architecture"]
    cpu = data["cpu_info"]
    numa = data["numa_topology"]
    verdict = data["verdict"]

    lines.append("\n[1] Architecture & Environment")
    lines.append(f"  Machine Arch:     {arch['machine']}")
    lines.append(f"  Operating System: {arch['system']}")
    lines.append(f"  PowerPC Family:   {'Yes' if arch['is_powerpc'] else 'No'}")
    lines.append(f"  Python Runtime:   {arch['python_version']}")

    lines.append("\n[2] CPU & SIMD Capabilities")
    lines.append(f"  Model:            {cpu['model']}")
    lines.append(f"  Logical Cores:    {cpu['cores']}")
    lines.append(f"  Threads/Core:     {cpu['threads_per_core']}")
    lines.append(f"  AltiVec Support:  {'Yes' if cpu['has_altivec'] else 'No'}")
    lines.append(f"  VSX Support:      {'Yes' if cpu['has_vsx'] else 'No'}")
    lines.append(f"  Crypto Hardware:  {'Yes' if cpu['has_crypto'] else 'No'}")
    flags_snippet = ", ".join(cpu["isa_flags"][:12]) if cpu["isa_flags"] else "None detected"
    if len(cpu["isa_flags"]) > 12:
        flags_snippet += f" ... (+{len(cpu['isa_flags']) - 12} more)"
    lines.append(f"  Detected Flags:   {flags_snippet}")

    lines.append(f"\n[3] NUMA Topology ({numa['num_nodes']} Node(s) Detected)")
    for node in numa["nodes"]:
        cpus = node["cpus"]
        cpu_range = f"{min(cpus)}-{max(cpus)}" if cpus else "none"
        lines.append(
            f"  Node {node['node_id']}: {node['cpu_count']:2d} CPUs ({cpu_range:8s}) | "
            f"RAM: {node['size_mb'] / 1024:6.1f} GB"
        )

    lines.append("\n[4] Expected Compiler Flags")
    for target, flags in data["compiler_flags"].items():
        lines.append(f"  {target:18s}: {flags}")

    lines.append("\n" + "-" * 72)
    lines.append(f"PREFLIGHT VERDICT: [{verdict['status']}] (Mode: {verdict['benchmark_mode']})")
    lines.append(f"Can Reproduce 147.54 t/s: {'YES' if verdict['can_reproduce_canonical_147_ts'] else 'NO'}")
    lines.append(f"Summary: {verdict['summary']}")
    lines.append("Recommendations:")
    for rec in verdict["recommendations"]:
        lines.append(f"  - {rec}")
    lines.append("=" * 72)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Machine-readable topology & benchmark preflight for ram-coffers"
    )
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    report_data = run_preflight()

    if args.json:
        print(json.dumps(report_data, indent=2))
    else:
        print(format_text_report(report_data))


if __name__ == "__main__":
    main()
