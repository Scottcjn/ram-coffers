#!/usr/bin/env python3
"""
ram-coffers topology - NUMA topology visualization for RAM Coffers

Displays NUMA node layout, CPU assignment, memory distribution,
and weight allocation for LLM inference optimization.

Usage:
    python3 ram_coffers_topology.py                 # Text output
    python3 ram_coffers_topology.py --json          # JSON output
    python3 ram_coffers_topology.py --plan --model ./model.gguf
"""

import os
import sys
import json
import subprocess
from typing import Dict, List, Optional, Tuple


def detect_numa_linux() -> Optional[Dict]:
    """Detect NUMA topology on Linux systems."""
    try:
        # Try numactl first
        result = subprocess.run(
            ["numactl", "--hardware"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return _parse_numactl_output(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Fallback: parse /sys/devices/system/node/
    return _parse_sysfs_numa()


def _parse_numactl_output(output: str) -> Dict:
    """Parse numactl --hardware output."""
    nodes = {}

    for line in output.split('\n'):
        line = line.strip()
        parts = line.split()
        if len(parts) < 3 or parts[0] != 'node' or not parts[1].isdigit():
            continue

        node_id = int(parts[1])
        field = parts[2].rstrip(':')
        nodes.setdefault(
            node_id,
            {
                'cpus': [],
                'size_mb': 0,
                'free_mb': 0,
            },
        )

        if field == 'cpus':
            # Parse CPU list: "0 1 2 3 4 5 6 7" or "0-7"
            cpus = []
            for part in parts[3:]:
                if '-' in part:
                    start, end = part.split('-')
                    cpus.extend(range(int(start), int(end) + 1))
                else:
                    cpus.append(int(part))
            nodes[node_id]['cpus'] = cpus
        elif field == 'size' and len(parts) >= 4:
            nodes[node_id]['size_mb'] = int(parts[3])
        elif field == 'free' and len(parts) >= 4:
            nodes[node_id]['free_mb'] = int(parts[3])
    
    return {
        'system': 'linux',
        'num_nodes': len(nodes),
        'nodes': nodes,
    }


def _parse_sysfs_numa() -> Optional[Dict]:
    """Parse NUMA info from /sys filesystem."""
    nodes = {}
    node_dir = '/sys/devices/system/node'
    
    if not os.path.exists(node_dir):
        return None
    
    for entry in os.listdir(node_dir):
        if not entry.startswith('node'):
            continue
        node_id = int(entry[4:])
        node_path = os.path.join(node_dir, entry)
        
        node_info = {
            'cpus': [],
            'size_mb': 0,
            'free_mb': 0,
        }
        
        # Read CPU list
        cpulist_path = os.path.join(node_path, 'cpulist')
        if os.path.exists(cpulist_path):
            with open(cpulist_path, 'r') as f:
                cpu_str = f.read().strip()
                cpus = []
                for part in cpu_str.split(','):
                    if '-' in part:
                        start, end = part.split('-')
                        cpus.extend(range(int(start), int(end) + 1))
                    else:
                        cpus.append(int(part))
                node_info['cpus'] = cpus
        
        # Read memory info
        meminfo_path = os.path.join(node_path, 'meminfo')
        if os.path.exists(meminfo_path):
            with open(meminfo_path, 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        parts = line.split()
                        node_info['size_mb'] = int(parts[1]) // 1024
                    elif 'MemFree' in line:
                        parts = line.split()
                        node_info['free_mb'] = int(parts[1]) // 1024
        
        nodes[node_id] = node_info
    
    if not nodes:
        return None
    
    return {
        'system': 'linux',
        'num_nodes': len(nodes),
        'nodes': nodes,
    }


def detect_numa_macos() -> Dict:
    """Detect NUMA topology on macOS (unified memory)."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            size_mb = int(result.stdout.strip()) // (1024 * 1024)
            return {
                'system': 'macos',
                'num_nodes': 1,
                'nodes': {
                    0: {
                        'cpus': list(range(os.cpu_count() or 8)),
                        'size_mb': size_mb,
                        'free_mb': size_mb,  # Unified memory
                    }
                },
            }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return {
        'system': 'macos',
        'num_nodes': 1,
        'nodes': {
            0: {
                'cpus': [],
                'size_mb': 0,
                'free_mb': 0,
            }
        },
    }


def detect_numa() -> Dict:
    """Detect NUMA topology for current system."""
    if sys.platform == 'darwin':
        return detect_numa_macos()
    elif sys.platform == 'linux':
        result = detect_numa_linux()
        if result:
            return result
    
    # Fallback: single node
    return {
        'system': sys.platform,
        'num_nodes': 1,
        'nodes': {
            0: {
                'cpus': list(range(os.cpu_count() or 1)),
                'size_mb': 0,
                'free_mb': 0,
            }
        },
    }


def calculate_weights(topology: Dict) -> Dict[int, float]:
    """Calculate recommended weight distribution based on memory."""
    nodes = topology['nodes']
    total_memory = sum(n['size_mb'] for n in nodes.values())
    
    if total_memory == 0:
        # Equal distribution
        weight = 100.0 / len(nodes)
        return {node_id: weight for node_id in nodes}
    
    weights = {}
    for node_id, info in nodes.items():
        weights[node_id] = (info['size_mb'] / total_memory) * 100.0
    
    return weights


def build_placement_plan(
    topology: Dict,
    weights: Dict[int, float],
    model_path: Optional[str] = None,
) -> Dict:
    """Build a dry-run NUMA placement plan for model weights."""
    model_size_mb = None
    fallback_reason = None

    if model_path:
        if os.path.exists(model_path):
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        else:
            fallback_reason = f"model path not found: {model_path}"
    elif topology.get('num_nodes', 1) <= 1:
        fallback_reason = "single NUMA node detected"

    planned_nodes = {}
    for node_id, info in topology['nodes'].items():
        weight_percent = weights.get(node_id, 0.0)
        shard_mb = None
        if model_size_mb is not None:
            shard_mb = model_size_mb * (weight_percent / 100.0)

        planned_nodes[str(node_id)] = {
            'cpus': info.get('cpus', []),
            'available_memory_mb': info.get('free_mb', 0),
            'total_memory_mb': info.get('size_mb', 0),
            'weight_percent': weight_percent,
            'planned_shard_mb': shard_mb,
        }

    return {
        'mode': 'dry-run',
        'model_path': model_path,
        'model_size_mb': model_size_mb,
        'num_nodes': topology.get('num_nodes', len(topology['nodes'])),
        'fallback_reason': fallback_reason,
        'nodes': planned_nodes,
    }


def display_topology_text(topology: Dict, weights: Dict[int, float]) -> None:
    """Display NUMA topology as text tree."""
    print(f"\n{'='*60}")
    print(f"  NUMA Topology (ram-coffers)")
    print(f"{'='*60}\n")
    
    nodes = topology['nodes']
    num_nodes = topology['num_nodes']
    
    for i, (node_id, info) in enumerate(sorted(nodes.items())):
        is_last = (i == num_nodes - 1)
        prefix = "└──" if is_last else "├──"
        
        cpus = info.get('cpus', [])
        cpu_str = f"{cpus[0]}-{cpus[-1]}" if cpus else "N/A"
        memory_gb = info.get('size_mb', 0) / 1024
        weight = weights.get(node_id, 0)
        
        print(f"{prefix} Node {node_id}: CPUs {cpu_str} | Memory: {memory_gb:.1f}GB | Weight: {weight:.0f}%")
    
    # Status
    if num_nodes > 1:
        weight_values = list(weights.values())
        max_weight = max(weight_values)
        min_weight = min(weight_values)
        ratio = max_weight / min_weight if min_weight > 0 else 1.0
        
        if ratio > 1.3:
            print(f"└── Status: ⚠️ Imbalanced — Node with highest weight has {ratio:.1f}x weight of lowest")
        else:
            print(f"└── Status: ✓ Balanced")
    
    print()


def display_plan_text(plan: Dict) -> None:
    """Display dry-run NUMA placement plan as text."""
    print(f"\n{'='*60}")
    print("  Dry-run NUMA Placement Plan")
    print(f"{'='*60}\n")

    if plan.get('model_path'):
        print(f"Model: {plan['model_path']}")
    if plan.get('model_size_mb') is not None:
        print(f"Model size: {plan['model_size_mb']:.1f} MB")
    if plan.get('fallback_reason'):
        print(f"Fallback: {plan['fallback_reason']}")

    for node_id, info in sorted(plan['nodes'].items(), key=lambda item: int(item[0])):
        shard = info.get('planned_shard_mb')
        shard_text = f"{shard:.1f} MB" if shard is not None else "unknown"
        print(
            f"Node {node_id}: weight {info['weight_percent']:.1f}% | "
            f"planned shard {shard_text} | "
            f"free {info['available_memory_mb']} MB"
        )

    print()


def display_topology_json(topology: Dict, weights: Dict[int, float]) -> None:
    """Display NUMA topology as JSON."""
    output = {
        'system': topology['system'],
        'num_nodes': topology['num_nodes'],
        'nodes': {},
    }
    
    for node_id, info in topology['nodes'].items():
        output['nodes'][str(node_id)] = {
            'cpus': info.get('cpus', []),
            'cpu_count': len(info.get('cpus', [])),
            'memory_mb': info.get('size_mb', 0),
            'memory_gb': info.get('size_mb', 0) / 1024,
            'free_mb': info.get('free_mb', 0),
            'weight_percent': weights.get(node_id, 0),
        }
    
    print(json.dumps(output, indent=2))


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Display NUMA topology for ram-coffers weight optimization'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output in JSON format'
    )
    parser.add_argument(
        '--plan',
        action='store_true',
        help='Print a dry-run NUMA weight placement plan'
    )
    parser.add_argument(
        '--model',
        help='Optional model path used to estimate planned shard sizes'
    )
    
    args = parser.parse_args()
    
    # Detect topology
    topology = detect_numa()
    
    # Calculate weights
    weights = calculate_weights(topology)
    plan = build_placement_plan(topology, weights, args.model) if args.plan else None

    # Display
    if args.json:
        if plan:
            print(json.dumps(plan, indent=2))
        else:
            display_topology_json(topology, weights)
    else:
        if plan:
            display_plan_text(plan)
        else:
            display_topology_text(topology, weights)


if __name__ == '__main__':
    main()
