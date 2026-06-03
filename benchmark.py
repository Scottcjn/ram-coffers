#!/usr/bin/env python3
"""
RAM Coffers vs Stock llama.cpp Benchmark
=========================================

Reproducible benchmark comparing RAM Coffers NUMA-aware inference
against stock llama.cpp.

Requirements:
- Python 3.8+
- Multi-NUMA Linux system
- HuggingFace CLI (pip install huggingface_hub)
- llama.cpp built and available in PATH

Usage:
    python3 benchmark.py [--download-model] [--numa-nodes 2] [--pp 128] [--tg 32]

Output:
- Markdown table with results
- JSON file with detailed metrics
"""

import subprocess
import os
import sys
import json
import time
import argparse
import statistics
from pathlib import Path
from datetime import datetime

# Configuration
MODEL_NAME = "TinyLlama-1.1B-Chat-v1.0"
MODEL_FILE = "TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"
MODEL_URL = f"https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/{MODEL_FILE}"
DEFAULT_PROMPT = "Explain the concept of Proof of Antiquity in blockchain technology."
DEFAULT_PP = 128  # prompt processing tokens
DEFAULT_TG = 32   # text generation tokens

def get_numa_info():
    """Detect NUMA topology."""
    try:
        result = subprocess.run(['numactl', '--hardware'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        numa_nodes = []
        for line in lines:
            if 'available:' in line:
                count = int(line.split(':')[1].strip().split()[0])
                numa_nodes = list(range(count))
            elif 'node' in line and 'cpus:' in line:
                parts = line.split()
                node_id = int(parts[0].replace('node', '').replace(':', ''))
                cpus = [int(x) for x in parts[2:]]
                if node_id < len(numa_nodes):
                    numa_nodes[node_id] = {'id': node_id, 'cpus': cpus}
        return numa_nodes
    except FileNotFoundError:
        print("Warning: numactl not found, assuming single NUMA node")
        return [{'id': 0, 'cpus': list(range(os.cpu_count() or 1))}]

def download_model(model_dir="./models"):
    """Download TinyLlama Q4 model."""
    model_path = Path(model_dir) / MODEL_FILE
    if model_path.exists():
        print(f"Model already exists: {model_path}")
        return str(model_path)
    
    print(f"Downloading {MODEL_FILE}...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try huggingface-cli first
    try:
        subprocess.run([
            'huggingface-cli', 'download',
            f'TheBloke/{MODEL_NAME}-GGUF',
            MODEL_FILE, '--local-dir', model_dir
        ], check=True, timeout=600)
        return str(model_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Fallback to wget
    try:
        subprocess.run([
            'wget', '-q', '--show-progress',
            MODEL_URL, '-O', str(model_path)
        ], check=True, timeout=600)
        return str(model_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Fallback to curl
    subprocess.run([
        'curl', '-L', '--progress-bar',
        MODEL_URL, '-o', str(model_path)
    ], check=True, timeout=600)
    return str(model_path)

def run_benchmark_binary(binary, model_path, prompt, pp_tokens, tg_tokens, numa_node=None, threads=None):
    """Run a single benchmark iteration."""
    cmd = [
        binary,
        '-m', model_path,
        '-p', prompt,
        '-n', str(tg_tokens),
        '--pp', str(pp_tokens),
    ]
    
    if threads:
        cmd.extend(['-t', str(threads)])
    
    env = os.environ.copy()
    if numa_node is not None:
        cmd = ['numactl', '--cpunbind', str(numa_node), '--membind', str(numa_node)] + cmd
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    elapsed = time.time() - start
    
    # Parse output for timing info
    output = result.stderr + result.stdout
    
    pp_tok_s = 0
    tg_tok_s = 0
    
    for line in output.split('\n'):
        if 'pp' in line and 't/s' in line:
            try:
                pp_tok_s = float(line.split('t/s')[0].split()[-1])
            except (ValueError, IndexError):
                pass
        if 'tg' in line and 't/s' in line:
            try:
                tg_tok_s = float(line.split('t/s')[0].split()[-1])
            except (ValueError, IndexError):
                pass
    
    # Fallback: calculate from elapsed time
    if pp_tok_s == 0:
        pp_tok_s = pp_tokens / elapsed if elapsed > 0 else 0
    if tg_tok_s == 0:
        tg_tok_s = tg_tokens / elapsed if elapsed > 0 else 0
    
    return {
        'elapsed': round(elapsed, 3),
        'pp_tok_s': round(pp_tok_s, 2),
        'tg_tok_s': round(tg_tok_s, 2),
        'total_tokens': pp_tokens + tg_tokens,
        'total_tok_s': round((pp_tokens + tg_tokens) / elapsed, 2) if elapsed > 0 else 0
    }

def run_benchmark_suite(binary_name, model_path, prompt, pp, tg, numa_nodes, iterations=3):
    """Run full benchmark suite."""
    results = []
    
    for i in range(iterations):
        for node in numa_nodes:
            node_id = node['id'] if isinstance(node, dict) else node
            print(f"  Iteration {i+1}/{iterations}, NUMA node {node_id}...")
            
            result = run_benchmark_binary(
                binary_name, model_path, prompt, pp, tg,
                numa_node=node_id
            )
            result['iteration'] = i + 1
            result['numa_node'] = node_id
            results.append(result)
    
    return results

def generate_markdown_table(stock_results, coffers_results):
    """Generate markdown comparison table."""
    lines = [
        "",
        "## Benchmark Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model:** {MODEL_FILE}",
        f"**Prompt tokens (pp):** {DEFAULT_PP}",
        f"**Generation tokens (tg):** {DEFAULT_TG}",
        "",
        "### Throughput Comparison (tokens/second)",
        "",
        "| Metric | Stock llama.cpp | RAM Coffers | Speedup |",
        "|--------|----------------|-------------|---------|",
    ]
    
    # Calculate averages
    stock_pp = statistics.mean([r['pp_tok_s'] for r in stock_results]) if stock_results else 0
    stock_tg = statistics.mean([r['tg_tok_s'] for r in stock_results]) if stock_results else 0
    stock_total = statistics.mean([r['total_tok_s'] for r in stock_results]) if stock_results else 0
    
    coffers_pp = statistics.mean([r['pp_tok_s'] for r in coffers_results]) if coffers_results else 0
    coffers_tg = statistics.mean([r['tg_tok_s'] for r in coffers_results]) if coffers_results else 0
    coffers_total = statistics.mean([r['total_tok_s'] for r in coffers_results]) if coffers_results else 0
    
    pp_speedup = coffers_pp / stock_pp if stock_pp > 0 else 0
    tg_speedup = coffers_tg / stock_tg if stock_tg > 0 else 0
    total_speedup = coffers_total / stock_total if stock_total > 0 else 0
    
    lines.append(f"| Prompt Processing (pp{DEFAULT_PP}) | {stock_pp:.2f} tok/s | {coffers_pp:.2f} tok/s | {pp_speedup:.2f}x |")
    lines.append(f"| Text Generation (tg{DEFAULT_TG}) | {stock_tg:.2f} tok/s | {coffers_tg:.2f} tok/s | {tg_speedup:.2f}x |")
    lines.append(f"| Combined | {stock_total:.2f} tok/s | {coffers_total:.2f} tok/s | {total_speedup:.2f}x |")
    lines.append("")
    
    return "\n".join(lines), {
        'stock': {'pp_tok_s': round(stock_pp, 2), 'tg_tok_s': round(stock_tg, 2), 'total_tok_s': round(stock_total, 2)},
        'coffers': {'pp_tok_s': round(coffers_pp, 2), 'tg_tok_s': round(coffers_tg, 2), 'total_tok_s': round(coffers_total, 2)},
        'speedup': {'pp': round(pp_speedup, 2), 'tg': round(tg_speedup, 2), 'total': round(total_speedup, 2)}
    }

def main():
    parser = argparse.ArgumentParser(description='RAM Coffers vs llama.cpp Benchmark')
    parser.add_argument('--download-model', action='store_true', help='Download TinyLlama model')
    parser.add_argument('--model-dir', default='./models', help='Model directory')
    parser.add_argument('--numa-nodes', type=int, default=0, help='Number of NUMA nodes to test (0=auto)')
    parser.add_argument('--pp', type=int, default=DEFAULT_PP, help='Prompt processing tokens')
    parser.add_argument('--tg', type=int, default=DEFAULT_TG, help='Text generation tokens')
    parser.add_argument('--iterations', type=int, default=3, help='Iterations per configuration')
    parser.add_argument('--stock-binary', default='./llama.cpp/main', help='Path to stock llama.cpp binary')
    parser.add_argument('--coffers-binary', default='./ram-coffers/main', help='Path to RAM Coffers binary')
    args = parser.parse_args()
    
    print("=" * 60)
    print("RAM Coffers vs Stock llama.cpp Benchmark")
    print("=" * 60)
    
    # System info
    numa_info = get_numa_info()
    print(f"NUMA nodes detected: {len(numa_info)}")
    for node in numa_info:
        cpus = node.get('cpus', []) if isinstance(node, dict) else []
        print(f"  Node {node['id'] if isinstance(node, dict) else node}: {len(cpus)} CPUs")
    
    # Download model if requested
    model_path = None
    if args.download_model:
        model_path = download_model(args.model_dir)
    
    if not model_path:
        model_path = str(Path(args.model_dir) / MODEL_FILE)
    
    if not Path(model_path).exists():
        print(f"\nModel not found at {model_path}")
        print("Run with --download-model to download TinyLlama Q4")
        print(f"Or manually download from: {MODEL_URL}")
        sys.exit(1)
    
    print(f"\nModel: {model_path}")
    print(f"Prompt: {DEFAULT_PROMPT[:50]}...")
    print(f"PP tokens: {args.pp}, TG tokens: {args.tg}")
    print(f"Iterations: {args.iterations}")
    
    # Run benchmarks
    numa_nodes = numa_info[:args.numa_nodes] if args.numa_nodes > 0 else numa_info
    
    stock_results = []
    coffers_results = []
    
    if Path(args.stock_binary).exists():
        print("\n--- Stock llama.cpp ---")
        stock_results = run_benchmark_suite(
            args.stock_binary, model_path, DEFAULT_PROMPT,
            args.pp, args.tg, numa_nodes, args.iterations
        )
    else:
        print(f"\nStock binary not found: {args.stock_binary}")
        print("Skipping stock benchmark")
    
    if Path(args.coffers_binary).exists():
        print("\n--- RAM Coffers ---")
        coffers_results = run_benchmark_suite(
            args.coffers_binary, model_path, DEFAULT_PROMPT,
            args.pp, args.tg, numa_nodes, args.iterations
        )
    else:
        print(f"\nCoffers binary not found: {args.coffers_binary}")
        print("Skipping coffers benchmark")
    
    # Generate results
    if stock_results or coffers_results:
        md_table, summary = generate_markdown_table(stock_results, coffers_results)
        print(md_table)
        
        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': MODEL_FILE,
                'pp_tokens': args.pp,
                'tg_tokens': args.tg,
                'iterations': args.iterations,
                'numa_nodes': len(numa_nodes)
            },
            'system': {'numa_nodes': [str(n) for n in numa_info]},
            'stock_results': stock_results,
            'coffers_results': coffers_results,
            'summary': summary
        }
        
        output_file = 'benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_file}")
    else:
        print("\nNo benchmarks were run. Please ensure binaries are available.")

if __name__ == '__main__':
    main()
