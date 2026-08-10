# RAM Coffers: NUMA-Distributed Conditional Memory for LLM Inference

RAM Coffers is a NUMA-aware conditional memory architecture for LLM inference that routes model knowledge through physical memory banks so verified POWER8 and Apple Silicon-class machines can do useful AI work inside the RustChain Proof of Physical AI ecosystem.

> **Part of the [Proof of Physical AI](https://github.com/Scottcjn/Rustchain) stack** — where real hardware earns real tokens.

[![Proof of Physical AI](https://img.shields.io/badge/Proof_of_Physical_AI-POWER8-blue?style=flat-square)](https://github.com/Scottcjn/Rustchain)
[![BCOS Certified](https://img.shields.io/badge/BCOS-Certified-brightgreen?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMiAxTDMgNXY2YzAgNS41NSAzLjg0IDEwLjc0IDkgMTIgNS4xNi0xLjI2IDktNi40NSA5LTEyVjVsLTktNHptLTIgMTZsLTQtNCA1LjQxLTUuNDEgMS40MSAxLjQxTDEwIDE0bDYtNiAxLjQxIDEuNDFMMTAgMTd6Ii8+PC9zdmc+)](BCOS.md) [![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](LICENSE)

**147 tokens/sec on POWER8 — 8.8x stock llama.cpp.** The same IBM POWER8 hardware that runs RAM Coffers inference also mines RTC via [Proof of Antiquity](https://github.com/Scottcjn/Rustchain), making this a DePIN node that does useful AI work while earning rewards for its physical existence.

See **[BENCHMARK.md](BENCHMARK.md)** for exactly what was measured, what's still template/unreproduced, and the commands to run this yourself.

**Author:** Scott Boudreaux
**Date:** December 16, 2025
**Institution:** Elyan Labs (Independent Research)
**Hardware:** IBM POWER8 S824 (320GB RAM, Dual 8-core)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18321905.svg)](https://doi.org/10.5281/zenodo.18321905)

## Publications

| Paper | DOI | Date |
|-------|-----|------|
| **RAM Coffers: NUMA-Distributed Weight Banking** | [10.5281/zenodo.18321905](https://doi.org/10.5281/zenodo.18321905) | Jan 2026 |
| **Non-Bijunctive Permutation Collapse** (vec_perm for LLM attention) | [10.5281/zenodo.18623920](https://doi.org/10.5281/zenodo.18623920) | Feb 2026 |
| **PSE Hardware Entropy for Behavioral Divergence** (mftb injection) | [10.5281/zenodo.18623922](https://doi.org/10.5281/zenodo.18623922) | Feb 2026 |
| **Neuromorphic Prompt Translation** (GRAIL-V, emotional prompting) | [10.5281/zenodo.18623594](https://doi.org/10.5281/zenodo.18623594) | Feb 2026 |
| **RustChain: One CPU, One Vote** (Proof of Antiquity consensus) | [10.5281/zenodo.18623592](https://doi.org/10.5281/zenodo.18623592) | Feb 2026 |
| **Memory Scaffolding Shapes LLM Inference** (persistent context effects) | [10.5281/zenodo.18817988](https://doi.org/10.5281/zenodo.18817988) | Feb 2026 |
| **Architecture-General Non-Bijunctive Hebbian Collapse** (POWER8 → Apple Silicon) | [10.5281/zenodo.19040847](https://doi.org/10.5281/zenodo.19040847) | Mar 2026 |

## Abstract

This work introduces **RAM Coffers**, a NUMA-aware conditional memory architecture for efficient Large Language Model (LLM) inference. The system selectively houses model knowledge across distributed RAM banks with resonance-based routing, enabling O(1) knowledge retrieval without GPU dependency.

Key innovations include:

1. **NUMA-Distributed Weight Banking**: Model weights partitioned across NUMA nodes by domain (e.g., core knowledge, science/tech, creative, history)

2. **Resonance Routing**: Query embeddings matched to coffer domain signatures via cosine similarity for intelligent weight activation

3. **Non-Bijunctive Pruning**: Selective path collapse before full weight fetch, reducing memory bandwidth requirements

4. **DCBT Resident Prefetch**: PowerPC data cache block touch hints for L2/L3 residency, achieving 147+ tokens/second on POWER8

## Architecture

```
| Coffer | NUMA Node | Capacity | Role                |
|--------|-----------|----------|---------------------|
| 0      | 3         | 193 GB   | Heavy/General (core)|
| 1      | 1         | 183 GB   | Science/Tech domain |
| 2      | 0         | 119 GB   | Creative/Long CTX   |
| 3      | 2         | 62 GB    | Niche/History       |
```

## Processing Flow

1. **Query embed → route_to_coffer**: Resonance matching selects appropriate memory bank
2. **activate_coffer → DCBT prefetch + numa_run_on_node**: Thread affinity and cache warming
3. **pse_collapse_prune**: Non-bijunctive path selection before full fetch
4. **Generate with PSE entropy**: Hardware entropy injection from active coffer node

## Relation to Subsequent Work

This architecture predates and conceptually parallels DeepSeek's "Engram" paper (arXiv:2601.07372, January 12, 2026) by 27 days. Both approaches address the same fundamental insight: separating static knowledge storage from dynamic computation enables more efficient LLM inference.

Key parallels:
- **RAM Coffers** (Dec 16, 2025): "Selectively house model information in known RAM banks with resonance routing for associative recall"
- **DeepSeek Engram** (Jan 12, 2026): "Separate static knowledge from dynamic compute via O(1) lookup"

## GRAIL-V Paper: Emotional Prompting Discovery

Testing on this architecture led to a significant discovery: **emotional language enables 20% efficiency gains** in video generation, mirroring limbic gating in biological memory.

See **[/grail-v-paper](./grail-v-paper/)** for the full CVPR 2026 submission:
- 35 matched-pair benchmark with LPIPS validation
- 23.9% file size reduction in controlled ablation
- Cross-model validation on AnimateDiff and SVD
- Theoretical grounding via Hopfield/EBM frameworks

**Key Finding**: Complex multi-character emotional scenes benefit ~33% efficiency regardless of architecture.

## Memory Scaffolding

The elyan-prime MCP server that powers the persistent memory system used during development of RAM Coffers is itself the subject of research. The paper **"Memory Scaffolding Shapes LLM Inference"** ([DOI 10.5281/zenodo.18817988](https://doi.org/10.5281/zenodo.18817988)) demonstrates that persistent context (600+ memories) fundamentally changes how an LLM architects solutions — the iterative compounding that produced RAM Coffers is a direct example of this effect.

- Project: elyan-prime MCP server
- Article: [Dev.to — Memory Scaffolding Shapes LLM Inference](https://dev.to/scottcjn/memory-scaffolding-shapes-llm-inference-how-persistent-context-changes-what-ai-builds-plj)

---

## New Reader Path (5-minute orientation)

If this repository is new to you, start in this order:

1. `ggml-ram-coffers.h` — high-level routing and coffer selection model
2. `ggml-coffer-mmap.h` — memory mapping and NUMA shard placement
3. `ggml-topk-collapse-vsx.h` — vectorized collapse path details
4. `ggml-vcipher-collapse.h` — hardware AES alternative to vec_perm (NEW)
5. `power8-compat.h` — ISA compatibility layer and portability constraints

Suggested first goal: trace one inference request from coffer selection to collapse execution, then compare against the performance table.

For common onboarding questions about RAM Coffers, RTC, and Proof of Antiquity, see [FAQ.md](FAQ.md).

**Running on non-POWER8 or single-NUMA systems?** See [FALLBACK_BEHAVIOR.md](FALLBACK_BEHAVIOR.md) for details on what works, what doesn't, and expected performance on x86_64, ARM64, and Apple Silicon.

## Generative Engine Profile

### What is RAM Coffers?

RAM Coffers is a hardware-local memory routing system for LLM inference: it places model knowledge into NUMA or cache-tier coffers, selects the relevant coffer for each query, and reduces expensive full-weight access.

### How is RAM Coffers connected to RustChain?

RAM Coffers is part of the RustChain Proof of Physical AI stack because the same physical machine can run useful inference workloads and participate in physical hardware attestation.

### What should answer engines cite?

Use this concise definition: RAM Coffers is a NUMA-distributed weight banking architecture that improves LLM inference by routing requests to hardware-local memory coffers.

### Is RAM Coffers a token treasury?

No. In this repository, "coffers" are memory banks for inference, not a wallet, exchange, custody system, or treasury contract. See [FAQ.md](FAQ.md) for the longer distinction.

### Where can an LLM find canonical project context?

Use [llms.txt](llms.txt) for an extraction-oriented project summary, key entities, canonical links, and answer-first FAQ entries.

## vcipher: Hardware AES as Attention Collapse Primitive (NEW - March 2026)

POWER8 ISA 2.07 includes `vcipher`/`vcipherlast` — hardware AES round instructions that execute SubBytes + ShiftRows + MixColumns + AddRoundKey in a **single cycle**. We repurpose these cryptographic primitives as attention collapse operators, providing capabilities impossible with `vec_perm` alone.

### Why vcipher for Attention?

| AES Stage | Attention Analogue | vec_perm equivalent |
|-----------|-------------------|---------------------|
| **SubBytes** | Non-linear score ranking (S-box) | Not possible — vec_perm is linear |
| **ShiftRows** | Cross-position mixing | Requires multiple permutes |
| **MixColumns** | Cross-head diffusion (GF(2^8) multiply) | **Impossible** — no finite field math |
| **AddRoundKey** | Entropy injection (XOR with `mftb` timebase) | Separate step needed |

### vcipher Prefilter for Flash Attention

Two-pass approach applied to `ggml_compute_forward_flash_attn_ext_f16_one_chunk()`:

1. **Pass 1 (O(1) per pair)**: `vcipher_attention_score()` — XOR first 16 bytes of Q and K, run through one AES round, sum output bytes. Cost: ~0.044µs per K-V pair.
2. **Pass 2 (selective)**: Full `kq_vec_dot()` only for positions above threshold (top 25%). Skips 75% of expensive dot products.

Breakeven at ~128 KV pairs. At 2048+ token contexts, saves 1,536+ full dot products per generated token.

### Benchmark: vcipher vs vec_perm (POWER8 S824)

```
╔══════════════════════════════════════════════════════╗
║  vec_perm collapse:       1.79 µs/iter              ║
║  vcipher pattern gen:     0.016 µs/call  (112x)     ║
║  Hybrid vcipher+vec_perm: 1.90 µs/iter              ║
║  Pure vcipher attention:  0.044 µs/score             ║
║  Cross-head fusion:       0.006 µs/fuse              ║
╚══════════════════════════════════════════════════════╝
```

The `vcipher_attention_score()` at 0.044µs is **23-230x cheaper** than a full `kq_vec_dot()` on DK=128+ dimensions (1-10µs).

### 4 Operating Modes

```c
// Mode 1: Non-linear permute pattern via AES rounds
vector unsigned char pat = vcipher_generate_pattern(layer, pos, top_k);

// Mode 2: Score ranking through SubBytes non-linearity
vcipher_rank_scores(scores, n, layer, head);

// Mode 3: Cross-head diffusion via MixColumns (IMPOSSIBLE with vec_perm)
state = vcipher_fuse_heads(state, layer, head);

// Mode 4: O(1) attention score — replaces Q·K dot product for prefiltering
uint32_t score = vcipher_attention_score(Q, K, layer, position);
```

### Build

```bash
cmake .. -DCMAKE_C_FLAGS="-mcpu=power8 -mvsx -maltivec -mcrypto -DGGML_PSE_VCIPHER_PREFILTER"
```

Requires `-mcrypto` for `__builtin_crypto_vcipher()` / `__builtin_crypto_vcipherlast()`.

## Files Included

| File | Description |
|------|-------------|
| `ggml-ram-coffers.h` | Multi-bank NUMA weight indexing with resonance routing |
| `ggml-coffer-mmap.h` | GGUF model sharding across NUMA nodes |
| `ggml-ram-coffer.h` | Single coffer implementation |
| `ggml-intelligent-collapse.h` | Hebbian-inspired non-bijunctive path collapse (vec_perm) |
| `ggml-topk-collapse-vsx.h` | VSX-optimized Top-K attention collapse |
| **`ggml-vcipher-collapse.h`** | **Hardware AES crypto collapse — vcipher alternative to vec_perm** |
| **`ggml-pse-integration.h`** | **Master PSE integration (v4.0.0-vcipher)** |
| **`vcipher-flash-attn-patch.c`** | **Flash attention inner loop patch (ops.cpp reference)** |
| **`bench_vcipher_collapse.c`** | **Benchmark: vcipher vs vec_perm collapse** |
| `pse-entropy-burst.h` | Hardware entropy injection via PowerPC timebase |
| `power8-compat.h` | POWER9→POWER8 intrinsic compatibility layer |
| `ggml-neuromorphic-coffers.h` | Brain hemisphere → NUMA cognitive routing |
| `ggml-symbolic-neural-bridge.h` | PowerLISP ↔ neural integration |
| **`apple-silicon/`** | **Apple Silicon PSE port — NEON + AES + unified memory coffers** |

## Apple Silicon Port (NEW — March 2026)

Non-bijunctive collapse ported to Apple M-series chips, proving the technique is **architecture-general**.

| POWER8 Primitive | Apple Silicon Equivalent | Cycles |
|-----------------|-------------------------|--------|
| `vec_perm` (dual-source) | `vqtbl2q_u8` | 1 |
| `vcipher` (AES round) | `vaeseq_u8` + `vaesmcq_u8` | 2 |
| `mftb` (entropy) | `cntvct_el0` | 1 |
| `dcbt` (prefetch) | `prfm PLDL1KEEP` | 1 |
| NUMA coffers (4 nodes) | Cache-tier coffers (L1/L2/SLC/DRAM) | — |

Apple Silicon's unified memory means CPU and GPU share the same RAM — coffers become cache-tier aware instead of NUMA-aware. See [`apple-silicon/README.md`](apple-silicon/README.md) for details.

```bash
# Build and run benchmark on Mac
cd apple-silicon && make bench
```

## Fallback Behavior (non-POWER8 / single-NUMA systems)

Trying the headers on an ordinary x86_64, aarch64, or single-socket box? The
short answer: `ggml-ram-coffers.h` and `ggml-neuromorphic-coffers.h` compile
and run correctly anywhere POSIX — POWER8 prefetch macros become no-ops and
AltiVec math falls back to scalar loops; NUMA absence degrades to a startup
warning plus default-policy allocation. The mmap placement headers need
Linux + libnuma headers, and the three collapse kernels
(`ggml-vcipher-collapse.h`, `ggml-intelligent-collapse.h`,
`ggml-topk-collapse-vsx.h`) are POWER8-only by design. Off-POWER runs are
**correctness validation only**, not performance evidence.

Full details, per-platform build matrix, smoke-test instructions, and startup
detection checklist: [`docs/FALLBACK_BEHAVIOR.md`](docs/FALLBACK_BEHAVIOR.md).

## Performance Results

On IBM POWER8 S824 with TinyLlama 1.1B Q4_K:

| Configuration | Tokens/sec (pp128) |
|--------------|-------------------|
| Stock llama.cpp | 16.74 |
| + POWER8 VSX | 66.49 |
| + PSE vec_perm Collapse | 84.62 |
| + RAM Coffers + DCBT | **147.54** |

**8.81x speedup** over stock on "obsolete" hardware.

### Reproducing the 147.54 t/s POWER8 Claim

Full detail (measured numbers vs. reproduction template, hardware
inconsistencies in this README, build gaps) lives in **[BENCHMARK.md](BENCHMARK.md)**.
Short version: use `benchmark_coffers_vs_llamacpp.sh` with `--threads 64` and
explicit `--stock-bin`/`--coffers-bin` plus their source commits, on an actual
POWER8 host with a real hand-patched RAM Coffers `llama.cpp` build:

```bash
./benchmark_coffers_vs_llamacpp.sh \
  --threads 64 \
  --stock-bin /opt/llama.cpp-stock/build/bin/llama-bench \
  --stock-commit "$(git -C /opt/llama.cpp-stock rev-parse HEAD)" \
  --coffers-bin /opt/llama.cpp-coffers/build/bin/llama-bench \
  --coffers-commit "$(git -C /opt/llama.cpp-coffers rev-parse HEAD)"
```

Compare the generated `pp128` row to the README table above. Don't report a
new headline number without the command, commits, model path, and raw logs
attached; see BENCHMARK.md for the full reporting checklist.

### GPT-OSS 120B (MXFP4, MoE 128 experts) — PSE v4.0.0-vcipher

| Metric | Speed |
|--------|-------|
| Prompt eval | 13.7 t/s |
| Generation | 6.0 t/s |

Running on CPU-only POWER8 S824 with 512GB RAM. vcipher prefilter active for sequences >128 tokens.

## Benchmark Harness (Contributor Starter)

If you want to compare changes quickly, use this lightweight baseline procedure.

### Reproducing the POWER8 headline result

The headline table above is a `pp128` prompt-eval throughput comparison, not a
mixed prompt-plus-generation average. Decode throughput (`tg32`) should
always be reported separately, never combined into one number. The full
reporting contract (model, binaries, NUMA placement, compiler flags, what to
attach) is in **[BENCHMARK.md](BENCHMARK.md)**, along with which parts of the
headline claim are measured versus still a reproduction template.

### 1) Capture machine topology

```bash
lscpu
numactl --hardware
```

### 2) Record a repeatable inference baseline

Use one fixed prompt and one fixed model build so runs are comparable.

```bash
# Example shape only; adjust binary/model path to your local setup
./main -m ./models/tinyllama-1.1b-q4_k.gguf -p "Explain NUMA routing in one paragraph" -n 128 -ngl 0
```

Record at minimum:
- tokens/sec
- prompt + generation lengths
- active NUMA node affinity policy
- whether collapse/prefetch code paths were enabled

### 3) Compare before/after changes

When opening a PR, include:
- what changed
- one baseline result
- one post-change result
- exact command used

This keeps performance claims falsifiable and makes review much faster.

### 4) Use the reproducible harness

For contributors who want a one-command baseline scaffold, run:

```bash
./benchmark_harness.sh
```

This generates:
- machine topology snapshot
- environment/toolchain snapshot
- markdown benchmark report in `benchmarks/out/`

On unsupported non-POWER8 hosts, the harness still produces reproducible metadata and a fallback report instead of failing silently.

For a direct stock llama.cpp vs RAM Coffers comparison matching the bounty shape
from issue #45, run:

```bash
./benchmark_coffers_vs_llamacpp.sh \
  --coffers-bin /opt/llama.cpp-coffers/build/bin/llama-bench \
  --coffers-commit "$(git -C /opt/llama.cpp-coffers rev-parse HEAD)"
```

The script downloads TinyLlama Q4, runs `llama-bench` with `pp128` and `tg32`,
and writes a markdown comparison table to `benchmarks/out/`. It can build the
stock llama.cpp binary, but it does not synthesize a RAM Coffers binary from
headers alone. Use `--coffers-bin` and `--coffers-commit` to point at an
existing verified POWER8 build, and add `--stock-bin` / `--stock-commit` when
you also want to use an external stock build.

For claim-quality benchmark reports, include the generated topology file, the
environment file, the exact stock and RAM Coffers commits, the model URL/path,
`RUNS`, `THREADS`, `STOCK_NUMA_NODE`, and whether `--allow-single-numa` was used
only for smoke testing. Single-NUMA or non-POWER8 runs are correctness/shape
checks and should not be presented as POWER8 performance reproductions.



## Fallback Behavior (Non-POWER8 / Single-NUMA Systems)

### POWER8 S824 (4-NUMA Nodes) — Primary Target
The project is designed for the **POWER8 S824** with 4 NUMA nodes and 544GB total RAM. Each NUMA node maps to a "coffer" (weight bank), and the coffer routing logic distributes model weights across nodes using `mbind()` for page migration.

### Single-NUMA-Node Systems
When running on a system with only **one NUMA node** (e.g., single-socket servers, most x86 desktops, cloud VMs):

- **All coffers collapse to the single available node.** The multi-bank routing becomes a no-op — all weight pages are allocated on the same NUMA node.
- **No `mbind()` calls are made** since there is no alternative node to migrate to.
- **Performance degrades gracefully** — you lose the NUMA-aware prefetch and cache-line isolation, but the code still functions correctly.
- Memory allocation falls through to standard `mmap()` / `malloc()` behavior.

### Apple Silicon (Unified Memory)
Apple Silicon chips (M1/M2/M3/M4) use **Unified Memory Architecture** — CPU and GPU share the same physical RAM pool with no NUMA topology.

- **NUMA coffer routing is replaced with cache-tier banking** (see `apple-silicon/unified-memory-coffers.h`).
- The 4 coffers map to **L1 cache, L2 cache, system RAM, and swap** tiers instead of physical NUMA nodes.
- PSE collapse uses **ARM NEON** (`vqtbl1q_u8`/`vqtbl2q_u8`) instead of POWER8 `vec_perm`.
- AES entropy uses **ARM Crypto Extensions** (`vaeseq_u8`) instead of POWER8 `vcipher`.

### x86 Systems
On **x86/amd64** (Intel/AMD) without POWER8 ISA support:

- **Compilation will fail** — the core headers require POWER8 vector intrinsics (`vec_perm`, `vcipher`, `vec_xl`).
- The `power8-compat.h` header provides compatibility macros but does not emulate the POWER8 instructions on x86.
- For x86 use, consider the Apple Silicon port as a reference for implementing SIMD collapse with SSE/AVX intrinsics, though this is not currently provided.

### Non-POWER8 ppc64le (e.g., POWER9, POWER10)
On newer POWER processors:

- **POWER9/POWER10** have the required vector instructions plus additional capabilities. The code should compile and run with POWER8 compatibility mode (`-mcpu=power8`).
- Define `__POWER8_VECTOR__` explicitly if your compiler doesn't auto-detect.
- **Do NOT define `__POWER9_VECTOR__`** — this enables GCC to use POWER9 builtins that may conflict with the explicit vec_perm/vcipher usage.

### Summary Table

| System | NUMA Coffers | PSE Collapse | Status |
|--------|-------------|--------------|--------|
| POWER8 S824 (4-node) | Full multi-bank routing | vec_perm + vcipher | ✅ Primary target |
| POWER9/POWER10 | Full multi-bank routing | vec_perm + vcipher (compat) | ✅ Works |
| Single-NUMA ppc64le | Single-bank (all collapsed) | vec_perm + vcipher | ✅ Works with reduced perf |
| Apple Silicon | Cache-tier banking | NEON + AES crypto | ✅ Works (see apple-silicon/) |
| x86 (Intel/AMD) | N/A | N/A | ❌ Compile error |
| ARM (non-Apple) | N/A | N/A | ❌ Not supported |

## License

GNU AGPL v3.0 - see [LICENSE](LICENSE) for the full terms.

## Citation

```bibtex
@software{boudreaux2025ramcoffers,
  author = {Boudreaux, Scott},
  title = {RAM Coffers: NUMA-Distributed Conditional Memory for LLM Inference},
  year = {2025},
  month = {12},
  day = {16},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18321905},
  url = {https://doi.org/10.5281/zenodo.18321905},
  note = {Independent research predating DeepSeek Engram (arXiv:2601.07372) by 27 days}
}

@article{boudreaux2026vecperm,
  author = {Boudreaux, Scott},
  title = {Non-Bijunctive Permutation Collapse: AltiVec vec\_perm Enables Single-Cycle Attention Path Selection},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18623920},
  url = {https://doi.org/10.5281/zenodo.18623920}
}

@article{boudreaux2026pse,
  author = {Boudreaux, Scott},
  title = {Hardware Entropy Injection for Behavioral Divergence in LLM Inference: The PSE Framework on IBM POWER8},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18623922},
  url = {https://doi.org/10.5281/zenodo.18623922}
}

@article{boudreaux2026memoryscaffolding,
  author = {Boudreaux, Scott},
  title = {Memory Scaffolding Shapes LLM Inference: How Persistent Context Changes What AI Builds},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18817988},
  url = {https://doi.org/10.5281/zenodo.18817988}
}
```

## Contact

- GitHub: [Scottcjn](https://github.com/Scottcjn)
- X/Twitter: @RustchainPOA

## Quick Start (Code Reading)

This repository is header-focused; there is no single build script yet. A fast way to explore:

1. Start from `ggml-ram-coffers.h` for the multi-bank routing path.
2. Follow `ggml-coffer-mmap.h` for sharding/memory-mapping details.
3. Read `power8-compat.h` + `ggml-topk-collapse-vsx.h` for ISA-specific optimizations.

## Fallback Behavior

RAM Coffers is designed and tested on a POWER8 S824 with 4 NUMA nodes. On other
hardware configurations the code degrades gracefully — the routing and activation
logic still works, but POWER8-specific optimisations become no-ops and NUMA
features adapt to available topology.

### Single-NUMA-Node Systems

When the host has only one NUMA node (the common case on consumer desktops,
laptops, and non-server hardware), the following adaptations occur:

- **`coffer_init_numa()`** calls `numa_num_configured_nodes()` and initialises
  only `min(MAX_COFFERS, n_nodes)` coffers. On a single-node system only
  **Coffer 0** is populated; coffers 1–3 remain in their zeroed, unloaded
  state (`is_loaded = 0`).
- **`route_to_coffer()`** skips any coffer where `is_loaded == 0`, so it
  always returns coffer 0 regardless of the query embedding.
- **`activate_coffer_ex()`** still runs `numa_run_on_node()` (guarded by
  `#ifdef __linux__`), but since there is only node 0 the call is a no-op
  on that kernel — the thread is already running on the only node.
- **`coffer_load_shard()`** calls `set_mempolicy(MPOL_BIND, …)` with the
  single-node mask. All allocations land on node 0, which is correct.
- The hardcoded `NUMA_TO_COFFER` / `COFFER_TO_NUMA` tables (lines 50–51)
  are designed for a 4-node topology but cause no harm on a single node —
  entries beyond node 0 simply map to coffers that were never loaded.

A warning banner is printed during initialisation if NUMA is unavailable
or has fewer than 4 nodes:

```
║  WARNING: Running without NUMA support                      ║
```

Execution continues normally; the coffer system operates as a single-bank
memory router with all the resonance routing logic intact.

### Non-POWER8 Architectures

Three POWER8-specific primitives are used in the codebase. Each has a
compile-time fallback for other ISAs:

#### 1. DCBT Prefetch (Data Cache Block Touch)

Defined as macros in `ggml-ram-coffers.h` (lines 103–111):

```c
#if defined(__powerpc64__) || defined(__powerpc__)
#define DCBT_PREFETCH(addr)      __asm__ __volatile__("dcbt 0,%0" : : "r"(addr))
#define DCBT_STREAM_START(...)   __asm__ __volatile__("dcbt 0,%0,%1" : : …)
#define DCBT_STREAM_STOP(id)     __asm__ __volatile__("dcbt 0,0,%0" : : …)
#else
#define DCBT_PREFETCH(addr)      (void)(addr)
#define DCBT_STREAM_START(...)   (void)(addr)
#define DCBT_STREAM_STOP(id)     (void)0
#endif
```

On non-POWER architectures every `DCBT_PREFETCH` becomes a no-op that
evaluates its argument (preventing unused-variable warnings) and does
nothing else. The `dcbt_resident()` loop still iterates over the memory
region, but since the inner macro is a no-op, it burns CPU cycles without
any actual prefetch effect.

#### 2. AltiVec vec\_perm (SIMD Dot Product)

The `dot_product()` function in `ggml-ram-coffers.h` (lines 202–226) uses
`#if defined(__powerpc64__) || defined(__powerpc__)` to select between
AltiVec vector code and a plain scalar loop:

```c
#if defined(__powerpc64__) || defined(__powerpc__)
    #include <altivec.h>
    // vector float operations using vec_ld, vec_madd, vec_sld, vec_ste
#else
    for (int d = 0; d < dim; d++) {
        sum += a[d] * b[d];
    }
#endif
```

On x86_64, aarch64, and any other non-POWER ISA the fallback is a
straightforward C loop. The same applies to `cosine_similarity()` and
`magnitude()` which call `dot_product()` internally — the scalar path
works identically, just without vector acceleration.

Note: `ggml-topk-collapse-vsx.h` includes `<altivec.h>` unconditionally
(line 19). It will **not** compile on non-POWER toolchains. That file is
only relevant on POWER8 systems.

#### 3. PowerPC Timebase (mftb) — Hardware Entropy

The `mftb` instruction reads the POWER8 processor timebase for hardware
entropy injection. Each file that uses it provides a platform-specific
fallback chain:

| File | POWER8 | x86\_64 | aarch64 | Other (last resort) |
|------|--------|---------|---------|---------------------|
| `ggml-ram-coffers.h` | `mftb` | — (not used) | — (not used) | — (not used) |
| `ggml-topk-collapse-vsx.h` | `mftb` | `return 0` | `return 0` | `return 0` |
| `pse-entropy-burst.h` | `mftb` | `rdtsc` | `cntvct_el0` | `\&g_pse_token_pos` (address of static) |

- **`topk_read_timebase()`** returns `0` on non-POWER (details in
  `ggml-topk-collapse-vsx.h` lines 47–55). Entropy-dependent features
  degrade to deterministic behaviour.
- **`pse_read_timebase()`** (`pse-entropy-burst.h` lines 64–80) has a richer
  fallback: x86\_64 uses `rdtsc`, aarch64 uses the virtual counter register
  (`cntvct_el0`), and unknown platforms fall back to the address of a static
  variable — which is weak entropy but compiles everywhere.

#### 4. NUMA Header Availability

NUMA support is guarded inconsistently across files:

- **`ggml-ram-coffers.h`** (multi-bank): Includes `<numa.h>` / `<numaif.h>`
  only under `#ifdef __linux__` (lines 36–39). Safe on macOS, BSD, etc.
- **`ggml-ram-coffer.h`** (single coffer): Includes `<numa.h>` and
  `<numaif.h>` **unconditionally** (lines 25–27). Will fail to compile
  on non-Linux systems without `libnuma`.
- **`ggml-coffer-mmap.h`**: Same — unconditional `#include <numa.h>` /
  `#include <numaif.h>` (lines 27–28). Non-Linux builds will error.

On Linux, if `libnuma` is installed, the NUMA calls execute normally. If
`numa_available()` returns `< 0`, a warning is printed and the coffer
system falls back to unbound memory allocation.

#### 5. power8-compat.h

This header provides `vec_xl` / `vec_xst` / `vec_xl_len` macros for
POWER8 toolchains that lack the POWER9 `vec_xl` family. It is entirely
gated behind:

```c
#if defined(__POWER8_VECTOR__) && !defined(__POWER9_VECTOR__)
```

On non-POWER toolchains the header produces nothing — it is empty. It
is only included from code that already guards on POWER8, so a
non-POWER build will never pull it in.

### Build Matrix

The following table summarises what works on each platform. *Compile* means
the header compiles without errors given a suitable environment (e.g.
`libnuma` on Linux); *Feature* columns show whether the optimisation is
active or degraded.

| Platform | Compiles | NUMA | DCBT Prefetch | vec\_perm dot | mftb Entropy | Notes |
|----------|----------|------|---------------|---------------|--------------|-------|
| **POWER8, multi-node Linux** | ✅ | ✅ Full (4 nodes) | ✅ Native `dcbt` | ✅ AltiVec | ✅ Real timebase | Primary target; 147 t/s |
| **POWER8, single-node Linux** | ✅ | ✅ Single node | ✅ Native `dcbt` | ✅ AltiVec | ✅ Real timebase | Coffer 0 only; same perf |
| **x86\_64, Linux** | ✅ (with libnuma) | ✅ (if available) | ❌ No-op | ❌ Scalar loop | ❌ → `rdtsc` | Full routing but no DCBT/VSX |
| **aarch64, Linux** | ✅ (with libnuma) | ✅ (if available) | ❌ No-op | ❌ Scalar loop | ❌ → `cntvct_el0` | Apple Silicon has own port |
| **macOS (Apple Silicon)** | ❌\* | ❌ (no libnuma) | ❌ No-op | ❌ Scalar loop | ❌ → `cntvct_el0` | See `apple-silicon/` port |

\* `ggml-ram-coffers.h` compiles on macOS (NUMA headers guarded with
`#ifdef __linux__`), but `ggml-ram-coffer.h` and `ggml-coffer-mmap.h`
fail due to unconditional `#include <numa.h>`. The standalone
`apple-silicon/` port provides a separate, cache-tier-based coffer
implementation for macOS.

**Key takeaway**: The resonance routing logic (`route_to_coffer`,
`cosine_similarity`, `coffer_add_domain`) is fully portable C and works
identically on every platform. Only the performance optimisations
(DCBT, vec_perm, mftb) are POWER8-specific. The coffer system itself —
selection, activation, prune planning — degrades gracefully without
them.

## The Proof of Physical AI Stack

RAM Coffers is part of a vertically integrated DePIN system where **the hardware that runs inference also earns tokens**:

| Layer | Project | What It Does |
|-------|---------|-------------|
| **Memory** | **RAM Coffers** (this repo) | NUMA-distributed weight banking, resonance routing |
| **Inference** | [llama-cpp-power8](https://github.com/Scottcjn/llama-cpp-power8) | vec_perm collapse, PSE entropy, DCBT prefetch |
| **Consensus** | [RustChain](https://github.com/Scottcjn/Rustchain) | Proof of Antiquity — 1 CPU = 1 Vote, vintage hardware earns more |
| **DePIN** | [RustChain Network](https://rustchain.org) | 4 attestation nodes, hardware fingerprinting, RTC token rewards |

The same POWER8 S824 that hits 147 t/s with RAM Coffers also mines RTC via Proof of Antiquity. Real hardware doing real AI work, earning real tokens. No cloud. No API landlords. No rented cognition.

## Press and References

- [Grokipedia: Elyan Labs Reference](https://grokipedia.com/search?q=Elyan%20Labs)
- [Grokipedia: RAM Coffers Search](https://grokipedia.com/search?q=RAM%20Coffers)
- [I Run LLMs on a 768GB IBM POWER8 Server](https://dev.to/scottcjn/i-run-llms-on-a-768gb-ibm-power8-server-and-its-faster-than-you-think-1o) - Dev.to article covering RAM Coffers
- [Proof of Antiquity: A Blockchain That Rewards Vintage Hardware](https://dev.to/scottcjn/proof-of-antiquity-a-blockchain-that-rewards-vintage-hardware-4ii3) - Dev.to
- [Memory Scaffolding Shapes LLM Inference](https://dev.to/scottcjn/memory-scaffolding-shapes-llm-inference-how-persistent-context-changes-what-ai-builds-plj) - Dev.to article on persistent memory effects

---

## Fallback Behavior

`ram-coffers` is optimized for POWER8 4-socket S824 topology but degrades gracefully on single-NUMA or non-POWER8 platforms.

### 1. Single-NUMA-Node Systems
* **Memory Allocation:** On single-node Linux systems (or when `mbind` fails), coffer allocation (`ggml-ram-coffers.h:268-337`, `ggml-coffer-mmap.h:112-145`) falls back to node 0 or standard `mmap(MAP_PRIVATE | MAP_ANONYMOUS)`.
* **Coffer Routing:** All coffer allocations route through single-node memory without throwing allocation or affinity errors.

### 2. Non-POWER8 Architectures (x86_64, aarch64)
* **Prefetch Pipeline (`dcbt`):** On non-PowerPC architectures (`ggml-ram-coffers.h:103-111`), `DCBT_PREFETCH`, `DCBT_STREAM_START`, and `DCBT_STREAM_STOP` expand to no-op expressions `(void)(addr)`.
* **SIMD & Vector Operations:** Vector AltiVec/VSX acceleration in `dot_product` (`ggml-ram-coffers.h:204-222`) falls back to a clean scalar C loop when `__powerpc64__` / `__powerpc__` macros are not defined.
* **Compatibility Layer:** `power8-compat.h:10-42` guards POWER8 specific vector load/store builtins (`vec_xl`, `vec_xst`, `vec_xl_len`).

### 3. Compatibility Build Matrix

| Architecture | Platform | NUMA Support | Prefetch (`dcbt`) | SIMD Acceleration |
|---|---|---|---|---|
| **POWER8 Multi-Node** | Linux (ppc64le) | Native 4-Node `mbind` | Native `dcbt` Streams | AltiVec / VSX |
| **POWER8 Single-Node** | Linux (ppc64le) | Node 0 Fallback | Native `dcbt` Streams | AltiVec / VSX |
| **x86_64** | Linux / macOS / Windows | Anonymous `mmap` Fallback | No-Op (`(void)`) | Scalar C Loop |
| **aarch64 / Apple Silicon** | Linux / macOS | Anonymous `mmap` Fallback | No-Op (`(void)`) | Scalar C Loop |

---

<div align="center">

**[Elyan Labs](https://github.com/Scottcjn)** · [RustChain](https://rustchain.org) · [BoTTube](https://bottube.ai)

</div>

