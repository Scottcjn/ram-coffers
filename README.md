# RAM Coffers: NUMA-Distributed Conditional Memory for LLM Inference

[![BCOS Certified](https://img.shields.io/badge/BCOS-Certified-brightgreen?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMiAxTDMgNXY2YzAgNS41NSAzLjg0IDEwLjc0IDkgMTIgNS4xNi0xLjI2IDktNi40NSA5LTEyVjVsLTktNHptLTIgMTZsLTQtNCA1LjQxLTUuNDEgMS40MSAxLjQxTDEwIDE0bDYtNiAxLjQxIDEuNDFMMTAgMTd6Ii8+PC9zdmc+)](BCOS.md)
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

- Repository: [Scottcjn/elyan-prime](https://github.com/Scottcjn/elyan-prime)
- Article: [Dev.to — Memory Scaffolding Shapes LLM Inference](https://dev.to/scottcjn/memory-scaffolding-shapes-llm-inference-how-persistent-context-changes-what-ai-builds-plj)

---

## Files Included

| File | Description |
|------|-------------|
| `ggml-ram-coffers.h` | Multi-bank NUMA weight indexing with resonance routing |
| `ggml-coffer-mmap.h` | GGUF model sharding across NUMA nodes |
| `ggml-ram-coffer.h` | Single coffer implementation |
| `ggml-intelligent-collapse.h` | Hebbian-inspired non-bijunctive path collapse |
| `ggml-topk-collapse-vsx.h` | VSX-optimized Top-K attention collapse |
| `pse-entropy-burst.h` | Hardware entropy injection via PowerPC timebase |
| `power8-compat.h` | POWER9→POWER8 intrinsic compatibility layer |

## Performance Results

On IBM POWER8 S824 with TinyLlama 1.1B Q4_K:

| Configuration | Tokens/sec (pp128) |
|--------------|-------------------|
| Stock llama.cpp | 16.74 |
| + POWER8 VSX | 66.49 |
| + PSE Collapse | 84.62 |
| + RAM Coffers + DCBT | **147.54** |

**8.81x speedup** over stock on "obsolete" hardware.

## License

MIT License - Free to use, modify, and distribute with attribution.

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

## Press and References

- [Grokipedia: Elyan Labs Reference](https://grokipedia.com/search?q=Elyan%20Labs)
- [Grokipedia: RAM Coffers Search](https://grokipedia.com/search?q=RAM%20Coffers)
- [I Run LLMs on a 768GB IBM POWER8 Server](https://dev.to/scottcjn/i-run-llms-on-a-768gb-ibm-power8-server-and-its-faster-than-you-think-1o) - Dev.to article covering RAM Coffers
- [Proof of Antiquity: A Blockchain That Rewards Vintage Hardware](https://dev.to/scottcjn/proof-of-antiquity-a-blockchain-that-rewards-vintage-hardware-4ii3) - Dev.to
- [Memory Scaffolding Shapes LLM Inference](https://dev.to/scottcjn/memory-scaffolding-shapes-llm-inference-how-persistent-context-changes-what-ai-builds-plj) - Dev.to article on persistent memory effects
# RAM Coffers Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAM Coffers System                       │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Frontend   │─────▶│   Backend    │─────▶│   Database   │
│  (Web UI)    │      │   (API)      │      │ (PostgreSQL) │
└──────────────┘      └──────────────┘      └──────────────┘
       │                     │                      │
       │                     │                      │
       ▼                     ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Browser    │      │   Server     │      │   Storage    │
│   Cache      │      │   Cache      │      │   Layer      │
└──────────────┘      └──────────────┘      └──────────────┘

## Components

### Frontend
- React/Vue.js UI
- Real-time updates
- Responsive design

### Backend
- RESTful API
- Authentication
- Business logic

### Database
- PostgreSQL
- Data persistence
- Query optimization

### Caching
- Redis for session
- Browser cache
- CDN integration
