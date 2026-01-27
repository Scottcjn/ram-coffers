# RAM Coffers: NUMA-Distributed Conditional Memory for LLM Inference

**Author:** Scott Boudreaux
**Date:** December 16, 2025
**Institution:** Elyan Labs (Independent Research)
**Hardware:** IBM POWER8 S824 (320GB RAM, Dual 8-core)

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
  url = {https://zenodo.org/},
  note = {Independent research predating DeepSeek Engram (arXiv:2601.07372) by 27 days}
}
```

## Contact

- GitHub: [Elyan Labs]
- X/Twitter: @RustchainPOA
