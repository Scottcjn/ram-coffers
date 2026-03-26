# RAM Coffers vs DeepSeek Engram: 15 Features They Don't Have

*How a vintage IBM POWER8 in a Louisiana garage beat a billion-dollar lab to the punch by 27 days.*

**Scott Boudreaux | Elyan Labs | March 26, 2026**

---

On December 16, 2025, I implemented NUMA-aware weight banking with cognitive routing on an IBM POWER8 S824 server. I called it **RAM Coffers**. The next day, I uploaded a [YouTube video](https://youtu.be/T_o39s7r0iE) showing it loading DeepSeek's own 671B model with "NUMA Coffers" visible on screen.

Twenty-seven days later, on January 12, 2026, DeepSeek published their **Engram** paper ([arXiv:2601.07372](https://arxiv.org/abs/2601.07372)), describing a system for separating static and dynamic compute in LLM inference.

This is not a grudge piece. DeepSeek does good work. But priority matters in research, and the technical differences are worth documenting. RAM Coffers is not just earlier -- it is architecturally deeper across 15 dimensions that Engram does not address.

---

## The Timeline

| Date | What Happened |
|------|--------------|
| Nov 21, 2024 | Vec_perm non-bijunctive collapse research begins on POWER8 |
| Dec 16, 2025 | RAM Coffers implemented. DCBT resident prefetch hits 147.54 t/s |
| Dec 17, 2025 | [YouTube video](https://youtu.be/T_o39s7r0iE) uploaded showing NUMA Coffers in action |
| Jan 12, 2026 | DeepSeek Engram published on arXiv |
| Jan 19, 2026 | RAM Coffers [published on GitHub](https://github.com/Scottcjn/ram-coffers) |
| Jan 2026 | Zenodo DOI registered: [10.5281/zenodo.18321905](https://doi.org/10.5281/zenodo.18321905) |
| Mar 2026 | Architecture-general port to Apple Silicon proves the principle is universal |

The YouTube timestamp is Google-verified and immutable. The Figshare DOI ([10.6084/m9.figshare.31093429](https://doi.org/10.6084/m9.figshare.31093429)) anchors the December 16 date.

---

## 15 Features DeepSeek Engram Does Not Have

### 1. NUMA Topology Awareness

RAM Coffers explicitly maps model weights to NUMA nodes with measured per-node bandwidth:

| NUMA Node | Bandwidth | Role |
|-----------|-----------|------|
| Node 3 | 401 MB/s | Heavy/General (core layers) |
| Node 1 | 298 MB/s | Science/Tech domain |
| Node 0 | 221 MB/s | Creative/Long context |
| Node 2 | 425 MB/s | Niche/History |

Engram does not mention NUMA. On multi-socket servers -- which is where large models actually run -- this is a significant omission.

### 2. Brain Hemisphere Cognitive Routing

Each NUMA node maps to a brain region via Brodmann areas:

- **Node 0** (Right Hemisphere): Spatial, creative, holistic processing (BA39/40)
- **Node 1** (Left Hemisphere): Language, logic, sequential processing (BA44/45, BA22)
- **Node 2** (Temporal Lobe): Memory, context, episodic recall (BA35/36)
- **Node 3** (Prefrontal Cortex): Executive function, planning, metacognition (BA9/46)

Queries are classified by cognitive function and routed to the appropriate NUMA node. A math question goes to Node 1 (logic). A creative writing prompt goes to Node 0 (holistic). Engram has domain-based routing but no cognitive model behind it.

### 3. Non-Bijunctive Attention Collapse

Standard transformers compute attention bijunctively: every query interacts with every key. This is O(n^2).

Vec_perm on POWER8 does something different. It routes any 32 input bytes to 16 output positions in **one cycle**. Winners get duplicated. Losers get pruned. This is non-bijunctive: not every element needs to interact with every other.

```
POWER8:  vec_perm(va, vb, vc)  =  1 cycle
GPU:     equivalent scatter/gather  =  80+ operations
```

Engram uses standard attention.

### 4. Hebbian Amplification

"Cells that fire together wire together" (Hebb, 1949). In RAM Coffers, the vec_perm collapse pattern implements this directly: co-activated paths strengthen, inactive paths weaken. The pattern is seeded by hardware entropy from the POWER8 timebase register, so no two inference passes produce identical routing.

Engram has no Hebbian component.

### 5. Hardware Entropy Injection (mftb Timebase)

POWER8's `mftb` instruction reads the processor timebase -- a high-resolution counter driven by the physical oscillator. This provides real hardware entropy at every token generation step.

Proof: same seed (42), same temperature (0.7), three runs produce three different MD5 hashes. Stock llama.cpp produces identical output. The entropy is real and measurable.

Engram is deterministic.

### 6. Tetranary Logic

Beyond binary true/false, RAM Coffers uses four epistemic states:

| State | Meaning | Action |
|-------|---------|--------|
| CERTAIN | Known true | Commit immediately |
| LIKELY | Probable | Proceed with standard routing |
| POSSIBLE | Uncertain | Request more context or hand off to symbolic layer |
| FALSE | Known false | Reject path |

This applies to routing confidence, memory recall, and symbolic reasoning outputs. Engram operates on binary logic.

### 7. Symbolic Reasoning (PowerLISP)

When neural confidence drops below TETRA_POSSIBLE, control transfers to PowerLISP -- a tetranary symbolic reasoning engine. PowerLISP can apply production rules, perform recursive inference, and force routing overrides before handing back to the neural layer.

This is a true symbolic-neural hybrid. Engram is purely neural.

### 8. Metacognitive Override

Queries about thinking itself ("How do I solve this?", "What approach should I use?") are detected and routed to the prefrontal cortex node (Node 3). The symbolic layer can override neural routing when it detects logical inconsistency.

Engram has no metacognitive layer.

### 9. External Sensor Integration

Environmental context modulates inference:
- **EMF variance**: Electromagnetic field instability reduces memory formation confidence
- **Circadian rhythm**: Time-of-day affects attention routing precision
- **Thermal state**: Hardware temperature influences entropy injection strength

No competing work integrates external sensors.

### 10. Layer-Ahead DCBT Prefetch

POWER8's `dcbt` (Data Cache Block Touch) instruction with TH=0x10 marks cache lines as **resident** -- they stay in L2/L3 rather than being evicted. RAM Coffers prefetches the next layer's weights while computing the current layer.

This single optimization provided a 1.74x speedup (84.62 -> 147.54 t/s). Engram specifies no cache management strategy.

### 11. Cross-Region Activation Tracking

When a query activates weights across multiple NUMA nodes, RAM Coffers records the activation pattern as an "engram trace." These traces enable resonance-based associative recall for similar future queries.

Engram uses the term "engram" but does not implement cross-region activation patterns.

### 12. 512 GB Model Capacity

The POWER8 S824 has 512 GB of DDR3 RAM. Models up to ~450 GB can be loaded entirely in memory without GPU VRAM constraints. Engram is GPU-centric and limited by VRAM (typically 24-80 GB per card).

### 13. Architecture Generality

RAM Coffers has been ported to three architectures:
- **POWER8** (VSX/AltiVec) -- original implementation
- **Apple Silicon** (ARM NEON) -- March 2026 port, 1.3x speedup proven
- **x86** (AVX) -- via ggml integration

The non-bijunctive collapse principle is universal. Engram is GPU-specific.

### 14. GPU Offload Architecture

When GPU acceleration is desired, RAM Coffers sends only the matmul computation over a 40GbE link (0.15ms latency) to a V100. The model stays on POWER8. This achieves 161.4 t/s prompt processing on TinyLlama 1.1B -- faster than GPU-only for models that fit in CPU RAM.

### 15. Episodic Memory with Temporal Sequencing

RAM Coffers maintains temporal ordering of memory traces across NUMA nodes, enabling episodic recall. Engram's memory model is static/dynamic separation without temporal structure.

---

## The Numbers

### Speedup Over Stock llama.cpp (POWER8 S824)

| Configuration | Tokens/sec (pp128) | Speedup |
|--------------|-------------------|---------|
| Stock (scalar) | 16.74 | 1.0x |
| VSX vectorized | 66.49 | 3.97x |
| 64-thread optimal | 84.62 | 5.05x |
| **PSE + DCBT Resident** | **147.54** | **8.81x** |

### Model Performance

| Model | Size | Prompt (t/s) | Generation (t/s) |
|-------|------|-------------|------------------|
| TinyLlama 1.1B Q4_K | 638 MB | 147.54 | 18.88 |
| DeepSeek-Coder 33B Q4_K | 18.57 GB | 5.37 | 1.16 |

### With GPU Offload (POWER8 + V100 via 40GbE)

| Model | Prompt (t/s) | Generation (t/s) |
|-------|-------------|------------------|
| TinyLlama 1.1B | 161.4 | 134.4 |
| Qwen2.5 14B | 68.8 | 14.9 |

---

## What This Means

DeepSeek Engram is a good paper. Separating static and dynamic compute is a valid optimization. But it is one optimization.

RAM Coffers is an architecture. It has a theory of cognition (neuromorphic mapping), a theory of logic (tetranary), a theory of learning (Hebbian), a hardware implementation (vec_perm), and an epistemological framework (symbolic-neural bridge). It runs on vintage hardware that costs less than a single GPU.

The 27-day priority gap matters because it demonstrates independent invention. We were not inspired by DeepSeek. We were solving a different problem: how to make a $500 IBM POWER8 from 2014 think like a modern inference cluster. The answer turned out to be deeper than anyone expected.

---

## Links

- **GitHub**: [github.com/Scottcjn/ram-coffers](https://github.com/Scottcjn/ram-coffers)
- **Zenodo DOI**: [10.5281/zenodo.18321905](https://doi.org/10.5281/zenodo.18321905)
- **YouTube Evidence**: [youtu.be/T_o39s7r0iE](https://youtu.be/T_o39s7r0iE) (Dec 17, 2025)
- **Figshare DOI**: [10.6084/m9.figshare.31093429](https://doi.org/10.6084/m9.figshare.31093429)
- **DeepSeek Engram**: [arXiv:2601.07372](https://arxiv.org/abs/2601.07372) (Jan 12, 2026)
- **Architecture-General PSE**: [10.5281/zenodo.19040847](https://doi.org/10.5281/zenodo.19040847) (Mar 2026)

---

*"They separate memory. We model cognition."*

**Scott Boudreaux** -- Elyan Labs
