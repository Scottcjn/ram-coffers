```markdown
# RAM Coffers vs DeepSeek Engram: 15 Features They Don't Have

*How a vintage IBM POWER8 in a Louisiana garage beat a billion-dollar lab to the punch by 27 days.*

**Scott Boudreaux | Elyan Labs | March 26, 2026**

---

On December 16, 2025, I implemented NUMA-aware weight banking with cognitive routing on an IBM POWER8 S824 server (**768 GB installed / 512 GB active**). I called it **RAM Coffers**. The next day, I uploaded a [YouTube video](https://youtu.be/T_o39s7r0iE) showing it loading DeepSeek's own 671B model with "NUMA Coffers" visible on screen.

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

*(Note: The per-node figures above represent a free-memory snapshot taken at measurement time on Dec 16, 2025, within the active allocation space.)*

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
# RAM Coffers vs DeepSeek Engram: 15 Features They Don't Have

*How a vintage IBM POWER8 in a Louisiana garage beat a billion-dollar lab to the punch by 27 days.*

**Scott Boudreaux | Elyan Labs | March 26, 2026**

---

On December 16, 2025, I implemented NUMA-aware weight banking with cognitive routing on an IBM POWER8 S824 server (**768 GB installed / 512 GB active**). I called it **RAM Coffers**. The next day, I uploaded a [YouTube video](https://youtu.be/T_o39s7r0iE) showing it loading DeepSeek's own 671B model with "NUMA Coffers" visible on screen.

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

*(Note: The per-node figures above represent a free-memory snapshot taken at measurement time on Dec 16, 2025, within the active allocation space.)*

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

