# Apple Silicon PSE — Non-Bijunctive Collapse via ARM NEON

Port of the POWER8 Proto-Sentient Engine to Apple M-series chips.

## Architecture Mapping

| POWER8 | Apple Silicon | Cycles | Purpose |
|--------|--------------|--------|---------|
| `vec_perm` | `vqtbl1q_u8` | 1 | Single-source byte permute |
| `vec_perm` (dual) | `vqtbl2q_u8` | 1 | Dual-source 32→16 byte permute |
| `vcipher` | `vaeseq_u8` + `vaesmcq_u8` | 2 | AES round (entropy diffusion) |
| `mftb` | `cntvct_el0` | 1 | Hardware entropy counter |
| `dcbt` resident | `prfm PLDL1KEEP` | 1 | Cache-line prefetch |
| NUMA coffers | Cache-tier coffers | — | L1/L2/SLC/DRAM weight banking |

## Key Insight

Non-bijunctive collapse is **architecture-general**. Any CPU with:
- 128-bit byte-level permutation (1 cycle)
- Hardware entropy source
- Threshold + select operations

...can implement Hebbian attention collapse. This isn't a POWER8 trick.

## Files

| File | Purpose |
|------|---------|
| `apple-pse-config.h` | Detection, parameters, timebase |
| `neon-collapse.h` | vqtbl1q/vqtbl2q non-bijunctive collapse |
| `aes-entropy-collapse.h` | AES-based entropy injection |
| `unified-memory-coffers.h` | Cache-tier weight banking |
| `apple-pse-integration.h` | Master header (include this one) |
| `bench-pse-apple.c` | Benchmark: stock vs PSE |

## Build & Run

```bash
# On Apple Silicon Mac
make bench

# On Linux aarch64
make CC=gcc bench

# Cross-compile (for testing on x86)
# Will run in compatibility mode without NEON
make CC=gcc CFLAGS="-O3" bench
```

## Unified Memory Coffers

Apple Silicon has no NUMA — CPU and GPU share the same RAM. Instead of
NUMA-node coffers, we use **cache-tier coffers**:

```
Coffer 0 (HOT)  → L2-pinned    — attention heads, Q/K/V
Coffer 1 (WARM) → SLC-resident — FFN weights
Coffer 2 (COOL) → DRAM         — embeddings
Coffer 3 (COLD) → demand-load  — rare layers
```

The GPU can read any coffer without copy (unified memory advantage).

## Integration with llama.cpp

```c
#include "apple-pse-integration.h"

// Init (during model load)
pse_apple_init(model->data, model->size, model->n_layers, layer_size);

// Attention collapse (after QK^T, before softmax×V)
pse_apple_collapse_attention(scores, n, layer, head, pos);

// Entropy burst (before sampling)
pse_apple_entropy_burst(logits, n_vocab);
```
