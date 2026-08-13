# Apple Silicon PSE — Non-Bijunctive Collapse via ARM NEON

Port of the POWER8 Proto-Sentient Engine to Apple M-series chips.

## Architecture Mapping

| POWER8 | Apple Silicon | Cycles | Purpose |
|--------|--------------|--------|---------|
| `vec_perm` | `vqtbl1q_u8` | 1 | Single-source byte permute |
| `vec_perm` (dual) | `vqtbl2q_u8` | 1 | Dual-source 32→16 byte permute |
| `vcipher` | `vaeseq_u8` + `vaesmcq_u8` | 2 | AES round (entropy diffusion) — **NOT bit-identical to `vcipher`, see below** |
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


### `vaeseq_u8` + `vaesmcq_u8` is not `vcipher`

The table above pairs these for the equivalent *purpose*, not for equivalent
*output*. They compute different functions of `(state, key)`:

```
POWER8   vcipher(s,k)              = MixColumns(ShiftRows(SubBytes(s))) XOR k
x86      aesenc(s,k)               = MixColumns(ShiftRows(SubBytes(s))) XOR k
ARM      vaesmcq_u8(vaeseq_u8(s,k)) = MixColumns(ShiftRows(SubBytes(s XOR k)))
```

`AESE` applies the round key **first** and `AESMC` adds no final XOR, so the
ARM pair computes `aesenc(s XOR k, 0)`. That agrees with `vcipher(s, k)` only
when the key is zero.

Diffusion quality is unaffected and nothing here is insecure — but **the ports
do not produce the same bytes for the same inputs**, so any test that pins
exact output cannot be shared across them, and a value collapsed on Apple
Silicon will not match the same value collapsed on POWER8 or x86.

Found by @erkinalp while porting the collapse to AES-NI (ram-coffers#685).
`x86_aes_arm_order()` in `x86-64/` exists for anyone who needs to match the
Apple ordering deliberately.
