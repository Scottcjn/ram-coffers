# RAM Coffers + Proof of Physical AI (PPA) Integration

## How RAM Coffers Relates to PPA

RAM Coffers provides NUMA-aware weight banking for LLM inference on POWER8. PPA (Proof of Physical AI) verifies that the hardware performing inference is real, physical silicon — not a VM or emulated environment.

Together they form a verified inference stack:
1. **PPA** proves the hardware is real (7 CPU + 8 GPU fingerprint channels)
2. **RAM Coffers** optimizes inference on that verified hardware (NUMA routing, cache-resident prefetch)
3. **Tensor Core Precision Drift** proves which GPU generation ran the inference (deterministic LSB differences)

## Performance on Verified Hardware

| Model | Hardware (PPA-verified) | Speed | Notes |
|-------|------------------------|-------|-------|
| TinyLlama 1.1B Q4 | POWER8 S824 (PPA: 7/7 pass) | 147.54 t/s | PSE + full resident prefetch |
| DeepSeek-33B Q4_K | POWER8 S824 + V100 RPC | 5.37 t/s | NUMA interleave, 64 threads |
| Qwen2.5-14B | POWER8 + C4130 V100 (PPA: 5/5 GPU pass) | 68.8 t/s | RPC GPU offload over 40GbE |

All benchmarks run on PPA-attested hardware with verified silicon identity.

## Reference

- [RIP-0308: Proof of Physical AI](https://doi.org/10.5281/zenodo.19442753)
- [RAM Coffers: Neuromorphic NUMA Weight Banking](https://github.com/Scottcjn/ram-coffers)
