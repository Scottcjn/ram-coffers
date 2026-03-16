# The Matmul Collapse Equation

## Core Equation (Element Form)

Let **x** ∈ R^K be the post-ReLU activation vector (~50-92% zeros).

Let I = { i : |x_i| > ε }, k' = |I|

Then:
- **x'** = x[I] ∈ R^(k')
- **W'** = W[:, I] ∈ R^(M × k')
- **y** = W' x'

Exact mathematical equivalent to y = Wx. Not an approximation.
Problem is 5-10x smaller and **100% dense**.

## Block Form (Q4_K Compatible — The Version That Ships)

Per-element gather breaks Q4_K packing (256 elements share scales).
Instead, partition K into blocks B_j of width b=256 (matching Q4_K).

Define block score:
- s_j = ||x[B_j]||_1

Active set:
- J = { j : s_j > τ }

Compacted multiply:
- **y** = Σ_{j ∈ J} W[:, B_j] × x[B_j]

Equivalent packed form:
- W' = concat_cols(W[:, B_j], j ∈ J)
- x' = concat(x[B_j], j ∈ J)
- y = W' x'

## Why Block-256

- Q4_K block = 256 elements with shared d/dmin/scales
- Gathering whole Q4_K blocks preserves dequant locality
- No partial dequant needed
- Aligned to Metal simdgroup tile boundaries

## Example (Qwen2.5-7B FFN)

- K = 18944 (FFN intermediate dim)
- Block size = 256
- Total blocks = 74
- With 50% sparsity: ~37 active blocks
- Effective K' = 37 × 256 = 9472 (2x smaller)
- With 90% sparsity: ~7 active blocks
- Effective K' = 7 × 256 = 1792 (10x smaller)

## Cost Model

Win condition:

T_mask + T_gather + T_dense(K') < T_dense(K)

Where:
- T_mask ≈ 0 (CPU builds list while GPU runs previous layer, unified memory)
- T_gather ≈ 0 (no physical gather — just iterate active block IDs)
- T_dense(K') = T_dense(K) × (K'/K)

So win condition simplifies to: K'/K < 1 — which is always true when sparsity > 0.

## The Key Implementation Insight

**Do NOT physically rebuild W_compact.** Instead:

```metal
// Loop only over active block IDs (no branch, uniform across SIMD)
for (uint i = 0; i < active->num_active; ++i) {
    uint b = active->block_ids[i];
    // Full stock Q4_K dequant + MMA for block b only
    dequantize_q4_K(weights + b, ...);
    simdgroup_multiply_accumulate(...);
}
```

Same dequant code. Same MMA code. Just fewer iterations.
No temporary buffers. No column copying. No graph changes.

## Three-AI Consensus

- **Claude Opus 4.6**: Designed the gather+compact approach, built the headers
- **GPT-5.4**: Formalized block-indexed collapse, showed Q4_K alignment requirement
- **Grok 4.2**: Confirmed block-iterate approach, provided kernel diff structure

All three independently concluded: stop fighting dense hardware.
Make the dense problem smaller. That's the equation Metal was missing.

## References

- DOI: 10.5281/zenodo.19040847 (Architecture-General PSE paper)
- Repo: github.com/Scottcjn/ram-coffers/tree/main/apple-silicon
