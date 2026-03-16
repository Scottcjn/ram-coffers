# Briefing: Sparse-Aware Metal Matmul for llama.cpp on Apple Silicon

## Objective

Accelerate llama.cpp on Apple Silicon by exploiting activation sparsity created through PSE Hebbian collapse and ReLU gating. Sparsity is proven (92% attention, ~50% FFN). Overhead is zero. But **Metal's stock matmul remains fully dense** — zero blocks still cause weight loads, dequant, and FMA. The next step is sparsity-aware execution inside the Metal kernel.

## Proven Results

| Metric | Value | Source |
|--------|-------|--------|
| Attention sparsity | 92% | PSE standalone bench (M2 native) |
| FFN activation zeros | ~50% | ReLU before down_proj |
| PSE overhead | 0% | llama-bench stock vs PSE |
| ReLU overhead | -0.6% | Extra Metal dispatch, no matmul savings |
| Persona divergence | 0.865 cosine | Hebbian logit collapse |

## Root Problem

Stock Metal kernel `kernel_mul_mv_q4_K_f32` iterates all K elements:
- Dequantizes every Q4_K block (256 elements)
- Loads weight rows regardless of activation value
- Runs `simdgroup_multiply_accumulate` on zero×weight=0
- Sparsity exists in memory but NOT in execution

## Required Engineering Change

A **sparse-aware Q4_K-compatible Metal matmul kernel** that:
1. Inspects activation blocks (32 or 256 elements)
2. Tests if block is entirely zero
3. Skips weight load + dequant + FMA for zero blocks
4. Preserves Q4_K dequantization compatibility

## Implementation Plan (from GPT-5.4 + Grok 4.2 analysis)

### Step 1: Mask struct in Metal shader

```metal
struct pse_ffn_mask {
    bool is_valid;
    uint mask[16];  // 512 blocks max (covers K up to ~16k)
};

inline bool block_active(const device pse_ffn_mask &m, uint b) {
    if (!m.is_valid) return true;
    return (m.mask[b >> 5] & (1u << (b & 31))) != 0;
}
```

### Step 2: Sparse inner loop in Q4_K kernel

Replace dense K iteration with block-skip:
```metal
for (uint block = 0; block < K / BLOCK_SIZE; ++block) {
    if (!block_active(*sparsity, block)) {
        continue;  // Skip dequant + weight load + FMA
    }
    // Existing Q4_K dequant + multiply for this block only
}
```

### Step 3: Dispatch in ggml-metal-ops.cpp

```cpp
if (use_sparse) {
    // Set mask buffer at slot 4
    ggml_metal_encoder_set_buffer(enc, mask_buffer_id, 4);
    // Dispatch sparse pipeline
}
```

### Step 4: CPU mask generation (unified memory)

```cpp
void generate_ffn_mask(const float* activations, uint K, pse_ffn_mask* mask) {
    mask->is_valid = true;
    for (uint b = 0; b < K/32; ++b) {
        float sum = 0.0f;
        for (uint j = 0; j < 32; ++j)
            sum += fabs(activations[b*32 + j]);
        if (sum > 1e-6f) mask->mask[b/32] |= (1u << (b%32));
        else              mask->mask[b/32] &= ~(1u << (b%32));
    }
}
```

## Key Design Questions

1. **Block granularity**: 32 elements (matches ReLU patterns + simd alignment) vs 256 (matches Q4_K block size, fewer branches). Recommendation: start with 256 to match Q4_K.

2. **Mask computation**: CPU precompute via unified memory (zero-cost transfer) vs GPU in-kernel check. Recommendation: CPU precompute — avoids branch divergence.

3. **Dispatch policy**: Dense if sparsity < threshold, sparse if above. Need switching threshold benchmark.

## Files Involved

| File | Change |
|------|--------|
| `ggml/src/ggml-metal/ggml-metal.metal` | Add sparse Q4_K kernel |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | Dispatch to sparse kernel |
| `src/llama-graph.cpp` | Already patched (ReLU before down_proj) |
| `common/sampling.cpp` | Already patched (Hebbian logit collapse) |

## Expected Performance

With 50% FFN sparsity and FFN being 60-70% of total compute:
- **20-40% faster FFN**
- **12-28% overall speedup**

## Repository

https://github.com/Scottcjn/ram-coffers/tree/main/apple-silicon

## Multi-AI Contributors

- **Claude Opus 4.6**: Architecture design, POWER8→NEON port, all code written
- **GPT-5.4**: Detailed kernel patch, Q4_K dequant integration, dispatch policy
- **Grok 4.2**: Block mask design, alignment analysis, PR-ready kernel diff (in progress)

## DOI

10.5281/zenodo.19040847 — "Architecture-General Non-Bijunctive Hebbian Collapse"
