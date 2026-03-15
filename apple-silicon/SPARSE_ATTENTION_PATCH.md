# PSE Sparse Attention — Metal Shader Integration

## The Optimization

Stock flash attention computes V accumulation for ALL KV positions:
```metal
FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
    s8x8_t vs;
    simdgroup_load(vs, ss + 8*cc, SH, 0, false);    // Load 8 scores
    simdgroup_load(mv, pv, NS20, 0, false);           // Load V block
    simdgroup_multiply_accumulate(lo, vs, mv, lo);     // matmul
    pv += 8*NS20;
}
```

PSE sparse attention **skips blocks where all scores ≈ 0**:
```metal
FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
    // PSE: Check if this block survived collapse
    bool active = pse_block_is_active(ss, 8*cc, SH, j, tiisg);

    if (!active) {
        pv += 8*NS20;  // Advance pointer, skip everything else
        continue;
    }

    s8x8_t vs;
    simdgroup_load(vs, ss + 8*cc, SH, 0, false);
    simdgroup_load(mv, pv, NS20, 0, false);
    simdgroup_multiply_accumulate(lo, vs, mv, lo);
    pv += 8*NS20;
}
```

## What Gets Skipped Per Block

Each skipped block saves:
- 1x `simdgroup_load` of V from device memory (bandwidth)
- 1x `simdgroup_multiply_accumulate` (compute)
- 1x `simdgroup_load` of scores from shared memory

With 91% sparsity: ~90% of blocks skipped → ~10x fewer V operations.

## Where to Patch

File: `ggml/src/ggml-metal/ggml-metal.metal`

### Patch Point 1: After softmax, before V accumulation (~line 5880)

Add PSE collapse call after `threadgroup_barrier(mem_flags::mem_threadgroup);`:
```metal
// === PSE COLLAPSE (add after softmax barrier) ===
#ifdef PSE_ENABLED
FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
    const short j = jj*NSG + sgitg;
    pse_collapse_scores(ss, C, SH, tiisg, sgitg, j, int(iq2));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
```

### Patch Point 2: V accumulation loop (~line 5900)

Add block skip check inside the `cc` loop:
```metal
// Replace inner V loop with sparse version
FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
#ifdef PSE_ENABLED
    // Check if ALL 8 scores in this block are near-zero
    threadgroup float * block_scores = ss + 8*cc;
    float bmax = 0.0f;
    for (short b = 0; b < 8; b++) {
        bmax = max(bmax, abs(block_scores[j*SH + b]));
    }
    if (bmax < 0.001f) {
        pv += 8*NS20;
        continue;  // SKIP: no significant attention here
    }
#endif
    // ... rest of original V accumulation ...
}
```

## Build

```bash
cd ~/llama.cpp/build-pse
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-DPSE_ENABLED=1" \
    -DCMAKE_METAL_FLAGS="-DPSE_ENABLED=1"
make -j8
```

## Expected Results

| Metric | Stock | PSE Sparse | Speedup |
|--------|-------|------------|---------|
| V matmuls per token | 100% | ~10% | 10x fewer |
| Memory bandwidth (V) | 100% | ~10% | 10x less |
| Overall attention | 100% | ~30-50% | 2-3x faster |
| Text quality | baseline | +persona | better |

The 2-3x overall (not 10x) because Q·K computation and softmax are unchanged.
The speedup grows with context length (more KV = more blocks to skip).
