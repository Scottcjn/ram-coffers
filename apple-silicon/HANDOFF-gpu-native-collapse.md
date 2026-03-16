# Handoff: GPU-Native Collapsed Q4_K Kernel

## Task

Write a Metal kernel that replaces the stock Q4_K mat-vec inner loop with one that iterates only active block IDs. No CPU sync. No custom ops. Pure GPU.

## Why

CPU-side block collapse (`ggml_map_custom1_inplace`) forces GPU↔CPU sync per layer:
- pp128: -5.5%, tg32: -41%. Pipeline destroyed.

The math is proven (CPU benchmark: 10x speedup at 10% blocks kept). The execution must stay on GPU.

## Exact Code Needed

### 1. Active Block List (add to ggml-metal.metal or ggml-metal-impl.h)

```metal
struct pse_active_blocks {
    uint  num_active;
    uint  block_ids[128];  // max K/256 blocks for typical models (74 for Qwen 7B)
};
```

### 2. The kargs struct (already exists at ggml-metal-impl.h:455)

```c
typedef struct {
    int32_t  ne00;   // K dimension (e.g. 18944)
    int32_t  ne01;   // M dimension (e.g. 4096)
    int32_t  ne02;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t  ne10, ne11, ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t  ne0, ne1;
    int16_t  r2, r3;
} ggml_metal_kargs_mul_mv_ext;
```

### 3. The stock dequant function (ggml-metal.metal:615)

```metal
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;
    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}
```

`block_q4_K` has: `half d`, `half dmin`, `uchar scales[12]`, `uchar qs[128]`. Total 144 bytes per 256 elements.

`dequantize_q4_K` is called with `il` from 0 to 15 (epb/16 = 256/16 = 16 chunks per block). Each call dequantizes 16 elements into a `float4x4` (type4x4).

### 4. The stock inner loop (ggml-metal.metal:3608)

This is what runs today for every Q4_K block:
```metal
for (int ich = tx; 16*ich < args.ne00; ich += chpt*nxpsg) {
    float4x4 lx[chpt];
    for (short ch = 0; ch < chpt; ++ch) {
        deq_t4x4(xq, cch, lx[ch]);        // dequantize weight chunk
        cch += nxpsg;
        if (cch >= chpb) { xq += cch/chpb; cch %= chpb; }
    }
    for (short ch = 0; ch < chpt; ++ch) {
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            sumf[ir1] +=
                dot(lx[ch][0], y4x4[ir1][ch*nxpsg][0]) +
                dot(lx[ch][1], y4x4[ir1][ch*nxpsg][1]) +
                dot(lx[ch][2], y4x4[ir1][ch*nxpsg][2]) +
                dot(lx[ch][3], y4x4[ir1][ch*nxpsg][3]);
        }
    }
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4x4[ir1] += chpt*nxpsg;
    }
}
```

**Key variables:**
- `tx` = thread position in simdgroup (0-31)
- `nxpsg` = threads per simdgroup handling x-dimension
- `chpt` = chunks per thread per iteration (= 1)
- `chpb` = chunks per Q4_K block (= 256/16 = 16)
- `xq` = weight pointer (advances through Q4_K blocks)
- `y4x4` = activation pointer array (float4x4, advances in lockstep)
- `cch` = current chunk within current Q4_K block

### 5. What needs to change

The `ich` loop iterates ALL chunks sequentially. For collapse, we need to:

1. **Convert sequential iteration to block-indexed iteration**
2. **For each active block ID, compute the absolute `xq` and `y4x4` positions**
3. **Run the same dequant + dot product on only those blocks**

The challenge: `xq` and `y4x4` use sequential pointer advancement, not absolute addressing. The collapsed version needs absolute positioning:

```metal
// COLLAPSED version (conceptual — needs exact pointer math)
for (uint ab = 0; ab < active->num_active; ++ab) {
    uint block_id = active->block_ids[ab];

    // Absolute position for this Q4_K block
    device const q_t * xq_block = (device const q_t *)(src0 + row_offset) + block_id;

    // Activation position (16 float4x4 chunks per Q4_K block)
    device const float4x4 * y_block = (device const float4x4 *)(src1 + col_offset)
                                    + block_id * (256/16);  // 16 chunks per block

    // Dequant all 16 chunks of this block and accumulate
    for (short il = tx; il < 16; il += nxpsg) {
        float4x4 lx;
        dequantize_q4_K(xq_block, il, lx);

        sumf[0] +=
            dot(lx[0], y_block[il][0]) +
            dot(lx[1], y_block[il][1]) +
            dot(lx[2], y_block[il][2]) +
            dot(lx[3], y_block[il][3]);
    }
}
```

### 6. Active block list generation (CPU side, at graph build time)

```c
// Call ONCE before Metal encoder starts — not per layer during execution
void pse_build_active_blocks(
    const float *activations,  // FFN intermediate after SwiGLU
    int K,                     // FFN dimension
    pse_active_blocks *out,
    float keep_ratio           // 0.5 = keep top 50% blocks
) {
    const int BLOCK = 256;
    int n_blocks = K / BLOCK;
    float norms[128];

    for (int b = 0; b < n_blocks; b++) {
        float s = 0;
        for (int j = 0; j < BLOCK; j++) {
            float v = activations[b*BLOCK + j];
            s += v * v;
        }
        norms[b] = s;
    }

    // Find threshold for top keep_ratio
    float sorted[128];
    memcpy(sorted, norms, n_blocks * sizeof(float));
    // ... sort descending, threshold = sorted[n_keep-1] ...

    out->num_active = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (norms[b] >= threshold)
            out->block_ids[out->num_active++] = b;
    }
}
```

This writes to a Metal shared buffer (unified memory). GPU reads it with zero cost.

### 7. Dispatch (ggml-metal-ops.cpp)

In `ggml_metal_op_mul_mat`, for the Q4_K ne11==1 (single-token) path:

```objc
if (pse_enabled && src0_type == GGML_TYPE_Q4_K && ne11 == 1) {
    [encoder setComputePipelineState:ctx->pipeline_pse_collapsed_q4k];
    [encoder setBuffer:src0_buf offset:src0_off atIndex:0];
    [encoder setBuffer:src1_buf offset:src1_off atIndex:1];
    [encoder setBuffer:dst_buf  offset:dst_off  atIndex:2];
    [encoder setBuffer:active_blocks_buf        atIndex:3];
    [encoder setBytes:&kargs length:sizeof(kargs) atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(ne01,1,1)
       threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    return 1;
}
```

## Expected Results

With 50% block sparsity (proven achievable with block-norm thresholding):
- down_proj matmul: **2x faster** (half the blocks)
- FFN total: **~1.5x faster** (down_proj is ~33% of FFN)
- Overall inference: **~1.2-1.3x faster** (FFN is ~65% of total)

With 75% block sparsity (aggressive thresholding):
- **3-4x faster FFN, 2x overall**

## Files to Modify

1. `ggml/src/ggml-metal/ggml-metal.metal` — add collapsed kernel
2. `ggml/src/ggml-metal/ggml-metal-impl.h` — add `pse_active_blocks` struct
3. `ggml/src/ggml-metal/ggml-metal-ops.cpp` — dispatch to collapsed kernel
4. `ggml/src/ggml-metal/ggml-metal-device.cpp` — create pipeline for new kernel

## Critical Constraint

The `dequantize_q4_K` pointer arithmetic assumes sequential chunk access within a block. The collapsed kernel must address each Q4_K block absolutely (by block_id), not sequentially. This means replacing `xq += cch/chpb` pointer walks with `xq_block = weights_base + block_id` absolute addressing.
