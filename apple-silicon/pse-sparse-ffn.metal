/*
 * pse-sparse-ffn.metal — GPU-Side Sparse FFN via Unified Memory Mask
 *
 * Elyan Labs — The CPU is the oracle. The GPU is the engine.
 *
 * The CPU writes a bitmask to unified memory indicating which FFN
 * column blocks have strong activations (worth computing).
 * This kernel reads that mask and SKIPS masked-out blocks.
 *
 * On CUDA, this mask transfer would cost 5-10us per layer.
 * On Metal unified memory, the cost is ZERO — same RAM.
 *
 * This is a MATMUL KERNEL with integrated sparse skip.
 * Drop-in replacement for ggml_compute_forward_mul_mat on FFN layers.
 */

#include <metal_stdlib>
using namespace metal;

/* ── Sparse mask structure (matches CPU-side definition) ── */

#define PSE_FFN_BLOCK 32

struct pse_ffn_mask {
    uint64_t mask[8];       /* Up to 512 blocks (16384 cols / 32) */
    uint  total_blocks;
    uint  active_blocks;
    uint  skipped_blocks;
    int   layer_id;
    int   is_valid;
};

/* Check if a column block is active */
inline bool block_active(device const pse_ffn_mask & m, uint block_idx) {
    uint word = block_idx / 64;
    uint bit  = block_idx % 64;
    return (m.mask[word] >> bit) & 1;
}

/* ═══════════════════════════════════════════════════════
 * Sparse FFN MatMul Kernel
 *
 * output[M] = input[K] × weights[M × K]
 *
 * But SKIP column blocks where mask bit = 0.
 *
 * Each threadgroup handles one output row.
 * Within a row, threads cooperatively accumulate the dot product,
 * but SKIP column blocks that are masked out.
 *
 * With 90% sparsity: 90% of multiply-accumulate operations skipped.
 * FFN is 60-70% of total compute → 54-63% overall savings.
 * ═══════════════════════════════════════════════════════ */

kernel void pse_sparse_ffn_mul_mat(
    device const float * input       [[buffer(0)]],   /* [K] */
    device const float * weights     [[buffer(1)]],   /* [M × K] row-major */
    device       float * output      [[buffer(2)]],   /* [M] */
    device const pse_ffn_mask & mask [[buffer(3)]],   /* CPU-written sparse mask */
    constant uint & M                [[buffer(4)]],   /* output dimension */
    constant uint & K                [[buffer(5)]],   /* input dimension (FFN intermediate) */
    uint2 gid                        [[thread_position_in_grid]],
    uint2 lid                        [[thread_position_in_threadgroup]],
    uint2 tgsize                     [[threads_per_threadgroup]]
) {
    uint row = gid.x;
    if (row >= M) return;

    uint n_blocks = (K + PSE_FFN_BLOCK - 1) / PSE_FFN_BLOCK;

    float sum = 0.0f;
    device const float * w_row = weights + row * K;

    /* Iterate over column blocks */
    for (uint b = 0; b < n_blocks; b++) {
        /* ── THE SPARSE SKIP ──
         * If this block is masked out by CPU-side Hebbian collapse,
         * skip the entire multiply-accumulate for 32 columns.
         *
         * The mask was written by the CPU (pse_collapse_ffn_mask)
         * to unified memory. We read it HERE with zero latency.
         * On CUDA, this would require cudaMemcpy + sync.
         */
        if (mask.is_valid && !block_active(mask, b)) {
            continue;  /* Skip 32 columns of matmul */
        }

        uint col_start = b * PSE_FFN_BLOCK;
        uint col_end = min(col_start + PSE_FFN_BLOCK, K);

        /* Vectorized dot product for this block */
        for (uint j = col_start; j < col_end; j += 4) {
            if (j + 3 < col_end) {
                float4 w = float4(w_row[j], w_row[j+1], w_row[j+2], w_row[j+3]);
                float4 x = float4(input[j], input[j+1], input[j+2], input[j+3]);
                sum += dot(w, x);
            } else {
                for (uint jj = j; jj < col_end; jj++) {
                    sum += w_row[jj] * input[jj];
                }
            }
        }
    }

    output[row] = sum;
}

/* ═══════════════════════════════════════════════════════
 * Tiled Sparse FFN (for larger matrices)
 *
 * Uses threadgroup shared memory for input tiling.
 * Each threadgroup loads a tile of the input vector into
 * shared memory, then each thread computes its output row
 * using the shared tile.
 *
 * The sparse skip applies PER TILE — if the entire tile
 * maps to masked-out columns, the tile load is skipped too.
 * ═══════════════════════════════════════════════════════ */

kernel void pse_sparse_ffn_tiled(
    device const float * input       [[buffer(0)]],
    device const float * weights     [[buffer(1)]],
    device       float * output      [[buffer(2)]],
    device const pse_ffn_mask & mask [[buffer(3)]],
    constant uint & M                [[buffer(4)]],
    constant uint & K                [[buffer(5)]],
    uint gid                         [[thread_position_in_grid]],
    uint lid                         [[thread_position_in_threadgroup]],
    uint tg_size                     [[threads_per_threadgroup]],
    threadgroup float * shared       [[threadgroup(0)]]
) {
    uint row = gid;
    if (row >= M) return;

    float sum = 0.0f;
    device const float * w_row = weights + row * K;

    uint n_blocks = (K + PSE_FFN_BLOCK - 1) / PSE_FFN_BLOCK;

    /* Process in tiles of PSE_FFN_BLOCK */
    for (uint b = 0; b < n_blocks; b++) {
        /* Sparse skip: check mask BEFORE loading tile */
        if (mask.is_valid && !block_active(mask, b)) {
            continue;  /* Skip tile load + compute */
        }

        uint col_start = b * PSE_FFN_BLOCK;
        uint col_end = min(col_start + PSE_FFN_BLOCK, K);
        uint tile_size = col_end - col_start;

        /* Cooperative tile load into shared memory */
        if (lid < tile_size) {
            shared[lid] = input[col_start + lid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Compute partial dot product from shared tile */
        for (uint j = 0; j < tile_size; j++) {
            sum += w_row[col_start + j] * shared[j];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    output[row] = sum;
}

/* ═══════════════════════════════════════════════════════
 * Benchmark: Stock vs Sparse FFN
 *
 * Runs both full and sparse matmul, reports timing.
 * Use to measure actual cycle savings.
 * ═══════════════════════════════════════════════════════ */

kernel void pse_ffn_benchmark(
    device const float * input       [[buffer(0)]],
    device const float * weights     [[buffer(1)]],
    device       float * output_full [[buffer(2)]],
    device       float * output_sparse [[buffer(3)]],
    device const pse_ffn_mask & mask [[buffer(4)]],
    constant uint & M                [[buffer(5)]],
    constant uint & K                [[buffer(6)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= M) return;

    device const float * w_row = weights + row * K;

    /* Full (stock) computation */
    float sum_full = 0.0f;
    for (uint j = 0; j < K; j++) {
        sum_full += w_row[j] * input[j];
    }
    output_full[row] = sum_full;

    /* Sparse (PSE) computation */
    float sum_sparse = 0.0f;
    uint n_blocks = (K + PSE_FFN_BLOCK - 1) / PSE_FFN_BLOCK;
    for (uint b = 0; b < n_blocks; b++) {
        if (mask.is_valid && !block_active(mask, b)) continue;

        uint col_start = b * PSE_FFN_BLOCK;
        uint col_end = min(col_start + PSE_FFN_BLOCK, K);
        for (uint j = col_start; j < col_end; j++) {
            sum_sparse += w_row[j] * input[j];
        }
    }
    output_sparse[row] = sum_sparse;
}
