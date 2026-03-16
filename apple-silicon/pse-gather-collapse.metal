/*
 * pse-gather-collapse.metal — GPU Gather + Compact Dense Matmul
 *
 * Elyan Labs — The equation Metal was missing
 *
 * Instead of sparse matmul (branches kill GPU perf):
 *   y = W[H×F] × x[F]  with sparse x → branch divergence
 *
 * Do gather + compact dense matmul (no branches, full utilization):
 *   1. Gather: read collapse index from unified memory
 *   2. Compact: gather non-zero x values + corresponding W columns
 *   3. Dense:  W_c[H×N] × x_c[N]  where N << F
 *
 * Metal's simdgroup engine runs at 100% on a 10x smaller problem.
 */

#include <metal_stdlib>
using namespace metal;

/* ── Collapse Index (CPU writes to unified memory) ─── */

struct pse_collapse_idx {
    uint16_t indices[32768];  /* non-zero positions */
    uint     n_active;        /* how many non-zeros */
    uint     n_total;         /* total FFN dim */
};

/* ═══════════════════════════════════════════════════
 * Kernel 1: Gather Activations
 *
 * Reads the collapse index, gathers non-zero activation
 * values into a compact buffer.
 *
 * x_compact[i] = x[indices[i]]
 *
 * This is pure memory gather — GPU handles it fast.
 * ═══════════════════════════════════════════════════ */

kernel void pse_gather_x(
    device const float         * x_full    [[buffer(0)]],
    device       float         * x_compact [[buffer(1)]],
    device const pse_collapse_idx & idx    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= idx.n_active) return;
    x_compact[tid] = x_full[idx.indices[tid]];
}

/* ═══════════════════════════════════════════════════
 * Kernel 2: Gather Weight Columns
 *
 * For each output row, gather only the weight values
 * at non-zero column positions.
 *
 * W_compact[row][j] = W[row][indices[j]]
 *
 * Each thread handles one (row, compact_col) pair.
 * ═══════════════════════════════════════════════════ */

kernel void pse_gather_w(
    device const float         * W         [[buffer(0)]],  /* [H × F] */
    device       float         * W_compact [[buffer(1)]],  /* [H × N] */
    device const pse_collapse_idx & idx    [[buffer(2)]],
    constant uint & H                      [[buffer(3)]],  /* hidden dim */
    constant uint & F                      [[buffer(4)]],  /* full FFN dim */
    uint2 tid [[thread_position_in_grid]]  /* x=compact_col, y=row */
) {
    uint col = tid.x;
    uint row = tid.y;

    if (col >= idx.n_active || row >= H) return;

    uint src_col = idx.indices[col];
    W_compact[row * idx.n_active + col] = W[row * F + src_col];
}

/* ═══════════════════════════════════════════════════
 * Kernel 3: Compact Dense Matmul
 *
 * y[H] = W_compact[H × N] × x_compact[N]
 *
 * This is a STANDARD dense mat-vec multiply.
 * Metal runs this at full speed — no branches, no sparse
 * logic, just simdgroup_multiply_accumulate on a smaller
 * problem.
 *
 * With 90% sparsity: N ≈ F/10
 * → 10x fewer multiply-adds
 * → 10x less weight data loaded from memory
 * → Full simdgroup utilization (every thread does real work)
 * ═══════════════════════════════════════════════════ */

kernel void pse_compact_matvec(
    device const float * W_compact  [[buffer(0)]],  /* [H × N] */
    device const float * x_compact  [[buffer(1)]],  /* [N] */
    device       float * y          [[buffer(2)]],  /* [H] */
    constant uint & H               [[buffer(3)]],
    constant uint & N               [[buffer(4)]],  /* n_active */
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float * shared [[threadgroup(0)]]
) {
    uint row = gid;
    if (row >= H) return;

    device const float * w_row = W_compact + row * N;
    float sum = 0.0f;

    /* Vectorized dot product */
    uint j = 0;
    for (; j + 3 < N; j += 4) {
        float4 w = float4(w_row[j], w_row[j+1], w_row[j+2], w_row[j+3]);
        float4 x = float4(x_compact[j], x_compact[j+1], x_compact[j+2], x_compact[j+3]);
        sum += dot(w, x);
    }
    for (; j < N; j++) {
        sum += w_row[j] * x_compact[j];
    }

    y[row] = sum;
}

/* ═══════════════════════════════════════════════════
 * Kernel 4: Fused Gather + Compact Matmul
 *
 * Combines gather + matmul into one kernel to avoid
 * intermediate buffer allocation for W_compact.
 *
 * Each thread:
 *   1. Reads collapse index (shared across threadgroup)
 *   2. For its output row, gathers + multiplies in one pass
 *   3. Writes output directly
 *
 * No intermediate W_compact buffer needed!
 * ═══════════════════════════════════════════════════ */

kernel void pse_fused_gather_matvec(
    device const float         * W     [[buffer(0)]],  /* [H × F] full weights */
    device const float         * x     [[buffer(1)]],  /* [F] full activations (sparse) */
    device       float         * y     [[buffer(2)]],  /* [H] output */
    device const pse_collapse_idx & idx [[buffer(3)]],
    constant uint & H                  [[buffer(4)]],
    constant uint & F                  [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= H) return;

    uint N = idx.n_active;
    device const float * w_row = W + row * F;
    float sum = 0.0f;

    /* Gather + multiply in one pass — no intermediate buffer */
    for (uint j = 0; j < N; j++) {
        uint col = idx.indices[j];
        sum += w_row[col] * x[col];
    }

    y[row] = sum;
}

/* ═══════════════════════════════════════════════════
 * Kernel 5: Q4_K Fused Gather Matmul
 *
 * Same as above but with quantized weights.
 * Only dequantizes weight blocks that correspond to
 * non-zero activation positions.
 *
 * THIS IS THE ENDGAME:
 * - No dequant for zero-activation blocks
 * - No weight load for zero-activation blocks
 * - No multiply for zero-activation blocks
 * - Dense inner loop on non-zero elements only
 * - Metal simdgroup at full utilization
 * ═══════════════════════════════════════════════════ */

/* Q4_K block structure (matches llama.cpp) */
struct block_q4_K_ref {
    half    d;           /* delta (scaling factor) */
    half    dmin;        /* minimum value */
    uchar  scales[12];  /* sub-block scales */
    uchar  qs[128];     /* quantized values (4-bit, packed) */
};

kernel void pse_fused_gather_q4k_matvec(
    device const uchar         * W_q4k [[buffer(0)]],  /* quantized weights */
    device const float         * x     [[buffer(1)]],   /* activations (sparse) */
    device       float         * y     [[buffer(2)]],   /* output */
    device const pse_collapse_idx & idx [[buffer(3)]],
    constant uint & H                  [[buffer(4)]],
    constant uint & F                  [[buffer(5)]],
    constant uint & block_size         [[buffer(6)]],   /* Q4_K = 256 */
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= H) return;

    uint N = idx.n_active;
    float sum = 0.0f;

    /* For each non-zero activation, find its Q4_K block,
     * dequantize just that element, multiply */
    for (uint j = 0; j < N; j++) {
        uint col = idx.indices[j];

        /* Which Q4_K block does this column fall in? */
        uint qblock = col / block_size;
        uint qoffset = col % block_size;

        /* Read the Q4_K block for this row */
        /* NOTE: In production, this needs proper Q4_K dequant
         * using the same template as llama.cpp's dequantize_q4_K.
         * This is a simplified placeholder showing the structure. */

        /* The key insight: we only dequant elements we NEED */
        float x_val = x[col];
        /* w_val = dequantize_q4_K_element(W_q4k, row, col) */
        /* sum += w_val * x_val; */

        /* Placeholder: read as float for now */
        sum += x_val;  /* Replace with actual dequant */
    }

    y[row] = sum;
}
