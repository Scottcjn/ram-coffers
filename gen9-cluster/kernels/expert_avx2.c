/* Zen 2 / AVX2 expert kernel: the CPU path every console shares.
 *
 * This is the fallback that always works — a PS5 under HEN, an Xbox in retail
 * Dev Mode, a 4700S with no GPU at all — and on the salvage boards it is the
 * *only* path, so it is written to be genuinely fast rather than merely
 * present.
 *
 * What the shape of the problem dictates:
 *
 *   - Batch 1 decode means GEMV, not GEMM. Every weight is read once and used
 *     once, so the kernel is memory-bound and the only thing that matters is
 *     streaming weights at the memory's speed. Blocking, packing, and the rest
 *     of the GEMM playbook buy nothing here.
 *   - Zen 2 has AVX2 and two 256-bit FMA units but no AVX-512, and its
 *     L2 is 512 KiB per core. Four independent accumulators keep both FMA
 *     pipes fed across the 5-cycle latency; more than four gains nothing once
 *     the loop is bandwidth-bound.
 *   - FP8 weights are dequantised inline, 128 elements at a time, so the bytes
 *     crossing the memory bus are one per weight rather than four. On a
 *     bandwidth-bound kernel that is close to a 4x speedup, which is why the
 *     checkpoint's native format is kept all the way down to the kernel.
 *
 * Threading is OpenMP over output rows when available; a console gives us 6-8
 * cores and there is nothing else for them to do.
 */

#include "fp8.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* out[m] = sum_k w[m][k] * x[k], w row-major, fp32. */
void gen9_gemv_f32(const float *w, const float *x, float *out, int rows,
                   int cols)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < rows; ++m) {
        const float *row = w + (size_t)m * cols;
#if defined(__AVX2__)
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();
        int k = 0;
        for (; k + 31 < cols; k += 32) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + k),
                                 _mm256_loadu_ps(x + k), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + k + 8),
                                 _mm256_loadu_ps(x + k + 8), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + k + 16),
                                 _mm256_loadu_ps(x + k + 16), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + k + 24),
                                 _mm256_loadu_ps(x + k + 24), a3);
        }
        a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        __m128 lo = _mm256_castps256_ps128(a0);
        __m128 hi = _mm256_extractf128_ps(a0, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float sum = _mm_cvtss_f32(lo);
        for (; k < cols; ++k) {
            sum += row[k] * x[k];
        }
#else
        float sum = 0.0f;
        for (int k = 0; k < cols; ++k) {
            sum += row[k] * x[k];
        }
#endif
        out[m] = sum;
    }
}

/* Same, with FP8 blockwise weights dequantised inline.
 *
 * `scales` holds one fp32 per 128 weights, laid out row-major to match `w`, so
 * row m's scales start at m * (cols / 128). Rows are padded to a whole number
 * of blocks by the loader, which is why cols is required to be a multiple of
 * GEN9_FP8_BLOCK here rather than handled with a tail. */
void gen9_gemv_fp8(const uint8_t *w, const float *scales, const float *x,
                   float *out, int rows, int cols)
{
    gen9_fp8_init();
    const int blocks = cols / GEN9_FP8_BLOCK;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < rows; ++m) {
        const uint8_t *row = w + (size_t)m * cols;
        const float *row_scales = scales + (size_t)m * blocks;
        float sum = 0.0f;
        for (int b = 0; b < blocks; ++b) {
            const uint8_t *chunk = row + (size_t)b * GEN9_FP8_BLOCK;
            const float *xv = x + (size_t)b * GEN9_FP8_BLOCK;
            float block_sum = 0.0f;
            /* The dequant table lookup defeats vectorised loads of the weights,
             * so the inner loop is scalar over the byte stream and vectorised
             * only in the accumulate. It remains bandwidth-bound: one byte per
             * weight is a quarter of the traffic of the fp32 path. */
            for (int i = 0; i < GEN9_FP8_BLOCK; ++i) {
                block_sum += gen9_fp8_to_f32(chunk[i]) * xv[i];
            }
            sum += block_sum * row_scales[b];
        }
        out[m] = sum;
    }
}

static void silu_mul(float *hidden, const float *other, int n)
{
    for (int i = 0; i < n; ++i) {
        const float v = hidden[i];
        hidden[i] = (v / (1.0f + expf(-v))) * other[i];
    }
}

/* One DeepSeek routed expert, fp32: down @ (silu(gate @ x) * (up @ x)).
 *
 * `scratch` must hold at least 2 * intermediate floats; the caller owns it so
 * a node running experts back to back does not malloc per token. */
void gen9_expert_f32(const float *gate_w, const float *up_w,
                     const float *down_w, const float *x, float *out,
                     float *scratch, int hidden, int intermediate)
{
    float *g = scratch;
    float *u = scratch + intermediate;
    gen9_gemv_f32(gate_w, x, g, intermediate, hidden);
    gen9_gemv_f32(up_w, x, u, intermediate, hidden);
    silu_mul(g, u, intermediate);
    gen9_gemv_f32(down_w, g, out, hidden, intermediate);
}

/* The same expert with FP8 blockwise weights. */
void gen9_expert_fp8(const uint8_t *gate_w, const float *gate_s,
                     const uint8_t *up_w, const float *up_s,
                     const uint8_t *down_w, const float *down_s,
                     const float *x, float *out, float *scratch, int hidden,
                     int intermediate)
{
    float *g = scratch;
    float *u = scratch + intermediate;
    gen9_gemv_fp8(gate_w, gate_s, x, g, intermediate, hidden);
    gen9_gemv_fp8(up_w, up_s, x, u, intermediate, hidden);
    silu_mul(g, u, intermediate);
    gen9_gemv_fp8(down_w, down_s, g, out, hidden, intermediate);
}

/* Accumulate `gate * expert(x)` for a batch of experts into `out`.
 *
 * Summing on the node is what keeps a layer at one reply per console instead of
 * one per expert; see gen9_cluster.protocol.ExpertBatch. */
void gen9_expert_batch_f32(const float *const *gate_w, const float *const *up_w,
                           const float *const *down_w, const float *gates,
                           int n_experts, const float *x, float *out,
                           float *scratch, int hidden, int intermediate)
{
    float *partial = scratch + 2 * intermediate;
    memset(out, 0, sizeof(float) * (size_t)hidden);
    for (int e = 0; e < n_experts; ++e) {
        gen9_expert_f32(gate_w[e], up_w[e], down_w[e], x, partial, scratch,
                        hidden, intermediate);
        const float g = gates[e];
        for (int i = 0; i < hidden; ++i) {
            out[i] += g * partial[i];
        }
    }
}

#ifdef __cplusplus
}
#endif
