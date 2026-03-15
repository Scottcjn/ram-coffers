/*
 * neon-collapse.h — Non-Bijunctive Collapse via ARM NEON
 *
 * Elyan Labs — Apple Silicon PSE
 *
 * This is the core collapse engine. On POWER8 we used vec_perm to do
 * dual-source byte permutation in 1 cycle. On ARM we use:
 *
 *   vqtbl1q_u8  — Single-source 16-byte table lookup (16 → 16)
 *   vqtbl2q_u8  — Dual-source 32-byte table lookup (32 → 16)
 *
 * Both are non-bijunctive: multiple outputs can read the same source byte
 * (amplification/Hebbian strengthening), and source bytes can be skipped
 * (pruning/waste removal). This is the hardware-native Hebbian mechanism.
 *
 * The collapse pipeline:
 *   1. Generate entropy-seeded permute pattern (from cntvct_el0)
 *   2. Apply vqtbl2q to shuffle attention scores (prune + amplify)
 *   3. Threshold mask: zero weak survivors
 *   4. Scale strong survivors by amplification factor
 */

#ifndef NEON_COLLAPSE_H
#define NEON_COLLAPSE_H

#include "apple-pse-config.h"

#if PSE_HAS_NEON
#include <arm_neon.h>

/* ══════════════════════════════════════════════════════════
 * Pattern Generation — Non-Bijunctive Permute Indices
 *
 * Creates a 16-byte index vector where:
 *  - Multiple outputs can point to same source (duplication/amplify)
 *  - Some sources get no references (pruning)
 *  - 70% probability of selecting "strong" indices (0-7)
 *  - 30% probability of selecting "weak" indices (8-15)
 *
 * This IS the Hebbian rule in hardware:
 *  "Cells that fire together wire together"
 *  → Strong activations get duplicated (fire together)
 *  → Weak activations get pruned (don't fire, don't wire)
 * ══════════════════════════════════════════════════════════ */

static inline uint8x16_t neon_generate_collapse_pattern(
    int layer_id, int head_id, int position)
{
    uint64_t tb = pse_read_timebase();
    uint32_t seed = (uint32_t)(tb ^ (tb >> 32));
    seed ^= layer_id * PSE_PHI;
    seed ^= head_id * (PSE_PHI >> 8);
    seed ^= position * (PSE_PHI >> 16);

    uint8_t pattern[16];
    for (int i = 0; i < 16; i++) {
        seed = pse_xorshift32(seed);

        /* Non-bijunctive bias: favor strong indices */
        uint8_t idx;
        if ((seed & 0xFF) < PSE_STRONG_BIAS) {
            /* 70%: pick from strong range (0-7) → amplification */
            idx = (seed >> 8) & 0x07;
        } else {
            /* 30%: pick from full range (0-15) → some pruning */
            idx = (seed >> 8) & 0x0F;
        }
        pattern[i] = idx;
    }

    return vld1q_u8(pattern);
}

/* Dual-source pattern: indices 0-15 → first vector, 16-31 → second vector */
static inline uint8x16_t neon_generate_dual_collapse_pattern(
    int layer_id, int head_id, int position)
{
    uint64_t tb = pse_read_timebase();
    uint32_t seed = (uint32_t)(tb ^ (tb >> 32));
    seed ^= layer_id * PSE_PHI;
    seed ^= head_id * (PSE_PHI >> 8);
    seed ^= position * (PSE_PHI >> 16);

    uint8_t pattern[16];
    for (int i = 0; i < 16; i++) {
        seed = pse_xorshift32(seed);

        uint8_t idx;
        if ((seed & 0xFF) < PSE_STRONG_BIAS) {
            idx = (seed >> 8) % 8;       /* Strong from first vector */
        } else {
            idx = (seed >> 8) % 32;      /* Any byte from either vector */
        }
        pattern[i] = idx;
    }

    return vld1q_u8(pattern);
}

/* ══════════════════════════════════════════════════════════
 * Single-Source Collapse (vqtbl1q_u8)
 *
 * Takes one 16-byte vector, permutes non-bijunctively.
 * Equivalent to POWER8: vec_perm(v, v, pattern)
 *
 * Out-of-range indices (>15) produce 0 — natural pruning!
 * ══════════════════════════════════════════════════════════ */

static inline uint8x16_t neon_collapse_single(
    uint8x16_t input, uint8x16_t pattern)
{
    return vqtbl1q_u8(input, pattern);
}

/* ══════════════════════════════════════════════════════════
 * Dual-Source Collapse (vqtbl2q_u8)
 *
 * Takes TWO 16-byte vectors (32 bytes total), permutes into one.
 * Equivalent to POWER8: vec_perm(v_a, v_b, pattern)
 *
 * This is the TRUE non-bijunctive collapse:
 *  - 32 source bytes → 16 output bytes
 *  - Duplication: multiple outputs from same source
 *  - Pruning: source bytes with no references disappear
 *  - Index 0-15 → first vector, 16-31 → second vector
 * ══════════════════════════════════════════════════════════ */

static inline uint8x16_t neon_collapse_dual(
    uint8x16_t va, uint8x16_t vb, uint8x16_t pattern)
{
    uint8x16x2_t table = {{ va, vb }};
    return vqtbl2q_u8(table, pattern);
}

/* ══════════════════════════════════════════════════════════
 * Float-Level Collapse Operations
 *
 * The byte-level collapse above works on raw bytes. For attention
 * scores (float32), we need float-aware operations.
 *
 * Strategy: Reinterpret float vectors as bytes, collapse, reinterpret back.
 * This shuffles ENTIRE floats (4 bytes each) rather than individual bytes.
 * ══════════════════════════════════════════════════════════ */

/* Generate float-aligned pattern (indices in steps of 4) */
static inline uint8x16_t neon_generate_float_pattern(
    int layer_id, int head_id, int position)
{
    uint64_t tb = pse_read_timebase();
    uint32_t seed = (uint32_t)(tb ^ (tb >> 32));
    seed ^= layer_id * PSE_PHI;
    seed ^= head_id * (PSE_PHI >> 8);
    seed ^= position * (PSE_PHI >> 16);

    uint8_t pattern[16];
    for (int i = 0; i < 4; i++) {  /* 4 floats per vector */
        seed = pse_xorshift32(seed);

        /* Which float to select (0-3 for single source, 0-7 for dual) */
        int float_idx;
        if ((seed & 0xFF) < PSE_STRONG_BIAS) {
            float_idx = (seed >> 8) % 2;  /* Favor first 2 floats (strong) */
        } else {
            float_idx = (seed >> 8) % 4;  /* Any of 4 floats */
        }

        /* Fill 4 bytes for this float position */
        int base = float_idx * 4;
        pattern[i * 4 + 0] = base + 0;
        pattern[i * 4 + 1] = base + 1;
        pattern[i * 4 + 2] = base + 2;
        pattern[i * 4 + 3] = base + 3;
    }

    return vld1q_u8(pattern);
}

/* Generate float-aligned dual-source pattern (0-7 floats) */
static inline uint8x16_t neon_generate_float_dual_pattern(
    int layer_id, int head_id, int position)
{
    uint64_t tb = pse_read_timebase();
    uint32_t seed = (uint32_t)(tb ^ (tb >> 32));
    seed ^= layer_id * PSE_PHI;
    seed ^= head_id * (PSE_PHI >> 8);
    seed ^= position * (PSE_PHI >> 16);

    uint8_t pattern[16];
    for (int i = 0; i < 4; i++) {
        seed = pse_xorshift32(seed);

        int float_idx;
        if ((seed & 0xFF) < PSE_STRONG_BIAS) {
            float_idx = (seed >> 8) % 3;  /* Favor first 3 of 8 floats */
        } else {
            float_idx = (seed >> 8) % 8;  /* Any of 8 floats (across both vectors) */
        }

        int base = float_idx * 4;
        pattern[i * 4 + 0] = base + 0;
        pattern[i * 4 + 1] = base + 1;
        pattern[i * 4 + 2] = base + 2;
        pattern[i * 4 + 3] = base + 3;
    }

    return vld1q_u8(pattern);
}

/* Collapse float vector: permute + threshold + amplify */
static inline float32x4_t neon_collapse_float(
    float32x4_t input, uint8x16_t pattern, float threshold, float amplify)
{
    /* Step 1: Non-bijunctive permute (reinterpret as bytes) */
    uint8x16_t bytes_in = vreinterpretq_u8_f32(input);
    uint8x16_t bytes_out = vqtbl1q_u8(bytes_in, pattern);
    float32x4_t collapsed = vreinterpretq_f32_u8(bytes_out);

    /* Step 2: Threshold mask — prune weak activations */
    float32x4_t thresh_vec = vdupq_n_f32(threshold);
    float32x4_t abs_val = vabsq_f32(collapsed);
    uint32x4_t mask = vcgtq_f32(abs_val, thresh_vec);

    /* Step 3: Zero the weak, keep the strong */
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t pruned = vbslq_f32(mask, collapsed, zero);

    /* Step 4: Amplify survivors (Hebbian strengthening) */
    float32x4_t amp = vdupq_n_f32(amplify);
    return vmulq_f32(pruned, amp);
}

/* Dual-source float collapse: 8 floats → 4 floats */
static inline float32x4_t neon_collapse_float_dual(
    float32x4_t va, float32x4_t vb,
    uint8x16_t pattern, float threshold, float amplify)
{
    /* Pack both vectors as a 32-byte table */
    uint8x16_t ba = vreinterpretq_u8_f32(va);
    uint8x16_t bb = vreinterpretq_u8_f32(vb);
    uint8x16x2_t table = {{ ba, bb }};

    /* Non-bijunctive dual-source permute */
    uint8x16_t bytes_out = vqtbl2q_u8(table, pattern);
    float32x4_t collapsed = vreinterpretq_f32_u8(bytes_out);

    /* Threshold + amplify */
    float32x4_t thresh_vec = vdupq_n_f32(threshold);
    float32x4_t abs_val = vabsq_f32(collapsed);
    uint32x4_t mask = vcgtq_f32(abs_val, thresh_vec);

    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t pruned = vbslq_f32(mask, collapsed, zero);

    float32x4_t amp = vdupq_n_f32(amplify);
    return vmulq_f32(pruned, amp);
}

/* ══════════════════════════════════════════════════════════
 * Batch Collapse — Process attention score arrays
 *
 * This is the main entry point for collapsing attention scores.
 * Processes 4 floats at a time using NEON, with scalar fallback.
 * ══════════════════════════════════════════════════════════ */

static inline void neon_collapse_scores(
    float * restrict scores,
    int n,
    int layer_id,
    int head_id,
    int position)
{
    if (n < 4) return;

    uint8x16_t pattern = neon_generate_float_pattern(layer_id, head_id, position);

    float32x4_t thresh_vec = vdupq_n_f32(PSE_PRUNE_THRESH);
    float32x4_t amp_vec = vdupq_n_f32(PSE_AMPLIFY);
    float32x4_t zero = vdupq_n_f32(0.0f);

    int i = 0;

    /* Process 16 floats at a time (4 NEON vectors) */
    for (; i + 15 < n; i += 16) {
        /* Prefetch next cache line */
        __builtin_prefetch(&scores[i + 32], 0, 3);

        float32x4_t v0 = vld1q_f32(&scores[i + 0]);
        float32x4_t v1 = vld1q_f32(&scores[i + 4]);
        float32x4_t v2 = vld1q_f32(&scores[i + 8]);
        float32x4_t v3 = vld1q_f32(&scores[i + 12]);

        /* Dual-source collapse: v0+v1, v1+v2, v2+v3, v3+v0 */
        uint8x16_t dp = neon_generate_float_dual_pattern(layer_id, head_id, position + i);

        uint8x16x2_t t01 = {{ vreinterpretq_u8_f32(v0), vreinterpretq_u8_f32(v1) }};
        uint8x16x2_t t12 = {{ vreinterpretq_u8_f32(v1), vreinterpretq_u8_f32(v2) }};
        uint8x16x2_t t23 = {{ vreinterpretq_u8_f32(v2), vreinterpretq_u8_f32(v3) }};
        uint8x16x2_t t30 = {{ vreinterpretq_u8_f32(v3), vreinterpretq_u8_f32(v0) }};

        float32x4_t c0 = vreinterpretq_f32_u8(vqtbl2q_u8(t01, dp));
        float32x4_t c1 = vreinterpretq_f32_u8(vqtbl2q_u8(t12, dp));
        float32x4_t c2 = vreinterpretq_f32_u8(vqtbl2q_u8(t23, dp));
        float32x4_t c3 = vreinterpretq_f32_u8(vqtbl2q_u8(t30, dp));

        /* Threshold mask */
        uint32x4_t m0 = vcgtq_f32(vabsq_f32(c0), thresh_vec);
        uint32x4_t m1 = vcgtq_f32(vabsq_f32(c1), thresh_vec);
        uint32x4_t m2 = vcgtq_f32(vabsq_f32(c2), thresh_vec);
        uint32x4_t m3 = vcgtq_f32(vabsq_f32(c3), thresh_vec);

        /* Prune + amplify */
        c0 = vmulq_f32(vbslq_f32(m0, c0, zero), amp_vec);
        c1 = vmulq_f32(vbslq_f32(m1, c1, zero), amp_vec);
        c2 = vmulq_f32(vbslq_f32(m2, c2, zero), amp_vec);
        c3 = vmulq_f32(vbslq_f32(m3, c3, zero), amp_vec);

        vst1q_f32(&scores[i + 0],  c0);
        vst1q_f32(&scores[i + 4],  c1);
        vst1q_f32(&scores[i + 8],  c2);
        vst1q_f32(&scores[i + 12], c3);
    }

    /* Scalar remainder */
    for (; i < n; i++) {
        float val = scores[i];
        if (val > PSE_PRUNE_THRESH) {
            scores[i] = val * PSE_AMPLIFY;
        } else if (val < -PSE_PRUNE_THRESH) {
            scores[i] = val * PSE_AMPLIFY;
        } else {
            scores[i] = 0.0f;
        }
    }
}

/* ══════════════════════════════════════════════════════════
 * Top-K Collapse — Only collapse top-K scoring elements
 *
 * More selective than full collapse. Finds approximate top-K,
 * amplifies those, prunes everything else.
 * ══════════════════════════════════════════════════════════ */

static inline float neon_approx_topk_threshold(
    const float * restrict scores, int n, int k)
{
    /* Quick reservoir sample to estimate k-th largest */
    if (n <= k) return -1e30f;

    float samples[32];
    int ns = (n < 32) ? n : 32;

    /* Stride sampling for representative distribution */
    int stride = n / ns;
    for (int i = 0; i < ns; i++) {
        samples[i] = scores[i * stride];
    }

    /* Simple insertion sort on small sample */
    for (int i = 1; i < ns; i++) {
        float key = samples[i];
        int j = i - 1;
        while (j >= 0 && samples[j] < key) {
            samples[j + 1] = samples[j];
            j--;
        }
        samples[j + 1] = key;
    }

    /* Estimate threshold from sample */
    int sample_k = (k * ns) / n;
    if (sample_k >= ns) sample_k = ns - 1;
    return samples[sample_k];
}

static inline void neon_topk_collapse(
    float * restrict scores,
    int n,
    int top_k,
    int layer_id,
    int head_id,
    int position)
{
    if (n < 4) return;

    float threshold = neon_approx_topk_threshold(scores, n, top_k);
    uint8x16_t pattern = neon_generate_float_pattern(layer_id, head_id, position);

    float32x4_t thresh_vec = vdupq_n_f32(threshold);
    float32x4_t amp_vec = vdupq_n_f32(PSE_AMPLIFY);
    float32x4_t zero = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 3 < n; i += 4) {
        __builtin_prefetch(&scores[i + 16], 0, 3);

        float32x4_t v = vld1q_f32(&scores[i]);

        /* Collapse permute */
        uint8x16_t bv = vreinterpretq_u8_f32(v);
        float32x4_t collapsed = vreinterpretq_f32_u8(vqtbl1q_u8(bv, pattern));

        /* Top-K mask: keep only above threshold */
        uint32x4_t mask = vcgtq_f32(collapsed, thresh_vec);
        float32x4_t pruned = vbslq_f32(mask, collapsed, zero);

        /* Amplify survivors */
        vst1q_f32(&scores[i], vmulq_f32(pruned, amp_vec));
    }

    /* Scalar tail */
    for (; i < n; i++) {
        if (scores[i] >= threshold) {
            scores[i] *= PSE_AMPLIFY;
        } else {
            scores[i] = 0.0f;
        }
    }
}

#endif /* PSE_HAS_NEON */
#endif /* NEON_COLLAPSE_H */
