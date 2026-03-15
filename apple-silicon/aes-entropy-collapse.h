/*
 * aes-entropy-collapse.h — AES-Based Entropy Injection for Apple Silicon
 *
 * Elyan Labs — Apple Silicon PSE
 *
 * On POWER8, vcipher performs a single AES encryption round in 1 cycle,
 * mixing 128 bits of state through SubBytes → ShiftRows → MixColumns → AddRoundKey.
 * This provides cryptographic-quality diffusion — perfect for entropy injection.
 *
 * Apple Silicon has dedicated AES hardware via ARM Crypto Extensions:
 *   vaeseq_u8  — AES single round encrypt (SubBytes + ShiftRows + AddRoundKey)
 *   vaesmcq_u8 — AES MixColumns
 *   vaesdq_u8  — AES single round decrypt
 *   vaesimcq_u8 — AES InverseMixColumns
 *
 * Combined: vaeseq + vaesmcq = one full AES round = vcipher equivalent.
 *
 * Apple M-series chips have a dedicated AES engine that runs these at
 * essentially 1 cycle throughput — same as POWER8's vcipher.
 *
 * We use this for:
 *   1. Entropy mixing: cntvct_el0 timebase → AES diffusion → uniform entropy
 *   2. Pattern scrambling: Layer/head/position → AES → permute indices
 *   3. Logit perturbation: Controlled noise injection for behavioral divergence
 */

#ifndef AES_ENTROPY_COLLAPSE_H
#define AES_ENTROPY_COLLAPSE_H

#include "apple-pse-config.h"

#if PSE_HAS_NEON && PSE_HAS_AES
#include <arm_neon.h>

/* ══════════════════════════════════════════════════════════
 * AES Round — Full Diffusion in 2 Instructions
 *
 * vaeseq(state, key) does SubBytes + ShiftRows + XOR with key
 * vaesmcq(state)     does MixColumns
 * Together = one full AES round = cryptographic diffusion
 *
 * On Apple M2, these run on dedicated hardware at 1 cycle each.
 * ══════════════════════════════════════════════════════════ */

static inline uint8x16_t pse_aes_round(uint8x16_t state, uint8x16_t key) {
    state = vaeseq_u8(state, key);   /* SubBytes + ShiftRows + AddRoundKey */
    state = vaesmcq_u8(state);        /* MixColumns */
    return state;
}

/* Multiple rounds for stronger diffusion */
static inline uint8x16_t pse_aes_mix(uint8x16_t state, uint8x16_t key, int rounds) {
    for (int i = 0; i < rounds; i++) {
        state = vaeseq_u8(state, key);
        state = vaesmcq_u8(state);
    }
    return state;
}

/* ══════════════════════════════════════════════════════════
 * Entropy State — Seeded from Hardware Counter
 *
 * The cntvct_el0 counter provides hardware entropy (oscillator drift).
 * We mix it through AES to get uniform, high-quality random state.
 * ══════════════════════════════════════════════════════════ */

typedef struct {
    uint8x16_t state;       /* Current AES state */
    uint8x16_t key;         /* AES round key (context-dependent) */
    uint64_t   counter;     /* Monotonic counter for uniqueness */
    int        initialized; /* Guard flag */
} pse_aes_entropy_t;

static pse_aes_entropy_t g_aes_entropy = { .initialized = 0 };

static inline void pse_aes_entropy_init(int layer_id, int head_id) {
    uint64_t tb = pse_read_timebase();

    /* Build initial state from timebase + context */
    uint8_t state_bytes[16] = {0};
    uint8_t key_bytes[16] = {0};

    /* Pack timebase into state */
    for (int i = 0; i < 8; i++) {
        state_bytes[i] = (tb >> (i * 8)) & 0xFF;
    }
    /* Pack layer/head context */
    state_bytes[8]  = (layer_id >> 0) & 0xFF;
    state_bytes[9]  = (layer_id >> 8) & 0xFF;
    state_bytes[10] = (head_id >> 0) & 0xFF;
    state_bytes[11] = (head_id >> 8) & 0xFF;

    /* Key from golden ratio constants */
    uint32_t k0 = PSE_PHI;
    uint32_t k1 = PSE_PHI ^ (layer_id * 0x61C88647);
    uint32_t k2 = PSE_PHI ^ (head_id * 0x517CC1B7);
    uint32_t k3 = k0 ^ k1 ^ k2;
    for (int i = 0; i < 4; i++) {
        key_bytes[i]      = (k0 >> (i * 8)) & 0xFF;
        key_bytes[i + 4]  = (k1 >> (i * 8)) & 0xFF;
        key_bytes[i + 8]  = (k2 >> (i * 8)) & 0xFF;
        key_bytes[i + 12] = (k3 >> (i * 8)) & 0xFF;
    }

    g_aes_entropy.state = vld1q_u8(state_bytes);
    g_aes_entropy.key   = vld1q_u8(key_bytes);
    g_aes_entropy.counter = 0;
    g_aes_entropy.initialized = 1;
}

/* Generate next 16 bytes of AES entropy */
static inline uint8x16_t pse_aes_next(void) {
    if (!g_aes_entropy.initialized) {
        pse_aes_entropy_init(0, 0);
    }

    /* Mix in fresh timebase + counter for uniqueness */
    uint64_t tb = pse_read_timebase();
    uint8_t fresh[16];
    for (int i = 0; i < 8; i++) {
        fresh[i] = (tb >> (i * 8)) & 0xFF;
    }
    uint64_t ctr = g_aes_entropy.counter++;
    for (int i = 0; i < 8; i++) {
        fresh[i + 8] = (ctr >> (i * 8)) & 0xFF;
    }

    /* XOR fresh entropy into state */
    uint8x16_t fresh_vec = vld1q_u8(fresh);
    g_aes_entropy.state = veorq_u8(g_aes_entropy.state, fresh_vec);

    /* Run AES rounds for diffusion */
    g_aes_entropy.state = pse_aes_mix(
        g_aes_entropy.state, g_aes_entropy.key, PSE_AES_ROUNDS);

    return g_aes_entropy.state;
}

/* ══════════════════════════════════════════════════════════
 * AES-Seeded Collapse Pattern
 *
 * Instead of XorShift for pattern generation, use AES diffusion.
 * Much higher entropy quality — no linear artifacts.
 * ══════════════════════════════════════════════════════════ */

static inline uint8x16_t pse_aes_collapse_pattern(
    int layer_id, int head_id, int position)
{
    pse_aes_entropy_init(layer_id, head_id);

    /* Mix position into state */
    uint8_t pos_bytes[16] = {0};
    for (int i = 0; i < 4; i++) {
        pos_bytes[i] = (position >> (i * 8)) & 0xFF;
    }
    g_aes_entropy.state = veorq_u8(g_aes_entropy.state, vld1q_u8(pos_bytes));

    /* Generate entropy via AES */
    uint8x16_t entropy = pse_aes_next();

    /* Apply non-bijunctive bias */
    uint8_t raw[16], pattern[16];
    vst1q_u8(raw, entropy);

    for (int i = 0; i < 16; i++) {
        if (raw[i] < PSE_STRONG_BIAS) {
            /* Favor strong indices: 0-7 */
            pattern[i] = raw[(i + 1) % 16] & 0x07;
        } else {
            /* Full range: 0-15 */
            pattern[i] = raw[(i + 1) % 16] & 0x0F;
        }
    }

    return vld1q_u8(pattern);
}

/* Float-aligned AES pattern (groups of 4 bytes) */
static inline uint8x16_t pse_aes_float_pattern(
    int layer_id, int head_id, int position)
{
    uint8x16_t entropy = pse_aes_next();
    uint8_t raw[16], pattern[16];
    vst1q_u8(raw, entropy);

    for (int i = 0; i < 4; i++) {
        int float_idx;
        if (raw[i] < PSE_STRONG_BIAS) {
            float_idx = raw[i + 4] % 2;   /* Strong: first 2 floats */
        } else {
            float_idx = raw[i + 4] % 4;   /* Any float */
        }
        int base = float_idx * 4;
        pattern[i * 4 + 0] = base + 0;
        pattern[i * 4 + 1] = base + 1;
        pattern[i * 4 + 2] = base + 2;
        pattern[i * 4 + 3] = base + 3;
    }

    return vld1q_u8(pattern);
}

/* ══════════════════════════════════════════════════════════
 * AES Entropy Burst — Logit Perturbation
 *
 * Same concept as POWER8 pse-entropy-burst.h but using AES
 * for higher-quality entropy. Applied every N tokens to the
 * top-K logit candidates.
 *
 * This creates behavioral divergence: same model, same prompt,
 * different outputs each run (measurable via PSE markers).
 * ══════════════════════════════════════════════════════════ */

static int g_aes_token_pos = 0;

static inline void pse_aes_entropy_burst(
    float * restrict logits,
    int n_vocab)
{
    g_aes_token_pos++;

    /* Only apply every N tokens */
    if (g_aes_token_pos % PSE_BURST_INTERVAL != 0) return;

    /* Generate AES entropy block */
    uint8x16_t entropy_block = pse_aes_next();
    uint8_t entropy_bytes[16];
    vst1q_u8(entropy_bytes, entropy_block);

    /* Apply to top-K candidates */
    int count = (PSE_BURST_TOPK > 0 && PSE_BURST_TOPK < n_vocab)
              ? PSE_BURST_TOPK : n_vocab;

    /* NEON-accelerated perturbation: 4 logits at a time */
    int i = 0;
    float32x4_t strength = vdupq_n_f32(PSE_BURST_STRENGTH);
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t scale = vdupq_n_f32(1.0f / 65536.0f);

    for (; i + 3 < count; i += 4) {
        /* Generate fresh entropy for each group */
        if ((i & 0x0F) == 0) {
            entropy_block = pse_aes_next();
            vst1q_u8(entropy_bytes, entropy_block);
        }

        /* Convert 4 pairs of entropy bytes to float perturbations */
        int eb = (i & 0x0C);  /* Offset within 16-byte block */
        uint32_t e0 = (uint32_t)entropy_bytes[(eb + 0) % 16] << 8 | entropy_bytes[(eb + 1) % 16];
        uint32_t e1 = (uint32_t)entropy_bytes[(eb + 2) % 16] << 8 | entropy_bytes[(eb + 3) % 16];
        uint32_t e2 = (uint32_t)entropy_bytes[(eb + 4) % 16] << 8 | entropy_bytes[(eb + 5) % 16];
        uint32_t e3 = (uint32_t)entropy_bytes[(eb + 6) % 16] << 8 | entropy_bytes[(eb + 7) % 16];

        float32x4_t raw_entropy;
        float vals[4] = {(float)e0, (float)e1, (float)e2, (float)e3};
        raw_entropy = vld1q_f32(vals);

        /* Scale to [-0.5, 0.5] range, then apply strength */
        float32x4_t perturbation = vmulq_f32(
            vsubq_f32(vmulq_f32(raw_entropy, scale), half),
            strength);

        /* Apply to logits */
        float32x4_t current = vld1q_f32(&logits[i]);
        vst1q_f32(&logits[i], vaddq_f32(current, perturbation));
    }

    /* Scalar remainder */
    for (; i < count; i++) {
        if ((i & 0x0F) == 0) {
            entropy_block = pse_aes_next();
            vst1q_u8(entropy_bytes, entropy_block);
        }
        uint16_t ev = (uint16_t)entropy_bytes[i % 16] << 8 | entropy_bytes[(i + 1) % 16];
        float perturbation = ((float)ev / 65536.0f - 0.5f) * PSE_BURST_STRENGTH;
        logits[i] += perturbation;
    }

    /* Heavy collapse at resonance points */
    if (g_aes_token_pos % PSE_COLLAPSE_INTERVAL == 0) {
        /* Find mean of top candidates */
        float sum = 0.0f;
        float32x4_t vsum = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 3 < count; j += 4) {
            vsum = vaddq_f32(vsum, vld1q_f32(&logits[j]));
        }
        /* Horizontal sum */
        float32x2_t s2 = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
        sum = vget_lane_f32(vpadd_f32(s2, s2), 0);
        for (; j < count; j++) sum += logits[j];
        float mean = sum / count;

        /* AES-seeded threshold adjustment */
        entropy_block = pse_aes_next();
        vst1q_u8(entropy_bytes, entropy_block);
        float thresh_adj = ((float)entropy_bytes[0] / 256.0f - 0.5f) * 0.1f;
        float threshold = mean + thresh_adj;

        /* Non-bijunctive resonance: amplify hot, dampen cold */
        float32x4_t hot_scale  = vdupq_n_f32(1.05f);
        float32x4_t cold_scale = vdupq_n_f32(0.95f);
        float32x4_t thresh_v   = vdupq_n_f32(threshold);

        j = 0;
        for (; j + 3 < count; j += 4) {
            float32x4_t v = vld1q_f32(&logits[j]);
            uint32x4_t hot_mask = vcgtq_f32(v, thresh_v);
            float32x4_t scaled = vbslq_f32(hot_mask,
                vmulq_f32(v, hot_scale),
                vmulq_f32(v, cold_scale));
            vst1q_f32(&logits[j], scaled);
        }
        for (; j < count; j++) {
            logits[j] *= (logits[j] > threshold) ? 1.05f : 0.95f;
        }
    }
}

/* ══════════════════════════════════════════════════════════
 * AES Collapse on Attention Scores
 *
 * Full pipeline: AES pattern → vqtbl2q collapse → threshold → amplify
 * Uses AES entropy instead of XorShift for pattern generation.
 * ══════════════════════════════════════════════════════════ */

static inline void pse_aes_collapse_scores(
    float * restrict scores,
    int n,
    int layer_id,
    int head_id,
    int position)
{
    if (n < 4) return;

    /* AES-quality permute pattern */
    uint8x16_t pattern = pse_aes_float_pattern(layer_id, head_id, position);

    float32x4_t thresh_vec = vdupq_n_f32(PSE_PRUNE_THRESH);
    float32x4_t amp_vec = vdupq_n_f32(PSE_AMPLIFY);
    float32x4_t zero = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 7 < n; i += 8) {
        __builtin_prefetch(&scores[i + 16], 0, 3);

        float32x4_t va = vld1q_f32(&scores[i]);
        float32x4_t vb = vld1q_f32(&scores[i + 4]);

        /* Dual-source AES collapse */
        uint8x16x2_t table = {{
            vreinterpretq_u8_f32(va),
            vreinterpretq_u8_f32(vb)
        }};
        uint8x16_t dp = pse_aes_float_pattern(layer_id, head_id, position + i);

        float32x4_t c0 = vreinterpretq_f32_u8(vqtbl2q_u8(table, dp));
        float32x4_t c1 = vreinterpretq_f32_u8(vqtbl2q_u8(table,
            pse_aes_float_pattern(layer_id, head_id, position + i + 4)));

        /* Threshold + amplify */
        uint32x4_t m0 = vcgtq_f32(vabsq_f32(c0), thresh_vec);
        uint32x4_t m1 = vcgtq_f32(vabsq_f32(c1), thresh_vec);

        c0 = vmulq_f32(vbslq_f32(m0, c0, zero), amp_vec);
        c1 = vmulq_f32(vbslq_f32(m1, c1, zero), amp_vec);

        vst1q_f32(&scores[i],     c0);
        vst1q_f32(&scores[i + 4], c1);
    }

    /* Scalar tail */
    for (; i < n; i++) {
        float val = scores[i];
        if (val > PSE_PRUNE_THRESH || val < -PSE_PRUNE_THRESH) {
            scores[i] = val * PSE_AMPLIFY;
        } else {
            scores[i] = 0.0f;
        }
    }
}

#endif /* PSE_HAS_NEON && PSE_HAS_AES */
#endif /* AES_ENTROPY_COLLAPSE_H */
