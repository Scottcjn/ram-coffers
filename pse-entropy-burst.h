/*
 * pse-entropy-burst.h - Incremental Burst PSE Entropy for llama.cpp
 *
 * Optimized entropy injection that maintains VSX speed:
 * - Apply entropy every N tokens (not every token)
 * - Only perturb top-K candidates (not full 32K vocab)
 * - Stronger bursts compensate for lower frequency
 *
 * Result: Same behavioral divergence, minimal performance impact.
 */

#ifndef PSE_ENTROPY_BURST_H
#define PSE_ENTROPY_BURST_H

#include <stdint.h>
#include <cstdio>
#include <algorithm>

/*===========================================================================
 * Configuration
 *===========================================================================*/

/* Burst interval: Apply entropy every N tokens (1 = every token, 4 = every 4th) */
#ifndef PSE_BURST_INTERVAL
#define PSE_BURST_INTERVAL 4
#endif

/* Burst strength: Multiplied by interval to maintain total effect */
#ifndef PSE_BURST_STRENGTH
#define PSE_BURST_STRENGTH 0.08f  /* 4x normal (0.02) for interval=4 */
#endif

/* Top-K for entropy: Only perturb top candidates (0 = all) */
#ifndef PSE_TOPK_ENTROPY
#define PSE_TOPK_ENTROPY 512  /* Top 512 candidates get entropy */
#endif

/* Enable collapse resonance points (vec_perm activation) */
#ifndef PSE_COLLAPSE_ENABLED
#define PSE_COLLAPSE_ENABLED 1
#endif

/* Collapse interval: Heavy collapse every N tokens */
#ifndef PSE_COLLAPSE_INTERVAL
#define PSE_COLLAPSE_INTERVAL 16
#endif

/* Golden ratio for mixing */
#define PSE_PHI 0x9E3779B9U

/*===========================================================================
 * State Tracking
 *===========================================================================*/

static int64_t g_pse_token_pos = 0;
static uint64_t g_pse_bursts = 0;
static uint64_t g_pse_collapses = 0;
static bool g_pse_banner_shown = false;

/*===========================================================================
 * Hardware Timebase
 *===========================================================================*/

static inline uint64_t pse_read_timebase(void) {
#if defined(__powerpc64__) || defined(__powerpc__)
    uint64_t tb;
    __asm__ __volatile__("mftb %0" : "=r"(tb));
    return tb;
#elif defined(__x86_64__) || defined(__i386__)
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#elif defined(__aarch64__)
    uint64_t val;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#else
    return (uint64_t)&g_pse_token_pos;
#endif
}

/*===========================================================================
 * Burst Entropy Application
 *===========================================================================*/

/*
 * Apply burst entropy to token candidates.
 * Only activates every PSE_BURST_INTERVAL tokens.
 * Only affects top PSE_TOPK_ENTROPY candidates.
 *
 * Call this in set_logits() after copying logits.
 */
template<typename T>
static inline void pse_apply_entropy_burst(T* cur, size_t n_vocab) {
    g_pse_token_pos++;

    /* Show banner on first token */
    if (!g_pse_banner_shown) {
        g_pse_banner_shown = true;
        fprintf(stderr, "\n");
        fprintf(stderr, "╔═══════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║  PSE Burst Entropy Active                             ║\n");
        fprintf(stderr, "║  Interval: %d | Strength: %.3f | TopK: %d           ║\n",
                PSE_BURST_INTERVAL, PSE_BURST_STRENGTH, PSE_TOPK_ENTROPY);
        fprintf(stderr, "║  Platform: %-42s║\n",
#if defined(__powerpc64__)
                "POWER8 (mftb timebase)"
#elif defined(__x86_64__)
                "x86_64 (rdtsc)"
#elif defined(__aarch64__)
                "ARM64 (cntvct)"
#else
                "Generic"
#endif
        );
        fprintf(stderr, "║  Speed-optimized behavioral divergence                ║\n");
        fprintf(stderr, "╚═══════════════════════════════════════════════════════╝\n");
        fprintf(stderr, "\n");
    }

    /* Skip non-burst tokens for speed */
    if (g_pse_token_pos % PSE_BURST_INTERVAL != 0) {
        return;
    }

    if (!cur || n_vocab == 0) return;

    g_pse_bursts++;

    uint64_t tb = pse_read_timebase();
    uint32_t base_seed = (uint32_t)(tb ^ (tb >> 32));
    base_seed ^= (uint32_t)g_pse_token_pos * PSE_PHI;

    /* Determine how many candidates to affect */
    size_t entropy_count = (PSE_TOPK_ENTROPY > 0 && PSE_TOPK_ENTROPY < n_vocab)
                           ? PSE_TOPK_ENTROPY : n_vocab;

    /* Apply burst entropy to top candidates */
    for (size_t i = 0; i < entropy_count; i++) {
        uint32_t seed = base_seed ^ ((uint32_t)i * PSE_PHI);
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;

        float entropy = ((float)(seed & 0xFFFFU) / 65536.0f) - 0.5f;
        cur[i].logit += entropy * PSE_BURST_STRENGTH;
    }

#if PSE_COLLAPSE_ENABLED
    /* Heavy collapse at resonance points */
    if (g_pse_token_pos % PSE_COLLAPSE_INTERVAL == 0) {
        pse_apply_collapse_resonance(cur, entropy_count, base_seed);
        g_pse_collapses++;
    }
#endif
}

/*===========================================================================
 * Collapse Resonance (vec_perm style waste removal)
 *===========================================================================*/

#if PSE_COLLAPSE_ENABLED
/*
 * Apply collapse at resonance points.
 * Amplifies "hot" candidates (high logit) and prunes "cold" ones.
 * Non-bijunctive transformation - creates emergent patterns.
 */
template<typename T>
static inline void pse_apply_collapse_resonance(T* cur, size_t count, uint32_t seed) {
    if (count < 8) return;

    /* Find mean logit for hot/cold threshold */
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        sum += cur[i].logit;
    }
    float mean = sum / count;

    /* Entropy-seeded threshold adjustment */
    float threshold_adjust = ((float)(seed & 0xFF) / 256.0f - 0.5f) * 0.1f;
    float threshold = mean + threshold_adjust;

    /* Collapse: Amplify hot, dampen cold (non-bijunctive) */
    for (size_t i = 0; i < count; i++) {
        if (cur[i].logit > threshold) {
            /* Hot path: Amplify (dup effect from vec_perm) */
            cur[i].logit *= 1.05f;
        } else {
            /* Cold path: Dampen (prune effect) */
            cur[i].logit *= 0.95f;
        }
    }
}
#endif

/*===========================================================================
 * Reset and Metrics
 *===========================================================================*/

static inline void pse_reset(void) {
    g_pse_token_pos = 0;
    g_pse_bursts = 0;
    g_pse_collapses = 0;
}

static inline void pse_report_metrics(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  PSE Burst Entropy Metrics                            ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Tokens processed: %12ld                       ║\n",
            (long)g_pse_token_pos);
    fprintf(stderr, "║  Entropy bursts:   %12lu (every %d tokens)     ║\n",
            (unsigned long)g_pse_bursts, PSE_BURST_INTERVAL);
    fprintf(stderr, "║  Collapse events:  %12lu (every %d tokens)    ║\n",
            (unsigned long)g_pse_collapses, PSE_COLLAPSE_INTERVAL);
    fprintf(stderr, "║  Burst strength:   %12.4f                       ║\n",
            PSE_BURST_STRENGTH);
    fprintf(stderr, "║  TopK affected:    %12d                       ║\n",
            PSE_TOPK_ENTROPY);
    fprintf(stderr, "╚═══════════════════════════════════════════════════════╝\n");
}

/*===========================================================================
 * Compatibility alias for simple header
 *===========================================================================*/

/* Drop-in replacement for pse_apply_entropy */
template<typename T>
static inline void pse_apply_entropy(T* cur, size_t n_vocab) {
    pse_apply_entropy_burst(cur, n_vocab);
}

#endif /* PSE_ENTROPY_BURST_H */
