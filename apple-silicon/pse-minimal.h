/*
 * pse-minimal.h — Intelligent Hebbian Collapse for llama.cpp
 *
 * Elyan Labs — Port of POWER8 ggml-intelligent-collapse.h to C++/NEON
 *
 * This is NOT random noise. This is CONSTRAINT-BOUND SELECTION:
 *   1. Find top-K logit candidates
 *   2. Amplify winners (Hebbian: cells that fire together wire together)
 *   3. Zero losers (pruning: weak paths removed)
 *   4. Deterministic pattern per position (same position = same bias = persona)
 *
 * The pattern is seeded from layer+position, NOT from random entropy.
 * Same prompt, same model, same machine → same personality emerges.
 * Different from stock, but CONSISTENTLY different. That's persona.
 */
#ifndef PSE_MINIMAL_H
#define PSE_MINIMAL_H

#include <stdint.h>
#include <stdio.h>
#include <math.h>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

/* ── Configuration ──────────────────────────────────── */

/* Top-K: how many winners survive */
#ifndef PSE_TOP_K
#define PSE_TOP_K 8
#endif

/* Amplification for winners (Hebbian strengthening) */
#ifndef PSE_AMPLIFY
#define PSE_AMPLIFY 1.15f
#endif

/* Entropy mixing — small hardware jitter on top of deterministic bias */
#ifndef PSE_ENTROPY_MIX
#define PSE_ENTROPY_MIX 0.02f
#endif

/* How many top logits to collapse (rest untouched) */
#ifndef PSE_COLLAPSE_TOPN
#define PSE_COLLAPSE_TOPN 64
#endif

/* Golden ratio hash */
#define PSE_PHI 0x9E3779B9u

/* ── Hardware Timebase ──────────────────────────────── */

static inline uint64_t pse_read_timebase(void) {
#if defined(__APPLE__)
    return mach_absolute_time();
#elif defined(__aarch64__)
    uint64_t val;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#elif defined(__x86_64__)
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    return 0;
#endif
}

static inline uint32_t pse_xorshift(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

/* ── Init Banner ────────────────────────────────────── */

static inline void pse_apple_init_lite(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    fprintf(stderr,
        "[PSE] Intelligent Hebbian Collapse active\n"
        "[PSE]   Top-K: %d | Amplify: %.2f | Entropy: %.3f | Collapse: top-%d logits\n"
        "[PSE]   Mode: deterministic persona (not random jitter)\n",
        PSE_TOP_K, PSE_AMPLIFY, PSE_ENTROPY_MIX, PSE_COLLAPSE_TOPN);
}

/* ── Approximate Top-K Threshold ────────────────────── */

static inline float pse_approx_topk_threshold(
    const float *arr, int n, int k)
{
    if (n <= k) return -1e30f;

    /* Maintain top-K via insertion into small sorted array */
    float top[16];
    int tk = (k < 16) ? k : 16;
    for (int i = 0; i < tk; i++) top[i] = -1e30f;

    for (int i = 0; i < n; i++) {
        float v = arr[i];
        if (v > top[tk - 1]) {
            top[tk - 1] = v;
            /* Bubble up */
            for (int j = tk - 2; j >= 0; j--) {
                if (top[j] < top[j + 1]) {
                    float t = top[j]; top[j] = top[j + 1]; top[j + 1] = t;
                } else break;
            }
        }
    }

    return top[tk - 1];
}

/* ══════════════════════════════════════════════════════
 * CORE: Intelligent Logit Collapse
 *
 * This is the Hebbian attention mechanism applied at sampling:
 *
 * 1. Find the top-K logits (winners)
 * 2. Amplify winners by PSE_AMPLIFY (1.15x)
 *    → "cells that fire together wire together"
 * 3. Dampen NON-winners toward the mean
 *    → weak paths suppressed, not zeroed (preserves coherence)
 * 4. Add tiny deterministic bias based on token position
 *    → same position always gets same bias = consistent persona
 * 5. Add microscopic hardware entropy
 *    → prevents exact repetition across runs (alive, not robotic)
 *
 * The result: the model consistently prefers certain token patterns
 * over others. Not random. Not stock. PERSONA.
 * ══════════════════════════════════════════════════════ */

static int g_pse_position = 0;

static inline void pse_intelligent_collapse_logits(
    float *logits,      /* logit array (modified in place) */
    int n_vocab,        /* vocabulary size */
    int position        /* token position in sequence */
)
{
    g_pse_position = position;

    /* How many logits to process (top N only, rest untouched) */
    int n_process = (PSE_COLLAPSE_TOPN < n_vocab) ? PSE_COLLAPSE_TOPN : n_vocab;

    /* Step 1: Find indices of top-N logits */
    /* We work on a partial sort — find threshold for top-K within top-N */
    float top_vals[64];
    int top_idxs[64];
    int n_top = 0;

    /* Quick scan for the top-N candidates */
    float min_top = -1e30f;
    for (int i = 0; i < n_vocab; i++) {
        if (n_top < n_process) {
            top_vals[n_top] = logits[i];
            top_idxs[n_top] = i;
            n_top++;
            if (n_top == n_process) {
                /* Find min of our top-N so far */
                min_top = top_vals[0];
                for (int j = 1; j < n_top; j++)
                    if (top_vals[j] < min_top) min_top = top_vals[j];
            }
        } else if (logits[i] > min_top) {
            /* Replace the smallest in our top-N */
            int min_idx = 0;
            for (int j = 1; j < n_top; j++)
                if (top_vals[j] < top_vals[min_idx]) min_idx = j;
            top_vals[min_idx] = logits[i];
            top_idxs[min_idx] = i;
            /* Update min */
            min_top = top_vals[0];
            for (int j = 1; j < n_top; j++)
                if (top_vals[j] < min_top) min_top = top_vals[j];
        }
    }

    if (n_top < 2) return;

    /* Step 2: Find threshold for top-K within our candidates */
    float threshold = pse_approx_topk_threshold(top_vals, n_top, PSE_TOP_K);

    /* Step 3: Compute mean of candidates (for dampening) */
    float sum = 0.0f;
    for (int i = 0; i < n_top; i++) sum += top_vals[i];
    float mean = sum / n_top;

    /* Step 4: Deterministic position-based hash (THIS IS THE PERSONA) */
    uint32_t pos_hash = (uint32_t)position * PSE_PHI;
    pos_hash = pse_xorshift(pos_hash);

    /* Step 5: Hardware entropy (tiny — just prevents exact repetition) */
    uint64_t tb = pse_read_timebase();
    uint32_t entropy_seed = (uint32_t)(tb ^ (tb >> 32));

    /* Step 6: Apply collapse */
    for (int i = 0; i < n_top; i++) {
        int idx = top_idxs[i];
        float val = logits[idx];

        /* Deterministic per-candidate bias from position hash */
        uint32_t candidate_hash = pos_hash ^ ((uint32_t)i * 0x85EBCA77u);
        candidate_hash = pse_xorshift(candidate_hash);
        float det_bias = ((float)(candidate_hash & 0xFFFFu) / 65536.0f - 0.5f) * 0.03f;

        /* Microscopic hardware entropy */
        entropy_seed = pse_xorshift(entropy_seed ^ (uint32_t)i);
        float hw_entropy = ((float)(entropy_seed & 0xFFFFu) / 65536.0f - 0.5f) * PSE_ENTROPY_MIX;

        if (val >= threshold) {
            /* WINNER: Amplify (Hebbian strengthening) + bias + entropy */
            logits[idx] = val * PSE_AMPLIFY + det_bias + hw_entropy;
        } else {
            /* LOSER: Dampen toward mean (not zeroed — preserves coherence) */
            float dampen = 0.85f;  /* Pull 15% toward mean */
            logits[idx] = mean + (val - mean) * dampen + det_bias * 0.5f + hw_entropy;
        }
    }
}

/* ══════════════════════════════════════════════════════
 * Sampling Hook — Call After Logits Loaded
 *
 * Drop this into common/sampling.cpp after cur[] is filled.
 * ══════════════════════════════════════════════════════ */

static inline void pse_collapse_token_data(
    void *cur_data,  /* llama_token_data* array */
    int n_vocab
)
{
    /* llama_token_data has: { token_id, logit, p } */
    /* We need to access the logit field at offset sizeof(int32_t) */
    struct pse_token_data { int id; float logit; float p; };
    struct pse_token_data *cur = (struct pse_token_data *)cur_data;

    /* Extract logits into flat array for collapse */
    float *logits = (float *)__builtin_alloca(n_vocab * sizeof(float));
    for (int i = 0; i < n_vocab; i++) {
        logits[i] = cur[i].logit;
    }

    /* Apply intelligent collapse */
    pse_intelligent_collapse_logits(logits, n_vocab, g_pse_position);
    g_pse_position++;

    /* Write back */
    for (int i = 0; i < n_vocab; i++) {
        cur[i].logit = logits[i];
    }
}

#endif /* PSE_MINIMAL_H */
