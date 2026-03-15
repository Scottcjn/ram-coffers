/*
 * apple-pse-integration.h — Master PSE Integration for Apple Silicon
 *
 * Elyan Labs — Proto-Sentient Engine
 *
 * This is the top-level header that wires everything together.
 * Include this ONE file to get the full PSE pipeline:
 *
 *   1. NEON vqtbl2q collapse (non-bijunctive permute)
 *   2. AES entropy injection (hardware crypto diffusion)
 *   3. Unified memory coffers (cache-tier weight banking)
 *   4. Entropy burst (per-token behavioral divergence)
 *
 * Usage in llama.cpp:
 *   #include "apple-pse-integration.h"
 *
 *   // During model load:
 *   pse_apple_init(model_weights, model_size, n_layers, layer_size);
 *
 *   // During attention:
 *   pse_apple_collapse_attention(scores, n_scores, layer, head, pos);
 *
 *   // During sampling:
 *   pse_apple_entropy_burst(logits, n_vocab);
 *
 *   // End of inference:
 *   pse_apple_print_stats();
 */

#ifndef APPLE_PSE_INTEGRATION_H
#define APPLE_PSE_INTEGRATION_H

#include "apple-pse-config.h"
#include "neon-collapse.h"
#include "aes-entropy-collapse.h"
#include "unified-memory-coffers.h"

/* ══════════════════════════════════════════════════════════
 * Global PSE State
 * ══════════════════════════════════════════════════════════ */

typedef struct {
    int enabled;
    int use_aes;            /* Use AES entropy (requires ARM Crypto) */
    int use_coffers;        /* Use cache-tier coffers */
    int use_collapse;       /* Use non-bijunctive collapse */
    int use_entropy_burst;  /* Use per-token entropy */

    /* Counters */
    uint64_t collapse_calls;
    uint64_t entropy_bursts;
    uint64_t total_pruned;
    uint64_t total_amplified;

    /* Timing (cntvct_el0 ticks) */
    uint64_t collapse_ticks;
    uint64_t entropy_ticks;
    uint64_t prefetch_ticks;
} pse_apple_state_t;

static pse_apple_state_t g_pse_apple = {
    .enabled = 0,
    .use_aes = PSE_HAS_AES,
    .use_coffers = 1,
    .use_collapse = 1,
    .use_entropy_burst = 1,
};

/* ══════════════════════════════════════════════════════════
 * Initialization
 * ══════════════════════════════════════════════════════════ */

static inline void pse_apple_init(
    void *model_weights,
    size_t model_size,
    int n_layers,
    size_t layer_size)
{
#if !APPLE_SILICON_PSE
    PSE_LOG("WARNING: Not Apple Silicon — PSE running in compatibility mode");
#endif

#if !PSE_HAS_NEON
    PSE_LOG("ERROR: NEON not available — PSE disabled");
    g_pse_apple.enabled = 0;
    return;
#endif

    PSE_LOG("╔══════════════════════════════════════════════╗");
    PSE_LOG("║  Elyan Labs PSE — Apple Silicon Edition      ║");
    PSE_LOG("║  Non-Bijunctive Collapse via ARM NEON        ║");
    PSE_LOG("╠══════════════════════════════════════════════╣");
    PSE_LOG("║  vqtbl2q: dual-source permute (Hebbian)      ║");
#if PSE_HAS_AES
    PSE_LOG("║  vaese+vaesmc: AES entropy diffusion         ║");
#else
    PSE_LOG("║  XorShift: software entropy (no AES hw)      ║");
#endif
    PSE_LOG("║  Unified Memory: cache-tier coffers           ║");
    PSE_LOG("║  Top-K: %d | Amplify: %.2f | Entropy: cntvct ║",
            PSE_TOPK, PSE_AMPLIFY);
    PSE_LOG("╚══════════════════════════════════════════════╝");

    /* Initialize coffers */
    if (g_pse_apple.use_coffers && model_weights && n_layers > 0) {
        coffers_init_auto(model_weights, model_size, n_layers, layer_size);
    }

#if PSE_HAS_AES
    /* Initialize AES entropy state */
    if (g_pse_apple.use_aes) {
        pse_aes_entropy_init(0, 0);
        PSE_LOG("AES entropy initialized (cntvct_el0 + %d AES rounds)", PSE_AES_ROUNDS);
    }
#endif

    g_pse_apple.enabled = 1;
    PSE_LOG("PSE Apple Silicon ready.");
}

/* Lightweight init (no model weights — for standalone benchmarks) */
static inline void pse_apple_init_lite(void) {
    g_pse_apple.use_coffers = 0;

#if PSE_HAS_NEON
    PSE_LOG("PSE Apple Silicon (lite mode) — NEON collapse + entropy");
  #if PSE_HAS_AES
    pse_aes_entropy_init(0, 0);
    PSE_LOG("AES entropy active (%d rounds)", PSE_AES_ROUNDS);
  #endif
    g_pse_apple.enabled = 1;
#else
    PSE_LOG("NEON not available — PSE disabled");
    g_pse_apple.enabled = 0;
#endif
}

/* ══════════════════════════════════════════════════════════
 * Attention Collapse — Main Entry Point
 *
 * Call this on attention scores AFTER softmax but BEFORE
 * multiplying by V. This is where non-bijunctive collapse
 * removes noise and amplifies signal.
 *
 * The choice of AES vs XorShift entropy is automatic:
 *   - AES available → use AES (higher quality diffusion)
 *   - No AES → fall back to XorShift (still works, less diffusion)
 * ══════════════════════════════════════════════════════════ */

static inline void pse_apple_collapse_attention(
    float * restrict scores,
    int n_scores,
    int layer_id,
    int head_id,
    int position)
{
    if (!g_pse_apple.enabled || !g_pse_apple.use_collapse) return;
    if (n_scores < 4) return;

#if PSE_HAS_NEON
    uint64_t t0 = pse_read_timebase();

    /* Activate coffer for this layer (prefetch weights) */
    if (g_pse_apple.use_coffers) {
        for (int c = 0; c < PSE_NUM_COFFERS; c++) {
            if (g_coffers[c].is_loaded &&
                layer_id >= g_coffers[c].first_layer &&
                layer_id <= g_coffers[c].last_layer) {
                coffer_activate(c, layer_id);
                break;
            }
        }
    }

    /* Choose collapse method based on AES availability */
#if PSE_HAS_AES
    if (g_pse_apple.use_aes) {
        pse_aes_collapse_scores(scores, n_scores, layer_id, head_id, position);
    } else
#endif
    {
        /* XorShift fallback — still non-bijunctive, just less entropy quality */
        neon_collapse_scores(scores, n_scores, layer_id, head_id, position);
    }

    uint64_t t1 = pse_read_timebase();
    g_pse_apple.collapse_ticks += (t1 - t0);
    g_pse_apple.collapse_calls++;
#endif
}

/* Top-K variant: only collapse the top-K attention heads */
static inline void pse_apple_topk_collapse(
    float * restrict scores,
    int n_scores,
    int layer_id,
    int head_id,
    int position)
{
    if (!g_pse_apple.enabled || !g_pse_apple.use_collapse) return;

#if PSE_HAS_NEON
    uint64_t t0 = pse_read_timebase();

    neon_topk_collapse(scores, n_scores, PSE_TOPK,
                       layer_id, head_id, position);

    uint64_t t1 = pse_read_timebase();
    g_pse_apple.collapse_ticks += (t1 - t0);
    g_pse_apple.collapse_calls++;
#endif
}

/* ══════════════════════════════════════════════════════════
 * Entropy Burst — Per-Token Logit Perturbation
 *
 * Call this on logits BEFORE sampling. Creates behavioral
 * divergence — same prompt, different outputs each run.
 * ══════════════════════════════════════════════════════════ */

static inline void pse_apple_entropy_burst(
    float * restrict logits,
    int n_vocab)
{
    if (!g_pse_apple.enabled || !g_pse_apple.use_entropy_burst) return;

#if PSE_HAS_NEON
    uint64_t t0 = pse_read_timebase();

#if PSE_HAS_AES
    if (g_pse_apple.use_aes) {
        pse_aes_entropy_burst(logits, n_vocab);
    } else
#endif
    {
        /* XorShift entropy burst fallback */
        static int token_pos = 0;
        token_pos++;
        if (token_pos % PSE_BURST_INTERVAL != 0) return;

        uint64_t tb = pse_read_timebase();
        uint32_t seed = (uint32_t)(tb ^ (tb >> 32));
        seed ^= (uint32_t)token_pos * PSE_PHI;

        int count = (PSE_BURST_TOPK > 0 && PSE_BURST_TOPK < n_vocab)
                  ? PSE_BURST_TOPK : n_vocab;

        for (int i = 0; i < count; i++) {
            seed = pse_xorshift32(seed);
            float entropy = ((float)(seed & 0xFFFFU) / 65536.0f) - 0.5f;
            logits[i] += entropy * PSE_BURST_STRENGTH;
        }
    }

    uint64_t t1 = pse_read_timebase();
    g_pse_apple.entropy_ticks += (t1 - t0);
    g_pse_apple.entropy_bursts++;
#endif
}

/* ══════════════════════════════════════════════════════════
 * Layer Hook — Call at each transformer layer
 *
 * Handles:
 *   - Coffer activation + prefetch
 *   - Layer-ahead prefetch pipeline
 *   - Per-layer collapse if enabled
 * ══════════════════════════════════════════════════════════ */

static inline void pse_apple_layer_begin(int layer_id) {
    if (!g_pse_apple.enabled) return;

    if (g_pse_apple.use_coffers) {
        /* Find and activate the right coffer */
        for (int c = 0; c < PSE_NUM_COFFERS; c++) {
            if (g_coffers[c].is_loaded &&
                layer_id >= g_coffers[c].first_layer &&
                layer_id <= g_coffers[c].last_layer) {
                coffer_activate(c, layer_id);
                break;
            }
        }
    }
}

/* ══════════════════════════════════════════════════════════
 * Statistics
 * ══════════════════════════════════════════════════════════ */

static inline void pse_apple_print_stats(void) {
    if (!g_pse_apple.enabled) {
        PSE_LOG("PSE Apple Silicon: DISABLED");
        return;
    }

    PSE_LOG("=== PSE Apple Silicon Statistics ===");
    PSE_LOG("  Collapse calls:  %llu", (unsigned long long)g_pse_apple.collapse_calls);
    PSE_LOG("  Entropy bursts:  %llu", (unsigned long long)g_pse_apple.entropy_bursts);
    PSE_LOG("  Collapse ticks:  %llu", (unsigned long long)g_pse_apple.collapse_ticks);
    PSE_LOG("  Entropy ticks:   %llu", (unsigned long long)g_pse_apple.entropy_ticks);

    if (g_pse_apple.collapse_calls > 0) {
        PSE_LOG("  Avg ticks/collapse: %llu",
            (unsigned long long)(g_pse_apple.collapse_ticks / g_pse_apple.collapse_calls));
    }

    if (g_pse_apple.use_coffers) {
        coffers_print_stats();
    }
}

/* ══════════════════════════════════════════════════════════
 * Feature Query
 * ══════════════════════════════════════════════════════════ */

static inline int pse_apple_has_neon(void)    { return PSE_HAS_NEON; }
static inline int pse_apple_has_aes(void)     { return PSE_HAS_AES; }
static inline int pse_apple_is_enabled(void)  { return g_pse_apple.enabled; }
static inline int pse_apple_has_coffers(void) { return g_pse_apple.use_coffers && g_coffers_initialized; }

/* ══════════════════════════════════════════════════════════
 * Architecture Comparison Table
 *
 * For reference — how Apple Silicon PSE maps to POWER8 PSE:
 *
 * ┌─────────────────┬──────────────────┬──────────────────┐
 * │ Operation       │ POWER8           │ Apple Silicon     │
 * ├─────────────────┼──────────────────┼──────────────────┤
 * │ Byte permute    │ vec_perm (1 cyc) │ vqtbl1q (1 cyc)  │
 * │ Dual permute    │ vec_perm 2-src   │ vqtbl2q 2-src    │
 * │ AES round       │ vcipher (1 cyc)  │ vaese+mc (2 cyc) │
 * │ Threshold mask  │ vec_cmpgt+sel    │ vcgtq+vbslq      │
 * │ Fused mul-add   │ vec_madd         │ vfmaq_f32        │
 * │ Entropy source  │ mftb timebase    │ cntvct_el0       │
 * │ Prefetch        │ dcbt resident    │ prfm PLDL1KEEP   │
 * │ Cache line      │ 128 bytes        │ 128 bytes (same!)│
 * │ Weight banking  │ NUMA 4-node      │ Cache-tier 4-lvl │
 * │ Vector width    │ 128-bit VSX      │ 128-bit NEON     │
 * └─────────────────┴──────────────────┴──────────────────┘
 *
 * Key insight: Both architectures have 128-bit vectors with
 * 1-cycle byte permutation. The non-bijunctive collapse is
 * architecture-GENERAL, not a POWER8 trick.
 * ══════════════════════════════════════════════════════════ */

#endif /* APPLE_PSE_INTEGRATION_H */
