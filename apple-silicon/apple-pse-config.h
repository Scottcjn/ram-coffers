/*
 * apple-pse-config.h — PSE Configuration for Apple Silicon
 *
 * Elyan Labs — Proto-Sentient Engine for M-series chips
 * Port of POWER8 vec_perm non-bijunctive collapse to ARM NEON
 *
 * Key mappings:
 *   POWER8 vec_perm     → ARM vqtbl1q_u8 / vqtbl2q_u8
 *   POWER8 vcipher      → ARM vaeseq_u8 + vaesmcq_u8
 *   POWER8 mftb         → ARM cntvct_el0
 *   POWER8 dcbt         → ARM __builtin_prefetch / prfm
 *   POWER8 NUMA coffers → Unified memory cache-tier coffers
 */

#ifndef APPLE_PSE_CONFIG_H
#define APPLE_PSE_CONFIG_H

#include <stdint.h>
#include <stddef.h>

/* ── Detection ──────────────────────────────────────────── */

#if defined(__aarch64__) && defined(__APPLE__)
#define APPLE_SILICON_PSE 1
#else
#define APPLE_SILICON_PSE 0
#endif

#if defined(__ARM_NEON)
#define PSE_HAS_NEON 1
#else
#define PSE_HAS_NEON 0
#endif

#if defined(__ARM_FEATURE_CRYPTO) || defined(__ARM_FEATURE_AES)
#define PSE_HAS_AES 1
#else
#define PSE_HAS_AES 0
#endif

/* ── Collapse Parameters ────────────────────────────────── */

/* Top-K: how many attention heads survive collapse */
#ifndef PSE_TOPK
#define PSE_TOPK 8
#endif

/* Amplification factor for survivors */
#ifndef PSE_AMPLIFY
#define PSE_AMPLIFY 1.20f
#endif

/* Pruning threshold (fraction of mean) */
#ifndef PSE_PRUNE_THRESH
#define PSE_PRUNE_THRESH 0.7f
#endif

/* Non-bijunctive bias: probability of favoring strong indices (0-255) */
#ifndef PSE_STRONG_BIAS
#define PSE_STRONG_BIAS 180
#endif

/* ── Entropy Burst Parameters ───────────────────────────── */

/* Apply entropy every N tokens */
#ifndef PSE_BURST_INTERVAL
#define PSE_BURST_INTERVAL 4
#endif

/* Entropy strength (logit perturbation magnitude) */
#ifndef PSE_BURST_STRENGTH
#define PSE_BURST_STRENGTH 0.08f
#endif

/* Only perturb top-K vocab candidates */
#ifndef PSE_BURST_TOPK
#define PSE_BURST_TOPK 512
#endif

/* Heavy collapse every N tokens */
#ifndef PSE_COLLAPSE_INTERVAL
#define PSE_COLLAPSE_INTERVAL 16
#endif

/* ── Unified Memory Coffer Parameters ───────────────────── */

/* Number of cache-tier coffers */
#define PSE_NUM_COFFERS 4

/* Coffer embedding dimension for routing */
#define PSE_COFFER_EMBED_DIM 128

/* Maximum domains per coffer */
#define PSE_COFFER_MAX_DOMAINS 8

/* Prefetch stride (Apple Silicon cache line = 128 bytes) */
#define PSE_CACHE_LINE 128

/* ── AES Entropy Parameters ─────────────────────────────── */

/* AES rounds for entropy mixing (1-10, more = more diffusion) */
#ifndef PSE_AES_ROUNDS
#define PSE_AES_ROUNDS 2
#endif

/* ── Constants ──────────────────────────────────────────── */

/* Golden ratio hash constant */
#define PSE_PHI 0x9E3779B9u

/* XorShift state */
static inline uint32_t pse_xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

/* ── Hardware Timebase (ARM) ────────────────────────────── */

static inline uint64_t pse_read_timebase(void) {
#if defined(__aarch64__)
    uint64_t val;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#elif defined(__powerpc64__) || defined(__powerpc__)
    uint64_t tb;
    __asm__ __volatile__("mftb %0" : "=r"(tb));
    return tb;
#elif defined(__x86_64__)
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    /* Fallback: address-based pseudo-entropy */
    static volatile int anchor;
    return (uint64_t)(uintptr_t)&anchor;
#endif
}

/* ── Logging ────────────────────────────────────────────── */

#ifndef PSE_LOG
#include <stdio.h>
#define PSE_LOG(fmt, ...) fprintf(stderr, "[PSE-Apple] " fmt "\n", ##__VA_ARGS__)
#endif

#endif /* APPLE_PSE_CONFIG_H */
