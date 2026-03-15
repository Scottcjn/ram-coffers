/*
 * unified-memory-coffers.h — Cache-Tier Weight Banking for Apple Silicon
 *
 * Elyan Labs — Apple Silicon PSE
 *
 * On POWER8, RAM Coffers map to NUMA nodes (4 nodes, 544GB total).
 * Apple Silicon has UNIFIED MEMORY — CPU and GPU see the same RAM.
 * No NUMA topology to exploit.
 *
 * Instead, we exploit the CACHE HIERARCHY:
 *
 *   Apple M2 Cache Architecture:
 *   ┌─────────────────────────────────────┐
 *   │ L1 Data  : 128KB per P-core         │  ~1 ns    (fastest)
 *   │ L1 Instr : 192KB per P-core         │
 *   ├─────────────────────────────────────┤
 *   │ L2       : 16MB shared (P-cores)    │  ~4 ns
 *   │            4MB (E-cores)            │
 *   ├─────────────────────────────────────┤
 *   │ SLC      : ~16MB System Level Cache │  ~12 ns   (shared CPU/GPU)
 *   ├─────────────────────────────────────┤
 *   │ DRAM     : 8-24GB Unified (LPDDR5)  │  ~100 ns  (slowest)
 *   └─────────────────────────────────────┘
 *
 * Coffer Strategy:
 *   Coffer 0 (HOT)   → Pin to L2 via streaming prefetch  (attention heads)
 *   Coffer 1 (WARM)  → Keep in SLC via temporal prefetch  (FFN weights)
 *   Coffer 2 (COOL)  → Standard DRAM access               (embeddings)
 *   Coffer 3 (COLD)  → Demand-load from DRAM              (rare layers)
 *
 * Unified Memory Advantage:
 *   GPU can also read coffers without copy — Metal compute shaders
 *   could do matmul on the SAME memory the CPU is managing.
 *   This is impossible on discrete GPU systems (PCIe bottleneck).
 *
 * Prefetch Instructions (ARM):
 *   __builtin_prefetch(addr, rw, locality)
 *     rw: 0=read, 1=write
 *     locality: 0=non-temporal, 1=L3, 2=L2, 3=L1
 *
 *   ARM prfm variants:
 *     PLDL1KEEP  — prefetch for load, L1, keep (temporal)
 *     PLDL1STRM  — prefetch for load, L1, streaming (non-temporal)
 *     PLDL2KEEP  — prefetch for load, L2, keep
 *     PLDL2STRM  — prefetch for load, L2, streaming
 *     PLDL3KEEP  — prefetch for load, L3/SLC, keep
 *     PLDL3STRM  — prefetch for load, L3/SLC, streaming
 */

#ifndef UNIFIED_MEMORY_COFFERS_H
#define UNIFIED_MEMORY_COFFERS_H

#include "apple-pse-config.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if PSE_HAS_NEON
#include <arm_neon.h>
#endif

/* ══════════════════════════════════════════════════════════
 * Cache Tier Definitions
 * ══════════════════════════════════════════════════════════ */

typedef enum {
    COFFER_HOT  = 0,   /* L2-pinned: attention weights, Q/K/V projections */
    COFFER_WARM = 1,   /* SLC-resident: FFN weights, up/down projections */
    COFFER_COOL = 2,   /* DRAM standard: token embeddings, LM head */
    COFFER_COLD = 3,   /* Demand-load: rarely-used layers, overflow */
} coffer_tier_t;

static const char* COFFER_TIER_NAMES[] = {
    "HOT (L2)",
    "WARM (SLC)",
    "COOL (DRAM)",
    "COLD (demand)"
};

/* ══════════════════════════════════════════════════════════
 * Domain Signatures — Cognitive Routing
 *
 * Same concept as neuromorphic coffers on POWER8:
 * Each coffer specializes in a cognitive domain.
 * Query embeddings are routed to the best-matching coffer.
 * ══════════════════════════════════════════════════════════ */

typedef struct {
    float  embed[PSE_COFFER_EMBED_DIM];  /* Domain centroid embedding */
    char   name[64];                      /* Human-readable domain name */
    float  activation_count;              /* Usage counter */
} coffer_domain_t;

/* ══════════════════════════════════════════════════════════
 * Coffer Structure
 * ══════════════════════════════════════════════════════════ */

typedef struct {
    coffer_tier_t tier;
    int           coffer_id;
    char          name[64];

    /* Memory region */
    void  *base_ptr;            /* Base address of weight data */
    size_t size;                /* Total bytes */
    int    is_loaded;           /* Whether data is mapped */

    /* Layer mapping */
    int    first_layer;         /* First model layer in this coffer */
    int    last_layer;          /* Last model layer in this coffer */
    size_t layer_stride;        /* Bytes per layer */

    /* Domain routing */
    coffer_domain_t domains[PSE_COFFER_MAX_DOMAINS];
    int n_domains;

    /* Stats */
    uint64_t activations;       /* Times this coffer was activated */
    uint64_t prefetch_bytes;    /* Total bytes prefetched */
    uint64_t cache_hits;        /* Estimated cache hits */
    uint64_t prune_savings;     /* Bytes saved by pruning */

} unified_coffer_t;

/* Global coffer array */
static unified_coffer_t g_coffers[PSE_NUM_COFFERS];
static int g_coffers_initialized = 0;

/* ══════════════════════════════════════════════════════════
 * Prefetch Macros — Cache-Tier Aware
 *
 * Apple Silicon cache line = 128 bytes (same as POWER8!)
 * Different tiers get different prefetch hints.
 * ══════════════════════════════════════════════════════════ */

/* L1 temporal (strongest — keep in L1 as long as possible) */
#define PREFETCH_L1_KEEP(addr) \
    __builtin_prefetch((const void*)(addr), 0, 3)

/* L2 temporal (keep in L2) */
#define PREFETCH_L2_KEEP(addr) \
    __builtin_prefetch((const void*)(addr), 0, 2)

/* L3/SLC temporal */
#define PREFETCH_SLC_KEEP(addr) \
    __builtin_prefetch((const void*)(addr), 0, 1)

/* Non-temporal (streaming, don't pollute caches) */
#define PREFETCH_STREAM(addr) \
    __builtin_prefetch((const void*)(addr), 0, 0)

/* Tier-aware prefetch: picks the right hint based on coffer tier */
static inline void coffer_prefetch(const void *addr, coffer_tier_t tier) {
    switch (tier) {
        case COFFER_HOT:
            PREFETCH_L1_KEEP(addr);
            break;
        case COFFER_WARM:
            PREFETCH_L2_KEEP(addr);
            break;
        case COFFER_COOL:
            PREFETCH_SLC_KEEP(addr);
            break;
        case COFFER_COLD:
            PREFETCH_STREAM(addr);
            break;
    }
}

/* Prefetch a range of memory with tier-appropriate hints */
static inline void coffer_prefetch_range(
    const void *base, size_t bytes, coffer_tier_t tier)
{
    const char *p = (const char *)base;
    const char *end = p + bytes;

    while (p < end) {
        coffer_prefetch(p, tier);
        p += PSE_CACHE_LINE;
    }
}

/* ══════════════════════════════════════════════════════════
 * Layer-Ahead Prefetch — Pipeline Next Layer While Computing
 *
 * Same concept as POWER8 dcbt streaming, but using ARM prfm.
 * While computing layer N, prefetch layer N+1 into appropriate
 * cache tier based on which coffer owns it.
 * ══════════════════════════════════════════════════════════ */

static inline void coffer_prefetch_next_layer(int current_layer) {
    int next_layer = current_layer + 1;

    for (int c = 0; c < PSE_NUM_COFFERS; c++) {
        if (!g_coffers[c].is_loaded) continue;
        if (next_layer < g_coffers[c].first_layer) continue;
        if (next_layer > g_coffers[c].last_layer)  continue;

        /* Found the coffer for next layer */
        int layer_offset = next_layer - g_coffers[c].first_layer;
        const char *next_addr = (const char *)g_coffers[c].base_ptr
                              + (layer_offset * g_coffers[c].layer_stride);

        /* Prefetch the first 8 cache lines of next layer (1KB) */
        for (int i = 0; i < 8; i++) {
            coffer_prefetch(next_addr + i * PSE_CACHE_LINE, g_coffers[c].tier);
        }

        g_coffers[c].prefetch_bytes += 8 * PSE_CACHE_LINE;
        break;
    }
}

/* ══════════════════════════════════════════════════════════
 * Cosine Similarity — NEON Accelerated
 *
 * Used for routing queries to the best-matching coffer domain.
 * ══════════════════════════════════════════════════════════ */

#if PSE_HAS_NEON
static inline float coffer_cosine_similarity(
    const float * restrict a,
    const float * restrict b,
    int dim)
{
    float32x4_t dot_sum = vdupq_n_f32(0.0f);
    float32x4_t norm_a_sum = vdupq_n_f32(0.0f);
    float32x4_t norm_b_sum = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);

        dot_sum    = vfmaq_f32(dot_sum,    va, vb);
        norm_a_sum = vfmaq_f32(norm_a_sum, va, va);
        norm_b_sum = vfmaq_f32(norm_b_sum, vb, vb);
    }

    /* Horizontal reduction */
    float32x2_t d2 = vadd_f32(vget_low_f32(dot_sum), vget_high_f32(dot_sum));
    float dot = vget_lane_f32(vpadd_f32(d2, d2), 0);

    float32x2_t a2 = vadd_f32(vget_low_f32(norm_a_sum), vget_high_f32(norm_a_sum));
    float norm_a = vget_lane_f32(vpadd_f32(a2, a2), 0);

    float32x2_t b2 = vadd_f32(vget_low_f32(norm_b_sum), vget_high_f32(norm_b_sum));
    float norm_b = vget_lane_f32(vpadd_f32(b2, b2), 0);

    /* Scalar remainder */
    for (; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    return (denom > 1e-8f) ? (dot / denom) : 0.0f;
}
#else
static inline float coffer_cosine_similarity(
    const float *a, const float *b, int dim)
{
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float d = sqrtf(na) * sqrtf(nb);
    return (d > 1e-8f) ? (dot / d) : 0.0f;
}
#endif

/* ══════════════════════════════════════════════════════════
 * Coffer Routing — Query → Best Coffer
 *
 * Routes a query embedding to the coffer with the
 * most relevant domain. Returns coffer ID.
 * ══════════════════════════════════════════════════════════ */

static inline int coffer_route(const float *query_embed) {
    int best_coffer = 0;
    float best_score = -1e30f;

    for (int c = 0; c < PSE_NUM_COFFERS; c++) {
        if (!g_coffers[c].is_loaded) continue;

        for (int d = 0; d < g_coffers[c].n_domains; d++) {
            float score = coffer_cosine_similarity(
                query_embed,
                g_coffers[c].domains[d].embed,
                PSE_COFFER_EMBED_DIM);

            if (score > best_score) {
                best_score = score;
                best_coffer = c;
            }
        }
    }

    g_coffers[best_coffer].activations++;
    return best_coffer;
}

/* ══════════════════════════════════════════════════════════
 * Coffer Initialization
 *
 * Auto-assigns model layers to coffers based on importance:
 *   HOT  → First attention layers (most impactful)
 *   WARM → Middle FFN layers
 *   COOL → Embeddings and final layers
 *   COLD → Overflow / rarely-activated expert layers
 * ══════════════════════════════════════════════════════════ */

static inline void coffers_init_auto(
    void *model_weights,
    size_t total_bytes,
    int n_layers,
    size_t layer_size)
{
    if (g_coffers_initialized) return;

    /* Divide layers into 4 tiers */
    int hot_end   = n_layers / 4;         /* First 25%: attention-heavy */
    int warm_end  = n_layers / 2;         /* Next 25%: FFN-heavy */
    int cool_end  = (3 * n_layers) / 4;   /* Next 25%: later layers */
    /* Remaining 25%: cold */

    char *base = (char *)model_weights;

    /* Coffer 0: HOT — first attention layers */
    g_coffers[0] = (unified_coffer_t){
        .tier = COFFER_HOT,
        .coffer_id = 0,
        .base_ptr = base,
        .size = hot_end * layer_size,
        .is_loaded = 1,
        .first_layer = 0,
        .last_layer = hot_end - 1,
        .layer_stride = layer_size,
    };
    snprintf(g_coffers[0].name, sizeof(g_coffers[0].name), "HOT-attention");

    /* Coffer 1: WARM — middle FFN layers */
    g_coffers[1] = (unified_coffer_t){
        .tier = COFFER_WARM,
        .coffer_id = 1,
        .base_ptr = base + hot_end * layer_size,
        .size = (warm_end - hot_end) * layer_size,
        .is_loaded = 1,
        .first_layer = hot_end,
        .last_layer = warm_end - 1,
        .layer_stride = layer_size,
    };
    snprintf(g_coffers[1].name, sizeof(g_coffers[1].name), "WARM-ffn");

    /* Coffer 2: COOL — later layers */
    g_coffers[2] = (unified_coffer_t){
        .tier = COFFER_COOL,
        .coffer_id = 2,
        .base_ptr = base + warm_end * layer_size,
        .size = (cool_end - warm_end) * layer_size,
        .is_loaded = 1,
        .first_layer = warm_end,
        .last_layer = cool_end - 1,
        .layer_stride = layer_size,
    };
    snprintf(g_coffers[2].name, sizeof(g_coffers[2].name), "COOL-embed");

    /* Coffer 3: COLD — final layers */
    g_coffers[3] = (unified_coffer_t){
        .tier = COFFER_COLD,
        .coffer_id = 3,
        .base_ptr = base + cool_end * layer_size,
        .size = (n_layers - cool_end) * layer_size,
        .is_loaded = 1,
        .first_layer = cool_end,
        .last_layer = n_layers - 1,
        .layer_stride = layer_size,
    };
    snprintf(g_coffers[3].name, sizeof(g_coffers[3].name), "COLD-overflow");

    g_coffers_initialized = 1;

    PSE_LOG("Unified Memory Coffers initialized (%d layers)", n_layers);
    for (int c = 0; c < PSE_NUM_COFFERS; c++) {
        PSE_LOG("  Coffer %d [%s]: layers %d-%d, %.1f MB, tier=%s",
            c, g_coffers[c].name,
            g_coffers[c].first_layer, g_coffers[c].last_layer,
            (float)g_coffers[c].size / (1024*1024),
            COFFER_TIER_NAMES[g_coffers[c].tier]);
    }
}

/* ══════════════════════════════════════════════════════════
 * Coffer Activation — Prefetch + Collapse Plan
 *
 * When a coffer is activated:
 *   1. Prefetch weights into appropriate cache tier
 *   2. Plan non-bijunctive pruning (skip irrelevant weights)
 *   3. Track activation for adaptive rebalancing
 * ══════════════════════════════════════════════════════════ */

static inline void coffer_activate(int coffer_id, int layer_id) {
    if (coffer_id < 0 || coffer_id >= PSE_NUM_COFFERS) return;

    unified_coffer_t *cof = &g_coffers[coffer_id];
    if (!cof->is_loaded) return;

    cof->activations++;

    /* Calculate layer offset */
    int offset = layer_id - cof->first_layer;
    if (offset < 0 || offset > (cof->last_layer - cof->first_layer)) return;

    const char *layer_addr = (const char *)cof->base_ptr + (offset * cof->layer_stride);

    /* Prefetch with tier-appropriate hints */
    size_t prefetch_size = cof->layer_stride;
    if (prefetch_size > 64 * PSE_CACHE_LINE) {
        prefetch_size = 64 * PSE_CACHE_LINE;  /* Cap at 8KB initial prefetch */
    }

    coffer_prefetch_range(layer_addr, prefetch_size, cof->tier);
    cof->prefetch_bytes += prefetch_size;

    /* Also prefetch next layer (pipeline) */
    coffer_prefetch_next_layer(layer_id);
}

/* ══════════════════════════════════════════════════════════
 * Non-Bijunctive Skip Planning
 *
 * Before fetching a weight tensor, decide which blocks to skip.
 * Based on historical activation patterns and entropy score.
 *
 * Returns a bitmask: 1 = fetch this block, 0 = skip.
 * ══════════════════════════════════════════════════════════ */

static inline uint64_t coffer_plan_prune(
    int coffer_id,
    int layer_id,
    const float * restrict activation_history,
    int n_blocks)
{
    if (n_blocks > 64) n_blocks = 64;  /* Max 64 blocks per bitmask */

    uint64_t mask = 0;

    /* Generate entropy-seeded threshold */
    uint64_t tb = pse_read_timebase();
    uint32_t seed = (uint32_t)(tb ^ (tb >> 32)) ^ (layer_id * PSE_PHI);

    for (int b = 0; b < n_blocks; b++) {
        seed = pse_xorshift32(seed);

        /* Score: combination of historical activation + entropy */
        float hist_score = (activation_history != NULL)
                         ? activation_history[b]
                         : 1.0f;

        float entropy_bonus = ((float)(seed & 0xFFFF) / 65536.0f) * 0.2f;
        float total_score = hist_score + entropy_bonus;

        /* Fetch if score above pruning threshold */
        if (total_score > PSE_PRUNE_THRESH) {
            mask |= (1ULL << b);
        }
    }

    /* Track savings */
    int pruned = n_blocks - __builtin_popcountll(mask);
    g_coffers[coffer_id].prune_savings += pruned * (PSE_CACHE_LINE * 8);

    return mask;
}

/* ══════════════════════════════════════════════════════════
 * Coffer Statistics
 * ══════════════════════════════════════════════════════════ */

static inline void coffers_print_stats(void) {
    if (!g_coffers_initialized) return;

    PSE_LOG("=== Unified Memory Coffer Statistics ===");
    for (int c = 0; c < PSE_NUM_COFFERS; c++) {
        if (!g_coffers[c].is_loaded) continue;
        PSE_LOG("  Coffer %d [%s]: %llu activations, "
                "%.1f MB prefetched, %.1f MB pruned",
            c, g_coffers[c].name,
            (unsigned long long)g_coffers[c].activations,
            (float)g_coffers[c].prefetch_bytes / (1024*1024),
            (float)g_coffers[c].prune_savings / (1024*1024));
    }
}

/* ══════════════════════════════════════════════════════════
 * GPU Unified Memory Hints (Apple Metal)
 *
 * Since Apple Silicon shares memory between CPU and GPU,
 * we can hint the system about access patterns.
 *
 * This is a stub for Metal integration — would use
 * MTLResource.storageModeShared for zero-copy GPU access.
 * ══════════════════════════════════════════════════════════ */

typedef struct {
    int gpu_accessible;    /* Whether Metal can see this coffer */
    int gpu_preferred;     /* Whether GPU is faster for this coffer's ops */
    int cpu_preferred;     /* Whether CPU is faster (NEON collapse) */
} coffer_gpu_hint_t;

/* Default hints: CPU handles collapse, GPU handles matmul */
static const coffer_gpu_hint_t COFFER_GPU_HINTS[PSE_NUM_COFFERS] = {
    { .gpu_accessible = 1, .gpu_preferred = 1, .cpu_preferred = 1 },  /* HOT: both */
    { .gpu_accessible = 1, .gpu_preferred = 1, .cpu_preferred = 0 },  /* WARM: GPU matmul */
    { .gpu_accessible = 1, .gpu_preferred = 0, .cpu_preferred = 1 },  /* COOL: CPU embed */
    { .gpu_accessible = 0, .gpu_preferred = 0, .cpu_preferred = 1 },  /* COLD: CPU only */
};

#endif /* UNIFIED_MEMORY_COFFERS_H */
