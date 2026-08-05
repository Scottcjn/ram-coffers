/*
 * ggml-ram-coffers.h - Multi-Bank NUMA Weight Indexing for POWER8 S824
 *
 * Scott's Vision: "Selectively house model information in known RAM banks
 *                  with resonance routing for associative recall"
 *
 * Architecture (544GB free across 4 NUMA nodes):
 * | Coffer | Node | Free GB | Role                    |
 * |--------|------|---------|-------------------------|
 * | 0      | 3    | 193     | Heavy/General (core)    |
 * | 1      | 1    | 183     | Science/Tech domain     |
 * | 2      | 0    | 119     | Creative/Long CTX       |
 * | 3      | 2    | 62      | Niche/History           |
 *
 * Flow:
 * 1. Query embed → route_to_coffer (resonance match)
 * 2. activate_coffer → DCBT prefetch + numa_run_on_node
 * 3. pse_collapse_prune → Non-bijunctive prune before full fetch
 * 4. Generate with PSE entropy from active coffer node
 */

#ifndef GGML_RAM_COFFERS_H
#define GGML_RAM_COFFERS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <math.h>

/*
 * Capability detection + NUMA/intrinsic shims. The former guards here were
 * "#ifdef __linux__", which still required <numa.h> to exist on every Linux
 * box - so a Linux machine without libnuma-dev failed to compile.
 */
#include "coffers-portability.h"

/*===========================================================================
 * Configuration
 *===========================================================================*/

#define MAX_COFFERS 4
#define COFFER_EMBED_DIM 128    /* Resonance embedding dimension */
#define COFFER_MAX_DOMAINS 16   /* Domain signatures per coffer */

/*
 * POWER8 S824 topology (4 nodes, ordered largest-free-space first).
 * These remain as the reference layout, but are NO LONGER indexed directly:
 * on a 1- or 2-node machine they would name nodes that do not exist. Use
 * coffers_node_for(coffer_id) from coffers-portability.h, which derives the
 * mapping from the node count actually detected.
 */
static const int NUMA_TO_COFFER[4] = {2, 1, 3, 0};  /* Node 0→C2, 1→C1, 2→C3, 3→C0 */
static const int COFFER_TO_NUMA[4] = {3, 1, 0, 2};  /* C0→Node3, C1→Node1, etc */

/*===========================================================================
 * Domain Signatures for Resonance Routing
 *
 * Each coffer has domain signatures - embeddings that define what
 * queries should route to it. Simple cosine similarity routing.
 *===========================================================================*/

typedef struct {
    float embed[COFFER_EMBED_DIM];
    char label[32];
} domain_signature_t;

/*===========================================================================
 * RAM Coffer Structure
 *===========================================================================*/

typedef struct {
    /* NUMA/Memory */
    int numa_node;
    void* mmap_ptr;
    size_t mmap_size;
    int fd;

    /* Coffer identity */
    int coffer_id;
    char name[64];
    char gguf_path[256];

    /* Domain resonance */
    domain_signature_t domains[COFFER_MAX_DOMAINS];
    int n_domains;

    /* Statistics */
    uint64_t activations;
    uint64_t prefetch_bytes;
    uint64_t prune_savings;

    /* State */
    int is_loaded;
    int is_active;
} ram_coffer_t;

/* Global coffer array */
static ram_coffer_t g_coffers[MAX_COFFERS] = {0};
static int g_coffers_initialized = 0;

/*===========================================================================
 * POWER8 DCBT Prefetch Macros
 *===========================================================================*/

/*
 * POWER8 has dcbt with hardware prefetch STREAMS (start/stop), which no other
 * architecture exposes. Elsewhere we fall back to __builtin_prefetch for the
 * per-line touch and drop the stream hints, which are advisory anyway.
 */
#if GGML_COFFERS_IS_POWER
#define DCBT_PREFETCH(addr) __asm__ __volatile__("dcbt 0,%0" : : "r"(addr))
#define DCBT_STREAM_START(addr, id) __asm__ __volatile__("dcbt 0,%0,%1" : : "r"(addr), "i"(id))
#define DCBT_STREAM_STOP(id) __asm__ __volatile__("dcbt 0,0,%0" : : "i"(id | 0x10))
#else
#define DCBT_PREFETCH(addr) coffers_prefetch(addr)
#define DCBT_STREAM_START(addr, id) ((void)(addr), (void)(id))
#define DCBT_STREAM_STOP(id) ((void)(id))
#endif

/* Prefetch entire region to L2/L3 */
static inline void dcbt_resident(const void* addr, size_t size) {
    const size_t cache_line = COFFERS_CACHE_LINE;
    const char* p = (const char*)addr;
    const char* end = p + size;

    /* Start prefetch stream */
    DCBT_STREAM_START(p, 0);

    while (p < end) {
        DCBT_PREFETCH(p);
        p += cache_line * 8;  /* Skip ahead for stream */
    }

    /* Stop stream */
    DCBT_STREAM_STOP(0);
}

/*===========================================================================
 * Layer-Ahead Prefetch Pipeline
 *
 * FIX: Don't prefetch everything at activation - that thrashes cache.
 * Instead, prefetch layer N+1 while computing layer N.
 * Thanks to ng @ NYSE for this optimization.
 *===========================================================================*/

typedef struct {
    int current_layer;
    int total_layers;
    size_t layer_size;        /* Bytes per layer (approximate) */
    const char* base_ptr;     /* Base of weight memory */
} layer_prefetch_state_t;

static layer_prefetch_state_t g_prefetch_state = {0};

/* Initialize layer-ahead prefetch for a coffer */
static inline void layer_prefetch_init(int coffer_id, int total_layers) {
    if (coffer_id < 0 || coffer_id >= MAX_COFFERS) return;
    ram_coffer_t* coffer = &g_coffers[coffer_id];
    if (!coffer->is_loaded) return;

    g_prefetch_state.current_layer = -1;
    g_prefetch_state.total_layers = total_layers;
    g_prefetch_state.layer_size = coffer->mmap_size / total_layers;
    g_prefetch_state.base_ptr = (const char*)coffer->mmap_ptr;

    /* Prefetch first layer only */
    dcbt_resident(g_prefetch_state.base_ptr, g_prefetch_state.layer_size);
}

/*
 * Call this BEFORE starting computation on layer N.
 * It will prefetch layer N+1 while you compute N.
 */
static inline void layer_prefetch_ahead(int layer_id) {
    if (layer_id < 0 || layer_id >= g_prefetch_state.total_layers) return;

    g_prefetch_state.current_layer = layer_id;

    /* Prefetch NEXT layer (layer_id + 1) if it exists */
    int next_layer = layer_id + 1;
    if (next_layer < g_prefetch_state.total_layers) {
        const char* next_addr = g_prefetch_state.base_ptr +
                                (next_layer * g_prefetch_state.layer_size);
        size_t prefetch_size = g_prefetch_state.layer_size;

        /* Use stream 1 for look-ahead (stream 0 may be in use) */
        DCBT_STREAM_START(next_addr, 1);

        const size_t cache_line = COFFERS_CACHE_LINE;
        const char* p = next_addr;
        const char* end = p + prefetch_size;

        while (p < end) {
            DCBT_PREFETCH(p);
            p += cache_line * 8;
        }

        DCBT_STREAM_STOP(1);
    }
}

/*===========================================================================
 * Resonance Routing
 *
 * Simple cosine similarity between query embedding and domain signatures.
 * Returns best matching coffer ID.
 *===========================================================================*/

static inline float dot_product(const float* a, const float* b, int dim) {
    float sum = 0.0f;
#if GGML_COFFERS_HAVE_ALTIVEC
    /*
     * NOTE: <altivec.h> used to be #included HERE, inside the function body.
     * It is now included once at file scope by coffers-portability.h, which
     * is both correct and what let this compile under -Werror.
     */
    vector float vsum = vec_splats(0.0f);
    int d = 0;
    for (; d + 3 < dim; d += 4) {
        vector float va = vec_ld(0, &a[d]);
        vector float vb = vec_ld(0, &b[d]);
        vsum = vec_madd(va, vb, vsum);
    }
    /* Horizontal sum */
    vector float s1 = vec_add(vsum, vec_sld(vsum, vsum, 8));
    vector float s2 = vec_add(s1, vec_sld(s1, s1, 4));
    vec_ste(s2, 0, &sum);
    for (; d < dim; d++) {
        sum += a[d] * b[d];
    }
#else
    for (int d = 0; d < dim; d++) {
        sum += a[d] * b[d];
    }
#endif
    return sum;
}

static inline float magnitude(const float* v, int dim) {
    return sqrtf(dot_product(v, v, dim));
}

static inline float cosine_similarity(const float* a, const float* b, int dim) {
    float dot = dot_product(a, b, dim);
    float mag_a = magnitude(a, dim);
    float mag_b = magnitude(b, dim);
    if (mag_a < 1e-9f || mag_b < 1e-9f) return 0.0f;
    return dot / (mag_a * mag_b);
}

/* Route query to best coffer based on embedding */
static int route_to_coffer(const float* query_embed) {
    int best_coffer = 0;
    float best_score = -1e30f;

    for (int c = 0; c < MAX_COFFERS; c++) {
        if (!g_coffers[c].is_loaded) continue;

        /* Check against all domain signatures */
        for (int d = 0; d < g_coffers[c].n_domains; d++) {
            float score = cosine_similarity(query_embed,
                                           g_coffers[c].domains[d].embed,
                                           COFFER_EMBED_DIM);
            if (score > best_score) {
                best_score = score;
                best_coffer = c;
            }
        }
    }

    return best_coffer;
}

/*===========================================================================
 * Coffer Initialization
 *===========================================================================*/

/*
 * Number of coffers usable on this machine. 1 in uniform-memory mode.
 * Everything that used to loop to MAX_COFFERS should bound on this.
 */
static int g_coffers_usable = 0;

static inline int coffer_count(void) {
    if (g_coffers_usable > 0) return g_coffers_usable;
    const coffers_topology_t* t = coffers_topology();
    g_coffers_usable = t->n_coffers < MAX_COFFERS ? t->n_coffers : MAX_COFFERS;
    if (g_coffers_usable < 1) g_coffers_usable = 1;
    return g_coffers_usable;
}

/*
 * Initialise coffer→node bindings.
 *
 * Returns 0 in ALL supported configurations, including no-NUMA and
 * single-node. Uniform memory is a valid topology (1 coffer over all RAM),
 * not a failure - the old code returned -1 here and callers printed a
 * warning about running "without NUMA support" on perfectly ordinary
 * single-socket machines.
 */
static int coffer_init_numa(void) {
    coffers_report_mode();

    int n = coffer_count();
    for (int c = 0; c < n; c++) {
        g_coffers[c].coffer_id = c;
        g_coffers[c].numa_node = coffers_node_for(c);
        snprintf(g_coffers[c].name, sizeof(g_coffers[c].name), "Coffer-%d", c);
    }

    /* Coffers beyond the usable count stay unbound and unloaded. */
    for (int c = n; c < MAX_COFFERS; c++) {
        g_coffers[c].coffer_id = c;
        g_coffers[c].numa_node = COFFERS_UNIFORM_NODE;
        g_coffers[c].is_loaded = 0;
    }

    return 0;
}

/* Load a GGUF shard into a specific coffer */
static int coffer_load_shard(int coffer_id, const char* gguf_path) {
    if (coffer_id < 0 || coffer_id >= MAX_COFFERS) {
        return -1;
    }

    ram_coffer_t* coffer = &g_coffers[coffer_id];

    /* Bind to coffer's NUMA node for allocation (no-op in uniform mode) */
    coffers_run_on_node(coffer->numa_node);

    /* Open file */
    coffer->fd = open(gguf_path, O_RDONLY);
    if (coffer->fd < 0) {
        fprintf(stderr, "Coffers: Cannot open %s\n", gguf_path);
        return -1;
    }

    /* Get file size */
    struct stat st;
    fstat(coffer->fd, &st);
    coffer->mmap_size = st.st_size;

    /* mmap with huge pages if available */
    int mmap_flags = MAP_PRIVATE;

    /*
     * FIX: Set memory policy BEFORE mmap to allocate on correct node.
     * This avoids mbind(MPOL_MF_MOVE) which stalls on page migration.
     * Thanks to ng @ NYSE for this optimization.
     *
     * In uniform-memory mode this is a no-op and mmap proceeds normally.
     */
    coffers_set_membind(coffer->numa_node);

#ifdef MAP_HUGETLB
    /* Try huge pages first, fall back to normal */
    coffer->mmap_ptr = mmap(NULL, coffer->mmap_size, PROT_READ,
                            mmap_flags | MAP_HUGETLB, coffer->fd, 0);
    if (coffer->mmap_ptr == MAP_FAILED) {
        coffer->mmap_ptr = mmap(NULL, coffer->mmap_size, PROT_READ,
                                mmap_flags, coffer->fd, 0);
    }
#else
    coffer->mmap_ptr = mmap(NULL, coffer->mmap_size, PROT_READ,
                            mmap_flags, coffer->fd, 0);
#endif

    /* Reset to default policy for future allocations */
    coffers_reset_mempolicy();

    if (coffer->mmap_ptr == MAP_FAILED) {
        fprintf(stderr, "Coffers: mmap failed for %s\n", gguf_path);
        close(coffer->fd);
        return -1;
    }

    strncpy(coffer->gguf_path, gguf_path, sizeof(coffer->gguf_path) - 1);
    coffer->is_loaded = 1;

    fprintf(stderr, "Coffers: Loaded %s (%.2f GB) into Coffer-%d (Node %d)\n",
            gguf_path,
            coffer->mmap_size / (1024.0 * 1024.0 * 1024.0),
            coffer_id,
            coffer->numa_node);

    return 0;
}

/*===========================================================================
 * Domain Signature Registration
 *
 * Pre-compute domain embeddings for routing.
 *===========================================================================*/

static void coffer_add_domain(int coffer_id, const char* label, const float* embed) {
    if (coffer_id < 0 || coffer_id >= MAX_COFFERS) return;
    ram_coffer_t* coffer = &g_coffers[coffer_id];

    if (coffer->n_domains >= COFFER_MAX_DOMAINS) return;

    domain_signature_t* dom = &coffer->domains[coffer->n_domains++];
    strncpy(dom->label, label, sizeof(dom->label) - 1);
    memcpy(dom->embed, embed, COFFER_EMBED_DIM * sizeof(float));
}

/* Pre-built domain signatures (simple keyword hashing as placeholder) */
static void coffer_init_default_domains(void) {
    /* Generate pseudo-embeddings from domain keywords */
    /* Real implementation would use actual embedding model */

    float general[COFFER_EMBED_DIM] = {0};
    float science[COFFER_EMBED_DIM] = {0};
    float creative[COFFER_EMBED_DIM] = {0};
    float history[COFFER_EMBED_DIM] = {0};

    /* Simple pattern: different frequency patterns per domain */
    for (int i = 0; i < COFFER_EMBED_DIM; i++) {
        general[i]  = sinf(i * 0.1f);
        science[i]  = cosf(i * 0.3f);
        creative[i] = sinf(i * 0.2f) + cosf(i * 0.15f);
        history[i]  = sinf(i * 0.05f) * 0.5f;
    }

    coffer_add_domain(0, "general", general);
    coffer_add_domain(0, "code", general);
    coffer_add_domain(1, "science", science);
    coffer_add_domain(1, "math", science);
    coffer_add_domain(1, "tech", science);
    coffer_add_domain(2, "creative", creative);
    coffer_add_domain(2, "story", creative);
    coffer_add_domain(2, "art", creative);
    coffer_add_domain(3, "history", history);
    coffer_add_domain(3, "philosophy", history);
}

/*===========================================================================
 * Coffer Activation
 *
 * Activate a coffer: bind CPU, prefetch weights, prepare for inference
 *===========================================================================*/

/*
 * Activate coffer with optional layer count for pipelined prefetch.
 * If n_layers > 0, uses layer-ahead prefetch (recommended).
 * If n_layers <= 0, falls back to minimal prefetch (first 4MB only).
 */
static int activate_coffer_ex(int coffer_id, int n_layers) {
    if (coffer_id < 0 || coffer_id >= MAX_COFFERS) return -1;

    ram_coffer_t* coffer = &g_coffers[coffer_id];
    if (!coffer->is_loaded) return -1;

    /* Bind to coffer's NUMA node (no-op in uniform mode) */
    coffers_run_on_node(coffer->numa_node);

    if (n_layers > 0) {
        /*
         * Layer-ahead prefetch: Initialize pipeline, prefetch only layer 0.
         * Subsequent layers prefetched via layer_prefetch_ahead().
         */
        layer_prefetch_init(coffer_id, n_layers);
        coffer->prefetch_bytes += g_prefetch_state.layer_size;
    } else {
        /*
         * Fallback: Minimal prefetch (first 4MB for headers/metadata).
         * Avoid the old 64MB eager prefetch that thrashed cache.
         */
        size_t prefetch_size = 4 * 1024 * 1024;
        if (prefetch_size > coffer->mmap_size) {
            prefetch_size = coffer->mmap_size;
        }
        dcbt_resident(coffer->mmap_ptr, prefetch_size);
        coffer->prefetch_bytes += prefetch_size;
    }

    coffer->is_active = 1;
    coffer->activations++;

    return 0;
}

/* Backward-compatible wrapper */
static int activate_coffer(int coffer_id) {
    return activate_coffer_ex(coffer_id, 0);
}

/*===========================================================================
 * Non-Bijunctive Prune Before Fetch
 *
 * Uses PSE collapse logic to identify which weights to skip.
 * Returns a mask indicating which blocks to actually load.
 *===========================================================================*/

typedef struct {
    uint64_t* block_mask;     /* Bitmap: 1 = load, 0 = skip */
    int n_blocks;
    size_t block_size;
    size_t total_saved;
} prune_plan_t;

static prune_plan_t* coffer_plan_prune(int coffer_id, const float* query_embed, float threshold) {
    if (coffer_id < 0 || coffer_id >= MAX_COFFERS) return NULL;

    ram_coffer_t* coffer = &g_coffers[coffer_id];
    if (!coffer->is_loaded) return NULL;

    /* Allocate prune plan */
    prune_plan_t* plan = (prune_plan_t*)calloc(1, sizeof(prune_plan_t));
    if (!plan) return NULL;

    /* Divide weights into blocks (e.g., 1MB each) */
    plan->block_size = 1024 * 1024;
    plan->n_blocks = (coffer->mmap_size + plan->block_size - 1) / plan->block_size;

    size_t mask_size = (plan->n_blocks + 63) / 64;
    plan->block_mask = (uint64_t*)calloc(mask_size, sizeof(uint64_t));
    if (!plan->block_mask) {
        free(plan);
        return NULL;
    }

    /* Simple prune heuristic: Skip blocks with low "resonance" */
    /* Real implementation would analyze weight patterns */
    int loaded = 0;
    for (int b = 0; b < plan->n_blocks; b++) {
        /* Pseudo-resonance based on block position + query */
        float resonance = fabsf(sinf(b * 0.1f + query_embed[0]));

        if (resonance >= threshold) {
            /* Set bit to load this block */
            plan->block_mask[b / 64] |= (1ULL << (b % 64));
            loaded++;
        } else {
            plan->total_saved += plan->block_size;
        }
    }

    return plan;
}

static void coffer_free_prune_plan(prune_plan_t* plan) {
    if (plan) {
        free(plan->block_mask);
        free(plan);
    }
}

/*===========================================================================
 * Full Initialization
 *===========================================================================*/

static int init_ram_coffers(const char* gguf_paths[MAX_COFFERS]) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  RAM Coffers System - POWER8 S824 NUMA Weight Banking        ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════════╣\n");

    coffer_init_numa();

    if (coffers_topology()->uniform) {
        fprintf(stderr, "║  Uniform-memory mode: 1 coffer over all RAM (no affinity)    ║\n");
    }

    /* Load shards */
    int loaded = 0;
    for (int c = 0; c < MAX_COFFERS; c++) {
        if (gguf_paths && gguf_paths[c] && gguf_paths[c][0]) {
            if (coffer_load_shard(c, gguf_paths[c]) == 0) {
                loaded++;
            }
        }
    }

    /* Initialize default domain signatures */
    coffer_init_default_domains();

    fprintf(stderr, "║  Loaded %d coffer shards                                      ║\n", loaded);
    fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n\n");

    g_coffers_initialized = 1;
    return loaded;
}

/*===========================================================================
 * Statistics
 *===========================================================================*/

static void coffer_print_stats(void) {
    if (!g_coffers_initialized) return;

    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  RAM Coffers Statistics                                       ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════════╣\n");

    uint64_t total_activations = 0;
    uint64_t total_prefetch = 0;
    uint64_t total_prune_saved = 0;

    for (int c = 0; c < MAX_COFFERS; c++) {
        ram_coffer_t* coffer = &g_coffers[c];
        if (!coffer->is_loaded) continue;

        fprintf(stderr, "║  Coffer-%d (Node %d): %6.2f GB, %8lu activations         ║\n",
                c, coffer->numa_node,
                coffer->mmap_size / (1024.0 * 1024.0 * 1024.0),
                (unsigned long)coffer->activations);

        total_activations += coffer->activations;
        total_prefetch += coffer->prefetch_bytes;
        total_prune_saved += coffer->prune_savings;
    }

    fprintf(stderr, "╠═══════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Total activations: %12lu                               ║\n",
            (unsigned long)total_activations);
    fprintf(stderr, "║  Prefetch bytes:    %12.2f GB                           ║\n",
            total_prefetch / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "║  Prune savings:     %12.2f GB                           ║\n",
            total_prune_saved / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n");
}

/*===========================================================================
 * Cleanup
 *===========================================================================*/

static void shutdown_ram_coffers(void) {
    coffer_print_stats();

    for (int c = 0; c < MAX_COFFERS; c++) {
        ram_coffer_t* coffer = &g_coffers[c];
        if (!coffer->is_loaded) continue;   /* never opened - nothing to release */

        if (coffer->mmap_ptr && coffer->mmap_ptr != MAP_FAILED) {
            munmap(coffer->mmap_ptr, coffer->mmap_size);
            coffer->mmap_ptr = NULL;
        }
        /*
         * Guard on > 0, not >= 0: g_coffers is zero-initialised, so an
         * unopened coffer has fd == 0 and the old check closed STDIN.
         */
        if (coffer->fd > 0) {
            close(coffer->fd);
            coffer->fd = -1;
        }
        coffer->is_loaded = 0;
    }

    g_coffers_initialized = 0;
}

/*===========================================================================
 * Test Function
 *===========================================================================*/

static void coffer_test_routing(void) {
    fprintf(stderr, "\n=== Coffer Routing Test ===\n");

    /* Test embeddings */
    float general_query[COFFER_EMBED_DIM];
    float science_query[COFFER_EMBED_DIM];
    float creative_query[COFFER_EMBED_DIM];

    for (int i = 0; i < COFFER_EMBED_DIM; i++) {
        general_query[i]  = sinf(i * 0.1f) + 0.1f;
        science_query[i]  = cosf(i * 0.3f) + 0.1f;
        creative_query[i] = sinf(i * 0.2f) + cosf(i * 0.15f) + 0.1f;
    }

    fprintf(stderr, "General query → Coffer %d\n", route_to_coffer(general_query));
    fprintf(stderr, "Science query → Coffer %d\n", route_to_coffer(science_query));
    fprintf(stderr, "Creative query → Coffer %d\n", route_to_coffer(creative_query));

    fprintf(stderr, "=== Test Complete ===\n\n");
}

#endif /* GGML_RAM_COFFERS_H */
