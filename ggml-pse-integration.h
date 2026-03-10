/*
 * GGML PSE Integration - Master Header for POWER8/9
 * Elyan Labs 2025
 *
 * COMPLETE INTEGRATION including:
 * - Vec_perm non-bijunctive collapse
 * - PowerLISP symbolic layer
 * - Neuromorphic cognitive routing
 */

#ifndef GGML_PSE_INTEGRATION_H
#define GGML_PSE_INTEGRATION_H

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * PSE COMPONENT INCLUDES
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Core PSE components */
#include "power8-compat.h"
#include "ggml-intelligent-collapse.h"
#include "ggml-vcipher-collapse.h"     /* Hardware crypto collapse (ISA 2.07 vcipher) */
#include "ggml-topk-collapse-vsx.h"
#include "pse-entropy-burst.h"
#include "ggml-ram-coffers.h"
#include "ggml-dcbt-resident.h"
#include "ggml-sparse-softmax.h"  /* Skip-before-softmax pruning */  /* L2/L3 resident prefetch - THE 147 t/s enabler! */

/* PowerLISP Symbolic Integration (NEW!) */
#include "ggml-pse-symbolic-gate.h"
#include "ggml-neuromorphic-coffers.h"
#include "ggml-symbolic-neural-bridge.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * PSE VERSION - Now includes PowerLISP
 * ═══════════════════════════════════════════════════════════════════════════ */

#define PSE_VERSION_MAJOR 4
#define PSE_VERSION_MINOR 0
#define PSE_VERSION_PATCH 0
#define PSE_VERSION_STRING "4.0.0-vcipher"

/* Alias constants */
#define PSE_TOPK_DEFAULT INTELLIGENT_COLLAPSE_TOP_K
#define PSE_AMPLIFY_DEFAULT INTELLIGENT_COLLAPSE_AMPLIFY
#define PSE_PRUNE_THRESHOLD 0.01f
#define PSE_ENTROPY_STRENGTH PSE_BURST_STRENGTH

/* ═══════════════════════════════════════════════════════════════════════════
 * PSE STATE STRUCTURES
 * ═══════════════════════════════════════════════════════════════════════════ */

/* PSE collapse state */
typedef struct {
    int topk;
    float amplify_factor;
    float prune_threshold;
    float entropy_strength;
    int burst_interval;
    uint64_t token_count;
    uint64_t last_entropy;
} pse_collapse_state_t;

/* Global PSE state */
typedef struct {
    bool initialized;
    bool enabled;
    bool powerlisp_enabled;     /* NEW: Symbolic layer active */
    bool neuromorphic_enabled;  /* NEW: Cognitive routing active */
    int numa_nodes;
    size_t total_ram_gb;
    const char* arch_name;
} pse_global_state_t;

/* Global state instances */
#ifdef __cplusplus
extern "C" {
#endif
extern pse_collapse_state_t g_pse_state;
extern pse_global_state_t g_pse_global;
#ifdef __cplusplus
}
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * POWERLISP SYMBOLIC GATING
 *
 * Instead of collapsing EVERY softmax, use symbolic rules to decide WHEN
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Wrapper for collapse decision that uses symbolic gating
 */
static inline bool pse_should_collapse_gated(
    int ne00,           /* Attention window */
    int ne01,           /* Batch size */
    int head_id,
    int position
) {
    /* If PowerLISP disabled, always collapse */
    if (!g_pse_global.powerlisp_enabled) {
        return true;
    }

    /* Use symbolic gate rules */
    return pse_should_collapse(ne00, ne01, head_id, position);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * NEUROMORPHIC COGNITIVE ROUTING
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Route attention to appropriate cognitive function
 */
static inline int pse_cognitive_route(const float* query_embed, int embed_dim __attribute__((unused))) {
    if (!g_pse_global.neuromorphic_enabled) {
        return 0;  /* Default coffer */
    }

    /* Use neuromorphic routing */
    return route_to_coffer(query_embed);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void pse_print_banner(void) {
    fprintf(stderr, "\nPSE Vec_Perm Collapse Active - POWER8 S824\n");
    fprintf(stderr, " - Top-K: %d | Amplify: %.2f | Entropy: mftb\n",
           INTELLIGENT_COLLAPSE_TOP_K, (double)INTELLIGENT_COLLAPSE_AMPLIFY);
    fprintf(stderr, " - RAM Coffers: ENABLED | NUMA-aware weight banking\n");
    if (g_pse_global.powerlisp_enabled) {
        fprintf(stderr, " - PowerLISP Symbolic: ENABLED | Tetranary logic\n");
    }
    if (g_pse_global.neuromorphic_enabled) {
        fprintf(stderr, " - Neuromorphic Routing: ENABLED | Brain hemisphere mapping\n");
    }
    fprintf(stderr, "\n");
}

static inline void intelligent_collapse_init(void) {
    /* Initialize collapse stats */
}

static inline bool pse_env_disable(void) {
    const char * v = getenv("GGML_PSE_DISABLE");
    if (v && (v[0] == 0x31 || v[0] == 0x74 || v[0] == 0x54 || v[0] == 0x79 || v[0] == 0x59)) {
        return true;
    }
    return false;
}

static inline bool pse_env_disable_powerlisp(void) {
    const char * v = getenv("GGML_PSE_NO_POWERLISP");
    if (v && (v[0] == 0x31 || v[0] == 0x74 || v[0] == 0x54 || v[0] == 0x79 || v[0] == 0x59)) {
        return true;
    }
    return false;
}

/* Check if PSE is currently enabled */
static inline bool pse_is_enabled(void) {
    return g_pse_global.initialized && g_pse_global.enabled;
}

static inline int pse_init(void) {
    if (g_pse_global.initialized) return 0;
    g_pse_global.enabled = true;
    g_pse_global.powerlisp_enabled = true;
    g_pse_global.neuromorphic_enabled = true;

    if (pse_env_disable()) {
        g_pse_global.enabled = false;
    }

    if (pse_env_disable_powerlisp()) {
        g_pse_global.powerlisp_enabled = false;
        g_pse_global.neuromorphic_enabled = false;
    }

#ifdef __powerpc__
    g_pse_global.arch_name = "POWER8/9";
    if (g_pse_global.enabled) {
        coffer_init_numa();
        /* Count initialized coffers */
        for (int i = 0; i < MAX_COFFERS; i++) {
            if (g_coffers[i].numa_node >= 0) {
                g_pse_global.numa_nodes++;
                /* Convert bytes to GB */
                g_pse_global.total_ram_gb += g_coffers[i].mmap_size / (1024ULL * 1024 * 1024);
            }
        }
        intelligent_collapse_init();

        /* Initialize PowerLISP symbolic layer */
        if (g_pse_global.powerlisp_enabled) {
            init_symbolic_neural_bridge();
        }
    }
#endif
    g_pse_global.initialized = true;
    return 0;
}

static inline void pse_print_startup_banner(void) {
    fprintf(stderr, "\n════════════════════════════════════════════════════════════════\n");
    if (!g_pse_global.enabled) {
        fprintf(stderr, "  PSE Module Disabled\n");
        fprintf(stderr, "════════════════════════════════════════════════════════════════\n\n");
        return;
    }
    fprintf(stderr, "  POWER8 PSE v%s - vcipher HARDWARE CRYPTO COLLAPSE\n", PSE_VERSION_STRING);
    fprintf(stderr, "────────────────────────────────────────────────────────────────\n");
    fprintf(stderr, "   - Hebbian vcipher collapse: ACTIVE (ISA 2.07 hw crypto)\n");
    fprintf(stderr, "   - NUMA neuromorphic coffers: ACTIVE\n");
    fprintf(stderr, "   - L3 resident prefetch: ACTIVE (dcbt)\n");
    fprintf(stderr, "   - IBM MASS vsexp/vstanh: ACTIVE\n");
    if (g_pse_global.powerlisp_enabled) {
        fprintf(stderr, "   - PowerLISP Symbolic Layer: ACTIVE\n");
        fprintf(stderr, "   - Tetranary Logic (4-state): ACTIVE\n");
        fprintf(stderr, "   - Neural ↔ Symbolic Bridge: ACTIVE\n");
    }
    if (g_pse_global.neuromorphic_enabled) {
        fprintf(stderr, "   - Cognitive Hemisphere Routing: ACTIVE\n");
    }
    fprintf(stderr, "════════════════════════════════════════════════════════════════\n\n");
    pse_print_banner();
    vcipher_collapse_banner();
    fprintf(stderr, "PSE Burst Entropy Active\n");
    fprintf(stderr, " - Interval: %d | Strength: %.3f | TopK: %d\n",
           PSE_BURST_INTERVAL, (double)PSE_BURST_STRENGTH, PSE_TOPK_ENTROPY);
    fprintf(stderr, "\n");
#ifdef __powerpc__
    if (g_coffers_initialized) {
        coffer_print_stats();
    }
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
 * STATISTICS REPORTING
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void pse_print_all_stats(void) {
    /* Symbolic gate stats */
    pse_gate_report();

    /* Symbolic bridge stats */
    if (g_pse_global.powerlisp_enabled) {
        print_bridge_stats();
    }

    /* Coffer stats */
#ifdef __powerpc__
    if (g_coffers_initialized) {
        coffer_print_stats();
    }
#endif
}

#endif /* GGML_PSE_INTEGRATION_H */
