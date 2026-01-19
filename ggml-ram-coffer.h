/*
 * ggml-ram-coffer.h - NUMA-Aware RAM Weight Indexing for POWER8
 *
 * Scott's Vision: "Selectively house model information in known RAM banks"
 *
 * Instead of linear memory access across 576GB:
 * 1. INDEX where each layer/tensor lives (which NUMA node)
 * 2. PREFETCH from the right bank before computation
 * 3. SKIP weights we don't need (non-bijunctive)
 * 4. Process on CPUs LOCAL to that memory
 *
 * This enables running 70B-405B models at reasonable speeds by:
 * - Eliminating random memory access patterns
 * - Maximizing NUMA locality
 * - Using vec_perm collapse to reduce what we need to fetch
 */

#ifndef GGML_RAM_COFFER_H
#define GGML_RAM_COFFER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <numa.h>
#include <numaif.h>
#include <sched.h>

/*===========================================================================
 * POWER8 S824 NUMA Configuration
 *
 * Node 0: 130GB, CPUs 0-31   (distance to 1: 20, to 2-3: 40)
 * Node 1: 190GB, CPUs 32-63  (distance to 0: 20, to 2-3: 40)
 * Node 2:  65GB, CPUs 64-95  (distance to 3: 20, to 0-1: 40)
 * Node 3: 195GB, CPUs 96-127 (distance to 2: 20, to 0-1: 40)
 *
 * Strategy: Pair nodes for bandwidth
 * - Fast pair A: Node 0 + Node 1 (320GB, distance 20)
 * - Fast pair B: Node 2 + Node 3 (260GB, distance 20)
 *===========================================================================*/

#define NUM_NUMA_NODES 4
#define COFFER_MAX_LAYERS 128
#define COFFER_MAX_TENSORS 4096

/* NUMA node info */
typedef struct {
    int node_id;
    size_t total_bytes;
    size_t free_bytes;
    size_t used_bytes;
    int cpu_start;
    int cpu_end;
    int paired_node;     /* Fast pair partner */
} numa_node_info_t;

/* Tensor location in RAM coffer */
typedef struct {
    char name[64];       /* Tensor name (e.g., "layers.0.attention.wq") */
    int numa_node;       /* Which NUMA node holds this tensor */
    void* base_addr;     /* Base address in memory */
    size_t size_bytes;   /* Size of tensor */
    int layer_id;        /* Which layer (for prefetch planning) */
    int tensor_type;     /* 0=weight, 1=kv_cache, 2=activation */
} tensor_location_t;

/* RAM Coffer - the indexed weight store */
typedef struct {
    numa_node_info_t nodes[NUM_NUMA_NODES];
    tensor_location_t tensors[COFFER_MAX_TENSORS];
    int num_tensors;

    /* Layer → NUMA node mapping */
    int layer_to_node[COFFER_MAX_LAYERS];

    /* Statistics */
    uint64_t local_accesses;
    uint64_t remote_accesses;
    uint64_t prefetch_hits;
    uint64_t prefetch_misses;
} ram_coffer_t;

/* Global coffer instance */
static ram_coffer_t g_coffer = {0};

/*===========================================================================
 * Initialization
 *===========================================================================*/

static int coffer_init(void) {
    if (numa_available() < 0) {
        fprintf(stderr, "NUMA not available!\n");
        return -1;
    }

    int num_nodes = numa_num_configured_nodes();
    fprintf(stderr, "RAM Coffer: Detected %d NUMA nodes\n", num_nodes);

    for (int i = 0; i < num_nodes && i < NUM_NUMA_NODES; i++) {
        long long free_bytes, total_bytes;
        total_bytes = numa_node_size64(i, &free_bytes);

        g_coffer.nodes[i].node_id = i;
        g_coffer.nodes[i].total_bytes = total_bytes;
        g_coffer.nodes[i].free_bytes = free_bytes;
        g_coffer.nodes[i].used_bytes = 0;

        /* CPU ranges (POWER8 S824 specific) */
        g_coffer.nodes[i].cpu_start = i * 32;
        g_coffer.nodes[i].cpu_end = (i + 1) * 32 - 1;

        /* Paired nodes (fast access partners) */
        if (i == 0) g_coffer.nodes[i].paired_node = 1;
        else if (i == 1) g_coffer.nodes[i].paired_node = 0;
        else if (i == 2) g_coffer.nodes[i].paired_node = 3;
        else g_coffer.nodes[i].paired_node = 2;

        fprintf(stderr, "  Node %d: %.1f GB total, %.1f GB free, CPUs %d-%d, paired with %d\n",
                i,
                total_bytes / (1024.0 * 1024.0 * 1024.0),
                free_bytes / (1024.0 * 1024.0 * 1024.0),
                g_coffer.nodes[i].cpu_start,
                g_coffer.nodes[i].cpu_end,
                g_coffer.nodes[i].paired_node);
    }

    g_coffer.num_tensors = 0;
    return 0;
}

/*===========================================================================
 * Layer Placement Strategy
 *
 * For a 70B model with ~80 layers:
 * - Layers 0-19:  Node 0 (130GB) - embedding + early layers
 * - Layers 20-39: Node 1 (190GB) - middle layers
 * - Layers 40-59: Node 3 (195GB) - late layers
 * - Layers 60-79: Node 2 (65GB)  - output layers + lm_head
 * - KV Cache: Distributed across all nodes
 *===========================================================================*/

static int coffer_plan_layer_placement(int total_layers, size_t layer_size_bytes) {
    fprintf(stderr, "\nRAM Coffer: Planning placement for %d layers (%.1f MB each)\n",
            total_layers, layer_size_bytes / (1024.0 * 1024.0));

    /* Sort nodes by free space */
    int node_order[NUM_NUMA_NODES] = {1, 3, 0, 2};  /* Largest first */

    int layers_per_node = total_layers / NUM_NUMA_NODES;
    int remainder = total_layers % NUM_NUMA_NODES;

    int layer = 0;
    for (int n = 0; n < NUM_NUMA_NODES; n++) {
        int node = node_order[n];
        int node_layers = layers_per_node + (n < remainder ? 1 : 0);

        fprintf(stderr, "  Node %d: Layers %d-%d (%d layers, %.1f GB)\n",
                node, layer, layer + node_layers - 1, node_layers,
                node_layers * layer_size_bytes / (1024.0 * 1024.0 * 1024.0));

        for (int i = 0; i < node_layers && layer < COFFER_MAX_LAYERS; i++) {
            g_coffer.layer_to_node[layer++] = node;
        }
    }

    return 0;
}

/*===========================================================================
 * NUMA-Aware Allocation
 *===========================================================================*/

static void* coffer_alloc_on_node(size_t size, int numa_node, const char* name) {
    /* Allocate on specific NUMA node */
    void* ptr = numa_alloc_onnode(size, numa_node);
    if (!ptr) {
        fprintf(stderr, "Failed to allocate %.1f MB on node %d\n",
                size / (1024.0 * 1024.0), numa_node);
        return NULL;
    }

    /* Register in coffer */
    if (g_coffer.num_tensors < COFFER_MAX_TENSORS) {
        tensor_location_t* loc = &g_coffer.tensors[g_coffer.num_tensors++];
        strncpy(loc->name, name, sizeof(loc->name) - 1);
        loc->numa_node = numa_node;
        loc->base_addr = ptr;
        loc->size_bytes = size;
    }

    g_coffer.nodes[numa_node].used_bytes += size;

    return ptr;
}

/*===========================================================================
 * Prefetch - Tell the CPU to start loading data
 *
 * POWER8 prefetch instructions:
 * - dcbt: Data Cache Block Touch (L1)
 * - dcbtst: Data Cache Block Touch for Store
 * - dcbz: Data Cache Block Zero (allocate without fetch)
 *===========================================================================*/

/* Prefetch a cache line (128 bytes on POWER8) */
static inline void coffer_prefetch(const void* addr) {
#if defined(__powerpc64__) || defined(__powerpc__)
    __asm__ __volatile__("dcbt 0,%0" : : "r"(addr));
#endif
}

/* Prefetch an entire tensor (strided for cache efficiency) */
static inline void coffer_prefetch_tensor(const void* addr, size_t size) {
    const size_t cache_line = 128;
    const char* p = (const char*)addr;
    const char* end = p + size;

    /* Prefetch every cache line */
    while (p < end) {
        coffer_prefetch(p);
        p += cache_line;
    }
}

/* Prefetch layer weights before we need them */
static inline void coffer_prefetch_layer(int layer_id) {
    for (int i = 0; i < g_coffer.num_tensors; i++) {
        tensor_location_t* t = &g_coffer.tensors[i];
        if (t->layer_id == layer_id) {
            coffer_prefetch_tensor(t->base_addr, t->size_bytes);
            g_coffer.prefetch_hits++;
        }
    }
}

/*===========================================================================
 * CPU Affinity - Run computation on CPUs local to the memory
 *===========================================================================*/

static int coffer_bind_to_node(int numa_node) {
    struct bitmask* mask = numa_allocate_cpumask();
    numa_node_to_cpus(numa_node, mask);

    if (numa_sched_setaffinity(0, mask) < 0) {
        fprintf(stderr, "Failed to bind to node %d\n", numa_node);
        numa_free_cpumask(mask);
        return -1;
    }

    numa_free_cpumask(mask);
    return 0;
}

/* Bind current thread to the NUMA node containing a tensor */
static int coffer_bind_to_tensor(const char* tensor_name) {
    for (int i = 0; i < g_coffer.num_tensors; i++) {
        if (strcmp(g_coffer.tensors[i].name, tensor_name) == 0) {
            return coffer_bind_to_node(g_coffer.tensors[i].numa_node);
        }
    }
    return -1;
}

/*===========================================================================
 * Smart Access - Check if access is local or remote
 *===========================================================================*/

static int coffer_get_tensor_node(const void* addr) {
    int node = -1;
    get_mempolicy(&node, NULL, 0, (void*)addr, MPOL_F_NODE | MPOL_F_ADDR);
    return node;
}

static void coffer_record_access(const void* addr, int accessing_cpu) {
    int tensor_node = coffer_get_tensor_node(addr);
    int cpu_node = numa_node_of_cpu(accessing_cpu);

    if (tensor_node == cpu_node) {
        g_coffer.local_accesses++;
    } else {
        g_coffer.remote_accesses++;
    }
}

/*===========================================================================
 * Layer Processing with NUMA Awareness
 *
 * Key insight: Process layer on CPUs LOCAL to its weights
 *===========================================================================*/

typedef void (*layer_compute_fn)(void* layer_weights, void* input, void* output, int layer_id);

static void coffer_process_layer(
    int layer_id,
    void* input,
    void* output,
    layer_compute_fn compute_fn
) {
    /* Get NUMA node for this layer */
    int target_node = g_coffer.layer_to_node[layer_id];

    /* Prefetch next layer while processing this one */
    if (layer_id + 1 < COFFER_MAX_LAYERS) {
        coffer_prefetch_layer(layer_id + 1);
    }

    /* Find layer weights */
    void* weights = NULL;
    for (int i = 0; i < g_coffer.num_tensors; i++) {
        if (g_coffer.tensors[i].layer_id == layer_id &&
            g_coffer.tensors[i].tensor_type == 0) {
            weights = g_coffer.tensors[i].base_addr;
            break;
        }
    }

    if (!weights) {
        fprintf(stderr, "Layer %d weights not found in coffer!\n", layer_id);
        return;
    }

    /* Bind to local CPUs */
    coffer_bind_to_node(target_node);

    /* Process */
    compute_fn(weights, input, output, layer_id);
}

/*===========================================================================
 * Statistics
 *===========================================================================*/

static void coffer_print_stats(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  RAM Coffer Statistics                                    ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Tensors registered: %10d                           ║\n",
            g_coffer.num_tensors);
    fprintf(stderr, "║  Local accesses:     %10lu                           ║\n",
            (unsigned long)g_coffer.local_accesses);
    fprintf(stderr, "║  Remote accesses:    %10lu                           ║\n",
            (unsigned long)g_coffer.remote_accesses);
    fprintf(stderr, "║  Locality ratio:     %10.1f%%                          ║\n",
            g_coffer.local_accesses + g_coffer.remote_accesses > 0 ?
            100.0 * g_coffer.local_accesses /
            (g_coffer.local_accesses + g_coffer.remote_accesses) : 0);
    fprintf(stderr, "║  Prefetch hits:      %10lu                           ║\n",
            (unsigned long)g_coffer.prefetch_hits);
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  NUMA Node Usage:                                         ║\n");
    for (int i = 0; i < NUM_NUMA_NODES; i++) {
        fprintf(stderr, "║    Node %d: %6.1f GB / %6.1f GB (%.1f%%)                   ║\n",
                i,
                g_coffer.nodes[i].used_bytes / (1024.0 * 1024.0 * 1024.0),
                g_coffer.nodes[i].total_bytes / (1024.0 * 1024.0 * 1024.0),
                100.0 * g_coffer.nodes[i].used_bytes / g_coffer.nodes[i].total_bytes);
    }
    fprintf(stderr, "╚═══════════════════════════════════════════════════════════╝\n");
}

/*===========================================================================
 * Model Loading with Coffer Placement
 *
 * This would integrate with ggml model loading to place tensors
 * on appropriate NUMA nodes.
 *===========================================================================*/

typedef struct {
    int num_layers;
    size_t layer_size;
    size_t embedding_size;
    size_t lm_head_size;
    size_t kv_cache_per_layer;
} model_topology_t;

static int coffer_plan_model(model_topology_t* model) {
    size_t total_size = model->embedding_size +
                        model->num_layers * model->layer_size +
                        model->lm_head_size +
                        model->num_layers * model->kv_cache_per_layer;

    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  RAM Coffer Model Planning                                ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Model size:        %10.1f GB                        ║\n",
            total_size / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "║  Layers:            %10d                            ║\n",
            model->num_layers);
    fprintf(stderr, "║  Layer size:        %10.1f MB                        ║\n",
            model->layer_size / (1024.0 * 1024.0));
    fprintf(stderr, "║  KV cache/layer:    %10.1f MB                        ║\n",
            model->kv_cache_per_layer / (1024.0 * 1024.0));
    fprintf(stderr, "╚═══════════════════════════════════════════════════════════╝\n");

    /* Check if model fits */
    size_t total_free = 0;
    for (int i = 0; i < NUM_NUMA_NODES; i++) {
        total_free += g_coffer.nodes[i].free_bytes;
    }

    if (total_size > total_free) {
        fprintf(stderr, "ERROR: Model (%.1f GB) exceeds available RAM (%.1f GB)!\n",
                total_size / (1024.0 * 1024.0 * 1024.0),
                total_free / (1024.0 * 1024.0 * 1024.0));
        return -1;
    }

    /* Plan layer placement */
    coffer_plan_layer_placement(model->num_layers, model->layer_size);

    return 0;
}

#endif /* GGML_RAM_COFFER_H */
