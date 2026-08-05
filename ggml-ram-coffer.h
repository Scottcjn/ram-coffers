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

/*
 * Capability detection + NUMA/intrinsic shims. This replaces the former
 * unconditional <numa.h>/<numaif.h>/<sched.h> includes, which made this
 * header fail to COMPILE on any machine without libnuma-dev.
 */
#include "coffers-portability.h"

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

/*
 * NUM_NUMA_NODES is the array CAPACITY, not an assumption about the machine.
 * The number of nodes actually present is discovered at init and stored in
 * ram_coffer_t::n_nodes; every loop below bounds on that, never on this.
 */
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
    int n_nodes;         /* Nodes actually detected (>=1, <=NUM_NUMA_NODES) */
    int uniform;         /* 1 = uniform-memory mode (no NUMA / single node) */
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

/*
 * Initialise the coffer.
 *
 * NEVER fails just because NUMA is missing. When libnuma is absent, or
 * present but reporting a single node, we fall back to UNIFORM-MEMORY MODE:
 * one coffer covering all of RAM. Everything downstream (placement,
 * prefetch, load, stats) still works - it simply has no node affinity to
 * exploit. Returns 0 on success; -1 only on a genuine error.
 */
static int coffer_init(void) {
    const coffers_topology_t* topo = coffers_topology();
    coffers_report_mode();

    g_coffer.uniform = topo->uniform;
    g_coffer.n_nodes = topo->n_nodes < NUM_NUMA_NODES ? topo->n_nodes : NUM_NUMA_NODES;
    if (g_coffer.n_nodes < 1) g_coffer.n_nodes = 1;

    /* Distribute the machine's CPUs across the detected nodes. */
    int n_cpus = 1;
#if defined(_SC_NPROCESSORS_ONLN)
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online > 0) n_cpus = (int)online;
#endif
    int cpus_per_node = n_cpus / g_coffer.n_nodes;
    if (cpus_per_node < 1) cpus_per_node = 1;

    for (int i = 0; i < g_coffer.n_nodes; i++) {
        size_t free_bytes = 0;
        size_t total_bytes = coffers_node_memory(i, &free_bytes);

        g_coffer.nodes[i].node_id     = i;
        g_coffer.nodes[i].total_bytes = total_bytes;
        g_coffer.nodes[i].free_bytes  = free_bytes;
        g_coffer.nodes[i].used_bytes  = 0;

        g_coffer.nodes[i].cpu_start = i * cpus_per_node;
        g_coffer.nodes[i].cpu_end   = (i + 1) * cpus_per_node - 1;

        /*
         * Pair each node with its neighbour for bandwidth (POWER8: 0<->1,
         * 2<->3). With an odd node count the last node pairs with itself.
         */
        int partner = (i % 2 == 0) ? i + 1 : i - 1;
        if (partner >= g_coffer.n_nodes) partner = i;
        g_coffer.nodes[i].paired_node = partner;

        fprintf(stderr,
                "  Node %d: %.1f GB total, %.1f GB free, CPUs %d-%d, paired with %d\n",
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

    int n_nodes = g_coffer.n_nodes > 0 ? g_coffer.n_nodes : 1;

    /*
     * Sort nodes by free space, largest first. This used to be the hardcoded
     * literal {1,3,0,2} - correct only for the POWER8 S824, and an
     * out-of-bounds node id on any machine with fewer than 4 nodes.
     */
    int node_order[NUM_NUMA_NODES];
    for (int i = 0; i < n_nodes; i++) node_order[i] = i;
    for (int i = 0; i < n_nodes - 1; i++) {
        for (int j = i + 1; j < n_nodes; j++) {
            if (g_coffer.nodes[node_order[j]].free_bytes >
                g_coffer.nodes[node_order[i]].free_bytes) {
                int tmp = node_order[i]; node_order[i] = node_order[j]; node_order[j] = tmp;
            }
        }
    }

    int layers_per_node = total_layers / n_nodes;
    int remainder = total_layers % n_nodes;

    int layer = 0;
    for (int n = 0; n < n_nodes; n++) {
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
    /* Clamp to a node that actually exists (uniform mode always uses node 0) */
    if (numa_node < 0 || numa_node >= g_coffer.n_nodes) numa_node = COFFERS_UNIFORM_NODE;

    /* Allocate on specific NUMA node (plain malloc in uniform mode) */
    void* ptr = coffers_alloc_on_node(size, numa_node);
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

/*
 * Prefetch a cache line. dcbt on POWER8 (128-byte line), __builtin_prefetch
 * elsewhere, no-op on compilers offering neither. See coffers-portability.h.
 */
static inline void coffer_prefetch(const void* addr) {
    coffers_prefetch(addr);
}

/* Prefetch an entire tensor (strided for cache efficiency) */
static inline void coffer_prefetch_tensor(const void* addr, size_t size) {
    coffers_prefetch_range(addr, size);
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

/*
 * Bind the calling thread to the CPUs local to a node.
 * In uniform-memory mode this is a successful no-op - there is nowhere else
 * to run, so "already local" is the correct answer, not a failure.
 */
static int coffer_bind_to_node(int numa_node) {
    if (g_coffer.uniform) return 0;
    if (numa_node < 0 || numa_node >= g_coffer.n_nodes) return 0;

    if (coffers_bind_thread_to_node(numa_node) < 0) {
        fprintf(stderr, "Failed to bind to node %d\n", numa_node);
        return -1;
    }
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
    return coffers_node_of_addr(addr);
}

static void coffer_record_access(const void* addr, int accessing_cpu) {
    int tensor_node = coffer_get_tensor_node(addr);
    int cpu_node = coffers_node_of_cpu(accessing_cpu);

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
    fprintf(stderr, "║  %-56s ║\n",
            g_coffer.uniform ? "Memory Usage (uniform mode):" : "NUMA Node Usage:");
    for (int i = 0; i < g_coffer.n_nodes; i++) {
        double total = (double)g_coffer.nodes[i].total_bytes;
        fprintf(stderr, "║    Node %d: %6.1f GB / %6.1f GB (%.1f%%)                   ║\n",
                i,
                g_coffer.nodes[i].used_bytes / (1024.0 * 1024.0 * 1024.0),
                total / (1024.0 * 1024.0 * 1024.0),
                total > 0.0 ? 100.0 * g_coffer.nodes[i].used_bytes / total : 0.0);
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
    for (int i = 0; i < g_coffer.n_nodes; i++) {
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
