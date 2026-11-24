/*
 * ggml-coffer-mmap.h - NUMA-Aware GGUF mmap Sharding for POWER8
 *
 * Scott's Vision: "Shard weights across NUMA nodes via mmap"
 *
 * Strategy:
 * 1. Parse GGUF header to find tensor locations
 * 2. mmap the file with MAP_POPULATE for prefetch
 * 3. Use mbind() to migrate tensor pages to target NUMA nodes
 * 4. Layer-based placement: early layers → Node 0, late layers → Node 3
 *
 * This enables running huge models (70B-405B) by placing weights
 * close to the CPUs that will process them.
 */

#ifndef GGML_COFFER_MMAP_H
#define GGML_COFFER_MMAP_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <numa.h>
#include <numaif.h>

/*===========================================================================
 * GGUF Format Structures (minimal parser)
 *===========================================================================*/

#define GGUF_MAGIC 0x46554747  /* "GGUF" */

/* GGUF value types */
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

/* GGUF header */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
} gguf_header_t;

/* Tensor metadata after parsing */
typedef struct {
    char name[128];
    uint64_t offset;      /* Offset from tensor data start */
    uint64_t size_bytes;  /* Size of tensor data */
    int n_dims;
    uint64_t dims[4];
    int ggml_type;        /* Quantization type */
    int target_node;      /* NUMA node for this tensor */
    int layer_id;         /* Extracted layer number (-1 if not a layer tensor) */
} coffer_tensor_info_t;

/* Coffer mmap context */
typedef struct {
    int fd;
    void* mapped_addr;
    size_t file_size;

    /* GGUF parsing results */
    gguf_header_t header;
    uint64_t tensor_data_offset;  /* Where tensor data starts */

    /* Tensor registry */
    coffer_tensor_info_t* tensors;
    int n_tensors;

    /* NUMA placement stats */
    size_t bytes_per_node[4];
} coffer_mmap_ctx_t;

/*===========================================================================
 * Layer ID Extraction from Tensor Name
 *
 * Examples:
 *   "blk.0.attn_q.weight" → layer 0
 *   "blk.15.ffn_up.weight" → layer 15
 *   "token_embd.weight" → layer -1 (embedding)
 *   "output.weight" → layer -1 (output)
 *===========================================================================*/

static int extract_layer_id(const char* name) {
    /* Look for "blk.N." or "layers.N." pattern */
    const char* p;

    p = strstr(name, "blk.");
    if (p) {
        return atoi(p + 4);
    }

    p = strstr(name, "layers.");
    if (p) {
        return atoi(p + 7);
    }

    p = strstr(name, "layer.");
    if (p) {
        return atoi(p + 6);
    }

    return -1;  /* Not a layer tensor */
}

/*===========================================================================
 * NUMA Node Assignment Strategy
 *
 * POWER8 S824: 4 nodes with varying sizes
 * - Node 0: 130GB (embedding + early layers)
 * - Node 1: 190GB (middle layers - largest)
 * - Node 3: 195GB (late layers - largest)
 * - Node 2:  65GB (output + overflow)
 *
 * Strategy: Distribute layers evenly by memory, not count
 *===========================================================================*/

static int assign_numa_node(int layer_id, int total_layers, const char* tensor_name) {
    /* Special tensors */
    if (strstr(tensor_name, "token_embd") || strstr(tensor_name, "embed")) {
        return 0;  /* Embedding on Node 0 */
    }
    if (strstr(tensor_name, "output") || strstr(tensor_name, "lm_head")) {
        return 2;  /* Output on Node 2 (smallest) */
    }

    /* Layer-based distribution */
    if (layer_id < 0) {
        return 0;  /* Unknown goes to Node 0 */
    }

    /* Split layers across nodes */
    float progress = (float)layer_id / (total_layers > 0 ? total_layers : 1);

    if (progress < 0.25f) {
        return 0;  /* Early layers → Node 0 */
    } else if (progress < 0.50f) {
        return 1;  /* Quarter 2 → Node 1 */
    } else if (progress < 0.75f) {
        return 3;  /* Quarter 3 → Node 3 */
    } else {
        return 2;  /* Late layers → Node 2 */
    }
}

/*===========================================================================
 * GGUF Minimal Parser - Extract Tensor Locations
 *===========================================================================*/

static int gguf_read_bytes(
    const coffer_mmap_ctx_t* ctx,
    size_t* pos,
    void* out,
    size_t size
) {
    if (*pos > ctx->file_size || size > ctx->file_size - *pos) {
        return -1;
    }
    if (out && size > 0) {
        memcpy(out, (const uint8_t*)ctx->mapped_addr + *pos, size);
    }
    *pos += size;
    return 0;
}

static int gguf_read_u32_checked(
    const coffer_mmap_ctx_t* ctx,
    size_t* pos,
    uint32_t* value
) {
    return gguf_read_bytes(ctx, pos, value, sizeof(*value));
}

static int gguf_read_u64_checked(
    const coffer_mmap_ctx_t* ctx,
    size_t* pos,
    uint64_t* value
) {
    return gguf_read_bytes(ctx, pos, value, sizeof(*value));
}

static int gguf_read_string_checked(
    const coffer_mmap_ctx_t* ctx,
    size_t* pos,
    char* out,
    size_t out_size
) {
    uint64_t raw_len = 0;
    if (gguf_read_u64_checked(ctx, pos, &raw_len) < 0) {
        return -1;
    }
    if (*pos > ctx->file_size || raw_len > ctx->file_size - *pos) {
        return -1;
    }

    size_t len = (size_t)raw_len;
    if ((uint64_t)len != raw_len) {
        return -1;
    }
    if (out && out_size > 0) {
        size_t copy_len = len < out_size - 1 ? len : out_size - 1;
        if (copy_len > 0) {
            memcpy(out, (const uint8_t*)ctx->mapped_addr + *pos, copy_len);
        }
        out[copy_len] = '\0';
    }
    *pos += len;
    return 0;
}

static size_t gguf_scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return 8;
        default:
            return 0;
    }
}

static int gguf_skip_value(
    const coffer_mmap_ctx_t* ctx,
    size_t* pos,
    uint32_t type
) {
    if (type == GGUF_TYPE_STRING) {
        return gguf_read_string_checked(ctx, pos, NULL, 0);
    }

    if (type == GGUF_TYPE_ARRAY) {
        uint32_t element_type = 0;
        uint64_t raw_count = 0;
        if (gguf_read_u32_checked(ctx, pos, &element_type) < 0 ||
            gguf_read_u64_checked(ctx, pos, &raw_count) < 0) {
            return -1;
        }
        if (element_type == GGUF_TYPE_ARRAY) {
            return -1;
        }

        size_t count = (size_t)raw_count;
        if ((uint64_t)count != raw_count) {
            return -1;
        }
        if (element_type == GGUF_TYPE_STRING) {
            for (size_t i = 0; i < count; i++) {
                if (gguf_read_string_checked(ctx, pos, NULL, 0) < 0) {
                    return -1;
                }
            }
            return 0;
        }

        size_t element_size = gguf_scalar_size(element_type);
        if (element_size == 0 || count > SIZE_MAX / element_size) {
            return -1;
        }
        return gguf_read_bytes(ctx, pos, NULL, count * element_size);
    }

    size_t scalar_size = gguf_scalar_size(type);
    if (scalar_size == 0) {
        return -1;
    }
    return gguf_read_bytes(ctx, pos, NULL, scalar_size);
}

static int coffer_parse_gguf(coffer_mmap_ctx_t* ctx) {
    size_t pos = 0;
    uint32_t alignment = 32;

    if (!ctx || !ctx->mapped_addr || ctx->file_size < sizeof(gguf_header_t)) {
        return -1;
    }

    if (gguf_read_bytes(ctx, &pos, &ctx->header, sizeof(ctx->header)) < 0) {
        return -1;
    }
    if (ctx->header.magic != GGUF_MAGIC) {
        fprintf(stderr, "Coffer: Invalid GGUF magic (got 0x%08X)\n", ctx->header.magic);
        return -1;
    }
    if (ctx->header.version < 2 || ctx->header.version > 3) {
        fprintf(stderr, "Coffer: Unsupported GGUF version %u\n", ctx->header.version);
        return -1;
    }
    if (ctx->header.n_tensors > INT_MAX ||
        ctx->header.n_tensors > SIZE_MAX / sizeof(coffer_tensor_info_t)) {
        fprintf(stderr, "Coffer: Too many tensors in GGUF\n");
        return -1;
    }

    fprintf(stderr, "Coffer: GGUF v%u, %lu tensors, %lu KV pairs\n",
            ctx->header.version,
            (unsigned long)ctx->header.n_tensors,
            (unsigned long)ctx->header.n_kv);

    for (uint64_t i = 0; i < ctx->header.n_kv; i++) {
        char key[128];
        uint32_t type = 0;
        if (gguf_read_string_checked(ctx, &pos, key, sizeof(key)) < 0 ||
            gguf_read_u32_checked(ctx, &pos, &type) < 0) {
            goto parse_error;
        }

        if (strcmp(key, "general.alignment") == 0 && type == GGUF_TYPE_UINT32) {
            uint32_t parsed_alignment = 0;
            if (gguf_read_u32_checked(ctx, &pos, &parsed_alignment) < 0 ||
                parsed_alignment == 0) {
                goto parse_error;
            }
            alignment = parsed_alignment;
        } else if (gguf_skip_value(ctx, &pos, type) < 0) {
            goto parse_error;
        }
    }

    ctx->n_tensors = (int)ctx->header.n_tensors;
    if (ctx->n_tensors > 0) {
        ctx->tensors = (coffer_tensor_info_t*)calloc(
            (size_t)ctx->n_tensors,
            sizeof(coffer_tensor_info_t)
        );
        if (!ctx->tensors) {
            goto parse_error;
        }
    }

    for (int i = 0; i < ctx->n_tensors; i++) {
        coffer_tensor_info_t* tensor = &ctx->tensors[i];
        uint32_t n_dims = 0;
        uint32_t ggml_type = 0;

        if (gguf_read_string_checked(ctx, &pos, tensor->name, sizeof(tensor->name)) < 0 ||
            gguf_read_u32_checked(ctx, &pos, &n_dims) < 0 ||
            n_dims == 0 || n_dims > 4) {
            goto parse_error;
        }

        tensor->n_dims = (int)n_dims;
        for (uint32_t dim = 0; dim < n_dims; dim++) {
            if (gguf_read_u64_checked(ctx, &pos, &tensor->dims[dim]) < 0 ||
                tensor->dims[dim] == 0) {
                goto parse_error;
            }
        }
        if (gguf_read_u32_checked(ctx, &pos, &ggml_type) < 0 ||
            gguf_read_u64_checked(ctx, &pos, &tensor->offset) < 0) {
            goto parse_error;
        }
        tensor->ggml_type = (int)ggml_type;
        tensor->layer_id = extract_layer_id(tensor->name);
    }

    {
        size_t remainder = pos % alignment;
        size_t padding = remainder == 0 ? 0 : alignment - remainder;
        if (padding > ctx->file_size - pos) {
            goto parse_error;
        }
        pos += padding;
    }
    ctx->tensor_data_offset = (uint64_t)pos;

    if (ctx->tensor_data_offset > ctx->file_size) {
        goto parse_error;
    }

    size_t tensor_data_size = ctx->file_size - (size_t)ctx->tensor_data_offset;
    for (int i = 0; i < ctx->n_tensors; i++) {
        coffer_tensor_info_t* tensor = &ctx->tensors[i];
        if (tensor->offset > tensor_data_size) {
            goto parse_error;
        }

        /* Use the file span to the next tensor rather than a GGML type table.
         * This remains correct for placement and supports new quantization types. */
        uint64_t next_offset = tensor_data_size;
        for (int j = 0; j < ctx->n_tensors; j++) {
            uint64_t candidate = ctx->tensors[j].offset;
            if (candidate > tensor->offset && candidate < next_offset) {
                next_offset = candidate;
            }
        }
        if (next_offset < tensor->offset) {
            goto parse_error;
        }
        uint64_t span = next_offset - tensor->offset;
        if ((uint64_t)(size_t)span != span) {
            goto parse_error;
        }
        tensor->size_bytes = span;
        if (tensor->size_bytes == 0) {
            goto parse_error;
        }
    }

    return 0;

parse_error:
    fprintf(stderr, "Coffer: Malformed or truncated GGUF metadata\n");
    free(ctx->tensors);
    ctx->tensors = NULL;
    ctx->n_tensors = 0;
    ctx->tensor_data_offset = 0;
    return -1;
}

/*===========================================================================
 * MMAP with NUMA Placement
 *
 * Strategy:
 * 1. mmap entire file first
 * 2. Parse to find tensor boundaries
 * 3. Use mbind() to migrate pages to target nodes
 *===========================================================================*/

static coffer_mmap_ctx_t* coffer_mmap_open(const char* path) {
    coffer_mmap_ctx_t* ctx = (coffer_mmap_ctx_t*)calloc(1, sizeof(coffer_mmap_ctx_t));
    if (!ctx) return NULL;

    /* Open file */
    ctx->fd = open(path, O_RDONLY);
    if (ctx->fd < 0) {
        fprintf(stderr, "Coffer: Cannot open %s\n", path);
        free(ctx);
        return NULL;
    }

    /* Get file size */
    struct stat st;
    fstat(ctx->fd, &st);
    ctx->file_size = st.st_size;

    fprintf(stderr, "Coffer: Opening %s (%.2f GB)\n",
            path, ctx->file_size / (1024.0 * 1024.0 * 1024.0));

    /* mmap with MAP_POPULATE to prefetch */
    ctx->mapped_addr = mmap(NULL, ctx->file_size, PROT_READ,
                            MAP_PRIVATE | MAP_POPULATE, ctx->fd, 0);
    if (ctx->mapped_addr == MAP_FAILED) {
        fprintf(stderr, "Coffer: mmap failed\n");
        close(ctx->fd);
        free(ctx);
        return NULL;
    }

    /* Parse GGUF structure */
    if (coffer_parse_gguf(ctx) < 0) {
        munmap(ctx->mapped_addr, ctx->file_size);
        close(ctx->fd);
        free(ctx);
        return NULL;
    }

    return ctx;
}

/*===========================================================================
 * Page Migration to NUMA Nodes
 *
 * Uses mbind() to move pages to specific NUMA nodes.
 * This is the key to NUMA-aware inference!
 *===========================================================================*/

static int coffer_migrate_region(void* addr, size_t size, int target_node) {
    if (numa_available() < 0) {
        return -1;
    }

    /* Create node mask for target */
    unsigned long nodemask = 1UL << target_node;

    /* Align to page boundary */
    size_t page_size = sysconf(_SC_PAGESIZE);
    uintptr_t aligned_addr = (uintptr_t)addr & ~(page_size - 1);
    size_t aligned_size = size + ((uintptr_t)addr - aligned_addr);
    aligned_size = (aligned_size + page_size - 1) & ~(page_size - 1);

    /* Migrate pages */
    int ret = mbind((void*)aligned_addr, aligned_size, MPOL_BIND,
                    &nodemask, sizeof(nodemask) * 8, MPOL_MF_MOVE);

    if (ret < 0) {
        /* mbind can fail if pages are shared, try MPOL_PREFERRED instead */
        ret = mbind((void*)aligned_addr, aligned_size, MPOL_PREFERRED,
                    &nodemask, sizeof(nodemask) * 8, 0);
    }

    return ret;
}

/*===========================================================================
 * Smart Tensor Placement
 *
 * Given tensor info, place it on the optimal NUMA node
 *===========================================================================*/

static int coffer_place_tensors(coffer_mmap_ctx_t* ctx, int total_layers) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Coffer NUMA Tensor Placement                             ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════╣\n");

    size_t migrated[4] = {0, 0, 0, 0};
    int placed = 0;

    for (int i = 0; i < ctx->n_tensors; i++) {
        coffer_tensor_info_t* t = &ctx->tensors[i];

        /* Offset 0 is valid for the first tensor; size is the validity check. */
        if (t->size_bytes == 0) continue;

        /* Determine target node */
        t->layer_id = extract_layer_id(t->name);
        t->target_node = assign_numa_node(t->layer_id, total_layers, t->name);

        /* Calculate tensor address */
        void* tensor_addr = (uint8_t*)ctx->mapped_addr + ctx->tensor_data_offset + t->offset;

        /* Migrate to target node */
        if (coffer_migrate_region(tensor_addr, t->size_bytes, t->target_node) >= 0) {
            migrated[t->target_node] += t->size_bytes;
            placed++;
        }
    }

    for (int i = 0; i < 4; i++) {
        ctx->bytes_per_node[i] = migrated[i];
        fprintf(stderr, "║  Node %d: %8.2f GB placed                               ║\n",
                i, migrated[i] / (1024.0 * 1024.0 * 1024.0));
    }

    fprintf(stderr, "║  Total tensors placed: %d                                 ║\n", placed);
    fprintf(stderr, "╚═══════════════════════════════════════════════════════════╝\n");

    return placed;
}

/*===========================================================================
 * Simplified Loader for llama.cpp Integration
 *
 * This is a simplified version that works with the existing ggml mmap
 * by applying NUMA hints AFTER the file is already mapped.
 *===========================================================================*/

typedef struct {
    void* weights_base;
    size_t weights_size;
    int total_layers;
} coffer_model_hint_t;

static int coffer_apply_numa_hints(coffer_model_hint_t* hint) {
    if (numa_available() < 0) {
        fprintf(stderr, "Coffer: NUMA not available, skipping placement\n");
        return 0;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Coffer NUMA Hints Applied                                ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════╣\n");

    /* Divide weights among NUMA nodes */
    size_t per_node = hint->weights_size / 4;
    uint8_t* base = (uint8_t*)hint->weights_base;

    for (int node = 0; node < 4; node++) {
        size_t offset = node * per_node;
        size_t size = (node == 3) ? (hint->weights_size - offset) : per_node;

        if (coffer_migrate_region(base + offset, size, node) >= 0) {
            fprintf(stderr, "║  Node %d: %8.2f GB (offset %zu)                       ║\n",
                    node, size / (1024.0 * 1024.0 * 1024.0), offset);
        }
    }

    fprintf(stderr, "╚═══════════════════════════════════════════════════════════╝\n");
    return 0;
}

/*===========================================================================
 * DCBT Prefetch Integration with NUMA Awareness
 *
 * Prefetch from the correct NUMA node based on which layer we're about to
 * process. This works with the coffer layer mapping.
 *===========================================================================*/

static inline void coffer_prefetch_layer_weights(
    coffer_mmap_ctx_t* ctx,
    int layer_id
) {
#if defined(__powerpc64__) || defined(__powerpc__)
    for (int i = 0; i < ctx->n_tensors; i++) {
        coffer_tensor_info_t* t = &ctx->tensors[i];
        if (t->layer_id == layer_id && t->size_bytes > 0) {
            /* Get tensor address */
            const uint8_t* addr = (const uint8_t*)ctx->mapped_addr +
                                  ctx->tensor_data_offset + t->offset;

            /* Prefetch every 128 bytes (cache line) */
            size_t prefetch_stride = 128;
            size_t prefetch_count = t->size_bytes / prefetch_stride;

            /* Limit prefetch to first 1MB to avoid cache thrashing */
            if (prefetch_count > 8192) prefetch_count = 8192;

            for (size_t j = 0; j < prefetch_count; j++) {
                __asm__ __volatile__("dcbt 0,%0" : : "r"(addr + j * prefetch_stride));
            }
        }
    }
#endif
}

/*===========================================================================
 * Cleanup
 *===========================================================================*/

static void coffer_mmap_close(coffer_mmap_ctx_t* ctx) {
    if (!ctx) return;

    if (ctx->tensors) {
        free(ctx->tensors);
    }

    if (ctx->mapped_addr && ctx->mapped_addr != MAP_FAILED) {
        munmap(ctx->mapped_addr, ctx->file_size);
    }

    if (ctx->fd >= 0) {
        close(ctx->fd);
    }

    free(ctx);
}

/*===========================================================================
 * Test/Debug Function
 *===========================================================================*/

static void coffer_mmap_test(const char* gguf_path) {
    fprintf(stderr, "\n=== Coffer MMAP Test ===\n");

    coffer_mmap_ctx_t* ctx = coffer_mmap_open(gguf_path);
    if (!ctx) {
        fprintf(stderr, "Failed to open GGUF file\n");
        return;
    }

    /* Estimate layers from file size (rough heuristic) */
    int est_layers = 32;  /* Default assumption */

    coffer_place_tensors(ctx, est_layers);

    coffer_mmap_close(ctx);

    fprintf(stderr, "=== Test Complete ===\n\n");
}

#endif /* GGML_COFFER_MMAP_H */
