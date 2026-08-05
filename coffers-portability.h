/*
 * coffers-portability.h - Capability detection & fallback shims for RAM Coffers
 *
 * PROBLEM THIS SOLVES:
 *   The coffer headers unconditionally #include <numa.h>/<numaif.h> and
 *   <altivec.h>, and hard-failed when numa_available() < 0. On any machine
 *   without libnuma-dev, or any non-POWER machine, the project did not
 *   "degrade" - it failed to COMPILE. That is the hardest possible failure
 *   for someone evaluating the repo.
 *
 * WHAT THIS PROVIDES:
 *   1. GGML_COFFERS_HAVE_NUMA     - is libnuma usable?
 *   2. GGML_COFFERS_HAVE_ALTIVEC  - are POWER vector intrinsics usable?
 *   3. A coffers_* abstraction over the handful of NUMA calls actually used,
 *      with a uniform-memory implementation when NUMA is absent, so call
 *      sites stay readable instead of being shot through with #ifdefs.
 *   4. Portable prefetch (dcbt -> __builtin_prefetch -> no-op) and
 *      timebase entropy (mftb -> clock_gettime).
 *   5. Topology-adaptive coffer<->node mapping (1, 2, 4, or N nodes).
 *
 * DEFAULTS ARE DELIBERATE:
 *   NUMA is ON whenever <numa.h> is present on Linux. It is NOT opt-in,
 *   because making it opt-in would silently disable NUMA on the POWER8 boxes
 *   this project was built for. To force it off, build with
 *   -DGGML_COFFERS_NO_NUMA (or pre-define GGML_COFFERS_HAVE_NUMA yourself).
 */

#ifndef GGML_COFFERS_PORTABILITY_H
#define GGML_COFFERS_PORTABILITY_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#ifndef __has_include
#define __has_include(x) 0
#endif

/*===========================================================================
 * 1. NUMA CAPABILITY DETECTION
 *===========================================================================*/

#if defined(GGML_COFFERS_HAVE_NUMA)
    /* Caller forced the value explicitly - honour it, detect nothing. */
#elif defined(GGML_COFFERS_NO_NUMA)
#   define GGML_COFFERS_HAVE_NUMA 0
#elif defined(__linux__) && __has_include(<numa.h>) && __has_include(<numaif.h>)
#   define GGML_COFFERS_HAVE_NUMA 1
#else
#   define GGML_COFFERS_HAVE_NUMA 0
#endif

#if GGML_COFFERS_HAVE_NUMA
#   include <numa.h>
#   include <numaif.h>
#   include <sched.h>
#endif

#if defined(__linux__) || defined(__APPLE__)
#   include <unistd.h>
#endif

/*===========================================================================
 * 2. POWER VECTOR / INTRINSIC CAPABILITY DETECTION
 *
 * __ALTIVEC__ / __VEC__ are only defined when the compiler was actually
 * invoked with -maltivec/-mvsx, so this is stricter (and more correct) than
 * testing __powerpc__ alone: a PPC build without -maltivec now takes the
 * scalar path instead of failing on missing intrinsics.
 *===========================================================================*/

#if defined(__powerpc__) || defined(__powerpc64__) || defined(__PPC__) || defined(__PPC64__)
#   define GGML_COFFERS_IS_POWER 1
#else
#   define GGML_COFFERS_IS_POWER 0
#endif

#if defined(GGML_COFFERS_HAVE_ALTIVEC)
    /* Caller forced it. */
#elif defined(GGML_COFFERS_NO_ALTIVEC)
#   define GGML_COFFERS_HAVE_ALTIVEC 0
#elif GGML_COFFERS_IS_POWER && (defined(__ALTIVEC__) || defined(__VEC__)) && __has_include(<altivec.h>)
#   define GGML_COFFERS_HAVE_ALTIVEC 1
#else
#   define GGML_COFFERS_HAVE_ALTIVEC 0
#endif

#if GGML_COFFERS_HAVE_ALTIVEC
#   include <altivec.h>
#endif

/* ISA 2.07 crypto (vcipher). Requires AltiVec/VSX *and* the crypto builtins. */
#if defined(GGML_COFFERS_HAVE_VCIPHER)
    /* Caller forced it. */
#elif GGML_COFFERS_HAVE_ALTIVEC && defined(__CRYPTO__)
#   define GGML_COFFERS_HAVE_VCIPHER 1
#else
#   define GGML_COFFERS_HAVE_VCIPHER 0
#endif

/*===========================================================================
 * 3. CACHE LINE / PREFETCH
 *
 * POWER8 uses a 128-byte cache line; most x86-64 and aarch64 use 64.
 *===========================================================================*/

#if GGML_COFFERS_IS_POWER
#   define COFFERS_CACHE_LINE 128
#else
#   define COFFERS_CACHE_LINE 64
#endif

/* Single-line prefetch: dcbt on POWER, __builtin_prefetch elsewhere, else no-op. */
static inline void coffers_prefetch(const void* addr) {
#if GGML_COFFERS_IS_POWER
    __asm__ __volatile__("dcbt 0,%0" : : "r"(addr));
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 0 /* read */, 3 /* high temporal locality */);
#else
    (void)addr;
#endif
}

/* Prefetch a whole region, one touch per cache line. */
static inline void coffers_prefetch_range(const void* addr, size_t bytes) {
    const char* p   = (const char*)addr;
    const char* end = p + bytes;
    for (; p < end; p += COFFERS_CACHE_LINE) {
        coffers_prefetch(p);
    }
}

/*===========================================================================
 * 4. TIMEBASE ENTROPY
 *
 * POWER8 reads the hardware timebase register (mftb) - this is the entropy
 * source behind PSE's behavioural divergence. Elsewhere, CLOCK_MONOTONIC
 * gives an equivalently monotonic, non-repeating counter.
 *===========================================================================*/

static inline uint64_t coffers_read_timebase(void) {
#if GGML_COFFERS_IS_POWER
    uint64_t tb;
    __asm__ __volatile__("mftb %0" : "=r"(tb));
    return tb;
#elif defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
    return (uint64_t)clock();
#else
    return (uint64_t)clock();
#endif
}

/*===========================================================================
 * 5. SCALAR REFERENCE FOR vec_perm
 *
 * AltiVec vec_perm(a, b, c): result byte i = (a || b)[ c[i] & 0x1F ], where
 * (a || b) is the 32-byte big-endian concatenation of a then b. This routine
 * reproduces that on any architecture so the collapse path has a portable
 * equivalent producing the same logical result.
 *
 * NOTE: on little-endian POWER (ppc64le) the hardware intrinsic indexes the
 * concatenation in the opposite byte order. This function defines the
 * BIG-ENDIAN reference semantics; it is used only on non-POWER builds, where
 * there is no hardware result to diverge from.
 *===========================================================================*/

/*
 * Portable permute-control type. Using this in function signatures lets the
 * AltiVec and scalar implementations share one API, so callers do not need
 * to be #ifdef'd.
 */
#if GGML_COFFERS_HAVE_ALTIVEC
typedef vector unsigned char coffers_perm_t;
#else
typedef struct { unsigned char b[16]; } coffers_perm_t;
#endif

static inline void coffers_perm_bytes(
    const unsigned char a[16],
    const unsigned char b[16],
    const unsigned char ctrl[16],
    unsigned char out[16]
) {
    unsigned char cat[32];
    memcpy(cat,      a, 16);
    memcpy(cat + 16, b, 16);
    for (int i = 0; i < 16; i++) {
        out[i] = cat[ctrl[i] & 0x1F];
    }
}

/*===========================================================================
 * 6. NUMA ABSTRACTION LAYER
 *
 * Only the calls the coffer headers actually use. Two implementations:
 * a real one over libnuma, and a uniform-memory one that reports a single
 * node covering all of RAM and turns placement calls into successful no-ops.
 *===========================================================================*/

#define COFFERS_UNIFORM_NODE 0

#if GGML_COFFERS_HAVE_NUMA

static inline int coffers_numa_available(void) {
    return numa_available() < 0 ? 0 : 1;
}

static inline int coffers_num_nodes(void) {
    if (!coffers_numa_available()) return 1;
    int n = numa_num_configured_nodes();
    return n > 0 ? n : 1;
}

static inline size_t coffers_node_memory(int node, size_t* free_bytes_out) {
    long long free_bytes = 0;
    long long total = numa_node_size64(node, &free_bytes);
    if (total < 0) {                       /* node not present */
        if (free_bytes_out) *free_bytes_out = 0;
        return 0;
    }
    if (free_bytes_out) *free_bytes_out = (size_t)free_bytes;
    return (size_t)total;
}

static inline void* coffers_alloc_on_node(size_t size, int node) {
    void* p = numa_alloc_onnode(size, node);
    if (!p) p = malloc(size);              /* still succeed, just unbound */
    return p;
}

static inline void coffers_free_on_node(void* ptr, size_t size) {
    if (ptr) numa_free(ptr, size);
}

static inline int coffers_run_on_node(int node) {
    return numa_run_on_node(node);
}

static inline int coffers_node_of_cpu(int cpu) {
    return numa_node_of_cpu(cpu);
}

static inline int coffers_node_of_addr(const void* addr) {
    int node = -1;
    if (get_mempolicy(&node, NULL, 0, (void*)addr, MPOL_F_NODE | MPOL_F_ADDR) < 0) {
        return -1;
    }
    return node;
}

/* Bind an existing address range to a node. Falls back to PREFERRED. */
static inline int coffers_bind_range(void* addr, size_t size, int node) {
    unsigned long nodemask = 1UL << node;
    size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    uintptr_t aligned_addr = (uintptr_t)addr & ~(uintptr_t)(page_size - 1);
    size_t aligned_size = size + ((uintptr_t)addr - aligned_addr);
    aligned_size = (aligned_size + page_size - 1) & ~(size_t)(page_size - 1);

    int ret = mbind((void*)aligned_addr, aligned_size, MPOL_BIND,
                    &nodemask, sizeof(nodemask) * 8, MPOL_MF_MOVE);
    if (ret < 0) {
        ret = mbind((void*)aligned_addr, aligned_size, MPOL_PREFERRED,
                    &nodemask, sizeof(nodemask) * 8, 0);
    }
    return ret;
}

/* Set/reset the allocation policy for subsequent mmap/allocations. */
static inline int coffers_set_membind(int node) {
    unsigned long nodemask = 1UL << node;
    return set_mempolicy(MPOL_BIND, &nodemask, sizeof(nodemask) * 8);
}

static inline int coffers_reset_mempolicy(void) {
    return set_mempolicy(MPOL_DEFAULT, NULL, 0);
}

/* Bind the calling thread to the CPUs of a node. */
static inline int coffers_bind_thread_to_node(int node) {
    struct bitmask* mask = numa_allocate_cpumask();
    if (!mask) return -1;
    numa_node_to_cpus(node, mask);
    int ret = numa_sched_setaffinity(0, mask);
    numa_free_cpumask(mask);
    return ret;
}

#else  /* ---------------- uniform-memory (no libnuma) ---------------- */

static inline int coffers_numa_available(void) { return 0; }
static inline int coffers_num_nodes(void)      { return 1; }

static inline size_t coffers_node_memory(int node, size_t* free_bytes_out) {
    if (node != COFFERS_UNIFORM_NODE) {
        if (free_bytes_out) *free_bytes_out = 0;
        return 0;
    }
    size_t total = 0, freeb = 0;
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) total = (size_t)pages * (size_t)page_size;
#endif
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    long avail = sysconf(_SC_AVPHYS_PAGES);
    long ps    = sysconf(_SC_PAGESIZE);
    if (avail > 0 && ps > 0) freeb = (size_t)avail * (size_t)ps;
#endif
    if (free_bytes_out) *free_bytes_out = freeb ? freeb : total;
    return total;
}

static inline void* coffers_alloc_on_node(size_t size, int node) {
    (void)node;
    return malloc(size);
}

static inline void coffers_free_on_node(void* ptr, size_t size) {
    (void)size;
    free(ptr);
}

/* All placement operations succeed trivially: there is only one place. */
static inline int coffers_run_on_node(int node)            { (void)node; return 0; }
static inline int coffers_node_of_cpu(int cpu)             { (void)cpu;  return COFFERS_UNIFORM_NODE; }
static inline int coffers_node_of_addr(const void* addr)   { (void)addr; return COFFERS_UNIFORM_NODE; }
static inline int coffers_set_membind(int node)            { (void)node; return 0; }
static inline int coffers_reset_mempolicy(void)            { return 0; }
static inline int coffers_bind_thread_to_node(int node)    { (void)node; return 0; }

static inline int coffers_bind_range(void* addr, size_t size, int node) {
    (void)addr; (void)size; (void)node;
    return 0;
}

#endif /* GGML_COFFERS_HAVE_NUMA */

/*===========================================================================
 * 7. TOPOLOGY-ADAPTIVE COFFER MAPPING
 *
 * The original code hardcoded COFFER_TO_NUMA[4] = {3,1,0,2} - the POWER8
 * S824 layout, ordered largest-node-first. That indexes past the end of the
 * node list on any machine with fewer than 4 nodes.
 *
 * This derives the mapping from the node count actually detected:
 *   1 node  -> 1 coffer  (uniform mode)
 *   2 nodes -> 2 coffers -> nodes {0, 1}
 *   4 nodes -> 4 coffers -> nodes {3, 1, 0, 2}   (preserves POWER8 ordering)
 *   N nodes -> min(N, max_coffers) coffers -> nodes {0..}
 *===========================================================================*/

#ifndef COFFERS_TOPOLOGY_MAX
#define COFFERS_TOPOLOGY_MAX 8
#endif

typedef struct {
    int n_nodes;                             /* NUMA nodes detected (>=1)   */
    int n_coffers;                           /* Coffers actually usable     */
    int coffer_to_numa[COFFERS_TOPOLOGY_MAX];/* coffer id -> numa node id   */
    int uniform;                             /* 1 = single uniform region   */
    int numa_runtime;                        /* 1 = libnuma present+working */
} coffers_topology_t;

static coffers_topology_t g_coffers_topology = {0};
static int g_coffers_topology_ready = 0;

static inline const coffers_topology_t* coffers_topology(void) {
    if (g_coffers_topology_ready) return &g_coffers_topology;

    coffers_topology_t* t = &g_coffers_topology;
    t->numa_runtime = coffers_numa_available();
    t->n_nodes      = t->numa_runtime ? coffers_num_nodes() : 1;
    if (t->n_nodes < 1) t->n_nodes = 1;

    if (!t->numa_runtime || t->n_nodes == 1) {
        /* Uniform memory: exactly one coffer covering everything. */
        t->uniform            = 1;
        t->n_coffers          = 1;
        t->coffer_to_numa[0]  = COFFERS_UNIFORM_NODE;
    } else if (t->n_nodes == 4) {
        /* Preserve the tuned POWER8 S824 ordering (largest node first). */
        static const int p8[4] = {3, 1, 0, 2};
        t->uniform   = 0;
        t->n_coffers = 4;
        for (int i = 0; i < 4; i++) t->coffer_to_numa[i] = p8[i];
    } else {
        t->uniform   = 0;
        t->n_coffers = t->n_nodes < COFFERS_TOPOLOGY_MAX ? t->n_nodes : COFFERS_TOPOLOGY_MAX;
        for (int i = 0; i < t->n_coffers; i++) t->coffer_to_numa[i] = i;
    }

    g_coffers_topology_ready = 1;
    return t;
}

/* Safe accessor: never returns a node id past the real node count. */
static inline int coffers_node_for(int coffer_id) {
    const coffers_topology_t* t = coffers_topology();
    if (coffer_id < 0 || coffer_id >= t->n_coffers) return COFFERS_UNIFORM_NODE;
    return t->coffer_to_numa[coffer_id];
}

/*===========================================================================
 * 8. ONE-TIME MODE BANNER
 *
 * Uniform-memory mode is a supported configuration, not an error. Say so
 * once, clearly, and carry on.
 *===========================================================================*/

static int g_coffers_banner_shown = 0;

static inline void coffers_report_mode(void) {
    if (g_coffers_banner_shown) return;
    g_coffers_banner_shown = 1;

    const coffers_topology_t* t = coffers_topology();

    if (!GGML_COFFERS_HAVE_NUMA) {
        fprintf(stderr,
            "RAM Coffers: built without libnuma - running in uniform-memory mode (1 coffer)\n");
    } else if (!t->numa_runtime) {
        fprintf(stderr,
            "RAM Coffers: NUMA unavailable - running in uniform-memory mode (1 coffer)\n");
    } else if (t->uniform) {
        fprintf(stderr,
            "RAM Coffers: 1 NUMA node - running in uniform-memory mode (1 coffer)\n");
    } else {
        fprintf(stderr,
            "RAM Coffers: %d NUMA nodes - %d coffers with node affinity\n",
            t->n_nodes, t->n_coffers);
    }

    fprintf(stderr,
            "RAM Coffers: vector path = %s, prefetch = %s, entropy = %s\n",
            GGML_COFFERS_HAVE_ALTIVEC ? "AltiVec/VSX" : "scalar C",
#if GGML_COFFERS_IS_POWER
            "dcbt",
#elif defined(__GNUC__) || defined(__clang__)
            "__builtin_prefetch",
#else
            "none",
#endif
            GGML_COFFERS_IS_POWER ? "mftb timebase" : "CLOCK_MONOTONIC");
}

#endif /* GGML_COFFERS_PORTABILITY_H */
