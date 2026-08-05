/*
 * test_portability.c - Build + run smoke test for non-POWER / non-NUMA hosts
 *
 * Exercises: capability detection, coffer init, shard load (mmap), resonance
 * routing, prune planning, layer-ahead prefetch, and the collapse path.
 *
 * This must COMPILE AND RUN CLEAN on x86_64 with no libnuma, on x86_64 with
 * libnuma but a single node, and on POWER8 with 4 nodes.
 *
 * Build:  gcc -std=c11 -I.. -Wall -Wextra test_portability.c -o test_portability -lm [-lnuma]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "coffers-portability.h"
#include "ggml-ram-coffers.h"
#include "ggml-intelligent-collapse.h"
#include "ggml-topk-collapse-vsx.h"

static int g_failures = 0;

static void check(int cond, const char* what) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) g_failures++;
}

/* Create a small throwaway file to stand in for a GGUF shard. */
static int make_dummy_shard(const char* path, size_t bytes) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    unsigned char* buf = (unsigned char*)calloc(1, bytes);
    if (!buf) { fclose(f); return -1; }
    for (size_t i = 0; i < bytes; i++) buf[i] = (unsigned char)(i & 0xFF);
    size_t wrote = fwrite(buf, 1, bytes, f);
    free(buf);
    fclose(f);
    return wrote == bytes ? 0 : -1;
}

int main(void) {
    printf("=== RAM Coffers portability smoke test ===\n\n");

    /*-------------------------------------------------------------------
     * 1. Capability detection
     *-----------------------------------------------------------------*/
    printf("[1] Capability detection\n");
    printf("  GGML_COFFERS_HAVE_NUMA    = %d\n", GGML_COFFERS_HAVE_NUMA);
    printf("  GGML_COFFERS_HAVE_ALTIVEC = %d\n", GGML_COFFERS_HAVE_ALTIVEC);
    printf("  GGML_COFFERS_HAVE_VCIPHER = %d\n", GGML_COFFERS_HAVE_VCIPHER);
    printf("  GGML_COFFERS_IS_POWER     = %d\n", GGML_COFFERS_IS_POWER);
    printf("  COFFERS_CACHE_LINE        = %d\n", COFFERS_CACHE_LINE);

    const coffers_topology_t* topo = coffers_topology();
    printf("  nodes=%d coffers=%d uniform=%d numa_runtime=%d\n",
           topo->n_nodes, topo->n_coffers, topo->uniform, topo->numa_runtime);

    check(topo->n_nodes >= 1, "at least one node reported");
    check(topo->n_coffers >= 1, "at least one coffer available");
    check(topo->n_coffers <= COFFERS_TOPOLOGY_MAX, "coffer count within bounds");

    /* Never index past the real node count - the old hardcoded {3,1,0,2} did */
    int mapping_ok = 1;
    for (int c = 0; c < topo->n_coffers; c++) {
        int node = coffers_node_for(c);
        printf("  coffer %d -> node %d\n", c, node);
        if (node < 0 || node >= topo->n_nodes) mapping_ok = 0;
    }
    check(mapping_ok, "coffer->node mapping never exceeds real node count");

    /* Uniform mode must mean exactly one coffer */
    if (topo->uniform) {
        check(topo->n_coffers == 1, "uniform mode uses exactly 1 coffer");
    }

    /*-------------------------------------------------------------------
     * 2. Timebase entropy must actually vary
     *-----------------------------------------------------------------*/
    printf("\n[2] Timebase entropy\n");
    uint64_t t1 = coffers_read_timebase();
    for (volatile int spin = 0; spin < 200000; spin++) { }
    uint64_t t2 = coffers_read_timebase();
    printf("  tb1=%llu tb2=%llu\n",
           (unsigned long long)t1, (unsigned long long)t2);
    check(t1 != 0 || t2 != 0, "timebase is non-zero (was hardcoded 0 off POWER)");
    check(t2 >= t1, "timebase is monotonic");

    /*-------------------------------------------------------------------
     * 3. Prefetch must be safe on any arch
     *-----------------------------------------------------------------*/
    printf("\n[3] Prefetch\n");
    unsigned char* region = (unsigned char*)malloc(64 * 1024);
    check(region != NULL, "prefetch region allocated");
    if (region) {
        memset(region, 0xA5, 64 * 1024);
        coffers_prefetch(region);
        coffers_prefetch_range(region, 64 * 1024);
        dcbt_resident(region, 64 * 1024);
        check(region[0] == 0xA5, "memory intact after prefetch");
        free(region);
    }

    /*-------------------------------------------------------------------
     * 4. Coffer init + shard load (the path that used to hard-fail)
     *-----------------------------------------------------------------*/
    printf("\n[4] Coffer init and shard load\n");
    const char* shard_path = "/tmp/ram_coffers_test_shard.bin";
    check(make_dummy_shard(shard_path, 512 * 1024) == 0, "dummy shard created");

    const char* paths[MAX_COFFERS] = {shard_path, NULL, NULL, NULL};
    int loaded = init_ram_coffers(paths);
    check(loaded >= 1, "init_ram_coffers loaded at least one shard");
    check(g_coffers[0].is_loaded == 1, "coffer 0 reports loaded");
    check(g_coffers[0].mmap_ptr != NULL && g_coffers[0].mmap_ptr != MAP_FAILED,
          "coffer 0 mmap succeeded");
    check(g_coffers[0].numa_node >= 0 && g_coffers[0].numa_node < topo->n_nodes,
          "coffer 0 bound to a real node");

    /*-------------------------------------------------------------------
     * 5. Activation + layer-ahead prefetch
     *-----------------------------------------------------------------*/
    printf("\n[5] Activation and layer-ahead prefetch\n");
    check(activate_coffer_ex(0, 8) == 0, "activate_coffer_ex with layers");
    check(g_coffers[0].is_active == 1, "coffer marked active");
    layer_prefetch_ahead(0);
    layer_prefetch_ahead(3);
    check(activate_coffer(0) == 0, "activate_coffer (legacy wrapper)");

    /*-------------------------------------------------------------------
     * 6. Resonance routing
     *-----------------------------------------------------------------*/
    printf("\n[6] Resonance routing\n");
    float q[COFFER_EMBED_DIM];
    for (int i = 0; i < COFFER_EMBED_DIM; i++) q[i] = sinf(i * 0.1f) + 0.1f;
    int routed = route_to_coffer(q);
    printf("  routed to coffer %d\n", routed);
    check(routed >= 0 && routed < MAX_COFFERS, "routing returns a valid coffer");

    /*-------------------------------------------------------------------
     * 7. Non-bijunctive prune planning
     *-----------------------------------------------------------------*/
    printf("\n[7] Prune planning\n");
    prune_plan_t* plan = coffer_plan_prune(0, q, 0.5f);
    check(plan != NULL, "prune plan created");
    if (plan) {
        printf("  blocks=%d block_size=%zu saved=%zu bytes\n",
               plan->n_blocks, plan->block_size, plan->total_saved);
        check(plan->n_blocks > 0, "prune plan has blocks");
        coffer_free_prune_plan(plan);
    }

    /*-------------------------------------------------------------------
     * 8. Collapse path (vec_perm on POWER, scalar equivalent elsewhere)
     *-----------------------------------------------------------------*/
    printf("\n[8] Collapse\n");
    enum { N = 64 };
    float scores[N];
    for (int i = 0; i < N; i++) scores[i] = (float)((i * 37) % 100) / 10.0f;

    coffers_perm_t pattern = generate_intelligent_pattern(1, 2, ic_read_tb());
    intelligent_collapse_scores(scores, N, 8, pattern, 1.2f);

    int survivors = 0;
    int finite = 1;
    for (int i = 0; i < N; i++) {
        if (scores[i] != 0.0f) survivors++;
        if (isnan(scores[i]) || isinf(scores[i])) finite = 0;
    }
    printf("  survivors=%d/%d\n", survivors, N);
    check(finite, "collapse produced only finite values");
    check(survivors > 0, "collapse kept at least one winner");
    check(survivors < N, "collapse pruned at least one loser");

    /* Top-K mask */
    float tk[N];
    for (int i = 0; i < N; i++) tk[i] = (float)((i * 17) % 50);
    float thr = find_kth_largest(tk, N, 8);
    apply_topk_mask_vsx(tk, N, thr);
    int kept = 0;
    for (int i = 0; i < N; i++) if (tk[i] != 0.0f) kept++;
    printf("  topk threshold=%.2f kept=%d/%d\n", thr, kept, N);
    check(kept > 0 && kept <= N, "top-K mask kept a sane number of scores");

    /*-------------------------------------------------------------------
     * 9. Attention end-to-end
     *-----------------------------------------------------------------*/
    printf("\n[9] Attention end-to-end\n");
    enum { SEQ = 8, DIM = 16 };
    static float Q[SEQ * DIM], K[SEQ * DIM], V[SEQ * DIM], O[SEQ * DIM];
    for (int i = 0; i < SEQ * DIM; i++) {
        Q[i] = sinf((float)i * 0.11f);
        K[i] = cosf((float)i * 0.07f);
        V[i] = sinf((float)i * 0.03f) + 0.5f;
    }
    attention_topk_collapsed(O, Q, K, V, SEQ, DIM, 0, 0, 4);
    int attn_finite = 1;
    for (int i = 0; i < SEQ * DIM; i++) {
        if (isnan(O[i]) || isinf(O[i])) attn_finite = 0;
    }
    check(attn_finite, "attention_topk_collapsed produced finite output");

    memset(O, 0, sizeof(O));
    attention_intelligent(O, Q, K, V, SEQ, DIM, 0);
    attn_finite = 1;
    for (int i = 0; i < SEQ * DIM; i++) {
        if (isnan(O[i]) || isinf(O[i])) attn_finite = 0;
    }
    check(attn_finite, "attention_intelligent produced finite output");

    /*-------------------------------------------------------------------
     * 10. Teardown
     *-----------------------------------------------------------------*/
    printf("\n[10] Teardown\n");
    shutdown_ram_coffers();
    check(g_coffers_initialized == 0, "coffers shut down cleanly");
    remove(shard_path);

    printf("\n=== %s (%d failures) ===\n",
           g_failures == 0 ? "ALL TESTS PASSED" : "FAILURES PRESENT", g_failures);
    return g_failures == 0 ? 0 : 1;
}
