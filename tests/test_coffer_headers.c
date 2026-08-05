/*
 * test_coffer_headers.c - Smoke test for the single-coffer and mmap headers.
 *
 * These live in a SEPARATE translation unit from test_portability.c because
 * ggml-ram-coffer.h (singular) and ggml-ram-coffers.h (plural) both define
 * `ram_coffer_t` and `coffer_print_stats` and therefore cannot be included
 * together. That is pre-existing and unrelated to portability.
 *
 * Build: gcc -std=c11 -I.. -Wall -Wextra test_coffer_headers.c -o t -lm [-lnuma]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-ram-coffer.h"   /* singular: tensor-level NUMA indexing */
#include "ggml-coffer-mmap.h"  /* GGUF mmap sharding                   */

static int g_failures = 0;

static void check(int cond, const char* what) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) g_failures++;
}

int main(void) {
    printf("=== RAM Coffer (singular) + mmap header test ===\n\n");

    /*-------------------------------------------------------------------
     * coffer_init() used to return -1 and print "NUMA not available!" on
     * any machine without NUMA. It must now succeed in uniform mode.
     *-----------------------------------------------------------------*/
    printf("[1] coffer_init\n");
    int rc = coffer_init();
    check(rc == 0, "coffer_init succeeds (used to hard-fail without NUMA)");
    check(g_coffer.n_nodes >= 1, "at least one node recorded");
    check(g_coffer.n_nodes <= NUM_NUMA_NODES, "node count within array bounds");
    printf("  n_nodes=%d uniform=%d\n", g_coffer.n_nodes, g_coffer.uniform);

    int nodes_sane = 1;
    for (int i = 0; i < g_coffer.n_nodes; i++) {
        if (g_coffer.nodes[i].paired_node < 0 ||
            g_coffer.nodes[i].paired_node >= g_coffer.n_nodes) nodes_sane = 0;
        if (g_coffer.nodes[i].total_bytes == 0) nodes_sane = 0;
    }
    check(nodes_sane, "every node has memory and a valid pair partner");

    /*-------------------------------------------------------------------
     * Layer placement must never name a node that does not exist.
     *-----------------------------------------------------------------*/
    printf("\n[2] Layer placement\n");
    coffer_plan_layer_placement(32, 64 * 1024 * 1024);
    int placement_ok = 1;
    for (int l = 0; l < 32; l++) {
        int node = g_coffer.layer_to_node[l];
        if (node < 0 || node >= g_coffer.n_nodes) placement_ok = 0;
    }
    check(placement_ok, "all 32 layers mapped to real nodes");

    /*-------------------------------------------------------------------
     * Allocation on a node (numa_alloc_onnode -> malloc in uniform mode)
     *-----------------------------------------------------------------*/
    printf("\n[3] Node allocation\n");
    void* p = coffer_alloc_on_node(1024 * 1024, 0, "test.weight");
    check(p != NULL, "coffer_alloc_on_node returned memory");
    if (p) {
        memset(p, 0x5A, 1024 * 1024);
        check(((unsigned char*)p)[0] == 0x5A, "allocated memory is writable");
        coffer_prefetch_tensor(p, 1024 * 1024);
        check(((unsigned char*)p)[0] == 0x5A, "memory intact after prefetch");
        coffers_free_on_node(p, 1024 * 1024);
    }

    /* Out-of-range node id must be clamped, not crash */
    void* p2 = coffer_alloc_on_node(4096, 99, "oob.weight");
    check(p2 != NULL, "out-of-range node id clamped rather than failing");
    if (p2) coffers_free_on_node(p2, 4096);

    /*-------------------------------------------------------------------
     * Thread binding is a successful no-op in uniform mode
     *-----------------------------------------------------------------*/
    printf("\n[4] Thread binding\n");
    check(coffer_bind_to_node(0) == 0, "bind to node 0 succeeds");

    /*-------------------------------------------------------------------
     * Model planning
     *-----------------------------------------------------------------*/
    printf("\n[5] Model planning\n");
    model_topology_t model;
    memset(&model, 0, sizeof(model));
    model.num_layers        = 16;
    model.layer_size        = 8 * 1024 * 1024;
    model.embedding_size    = 16 * 1024 * 1024;
    model.lm_head_size      = 16 * 1024 * 1024;
    model.kv_cache_per_layer = 1 * 1024 * 1024;
    check(coffer_plan_model(&model) == 0, "small model plan fits and succeeds");

    coffer_print_stats();

    /*-------------------------------------------------------------------
     * mmap header: node assignment must stay in range for this topology
     *-----------------------------------------------------------------*/
    printf("\n[6] mmap tensor node assignment\n");
    const coffers_topology_t* topo = coffers_topology();
    int assign_ok = 1;
    const char* names[] = {
        "token_embd.weight", "blk.0.attn_q.weight", "blk.15.ffn_up.weight",
        "blk.31.attn_k.weight", "output.weight", "some.unknown.tensor"
    };
    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        int layer = extract_layer_id(names[i]);
        int node  = assign_numa_node(layer, 32, names[i]);
        printf("  %-24s layer=%3d -> node %d\n", names[i], layer, node);
        if (node < 0 || node >= topo->n_nodes) assign_ok = 0;
    }
    check(assign_ok, "every tensor assigned to a node that exists");

    /* coffer_migrate_region must succeed (no-op) in uniform mode */
    printf("\n[7] Region migration\n");
    void* region = malloc(256 * 1024);
    check(region != NULL, "region allocated");
    if (region) {
        int mrc = coffer_migrate_region(region, 256 * 1024, 0);
        check(mrc >= 0, "migrate to node 0 succeeds (no-op in uniform mode)");
        free(region);
    }

    coffer_model_hint_t hint;
    void* wbuf = malloc(1024 * 1024);
    check(wbuf != NULL, "weights buffer allocated");
    if (wbuf) {
        hint.weights_base = wbuf;
        hint.weights_size = 1024 * 1024;
        hint.total_layers = 32;
        check(coffer_apply_numa_hints(&hint) == 0, "numa hints applied without error");
        free(wbuf);
    }

    printf("\n=== %s (%d failures) ===\n",
           g_failures == 0 ? "ALL TESTS PASSED" : "FAILURES PRESENT", g_failures);
    return g_failures == 0 ? 0 : 1;
}
