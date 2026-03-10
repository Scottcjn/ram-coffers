/*
 * bench_vcipher_collapse.c — Benchmark: vcipher collapse vs vec_perm collapse
 *
 * Compares:
 *   1. vec_perm only (current PSE approach)
 *   2. vcipher pattern generation + vec_perm routing (hybrid)
 *   3. Pure vcipher attention score (experimental)
 *   4. 8-way pipelined vcipher
 *
 * Build:
 *   gcc -mcpu=power8 -mvsx -mcrypto -maltivec -O3 -fopenmp \
 *       bench_vcipher_collapse.c -o bench_vcipher_collapse -lm
 *
 * Run:
 *   ./bench_vcipher_collapse
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <altivec.h>

/* Include both implementations */
#include "ggml-vcipher-collapse.h"

/*===========================================================================
 * Reference: Original vec_perm collapse (from ggml-intelligent-collapse.h)
 *===========================================================================*/

static inline uint64_t ref_read_tb(void) {
    uint64_t tb;
    __asm__ __volatile__("mftb %0" : "=r"(tb));
    return tb;
}

static inline vector unsigned char ref_generate_pattern(int layer_id, int position, uint64_t tb) {
    uint32_t h = (uint32_t)(tb ^ (tb >> 32)) ^ (layer_id * 0x9E3779B9U) ^ (position * 0x85EBCA77U);
    unsigned char p[16] __attribute__((aligned(16)));
    for (int i = 0; i < 8; i++) p[i] = i;
    for (int i = 8; i < 16; i++) {
        h ^= h << 13; h ^= h >> 17; h ^= h << 5;
        p[i] = h % 4;
    }
    return *(vector unsigned char*)p;
}

static inline void ref_collapse_scores(float* scores, int n, vector unsigned char pattern, float amplify) {
    if (n < 4) return;

    /* Simple top-4 threshold */
    float top[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    for (int i = 0; i < n; i++) {
        if (scores[i] > top[3]) {
            top[3] = scores[i];
            for (int k = 2; k >= 0 && top[k+1] > top[k]; k--) {
                float t = top[k]; top[k] = top[k+1]; top[k+1] = t;
            }
        }
    }
    float threshold = top[3];

    vector float thresh_vec = vec_splats(threshold);
    vector float amp_vec = vec_splats(amplify);
    vector float zero_vec = vec_splats(0.0f);

    int i = 0;
    for (; i + 15 < n; i += 16) {
        vector float v0 = vec_ld(0,  &scores[i]);
        vector float v1 = vec_ld(16, &scores[i]);
        vector float v2 = vec_ld(32, &scores[i]);
        vector float v3 = vec_ld(48, &scores[i]);

        vector float c0 = vec_perm(v0, v1, pattern);
        vector float c1 = vec_perm(v1, v2, pattern);
        vector float c2 = vec_perm(v2, v3, pattern);
        vector float c3 = vec_perm(v3, v0, pattern);

        vector bool int m0 = vec_cmpgt(c0, thresh_vec);
        vector bool int m1 = vec_cmpgt(c1, thresh_vec);
        vector bool int m2 = vec_cmpgt(c2, thresh_vec);
        vector bool int m3 = vec_cmpgt(c3, thresh_vec);

        c0 = vec_madd(vec_sel(zero_vec, c0, m0), amp_vec, zero_vec);
        c1 = vec_madd(vec_sel(zero_vec, c1, m1), amp_vec, zero_vec);
        c2 = vec_madd(vec_sel(zero_vec, c2, m2), amp_vec, zero_vec);
        c3 = vec_madd(vec_sel(zero_vec, c3, m3), amp_vec, zero_vec);

        vec_st(c0, 0,  &scores[i]);
        vec_st(c1, 16, &scores[i]);
        vec_st(c2, 32, &scores[i]);
        vec_st(c3, 48, &scores[i]);
    }
    for (; i < n; i++) {
        if (scores[i] >= threshold) scores[i] *= amplify;
        else scores[i] = 0.0f;
    }
}

/*===========================================================================
 * Benchmark Helpers
 *===========================================================================*/

static void fill_random_scores(float* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

static double elapsed_us(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3;
}

#define BENCH_ITERS 100000
#define SEQ_LEN 256

int main() {
    srand(42);

    float* scores_ref    = (float*)aligned_alloc(16, SEQ_LEN * sizeof(float));
    float* scores_vciph  = (float*)aligned_alloc(16, SEQ_LEN * sizeof(float));
    float* scores_orig   = (float*)aligned_alloc(16, SEQ_LEN * sizeof(float));
    float* Q_test        = (float*)aligned_alloc(16, 4 * sizeof(float));
    float* K_test        = (float*)aligned_alloc(16, 4 * sizeof(float));

    fill_random_scores(scores_orig, SEQ_LEN);
    fill_random_scores(Q_test, 4);
    fill_random_scores(K_test, 4);

    struct timespec t0, t1;

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  POWER8 vcipher vs vec_perm Collapse Benchmark             ║\n");
    printf("║  Sequence length: %d  |  Iterations: %d            ║\n", SEQ_LEN, BENCH_ITERS);
    printf("╠══════════════════════════════════════════════════════════════╣\n");

    /* Benchmark 1: Reference vec_perm collapse */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int iter = 0; iter < BENCH_ITERS; iter++) {
        memcpy(scores_ref, scores_orig, SEQ_LEN * sizeof(float));
        uint64_t tb = ref_read_tb();
        vector unsigned char pat = ref_generate_pattern(0, iter, tb);
        ref_collapse_scores(scores_ref, SEQ_LEN, pat, 1.2f);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us_vecperm = elapsed_us(t0, t1) / BENCH_ITERS;

    /* Benchmark 2: vcipher pattern generation only */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int iter = 0; iter < BENCH_ITERS; iter++) {
        vector unsigned char pat __attribute__((unused));
        pat = vcipher_generate_pattern(0, iter, 8);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us_vcipher_patgen = elapsed_us(t0, t1) / BENCH_ITERS;

    /* Benchmark 3: Hybrid vcipher collapse (full pipeline) */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int iter = 0; iter < BENCH_ITERS; iter++) {
        memcpy(scores_vciph, scores_orig, SEQ_LEN * sizeof(float));
        vcipher_hybrid_collapse(scores_vciph, SEQ_LEN, 8, 0, iter);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us_hybrid = elapsed_us(t0, t1) / BENCH_ITERS;

    /* Benchmark 4: Pure vcipher attention score (per Q·K pair) */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    volatile uint32_t sink = 0;
    for (int iter = 0; iter < BENCH_ITERS * 10; iter++) {
        sink += vcipher_attention_score(Q_test, K_test, 0, iter);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us_pure_attn = elapsed_us(t0, t1) / (BENCH_ITERS * 10);

    /* Benchmark 5: vcipher cross-head fusion */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    vector unsigned long long fuse_state = {0x0102030405060708ULL, 0x090a0b0c0d0e0f10ULL};
    for (int iter = 0; iter < BENCH_ITERS * 10; iter++) {
        fuse_state = vcipher_fuse_heads(fuse_state, 0, iter);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us_fuse = elapsed_us(t0, t1) / (BENCH_ITERS * 10);
    /* Prevent optimization-out */
    unsigned char fb[16]; memcpy(fb, &fuse_state, 16);

    printf("║                                                              ║\n");
    printf("║  1. vec_perm collapse (current):    %8.2f us/iter         ║\n", us_vecperm);
    printf("║  2. vcipher pattern gen only:        %8.3f us/call         ║\n", us_vcipher_patgen);
    printf("║  3. Hybrid vcipher+vec_perm:        %8.2f us/iter         ║\n", us_hybrid);
    printf("║  4. Pure vcipher attention:          %8.3f us/score        ║\n", us_pure_attn);
    printf("║  5. Cross-head fusion (MixColumns):  %8.3f us/fuse         ║\n", us_fuse);
    printf("║                                                              ║\n");

    double speedup_patgen = us_vecperm / us_hybrid;
    printf("║  Hybrid vs vec_perm:  %.2fx %s                          ║\n",
           speedup_patgen > 1.0 ? speedup_patgen : 1.0/speedup_patgen,
           speedup_patgen > 1.0 ? "FASTER" : "slower");

    printf("║                                                              ║\n");

    /* Correctness: check that both approaches prune similar amounts */
    memcpy(scores_ref, scores_orig, SEQ_LEN * sizeof(float));
    memcpy(scores_vciph, scores_orig, SEQ_LEN * sizeof(float));

    ref_collapse_scores(scores_ref, SEQ_LEN,
                        ref_generate_pattern(0, 0, ref_read_tb()), 1.2f);
    vcipher_hybrid_collapse(scores_vciph, SEQ_LEN, 8, 0, 0);

    int zeros_ref = 0, zeros_vc = 0;
    for (int i = 0; i < SEQ_LEN; i++) {
        if (scores_ref[i] == 0.0f) zeros_ref++;
        if (scores_vciph[i] == 0.0f) zeros_vc++;
    }

    printf("║  Pruning comparison (%d scores):                           ║\n", SEQ_LEN);
    printf("║    vec_perm:  %3d pruned, %3d kept                         ║\n", zeros_ref, SEQ_LEN - zeros_ref);
    printf("║    vcipher:   %3d pruned, %3d kept                         ║\n", zeros_vc, SEQ_LEN - zeros_vc);
    printf("║                                                              ║\n");

    /* Entropy divergence test: same input, vcipher should differ each run */
    int bytes_differ = 0;
    vector unsigned long long state1, state2;
    memcpy(&state1, scores_orig, 16);
    state2 = state1;

    vector unsigned long long rk1 = vc_make_round_key(0, 0);
    state1 = __builtin_crypto_vcipher(state1, rk1);

    /* Tiny delay to get different mftb */
    for (volatile int x = 0; x < 1000; x++) {}

    vector unsigned long long rk2 = vc_make_round_key(0, 0);
    state2 = __builtin_crypto_vcipher(state2, rk2);

    unsigned char b1[16], b2[16];
    memcpy(b1, &state1, 16); memcpy(b2, &state2, 16);
    for (int i = 0; i < 16; i++) if (b1[i] != b2[i]) bytes_differ++;

    printf("║  Entropy divergence:  %d/16 bytes differ (mftb seeded)     ║\n", bytes_differ);
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    vcipher_collapse_banner();

    free(scores_ref);
    free(scores_vciph);
    free(scores_orig);
    free(Q_test);
    free(K_test);

    return 0;
}
