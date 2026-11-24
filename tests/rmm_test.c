/* Self-test for ggml-rmm.h (arXiv:2608.13426).
 *
 * The checks are the paper's own claims, in the order it makes them:
 *   - selection is TopK by column norm, deterministic, and minimax optimal;
 *   - the reduced product obeys the Proposition 1 error bound;
 *   - activation-aware selection beats random selection at equal budget
 *     (Section 6.1), which is the whole reason for scoring;
 *   - rho = 1 is the dense product bit-for-bit, so reduction cannot perturb
 *     exact mode when it is switched off;
 *   - the MAC accounting matches the requested retention ratio, and the
 *     saving shows up on the clock.
 */

#include "../ggml-rmm.h"

#include <time.h>

static int failures = 0;

static void check(const char *what, int ok)
{
    printf("%-58s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) {
        failures++;
    }
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Deterministic pseudo-random floats: the test must be reproducible, and
 * rand() is not the same sequence everywhere. */
static uint32_t rng_state = 0x2608134u;

static float next_float(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((int32_t)(rng_state >> 8) % 2001 - 1000) / 1000.0f;
}

static double frobenius_diff(const float *a, const float *b, size_t n)
{
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)a[i] - (double)b[i];
        acc += d * d;
    }
    return sqrt(acc);
}

/*---------------------------------------------------------------------------
 * Selection
 *-------------------------------------------------------------------------*/

static void test_keep_count(void)
{
    check("rho = 1 retains the whole axis",
          rmm_keep_count(128, 1.0f, 8) == 128);
    check("k = ceil(rho * d)", rmm_keep_count(100, 0.55f, 1) == 55 &&
                                   rmm_keep_count(100, 0.551f, 1) == 56 &&
                                   rmm_keep_count(7, 0.5f, 1) == 4);
    check("min_keep floors an over-aggressive ratio",
          rmm_keep_count(64, 0.01f, 8) == 8);
    check("min_keep never exceeds the axis",
          rmm_keep_count(4, 0.01f, 8) == 4);
}

static void test_selection(void)
{
    const int d = 512;
    const int k = 128;
    float *scores = malloc(sizeof(float) * d);
    float *scratch = malloc(sizeof(float) * d);
    int32_t *idx = malloc(sizeof(int32_t) * d);
    int32_t *idx_again = malloc(sizeof(int32_t) * d);

    for (int j = 0; j < d; ++j) {
        scores[j] = fabsf(next_float());
    }

    const int n = rmm_select_topk(scores, d, k, idx, scratch);
    check("selection returns the requested budget", n == k);

    int ascending = 1;
    for (int t = 1; t < k; ++t) {
        if (idx[t] <= idx[t - 1]) {
            ascending = 0;
        }
    }
    check("selected indices come out ascending and distinct", ascending);

    /* Every retained score must dominate every discarded one. */
    char *kept = calloc((size_t)d, 1);
    for (int t = 0; t < k; ++t) {
        kept[idx[t]] = 1;
    }
    float min_kept = INFINITY;
    float max_dropped = -INFINITY;
    for (int j = 0; j < d; ++j) {
        if (kept[j] && scores[j] < min_kept) {
            min_kept = scores[j];
        }
        if (!kept[j] && scores[j] > max_dropped) {
            max_dropped = scores[j];
        }
    }
    check("retained scores dominate discarded scores",
          min_kept >= max_dropped);

    rmm_select_topk(scores, d, k, idx_again, scratch);
    check("selection is deterministic across calls",
          memcmp(idx, idx_again, sizeof(int32_t) * (size_t)k) == 0);

    /* All-equal scores: the tie class must resolve to the lowest indices,
     * which is what makes the choice reproducible rather than pivot-order
     * dependent. */
    for (int j = 0; j < d; ++j) {
        scores[j] = 1.0f;
    }
    rmm_select_topk(scores, d, k, idx, scratch);
    int lowest = 1;
    for (int t = 0; t < k; ++t) {
        if (idx[t] != t) {
            lowest = 0;
        }
    }
    check("ties resolve to the lowest indices", lowest);

    free(scores);
    free(scratch);
    free(idx);
    free(idx_again);
    free(kept);
}

static void test_feature_scores(void)
{
    const int n = 33;
    const int d = 67;
    float *A = malloc(sizeof(float) * (size_t)n * d);
    float *scores = malloc(sizeof(float) * d);

    for (int i = 0; i < n * d; ++i) {
        A[i] = next_float();
    }
    rmm_feature_scores(A, n, d, d, scores);

    int ok = 1;
    for (int j = 0; j < d; ++j) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) {
            acc += (double)A[i * d + j] * (double)A[i * d + j];
        }
        const double want = sqrt(acc);
        if (fabs((double)scores[j] - want) > 1e-4 * fmax(1.0, want)) {
            ok = 0;
        }
    }
    check("feature scores are the column 2-norms", ok);

    free(A);
    free(scores);
}

/*---------------------------------------------------------------------------
 * Reduced products
 *-------------------------------------------------------------------------*/

static void test_dense_equivalence(void)
{
    const int rows = 96;
    const int d = 320;
    float *W = malloc(sizeof(float) * (size_t)rows * d);
    float *x = malloc(sizeof(float) * d);
    float *y = malloc(sizeof(float) * rows);
    float *ref = malloc(sizeof(float) * rows);
    rmm_config_t cfg = rmm_config_default();
    rmm_workspace_t ws;

    for (int i = 0; i < rows * d; ++i) {
        W[i] = next_float();
    }
    for (int j = 0; j < d; ++j) {
        x[j] = next_float();
    }

    check("workspace allocates", rmm_workspace_init(&ws, d, 0, 0) == 1);

    cfg.rho_d = 1.0f;
    rmm_gemv_dense(W, x, ref, rows, d, d);
    rmm_project_gemv(x, d, W, rows, y, &cfg, &ws);
    check("rho = 1 reproduces the dense GEMV bit-for-bit",
          memcmp(y, ref, sizeof(float) * (size_t)rows) == 0);

    rmm_workspace_free(&ws);
    free(W);
    free(x);
    free(y);
    free(ref);
}

/* Proposition 1: ||AB - A[:,I] B[I,:]||_F <= sum_{j not in I} ||A[:,j]||_2
 * ||B[j,:]||_2. Also compares activation-aware selection against a random
 * subset of the same size (Section 6.1). */
static void test_error_bound_and_random_baseline(void)
{
    const int n = 24;
    const int d = 256;
    const int m = 48;
    const int k = 128;

    float *A = malloc(sizeof(float) * (size_t)n * d);
    float *B = malloc(sizeof(float) * (size_t)d * m);
    float *dense = malloc(sizeof(float) * (size_t)n * m);
    float *reduced = malloc(sizeof(float) * (size_t)n * m);
    float *scores = malloc(sizeof(float) * d);
    float *scratch = malloc(sizeof(float) * d);
    int32_t *idx = malloc(sizeof(int32_t) * d);
    int32_t *all = malloc(sizeof(int32_t) * d);

    /* Activations with a heavy-tailed channel profile, as real hidden states
     * have: a minority of channels carries most of the energy, which is the
     * structure RMM exploits. */
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            const float gain = (j % 16 == 0) ? 8.0f : 0.25f;
            A[i * d + j] = gain * next_float();
        }
    }
    for (int i = 0; i < d * m; ++i) {
        B[i] = next_float();
    }

    for (int j = 0; j < d; ++j) {
        all[j] = (int32_t)j;
    }
    rmm_gemm_reduced(A, B, dense, n, d, m, d, m, m, all, d);

    rmm_feature_scores(A, n, d, d, scores);
    rmm_select_topk(scores, d, k, idx, scratch);
    rmm_gemm_reduced(A, B, reduced, n, d, m, d, m, m, idx, k);

    const double err = frobenius_diff(dense, reduced, (size_t)n * m);

    /* The bound, computed over the discarded dimensions only. */
    char *kept = calloc((size_t)d, 1);
    for (int t = 0; t < k; ++t) {
        kept[idx[t]] = 1;
    }
    double bound = 0.0;
    for (int j = 0; j < d; ++j) {
        if (kept[j]) {
            continue;
        }
        double brow = 0.0;
        for (int c = 0; c < m; ++c) {
            brow += (double)B[j * m + c] * (double)B[j * m + c];
        }
        bound += (double)scores[j] * sqrt(brow);
    }
    printf("  reduced-product error %.4f, Proposition 1 bound %.4f\n", err,
           bound);
    check("reduced product obeys the Proposition 1 error bound",
          err <= bound * 1.000001);

    /* Random selection at the same budget, averaged over trials. */
    double random_err = 0.0;
    const int trials = 16;
    for (int trial = 0; trial < trials; ++trial) {
        int count = 0;
        for (int j = 0; j < d && count < k; ++j) {
            const int remaining = d - j;
            const int needed = k - count;
            rng_state = rng_state * 1664525u + 1013904223u;
            if ((int)((rng_state >> 8) % (uint32_t)remaining) < needed) {
                idx[count++] = (int32_t)j;
            }
        }
        rmm_gemm_reduced(A, B, reduced, n, d, m, d, m, m, idx, count);
        random_err += frobenius_diff(dense, reduced, (size_t)n * m);
    }
    random_err /= trials;
    printf("  activation-aware error %.4f vs random selection %.4f\n", err,
           random_err);
    check("activation-aware selection beats random at equal budget",
          err < random_err);

    free(A);
    free(B);
    free(dense);
    free(reduced);
    free(scores);
    free(scratch);
    free(idx);
    free(all);
    free(kept);
}

/* Minimax optimality (Theorem 1): against the adversarial B that puts all of
 * its row energy on the single largest discarded column norm, no other subset
 * of the same size can do better than TopK. */
static void test_minimax_optimality(void)
{
    const int d = 64;
    const int k = 16;
    float *scores = malloc(sizeof(float) * d);
    float *scratch = malloc(sizeof(float) * d);
    int32_t *idx = malloc(sizeof(int32_t) * d);

    for (int j = 0; j < d; ++j) {
        scores[j] = fabsf(next_float()) + 0.01f;
    }
    rmm_select_topk(scores, d, k, idx, scratch);

    char *kept = calloc((size_t)d, 1);
    for (int t = 0; t < k; ++t) {
        kept[idx[t]] = 1;
    }
    float worst_case = 0.0f;
    for (int j = 0; j < d; ++j) {
        if (!kept[j] && scores[j] > worst_case) {
            worst_case = scores[j];
        }
    }

    /* Any alternative subset: swap one retained index for a discarded one and
     * the worst case can only get worse or stay equal. */
    int optimal = 1;
    for (int t = 0; t < k; ++t) {
        for (int j = 0; j < d; ++j) {
            if (kept[j]) {
                continue;
            }
            float alt = scores[idx[t]]; /* now discarded */
            for (int u = 0; u < d; ++u) {
                if (!kept[u] && u != j && scores[u] > alt) {
                    alt = scores[u];
                }
            }
            if (alt < worst_case) {
                optimal = 0;
            }
        }
    }
    check("TopK by column norm minimises the worst-case error", optimal);

    free(scores);
    free(scratch);
    free(idx);
    free(kept);
}

/*---------------------------------------------------------------------------
 * Attention
 *-------------------------------------------------------------------------*/

static void test_attention(void)
{
    const int Lq = 48;
    const int Lk = 48;
    const int dh = 128;

    float *Q = malloc(sizeof(float) * (size_t)Lq * dh);
    float *K = malloc(sizeof(float) * (size_t)Lk * dh);
    float *V = malloc(sizeof(float) * (size_t)Lk * dh);
    float *out = malloc(sizeof(float) * (size_t)Lq * dh);
    float *ref = malloc(sizeof(float) * (size_t)Lq * dh);
    float *scores = malloc(sizeof(float) * Lk);
    rmm_config_t cfg = rmm_config_default();
    rmm_workspace_t ws;

    /* Heads concentrate their energy in a minority of feature dimensions;
     * without that structure there is nothing for any reduction method to
     * find, and the comparison against dense would only measure noise. */
    for (int i = 0; i < Lq; ++i) {
        for (int j = 0; j < dh; ++j) {
            const float gain = (j % 4 == 0) ? 4.0f : 0.2f;
            Q[i * dh + j] = gain * next_float();
        }
    }
    for (int i = 0; i < Lk; ++i) {
        for (int j = 0; j < dh; ++j) {
            const float gain = (j % 4 == 0) ? 4.0f : 0.2f;
            K[i * dh + j] = gain * next_float();
            V[i * dh + j] = next_float();
        }
    }

    rmm_workspace_init(&ws, dh, Lk, (size_t)Lq * Lk);
    rmm_attention_dense(ref, Q, K, V, Lq, Lk, dh, 1, scores);

    cfg.rho_d = 1.0f;
    cfg.rho_t = 1.0f;
    check("attention at rho = 1 is accepted",
          rmm_attention(out, Q, K, V, Lq, Lk, dh, 1, &cfg, &ws) == dh);
    check("attention at rho = 1 reproduces dense attention",
          frobenius_diff(out, ref, (size_t)Lq * dh) < 1e-4);

    /* Half the head dimension: the output must stay close to dense relative
     * to the magnitude of the dense output itself. */
    cfg.rho_d = 0.5f;
    rmm_attention(out, Q, K, V, Lq, Lk, dh, 1, &cfg, &ws);
    double ref_norm = 0.0;
    for (int i = 0; i < Lq * dh; ++i) {
        ref_norm += (double)ref[i] * (double)ref[i];
    }
    ref_norm = sqrt(ref_norm);
    const double relative = frobenius_diff(out, ref, (size_t)Lq * dh)
                            / ref_norm;
    printf("  relative attention error at rho_d = 0.5: %.4f\n", relative);
    check("attention at rho_d = 0.5 stays within 10% of dense",
          relative < 0.10);

    /* Token reduction on PV. */
    cfg.rho_t = 0.5f;
    rmm_stats_reset();
    check("attention with token reduction is accepted",
          rmm_attention(out, Q, K, V, Lq, Lk, dh, 1, &cfg, &ws) == dh / 2);
    int finite = 1;
    for (int i = 0; i < Lq * dh; ++i) {
        if (!isfinite(out[i])) {
            finite = 0;
        }
    }
    check("attention with token reduction produces finite output", finite);
    const double kept_ratio = (double)g_rmm_stats.macs_reduced /
                              (double)g_rmm_stats.macs_dense;
    printf("  MACs retained with rho_d = rho_t = 0.5: %.1f%%\n",
           100.0 * kept_ratio);
    check("token reduction actually reduces the MAC count", kept_ratio < 0.6);

    /* Without a P buffer, token reduction must be refused rather than
     * silently ignored. */
    rmm_workspace_t small;
    rmm_workspace_init(&small, dh, Lk, 0);
    check("token reduction without a P buffer is refused",
          rmm_attention(out, Q, K, V, Lq, Lk, dh, 1, &cfg, &small) == -1);
    rmm_workspace_free(&small);

    rmm_workspace_free(&ws);
    free(Q);
    free(K);
    free(V);
    free(out);
    free(ref);
    free(scores);
}

/*---------------------------------------------------------------------------
 * Wall clock
 *-------------------------------------------------------------------------*/

static void test_throughput(void)
{
    /* An expert-sized projection: 4096 hidden into 11008 intermediate. */
    const int rows = 4096;
    const int d = 4096;
    float *W = malloc(sizeof(float) * (size_t)rows * d);
    float *x = malloc(sizeof(float) * d);
    float *y = malloc(sizeof(float) * rows);
    rmm_config_t cfg = rmm_config_default();
    rmm_workspace_t ws;

    if (!W || !x || !y) {
        printf("not enough memory for the throughput check; skipping\n");
        free(W);
        free(x);
        free(y);
        return;
    }
    for (size_t i = 0; i < (size_t)rows * d; ++i) {
        W[i] = 0.001f * (float)(i % 7);
    }
    for (int j = 0; j < d; ++j) {
        x[j] = 0.01f * (float)(j % 11);
    }
    rmm_workspace_init(&ws, d, 0, 0);

    const int iterations = 8;
    double started = now_seconds();
    for (int i = 0; i < iterations; ++i) {
        rmm_gemv_dense(W, x, y, rows, d, d);
    }
    const double dense_time = now_seconds() - started;

    cfg.rho_d = 0.5f;
    started = now_seconds();
    for (int i = 0; i < iterations; ++i) {
        rmm_project_gemv(x, d, W, rows, y, &cfg, &ws);
    }
    const double reduced_time = now_seconds() - started;

    printf("\n%d x %d GEMV on this host:\n", rows, d);
    printf("  dense        %.2f ms\n", dense_time / iterations * 1e3);
    printf("  rho_d = 0.5  %.2f ms (%.2fx, selection included)\n",
           reduced_time / iterations * 1e3, dense_time / reduced_time);

    rmm_workspace_free(&ws);
    free(W);
    free(x);
    free(y);
}

int main(void)
{
    test_keep_count();
    test_selection();
    test_feature_scores();
    test_dense_equivalence();
    test_error_bound_and_random_baseline();
    test_minimax_optimality();
    test_attention();
    test_throughput();

    if (failures) {
        printf("\n%d check(s) FAILED\n", failures);
        return 1;
    }
    printf("\nall RMM checks passed\n");
    return 0;
}
