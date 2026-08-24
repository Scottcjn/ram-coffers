/*
 * ggml-rmm.h - Reduced Matrix Multiplication (RMM)
 *
 * Implementation of "Reduced Matrix Multiplication: Input-Adaptive
 * Matrix-Product Reduction for LLM Inference", Lan, Li and Zhou,
 * arXiv:2608.13426.
 *
 * THE METHOD
 * ----------
 * Every heavy Transformer op is Y = A B with A in R^{n x d} (activations,
 * known at inference time) and B in R^{d x m} (weights, or K/V). RMM keeps
 * only a subset I of the shared contraction axis:
 *
 *     RMM_rho(A, B) = A[:, I] B[I, :],    |I| = ceil(rho * d)
 *
 * and picks I deterministically from the *current* activations by column
 * norm, s_j = ||A[:, j]||_2, I = TopK(s, ceil(rho*d)). That rule is minimax
 * optimal (paper, Appendix E.1): since AB = sum_j A[:,j] B[j,:] and dropping
 * j costs ||A[:,j]||_2 ||B[j,:]||_2, retaining the largest column norms
 * minimises the worst-case Frobenius error over every B consistent with the
 * budget. No weights are modified and nothing is trained; the retained
 * subspace is free to move between inputs, layers, heads and decode steps.
 *
 * WHY IT BELONGS HERE
 * -------------------
 * A retention ratio on the contraction axis is a retention ratio on *rows of
 * the weight matrix*: X[:, I] W[I, :] touches only |I| rows of W. On POWER8
 * with weights living in mmap'd NUMA coffers, that is the difference between
 * streaming a whole bank and streaming rho of it, so the FLOP saving shows up
 * as a bandwidth and page-residency saving too - the axis this project is
 * actually short of. rmm_prefetch_rows() warms just the retained rows.
 *
 * Relation to the collapse headers: ggml-topk-collapse-vsx.h prunes attention
 * *scores* after QK^T is already paid for. RMM instead shrinks the products
 * themselves - the head dimension inside QK^T, and (optionally) the token axis
 * inside PV - so the saving is in the multiply, not in the softmax.
 *
 * Portable C11: the scalar path builds anywhere (that is what CI exercises),
 * with a VSX/AltiVec path on POWER8 when the target has it.
 *
 * License: Apache 2.0
 */

#ifndef GGML_RMM_H
#define GGML_RMM_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* VSX only, not plain AltiVec: the loops below load at arbitrary float
 * offsets, and vec_xl tolerates that while vec_ld would silently truncate the
 * address to a 16-byte boundary. */
#if defined(__VSX__)
#include <altivec.h>
#include "power8-compat.h"
#define RMM_HAVE_VSX 1
#else
#define RMM_HAVE_VSX 0
#endif

/*===========================================================================
 * Configuration
 *
 * rho_d - retention ratio on the feature/contraction axis (hidden channels
 *         for projections, head dimension for QK^T).
 * rho_t - retention ratio on the token axis of PV. 1.0f leaves PV dense,
 *         which is the paper's default for the main results.
 *
 * The paper's component analysis found attention-side products far more
 * reducible than MLP ones, so the two axes are configured separately and
 * callers are expected to give MLPs a higher rho_d than attention.
 *===========================================================================*/

#ifndef RMM_RHO_D_DEFAULT
#define RMM_RHO_D_DEFAULT 0.75f
#endif

#ifndef RMM_RHO_T_DEFAULT
#define RMM_RHO_T_DEFAULT 1.0f
#endif

/* Never reduce below this many indices, whatever rho asks for: at tiny d a
 * ratio alone can collapse a product to noise. */
#ifndef RMM_MIN_KEEP
#define RMM_MIN_KEEP 8
#endif

typedef struct {
    float rho_d;
    float rho_t;
    int   min_keep;
} rmm_config_t;

static inline rmm_config_t rmm_config_default(void) {
    rmm_config_t cfg;
    cfg.rho_d    = RMM_RHO_D_DEFAULT;
    cfg.rho_t    = RMM_RHO_T_DEFAULT;
    cfg.min_keep = RMM_MIN_KEEP;
    return cfg;
}

/* ceil(rho * d), clamped to [min(min_keep, d), d]. rho >= 1 gives d, so
 * rho = 1 degenerates to the dense product - see rmm_gemv_reduced(). */
static inline int rmm_keep_count(int d, float rho, int min_keep) {
    if (d <= 0) return 0;
    if (!(rho > 0.0f)) return (min_keep < d) ? min_keep : d;
    if (rho >= 1.0f) return d;

    int k = (int)ceilf(rho * (float)d);
    if (min_keep > 0 && k < min_keep) k = min_keep;
    if (k > d) k = d;
    if (k < 1) k = 1;
    return k;
}

/*===========================================================================
 * Statistics
 *
 * Counts multiply-accumulates as they would have been paid densely versus as
 * RMM actually paid them, so a run can report the realised reduction rather
 * than the requested one (min_keep and causal masking make the two differ).
 *===========================================================================*/

typedef struct {
    uint64_t macs_dense;
    uint64_t macs_reduced;
    uint64_t selections;
} rmm_stats_t;

static rmm_stats_t g_rmm_stats = {0, 0, 0};

static inline void rmm_stats_reset(void) {
    memset(&g_rmm_stats, 0, sizeof(g_rmm_stats));
}

static inline void rmm_stats_report(void) {
    const double dense = (double)g_rmm_stats.macs_dense;
    const double kept  = (double)g_rmm_stats.macs_reduced;
    fprintf(stderr, "RMM: %llu selections, %.3g/%.3g MACs kept (%.1f%%)\n",
            (unsigned long long)g_rmm_stats.selections, kept, dense,
            dense > 0.0 ? 100.0 * kept / dense : 100.0);
}

/*===========================================================================
 * Feature scores: s_j = ||A[:, j]||_2
 *
 * A is row-major n x d with row stride lda. Accumulating squares column-wise
 * over a row-major walk keeps A streaming forwards, which matters more than
 * the vectorisation: at prefill n is large and A does not fit in L2.
 *===========================================================================*/

static inline void rmm_feature_scores(const float *A, int n, int d, int lda,
                                      float *scores) {
    memset(scores, 0, (size_t)d * sizeof(float));

    for (int i = 0; i < n; i++) {
        const float *row = A + (size_t)i * (size_t)lda;
        int j = 0;
#if RMM_HAVE_VSX
        for (; j + 3 < d; j += 4) {
            vector float v = vec_xl(0, &row[j]);
            vector float s = vec_xl(0, &scores[j]);
            vec_xst(vec_madd(v, v, s), 0, &scores[j]);
        }
#endif
        for (; j < d; j++) {
            scores[j] += row[j] * row[j];
        }
    }

    for (int j = 0; j < d; j++) {
        scores[j] = sqrtf(scores[j]);
    }
}

/*===========================================================================
 * Deterministic TopK selection
 *
 * Two steps, both O(d) on average:
 *   1. quickselect the k-th largest score (the retention threshold);
 *   2. sweep j ascending, taking everything above the threshold and then as
 *      many ties as the budget still allows.
 *
 * The ascending sweep is what makes selection reproducible: ties are broken
 * by lowest index rather than by quickselect's pivot order, so the same
 * activations always yield the same I, on any host and at any -O level. The
 * output indices come out ascending, which also keeps the gathers in
 * ggml-rmm's inner loops moving forwards through memory.
 *
 * scratch must hold d floats; scores itself is not modified.
 *===========================================================================*/

static inline float rmm_kth_largest(const float *scores, int d, int k,
                                    float *scratch) {
    if (k >= d) return -INFINITY;
    if (k <= 0) return INFINITY;

    memcpy(scratch, scores, (size_t)d * sizeof(float));

    int lo = 0, hi = d - 1;
    const int target = k - 1;

    while (lo < hi) {
        const float pivot = scratch[hi];
        int store = lo;
        for (int i = lo; i < hi; i++) {
            if (scratch[i] >= pivot) {
                const float tmp = scratch[store];
                scratch[store] = scratch[i];
                scratch[i] = tmp;
                store++;
            }
        }
        const float tmp = scratch[store];
        scratch[store] = scratch[hi];
        scratch[hi] = tmp;

        if (store == target) return scratch[store];
        if (store < target) lo = store + 1;
        else hi = store - 1;
    }
    return scratch[lo];
}

/* Returns the number of indices written to idx (always k for 0 < k <= d). */
static inline int rmm_select_topk(const float *scores, int d, int k,
                                  int32_t *idx, float *scratch) {
    if (k >= d) {
        for (int j = 0; j < d; j++) idx[j] = (int32_t)j;
        g_rmm_stats.selections++;
        return d;
    }
    if (k <= 0) return 0;

    const float threshold = rmm_kth_largest(scores, d, k, scratch);

    /* How much of the budget the tie class at the threshold gets to fill. */
    int strictly_above = 0;
    for (int j = 0; j < d; j++) {
        if (scores[j] > threshold) strictly_above++;
    }
    int ties_left = k - strictly_above;

    int count = 0;
    for (int j = 0; j < d && count < k; j++) {
        if (scores[j] > threshold) {
            idx[count++] = (int32_t)j;
        } else if (scores[j] == threshold && ties_left > 0) {
            idx[count++] = (int32_t)j;
            ties_left--;
        }
    }

    g_rmm_stats.selections++;
    return count;
}

/*===========================================================================
 * Reduced products
 *===========================================================================*/

/* Warm the retained rows of a weight matrix. On POWER8 this is the dcbt the
 * rest of the project uses; elsewhere it is the compiler's prefetch builtin.
 * Only the rows RMM will actually read are touched, which is the point: at
 * rho = 0.5 half the bank never enters cache. */
static inline void rmm_prefetch_rows(const float *B, int ldb,
                                     const int32_t *idx, int k) {
    for (int t = 0; t < k; t++) {
        const float *row = B + (size_t)idx[t] * (size_t)ldb;
#if defined(__powerpc64__) || defined(__powerpc__)
        __asm__ __volatile__("dcbt 0,%0" : : "r"(row) : "memory");
#else
        __builtin_prefetch(row, 0, 1);
#endif
    }
}

/*
 * y = W[:, I] x[I], W row-major rows x d with row stride ldw.
 *
 * This is the projection/FFN case with n = 1 (decode). Summation follows I
 * ascending, so at rho = 1 (I = 0..d-1) the result is bit-identical to the
 * dense kernel rather than merely close - reduction must not perturb exact
 * mode when it is switched off.
 */
static inline void rmm_gemv_reduced(const float *W, const float *x, float *y,
                                    int rows, int d, int ldw,
                                    const int32_t *idx, int k) {
    g_rmm_stats.macs_dense   += (uint64_t)rows * (uint64_t)d;
    g_rmm_stats.macs_reduced += (uint64_t)rows * (uint64_t)k;

    for (int m = 0; m < rows; m++) {
        const float *w = W + (size_t)m * (size_t)ldw;
        float sum = 0.0f;
        for (int t = 0; t < k; t++) {
            const int j = idx[t];
            sum += w[j] * x[j];
        }
        y[m] = sum;
    }
}

/*
 * Y = A[:, I] B[I, :], A row-major n x d, B row-major d x m, Y row-major
 * n x m.
 *
 * Accumulated as a sum of rank-one updates over the retained axis (the same
 * decomposition the minimax argument uses), so each retained row of B is
 * streamed once per output row and never re-gathered.
 */
static inline void rmm_gemm_reduced(const float *A, const float *B, float *Y,
                                    int n, int d, int m, int lda, int ldb,
                                    int ldy, const int32_t *idx, int k) {
    g_rmm_stats.macs_dense   += (uint64_t)n * (uint64_t)d * (uint64_t)m;
    g_rmm_stats.macs_reduced += (uint64_t)n * (uint64_t)k * (uint64_t)m;

    for (int i = 0; i < n; i++) {
        const float *a = A + (size_t)i * (size_t)lda;
        float *out = Y + (size_t)i * (size_t)ldy;
        memset(out, 0, (size_t)m * sizeof(float));

        for (int t = 0; t < k; t++) {
            const int j = idx[t];
            const float scale = a[j];
            if (scale == 0.0f) continue;
            const float *b = B + (size_t)j * (size_t)ldb;

            int c = 0;
#if RMM_HAVE_VSX
            vector float sv = vec_splats(scale);
            for (; c + 3 < m; c += 4) {
                vector float bv = vec_xl(0, &b[c]);
                vector float ov = vec_xl(0, &out[c]);
                vec_xst(vec_madd(bv, sv, ov), 0, &out[c]);
            }
#endif
            for (; c < m; c++) {
                out[c] += scale * b[c];
            }
        }
    }
}

/* S = (1/sqrt(dh)) Q[:, I] K[:, I]^T, one query row at a time.
 *
 * The scale stays 1/sqrt(dh), not 1/sqrt(k): RMM approximates the dense
 * logits, so rescaling by the reduced width would shift the softmax
 * temperature away from the model's calibration. */
static inline void rmm_scores_reduced(const float *q, const float *K,
                                      float *scores, int Lk, int dh, int ldk,
                                      const int32_t *idx, int k, int limit) {
    const float scale = 1.0f / sqrtf((float)dh);

    g_rmm_stats.macs_dense   += (uint64_t)limit * (uint64_t)dh;
    g_rmm_stats.macs_reduced += (uint64_t)limit * (uint64_t)k;
    (void)Lk;

    for (int t = 0; t < limit; t++) {
        const float *krow = K + (size_t)t * (size_t)ldk;
        float sum = 0.0f;
        for (int u = 0; u < k; u++) {
            const int j = idx[u];
            sum += q[j] * krow[j];
        }
        scores[t] = sum * scale;
    }
}

/*===========================================================================
 * Workspace
 *
 * Selection needs d scores, d indices and a scratch copy for quickselect;
 * token reduction additionally needs the full P matrix, because token scores
 * a_t = ||P[:, t]||_2 are taken over the whole query block and so cannot be
 * formed while streaming P row by row. Sizing is done once per model rather
 * than per call: these paths run inside the decode loop.
 *===========================================================================*/

typedef struct {
    float   *scores;
    float   *scratch;
    int32_t *idx;
    float   *tok_scores;
    float   *tok_scratch;
    int32_t *tok_idx;
    float   *probs;      /* Lq x Lk, only when token reduction is wanted */
    int      cap_d;
    int      cap_tokens;
    int      cap_probs;
} rmm_workspace_t;

/* max_probs is max_q_rows * max_tokens, or 0 to leave PV dense. */
static inline int rmm_workspace_init(rmm_workspace_t *ws, int max_d,
                                     int max_tokens, size_t max_probs) {
    memset(ws, 0, sizeof(*ws));

    ws->scores  = (float *)calloc((size_t)max_d, sizeof(float));
    ws->scratch = (float *)calloc((size_t)max_d, sizeof(float));
    ws->idx     = (int32_t *)calloc((size_t)max_d, sizeof(int32_t));
    ws->cap_d   = max_d;

    if (max_tokens > 0) {
        ws->tok_scores  = (float *)calloc((size_t)max_tokens, sizeof(float));
        ws->tok_scratch = (float *)calloc((size_t)max_tokens, sizeof(float));
        ws->tok_idx     = (int32_t *)calloc((size_t)max_tokens,
                                            sizeof(int32_t));
        ws->cap_tokens  = max_tokens;
    }
    if (max_probs > 0) {
        ws->probs     = (float *)calloc(max_probs, sizeof(float));
        ws->cap_probs = (int)max_probs;
    }

    if (!ws->scores || !ws->scratch || !ws->idx ||
        (max_tokens > 0 && (!ws->tok_scores || !ws->tok_scratch ||
                            !ws->tok_idx)) ||
        (max_probs > 0 && !ws->probs)) {
        return 0;
    }
    return 1;
}

static inline void rmm_workspace_free(rmm_workspace_t *ws) {
    free(ws->scores);
    free(ws->scratch);
    free(ws->idx);
    free(ws->tok_scores);
    free(ws->tok_scratch);
    free(ws->tok_idx);
    free(ws->probs);
    memset(ws, 0, sizeof(*ws));
}

/*===========================================================================
 * Section 3.3: linear and MLP projections
 *
 * Y = X[:, I] W[I, :] with I chosen from ||X[:, j]||_2. The same call covers
 * Q/K/V projections, the attention output projection, and the gate/up/down
 * matrices of an FFN - anything of the form activations x weights.
 *===========================================================================*/

static inline int rmm_project(const float *X, int L, int d, const float *W,
                              int dout, float *Y, const rmm_config_t *cfg,
                              rmm_workspace_t *ws) {
    if (d > ws->cap_d) return -1;

    const int k = rmm_keep_count(d, cfg->rho_d, cfg->min_keep);
    rmm_feature_scores(X, L, d, d, ws->scores);
    rmm_select_topk(ws->scores, d, k, ws->idx, ws->scratch);
    rmm_prefetch_rows(W, dout, ws->idx, k);
    rmm_gemm_reduced(X, W, Y, L, d, dout, d, dout, dout, ws->idx, k);
    return k;
}

/* Decode-step form of rmm_project() for a weight matrix stored row-major as
 * rows x d (the layout the coffer headers mmap), i.e. y = W x reduced over
 * the shared axis of a single activation vector. */
static inline int rmm_project_gemv(const float *x, int d, const float *W,
                                   int rows, float *y,
                                   const rmm_config_t *cfg,
                                   rmm_workspace_t *ws) {
    if (d > ws->cap_d) return -1;

    const int k = rmm_keep_count(d, cfg->rho_d, cfg->min_keep);
    rmm_feature_scores(x, 1, d, d, ws->scores);
    rmm_select_topk(ws->scores, d, k, ws->idx, ws->scratch);
    rmm_gemv_reduced(W, x, y, rows, d, d, ws->idx, k);
    return k;
}

/*===========================================================================
 * Section 3.3: attention
 *
 * Per head: score the head dimension with s_j = ||Q[:, j]||_2, reduce QK^T
 * over the retained dimensions, softmax (with the causal mask if asked), then
 * optionally reduce PV over the token axis using a_t = ||P[:, t]||_2.
 *
 * Grouped-query attention: selection is per head on Q, and the retained
 * dimensions are gathered from the shared K/V - so call this once per query
 * head, passing that head's Q with the K/V of its group.
 *
 * Token reduction drops the discarded attention mass instead of
 * renormalising, exactly as the paper defines PV -> P[:, T] V[T, :]; the rows
 * of the output are therefore slightly shrunk, which is part of the
 * approximation being measured rather than a bug to patch out.
 *===========================================================================*/

static inline int rmm_attention(float *out, const float *Q, const float *K,
                                const float *V, int Lq, int Lk, int dh,
                                int causal, const rmm_config_t *cfg,
                                rmm_workspace_t *ws) {
    if (dh > ws->cap_d) return -1;

    if (Lk > ws->cap_tokens) return -1;

    const int reduce_tokens = (cfg->rho_t < 1.0f);
    if (reduce_tokens &&
        (size_t)Lq * (size_t)Lk > (size_t)ws->cap_probs) {
        return -1;
    }

    /* Feature selection on the head dimension, shared by every query row of
     * this head (Q is the observed operand for the whole block). */
    const int k = rmm_keep_count(dh, cfg->rho_d, cfg->min_keep);
    rmm_feature_scores(Q, Lq, dh, dh, ws->scores);
    rmm_select_topk(ws->scores, dh, k, ws->idx, ws->scratch);

    /* With token reduction the whole of P is materialised; without it, one
     * row at a time is enough and tok_scores doubles as that row. */
    float *P = reduce_tokens ? ws->probs : ws->tok_scores;
    const int ldp = reduce_tokens ? Lk : 0;

    for (int i = 0; i < Lq; i++) {
        float *row = reduce_tokens ? P + (size_t)i * (size_t)ldp : P;
        const int limit = causal ? (i + 1 < Lk ? i + 1 : Lk) : Lk;

        rmm_scores_reduced(Q + (size_t)i * (size_t)dh, K, row, Lk, dh, dh,
                           ws->idx, k, limit);

        float max_score = -INFINITY;
        for (int t = 0; t < limit; t++) {
            if (row[t] > max_score) max_score = row[t];
        }
        float sum_exp = 0.0f;
        for (int t = 0; t < limit; t++) {
            row[t] = expf(row[t] - max_score);
            sum_exp += row[t];
        }
        const float inv = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
        for (int t = 0; t < limit; t++) {
            row[t] *= inv;
        }
        for (int t = limit; t < Lk; t++) {
            row[t] = 0.0f;
        }

        if (!reduce_tokens) {
            /* Stream PV straight out of the row; no P buffer needed. */
            float *o = out + (size_t)i * (size_t)dh;
            memset(o, 0, (size_t)dh * sizeof(float));
            g_rmm_stats.macs_dense   += (uint64_t)limit * (uint64_t)dh;
            g_rmm_stats.macs_reduced += (uint64_t)limit * (uint64_t)dh;
            for (int t = 0; t < limit; t++) {
                const float w = row[t];
                if (w == 0.0f) continue;
                const float *v = V + (size_t)t * (size_t)dh;
                for (int c = 0; c < dh; c++) {
                    o[c] += w * v[c];
                }
            }
        }
    }

    if (!reduce_tokens) return k;

    /* Token selection over the whole query block: a_t = ||P[:, t]||_2. */
    const int kt = rmm_keep_count(Lk, cfg->rho_t, cfg->min_keep);
    rmm_feature_scores(P, Lq, Lk, Lk, ws->tok_scores);
    rmm_select_topk(ws->tok_scores, Lk, kt, ws->tok_idx, ws->tok_scratch);
    rmm_prefetch_rows(V, dh, ws->tok_idx, kt);
    rmm_gemm_reduced(P, V, out, Lq, Lk, dh, Lk, dh, dh, ws->tok_idx, kt);
    return k;
}

/*===========================================================================
 * Dense references
 *
 * Kept in the header so callers (and the self-test) can compare a reduced
 * product against the dense one without writing a second kernel that might
 * differ in summation order.
 *===========================================================================*/

static inline void rmm_gemv_dense(const float *W, const float *x, float *y,
                                  int rows, int d, int ldw) {
    for (int m = 0; m < rows; m++) {
        const float *w = W + (size_t)m * (size_t)ldw;
        float sum = 0.0f;
        for (int j = 0; j < d; j++) sum += w[j] * x[j];
        y[m] = sum;
    }
}

static inline void rmm_attention_dense(float *out, const float *Q,
                                       const float *K, const float *V,
                                       int Lq, int Lk, int dh, int causal,
                                       float *scores) {
    const float scale = 1.0f / sqrtf((float)dh);

    for (int i = 0; i < Lq; i++) {
        const float *q = Q + (size_t)i * (size_t)dh;
        const int limit = causal ? (i + 1 < Lk ? i + 1 : Lk) : Lk;

        for (int t = 0; t < limit; t++) {
            const float *krow = K + (size_t)t * (size_t)dh;
            float sum = 0.0f;
            for (int j = 0; j < dh; j++) sum += q[j] * krow[j];
            scores[t] = sum * scale;
        }

        float max_score = -INFINITY;
        for (int t = 0; t < limit; t++) {
            if (scores[t] > max_score) max_score = scores[t];
        }
        float sum_exp = 0.0f;
        for (int t = 0; t < limit; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        const float inv = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;

        float *o = out + (size_t)i * (size_t)dh;
        memset(o, 0, (size_t)dh * sizeof(float));
        for (int t = 0; t < limit; t++) {
            const float w = scores[t] * inv;
            const float *v = V + (size_t)t * (size_t)dh;
            for (int c = 0; c < dh; c++) o[c] += w * v[c];
        }
    }
}

#endif /* GGML_RMM_H */
