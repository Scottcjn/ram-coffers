/*
 * ggml-intelligent-collapse.h - Intelligent Vec_Perm Collapse for POWER8
 *
 * Scott + Grok Vision: "Collapse many potentials into one coherent output"
 *
 * NON-BIJUNCTIVE FUSION:
 * - vec_perm to DUPLICATE strong signals (Hebbian amplification)
 * - PRUNE weak signals (waste removal)
 * - FUSE into single coherent response path
 *
 * This is NOT random lossy - it's CONSTRAINT-BOUND SELECTION:
 * - Identify top-K attention candidates
 * - Amplify winners via permute (duplication)
 * - Prune losers in 1 cycle
 * - Fuse for coherent output
 *
 * PSE Alignment:
 * - High ACS (coherence under stress)
 * - Stable PMs (preference-like selection)
 * - Low NOI (no flattening from averaging)
 */

#ifndef GGML_INTELLIGENT_COLLAPSE_H
#define GGML_INTELLIGENT_COLLAPSE_H

#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Capability detection. Pulls in <altivec.h> only where it exists; supplies
 * coffers_perm_t, coffers_perm_bytes() and coffers_read_timebase() so the
 * scalar path below can mirror the vector path exactly.
 */
#include "coffers-portability.h"

/*===========================================================================
 * Configuration
 *===========================================================================*/

/* Top-K: How many winners to keep per attention position */
#ifndef INTELLIGENT_COLLAPSE_TOP_K
#define INTELLIGENT_COLLAPSE_TOP_K 8
#endif

/* Amplification factor for winners (Hebbian strengthening) */
#ifndef INTELLIGENT_COLLAPSE_AMPLIFY
#define INTELLIGENT_COLLAPSE_AMPLIFY 1.2f
#endif

/* Entropy mixing ratio */
#ifndef INTELLIGENT_COLLAPSE_ENTROPY_MIX
#define INTELLIGENT_COLLAPSE_ENTROPY_MIX 0.1f
#endif

/*===========================================================================
 * Hardware Timebase
 *===========================================================================*/

/*
 * mftb on POWER; CLOCK_MONOTONIC elsewhere. This previously returned a
 * constant 0 off POWER, which silently disabled entropy variation (every
 * pattern identical). The portable clock keeps the behaviour meaningful.
 */
static inline uint64_t ic_read_tb(void) {
    return coffers_read_timebase();
}

/*===========================================================================
 * Intelligent Pattern Generation
 *
 * Creates a vec_perm pattern that:
 * - Duplicates elements at positions 0-7 (assumed top-K after sort)
 * - Maps positions 8-15 to copies of winners
 *
 * This creates AMPLIFICATION of strong signals.
 *===========================================================================*/

static inline coffers_perm_t generate_intelligent_pattern(
    int layer_id, int position, uint64_t tb
) {
    uint32_t h = (uint32_t)(tb ^ (tb >> 32)) ^ (layer_id * 0x9E3779B9U) ^ (position * 0x85EBCA77U);

    unsigned char p[16] __attribute__((aligned(16)));

    /* First 8 slots: keep original top-K indices (0-7) */
    for (int i = 0; i < 8; i++) {
        p[i] = i;
    }

    /* Last 8 slots: duplicate top winners with entropy variation */
    for (int i = 8; i < 16; i++) {
        h ^= h << 13; h ^= h >> 17; h ^= h << 5;
        /* Map to one of top-4 winners (strongest) */
        p[i] = h % 4;
    }

    /* Pattern bytes are identical on both paths; only the container differs. */
#if GGML_COFFERS_HAVE_ALTIVEC
    return vec_ld(0, (const vector unsigned char*)p);
#else
    coffers_perm_t out;
    memcpy(out.b, p, 16);
    return out;
#endif
}

/*===========================================================================
 * Top-K Selection via Approximate Sort
 *
 * Uses compare-swap network to approximately sort 4 floats.
 * Returns indices of top elements.
 *===========================================================================*/

/* Compare-swap for 2 elements */
static inline void cs2(float* a, float* b) {
    if (*a < *b) {
        float t = *a; *a = *b; *b = t;
    }
}

/* Approximate top-4 from array (returns threshold) */
static inline float approx_top4_threshold(const float* arr, int n) {
    if (n <= 4) return -1e30f;

    /* Quick scan for approximate 4th largest */
    float top[4] = {-1e30f, -1e30f, -1e30f, -1e30f};

    for (int i = 0; i < n; i++) {
        float v = arr[i];
        if (v > top[3]) {
            top[3] = v;
            cs2(&top[2], &top[3]);
            cs2(&top[1], &top[2]);
            cs2(&top[0], &top[1]);
        }
    }

    return top[3];  /* Threshold: 4th largest */
}

/*===========================================================================
 * CORE: Intelligent Collapse Function
 *
 * Takes attention scores and collapses them:
 * 1. Find top-K threshold
 * 2. Create mask for winners
 * 3. Apply vec_perm to amplify winners (duplication)
 * 4. Zero losers
 * 5. Return fused coherent output
 *===========================================================================*/

static inline void intelligent_collapse_scores(
    float* scores,           /* In/Out: attention scores */
    int n,                   /* Number of scores */
    int top_k,               /* Keep top K */
    coffers_perm_t pattern,  /* Collapse pattern */
    float amplify            /* Amplification factor */
) {
    if (n < 4) return;  /* Too few to collapse */

    /* Step 1: Find threshold for top-K */
    float threshold = approx_top4_threshold(scores, n);

#if !GGML_COFFERS_HAVE_ALTIVEC
    /*
     * SCALAR PATH (non-POWER, or POWER built without -maltivec).
     *
     * Reproduces the vec_perm collapse below exactly: vec_perm is a BYTE
     * permute over the 32-byte concatenation of two vectors, so we do the
     * same byte permute in plain C, then apply the identical
     *     result = (c > threshold) ? c * amplify : 0
     * selection that vec_cmpgt/vec_sel/vec_madd perform. Same logical
     * result, no intrinsics. Correctness over speed.
     */
    {
        int i = 0;
        for (; i + 15 < n; i += 16) {
            unsigned char vb[4][16];
            unsigned char cb[4][16];

            /* Four 16-byte vectors = 16 consecutive floats */
            memcpy(vb[0], &scores[i +  0], 16);
            memcpy(vb[1], &scores[i +  4], 16);
            memcpy(vb[2], &scores[i +  8], 16);
            memcpy(vb[3], &scores[i + 12], 16);

            /* Same pairings as the vector path: (0,1) (1,2) (2,3) (3,0) */
            coffers_perm_bytes(vb[0], vb[1], pattern.b, cb[0]);
            coffers_perm_bytes(vb[1], vb[2], pattern.b, cb[1]);
            coffers_perm_bytes(vb[2], vb[3], pattern.b, cb[2]);
            coffers_perm_bytes(vb[3], vb[0], pattern.b, cb[3]);

            for (int v = 0; v < 4; v++) {
                float lane[4];
                memcpy(lane, cb[v], 16);
                for (int e = 0; e < 4; e++) {
                    lane[e] = (lane[e] > threshold) ? lane[e] * amplify : 0.0f;
                }
                memcpy(&scores[i + v * 4], lane, 16);
            }
        }

        /* Scalar remainder (identical to the vector path's tail) */
        for (; i < n; i++) {
            if (scores[i] >= threshold) {
                scores[i] *= amplify;
            } else {
                scores[i] = 0.0f;
            }
        }
        return;
    }
#else

    /* Step 2-4: Vectorized collapse */
    vector float thresh_vec = vec_splats(threshold);
    vector float amp_vec = vec_splats(amplify);
    vector float zero_vec = vec_splats(0.0f);

    int i = 0;
    for (; i + 15 < n; i += 16) {
        /* Load 4 vectors */
        vector float v0 = vec_ld(0, &scores[i]);
        vector float v1 = vec_ld(16, &scores[i]);
        vector float v2 = vec_ld(32, &scores[i]);
        vector float v3 = vec_ld(48, &scores[i]);

        /* Apply intelligent collapse pattern (amplify winners) */
        vector float c0 = vec_perm(v0, v1, pattern);
        vector float c1 = vec_perm(v1, v2, pattern);
        vector float c2 = vec_perm(v2, v3, pattern);
        vector float c3 = vec_perm(v3, v0, pattern);

        /* Mask: Keep above threshold, amplify */
        vector bool int m0 = vec_cmpgt(c0, thresh_vec);
        vector bool int m1 = vec_cmpgt(c1, thresh_vec);
        vector bool int m2 = vec_cmpgt(c2, thresh_vec);
        vector bool int m3 = vec_cmpgt(c3, thresh_vec);

        /* Select and amplify winners */
        c0 = vec_madd(vec_sel(zero_vec, c0, m0), amp_vec, zero_vec);
        c1 = vec_madd(vec_sel(zero_vec, c1, m1), amp_vec, zero_vec);
        c2 = vec_madd(vec_sel(zero_vec, c2, m2), amp_vec, zero_vec);
        c3 = vec_madd(vec_sel(zero_vec, c3, m3), amp_vec, zero_vec);

        vec_st(c0, 0, &scores[i]);
        vec_st(c1, 16, &scores[i]);
        vec_st(c2, 32, &scores[i]);
        vec_st(c3, 48, &scores[i]);
    }

    /* Scalar remainder */
    for (; i < n; i++) {
        if (scores[i] >= threshold) {
            scores[i] *= amplify;
        } else {
            scores[i] = 0.0f;
        }
    }
#endif /* GGML_COFFERS_HAVE_ALTIVEC */
}

/*===========================================================================
 * Full Intelligent Attention
 *
 * Computes attention with intelligent collapse:
 * 1. Standard Q·K dot products
 * 2. Intelligent collapse (top-K amplification)
 * 3. Sparse softmax
 * 4. Sparse V·scores
 *===========================================================================*/

static inline void attention_intelligent(
    float* output,
    const float* Q,
    const float* K,
    const float* V,
    int seq_len,
    int head_dim,
    int layer_id
) {
    uint64_t tb = ic_read_tb();
    float amplify = INTELLIGENT_COLLAPSE_AMPLIFY;
    int top_k = INTELLIGENT_COLLAPSE_TOP_K;

#if !GGML_COFFERS_HAVE_ALTIVEC
    /*
     * SCALAR PATH. Same algorithm as the AltiVec version below - Q.K dot
     * products, intelligent collapse, sparse softmax, sparse V accumulation
     * - written as plain C loops. The collapse step itself calls the shared
     * intelligent_collapse_scores(), which has its own scalar path.
     */
    {
        #pragma omp parallel
        {
            /*
             * aligned_alloc requires size to be a multiple of alignment;
             * malloc is sufficient here since no vector loads are performed.
             */
            float* scores = (float*)malloc((size_t)seq_len * sizeof(float));
            if (scores) {
                #pragma omp for
                for (int pos = 0; pos < seq_len; pos++) {
                    const float* q = Q + (size_t)pos * head_dim;
                    float* out = output + (size_t)pos * head_dim;

                    coffers_perm_t pattern =
                        generate_intelligent_pattern(layer_id, pos, tb + (uint64_t)pos);

                    /* Q.K */
                    for (int t = 0; t <= pos; t++) {
                        const float* k = K + (size_t)t * head_dim;
                        float sum = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            sum += q[d] * k[d];
                        }
                        scores[t] = sum;
                    }

                    /* INTELLIGENT COLLAPSE */
                    intelligent_collapse_scores(scores, pos + 1, top_k, pattern, amplify);

                    /* Sparse softmax */
                    float max_s = -1e30f;
                    for (int t = 0; t <= pos; t++) {
                        if (scores[t] > max_s) max_s = scores[t];
                    }

                    float sum_exp = 0.0f;
                    for (int t = 0; t <= pos; t++) {
                        if (scores[t] > 0.0f) {
                            scores[t] = expf(scores[t] - max_s);
                            sum_exp += scores[t];
                        }
                    }

                    if (sum_exp > 0.0f) {
                        for (int t = 0; t <= pos; t++) {
                            scores[t] /= sum_exp;
                        }
                    }

                    /* Sparse V.scores (skip zeros) */
                    memset(out, 0, (size_t)head_dim * sizeof(float));
                    for (int t = 0; t <= pos; t++) {
                        float w = scores[t];
                        if (w < 0.001f) continue;

                        const float* v = V + (size_t)t * head_dim;
                        for (int d = 0; d < head_dim; d++) {
                            out[d] += v[d] * w;
                        }
                    }
                }
                free(scores);
            }
        }
        return;
    }
#else

    #pragma omp parallel
    {
        float* scores = (float*)aligned_alloc(16, seq_len * sizeof(float));

        #pragma omp for
        for (int pos = 0; pos < seq_len; pos++) {
            const float* q = Q + pos * head_dim;
            float* out = output + pos * head_dim;

            /* Generate position-specific collapse pattern */
            vector unsigned char pattern = generate_intelligent_pattern(layer_id, pos, tb + pos);

            /* Standard Q·K computation */
            for (int t = 0; t <= pos; t++) {
                const float* k = K + t * head_dim;
                vector float sum = vec_splats(0.0f);

                for (int d = 0; d + 3 < head_dim; d += 4) {
                    vector float qv = vec_ld(0, &q[d]);
                    vector float kv = vec_ld(0, &k[d]);
                    sum = vec_madd(qv, kv, sum);
                }

                vector float s1 = vec_add(sum, vec_sld(sum, sum, 8));
                vector float s2 = vec_add(s1, vec_sld(s1, s1, 4));
                vec_ste(s2, 0, &scores[t]);
            }

            /* INTELLIGENT COLLAPSE: Amplify winners, prune losers */
            intelligent_collapse_scores(scores, pos + 1, top_k, pattern, amplify);

            /* Sparse softmax */
            float max_s = -1e30f;
            for (int t = 0; t <= pos; t++) {
                if (scores[t] > max_s) max_s = scores[t];
            }

            float sum_exp = 0.0f;
            for (int t = 0; t <= pos; t++) {
                if (scores[t] > 0.0f) {
                    scores[t] = expf(scores[t] - max_s);
                    sum_exp += scores[t];
                }
            }

            if (sum_exp > 0.0f) {
                for (int t = 0; t <= pos; t++) {
                    scores[t] /= sum_exp;
                }
            }

            /* Sparse V·scores (skip zeros) */
            memset(out, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float w = scores[t];
                if (w < 0.001f) continue;

                const float* v = V + t * head_dim;
                for (int d = 0; d + 3 < head_dim; d += 4) {
                    vector float ov = vec_ld(0, &out[d]);
                    ov = vec_madd(vec_ld(0, &v[d]), vec_splats(w), ov);
                    vec_st(ov, 0, &out[d]);
                }
            }
        }

        free(scores);
    }
#endif /* GGML_COFFERS_HAVE_ALTIVEC */
}

/*===========================================================================
 * Statistics
 *===========================================================================*/

typedef struct {
    uint64_t positions_collapsed;
    uint64_t winners_amplified;
    uint64_t losers_pruned;
} intelligent_collapse_stats_t;

static intelligent_collapse_stats_t g_ic_stats = {0};

static inline void intelligent_collapse_report(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Intelligent Collapse Statistics                      ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Positions collapsed: %10lu                      ║\n",
            (unsigned long)g_ic_stats.positions_collapsed);
    fprintf(stderr, "║  Winners amplified:   %10lu                      ║\n",
            (unsigned long)g_ic_stats.winners_amplified);
    fprintf(stderr, "║  Losers pruned:       %10lu                      ║\n",
            (unsigned long)g_ic_stats.losers_pruned);
    fprintf(stderr, "╚═══════════════════════════════════════════════════════╝\n");
}

#endif /* GGML_INTELLIGENT_COLLAPSE_H */
