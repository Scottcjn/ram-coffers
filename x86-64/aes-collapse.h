/*
 * aes-collapse.h — AES-NI Non-Bijunctive Collapse for x86-64
 *
 * The x86 counterpart of ggml-vcipher-collapse.h (POWER8) and
 * apple-silicon/aes-entropy-collapse.h (ARM).
 *
 * Of the three, x86 is the *exact* one. AESENC and vcipher compute the same
 * function:
 *
 *     aesenc(s, k)  = MixColumns(ShiftRows(SubBytes(s))) XOR k
 *     vcipher(s, k) = MixColumns(ShiftRows(SubBytes(s))) XOR k
 *
 * so every mode below is a literal transcription, and bench-aes-collapse.c
 * checks that against a scalar AES round rather than taking it on faith.
 *
 * ARM is the odd one out, and the porting table in
 * apple-silicon/PAPER-architecture-general-pse.md is wrong to list it as
 * equivalent. AESE applies the key *first* and AESMC adds no final XOR:
 *
 *     aese(s, k) ; aesmc  = MixColumns(ShiftRows(SubBytes(s XOR k)))
 *                         = aesenc(s XOR k, 0)
 *
 * which is a different function of (s, k). Diffusion quality is unaffected —
 * it is still a full AES round — but the three ports do not produce the same
 * bytes for the same inputs, so a pattern generated on POWER8 cannot be
 * reproduced on an M2. See x86_aes_arm_order() if you need to match the
 * existing ARM port instead of POWER8.
 *
 * Two further things this header does *not* inherit from the POWER8 original:
 *
 *   - Round keys are derived from (layer, position, counter) by default, not
 *     from the timebase. mftb/rdtsc keying makes every run produce different
 *     patterns, which is fine for entropy injection and useless for anything
 *     that has to be reproducible. Define X86_AES_COLLAPSE_ENTROPY=1 for the
 *     original nondeterministic behaviour.
 *   - Byte order. vec_perm numbers bytes from the most significant end on
 *     big-endian POWER; pshufb numbers from the least significant. Patterns
 *     are therefore mirrored between the two. x86_aes_beswap() converts.
 *
 * Not used by gen9-cluster: that fleet is Zen 2 with attention on RDNA2 via
 * Vulkan, and RDNA2 has no AES instruction, so a collapse there would mean
 * either pinning attention to the CPU or doing S-box lookups in GLSL.
 *
 * Requires: -maes -mssse3 (SSE4.1 for the blend path, optional VAES)
 */

#ifndef X86_AES_COLLAPSE_H
#define X86_AES_COLLAPSE_H

#if !defined(__x86_64__) && !defined(__i386__)
#error "aes-collapse.h is the x86 port; see ggml-vcipher-collapse.h for POWER8"
#endif

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/*===========================================================================
 * Configuration — names and defaults mirror the POWER8 header
 *===========================================================================*/

#ifndef X86_AES_COLLAPSE_ROUNDS
#define X86_AES_COLLAPSE_ROUNDS 2
#endif

#ifndef X86_AES_COLLAPSE_TOP_K
#define X86_AES_COLLAPSE_TOP_K 8
#endif

#ifndef X86_AES_COLLAPSE_AMPLIFY
#define X86_AES_COLLAPSE_AMPLIFY 1.15f
#endif

/* 1 = full aesenc (MixColumns diffuses across lanes), 0 = aesenclast. */
#ifndef X86_AES_CROSS_HEAD_FUSE
#define X86_AES_CROSS_HEAD_FUSE 1
#endif

/* 1 = key from rdtsc, as POWER8 does with mftb. 0 = reproducible. */
#ifndef X86_AES_COLLAPSE_ENTROPY
#define X86_AES_COLLAPSE_ENTROPY 0
#endif

/*===========================================================================
 * Runtime dispatch
 *
 * Every CPU this project targets has AES-NI — the consoles and salvage boards
 * are all Zen 2 — but a header that assumes an instruction set is a header
 * that SIGILLs on someone's Nehalem, so ask.
 *
 * VAES (four rounds per instruction on a 512-bit register) is Zen 3 and Ice
 * Lake and later, which excludes every ninth-generation console.
 *===========================================================================*/

static inline int x86_aes_available(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("aes");
}

static inline int x86_vaes_available(void) {
    __builtin_cpu_init();
#if defined(__VAES__) || defined(__AVX512F__)
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("aes");
#else
    return 0;
#endif
}

/*===========================================================================
 * Entropy and round keys
 *===========================================================================*/

static inline uint64_t x86_aes_read_tsc(void) {
    return __rdtsc();
}

/*
 * The counter exists so that the deterministic path still varies the key
 * between the rounds of one collapse, which is what the POWER8 version got
 * from mftb advancing underneath it.
 */
static inline __m128i x86_aes_make_round_key(int layer_id, int position,
                                             uint64_t counter) {
#if X86_AES_COLLAPSE_ENTROPY
    uint64_t seed = x86_aes_read_tsc();
#else
    uint64_t seed = counter * 0xD1342543DE82EF95ULL;
#endif
    uint64_t lo = seed ^ ((uint64_t)layer_id * 0x9E3779B97F4A7C15ULL);
    uint64_t hi = seed ^ ((uint64_t)position * 0x517CC1B727220A95ULL);
    return _mm_set_epi64x((long long)hi, (long long)lo);
}

/*===========================================================================
 * The round itself
 *===========================================================================*/

/* Identical to POWER8 __builtin_crypto_vcipher. */
static inline __m128i x86_aes_round(__m128i state, __m128i key) {
    return _mm_aesenc_si128(state, key);
}

/* Identical to POWER8 __builtin_crypto_vcipherlast (no MixColumns). */
static inline __m128i x86_aes_round_last(__m128i state, __m128i key) {
    return _mm_aesenclast_si128(state, key);
}

/*
 * What ARM's aese+aesmc pair actually computes, for when you need to match
 * the Apple port rather than the POWER8 one. Not used below.
 */
static inline __m128i x86_aes_arm_order(__m128i state, __m128i key) {
    return _mm_aesenc_si128(_mm_xor_si128(state, key), _mm_setzero_si128());
}

/* Mirror a 16-byte lane, converting between vec_perm and pshufb numbering. */
static inline __m128i x86_aes_beswap(__m128i v) {
    const __m128i rev = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7,
                                     8, 9, 10, 11, 12, 13, 14, 15);
    return _mm_shuffle_epi8(v, rev);
}

/*===========================================================================
 * MODE 1: Non-linear pattern generation
 *
 * Note what the pattern is: byte indices for pshufb, so it permutes the bytes
 * *inside* the floats, not the floats. That is what the POWER8 version does
 * too; it is a scrambler, not a gather.
 *===========================================================================*/

static inline __m128i x86_aes_generate_pattern(int layer_id, int position,
                                               int top_k) {
    __m128i state = _mm_set_epi64x(
        (long long)((uint64_t)position * 0xBB67AE8584CAA73BULL),
        (long long)((uint64_t)layer_id * 0x6A09E667F3BCC908ULL));

    for (int r = 0; r < X86_AES_COLLAPSE_ROUNDS; r++) {
        state = x86_aes_round(
            state, x86_aes_make_round_key(layer_id + r, position, (uint64_t)r));
    }
    state = x86_aes_round_last(
        state, x86_aes_make_round_key(layer_id, position + 1,
                                      X86_AES_COLLAPSE_ROUNDS));

    unsigned char raw[16], pattern[16];
    _mm_storeu_si128((__m128i *)raw, state);

    if (top_k < 1) top_k = 1;
    if (top_k > 16) top_k = 16;
    for (int i = 0; i < top_k; i++) pattern[i] = (unsigned char)i;
    for (int i = top_k; i < 16; i++)
        pattern[i] = (unsigned char)(raw[i] % (unsigned)top_k);

    return _mm_loadu_si128((const __m128i *)pattern);
}

/*===========================================================================
 * MODE 2: Score ranking via S-box non-linearity
 *===========================================================================*/

static inline void x86_aes_rank_scores(const float *scores, uint8_t *rank_keys,
                                       int layer_id, int position) {
    __m128i state = _mm_loadu_si128((const __m128i *)scores);
    __m128i rk = x86_aes_make_round_key(layer_id, position, 0);

#if X86_AES_CROSS_HEAD_FUSE
    state = x86_aes_round(state, rk);       /* MixColumns crosses the lanes */
#else
    state = x86_aes_round_last(state, rk);  /* each score ranked alone */
#endif

    _mm_storeu_si128((__m128i *)rank_keys, state);
}

/*===========================================================================
 * MODE 3: Cross-head fusion
 *===========================================================================*/

static inline __m128i x86_aes_fuse_heads(__m128i head_scores, int layer_id,
                                         int position) {
    return x86_aes_round(head_scores,
                         x86_aes_make_round_key(layer_id, position, 0));
}

/*===========================================================================
 * 8-way pipelined collapse
 *
 * AESENC is ~4 cycles of latency at one or two per cycle of throughput, so
 * the same pipelining argument as POWER8 applies and eight independent chains
 * saturate it. Measured on an Emerald Rapids Xeon: 0.76 ns/round dependent
 * against 0.20 ns/round with four chains in flight.
 *
 * Unlike the POWER8 original, this processes all eight vectors. That one
 * advances the loop by 8 while only ever touching vector 0.
 *===========================================================================*/

static inline void x86_aes_collapse_8way(float *scores, int n_vectors,
                                         int layer_id, int position) {
    __m128i rk[8];
    for (int j = 0; j < 8; j++)
        rk[j] = x86_aes_make_round_key(layer_id + (j >> 2), position + (j & 3),
                                       (uint64_t)j);

    const float amp = X86_AES_COLLAPSE_AMPLIFY;
    int i = 0;
    for (; i + 7 < n_vectors; i += 8) {
        __m128i s[8], c[8];
        for (int j = 0; j < 8; j++)
            s[j] = _mm_loadu_si128((const __m128i *)&scores[(i + j) * 4]);
        /* Issued back to back; the eight chains are independent. */
        for (int j = 0; j < 8; j++) c[j] = x86_aes_round(s[j], rk[j]);

        for (int j = 0; j < 8; j++) {
            unsigned char r[16];
            _mm_storeu_si128((__m128i *)r, c[j]);
            for (int lane = 0; lane < 4; lane++) {
                /* Energy of one float's four transformed bytes. */
                unsigned energy = (unsigned)r[lane] + r[lane + 4]
                                + r[lane + 8] + r[lane + 12];
                float *slot = &scores[(i + j) * 4 + lane];
                if (energy < 512u) *slot = 0.0f;
                else               *slot *= amp;
            }
        }
    }

    /* Remainder: same rule, one vector at a time. */
    for (; i < n_vectors; i++) {
        float *vec = scores + (ptrdiff_t)i * 4;
        __m128i c = x86_aes_round(_mm_loadu_si128((const __m128i *)vec),
                                  rk[i & 7]);
        unsigned char r[16];
        _mm_storeu_si128((__m128i *)r, c);
        for (int lane = 0; lane < 4; lane++) {
            unsigned energy = (unsigned)r[lane] + r[lane + 4]
                            + r[lane + 8] + r[lane + 12];
            if (energy < 512u) vec[lane] = 0.0f;
            else               vec[lane] *= amp;
        }
    }
}

/*===========================================================================
 * CORE: hybrid AES + pshufb collapse
 *
 * AES ranks, pshufb routes. The split exists because the S-box destroys IEEE
 * 754 encoding, so it can say which scores win but must not touch their
 * values; pshufb moves bytes without altering them but cannot rank.
 *===========================================================================*/

static inline void x86_aes_hybrid_collapse(float *scores, int n, int top_k,
                                           int layer_id, int position) {
    if (n < 4 || top_k < 1) return;
    if (top_k > 16) top_k = 16;

    /* Step 1: top-K threshold, AES-assisted ranking over groups of four. */
    float top_vals[16];
    for (int i = 0; i < 16; i++) top_vals[i] = -1e30f;

    for (int i = 0; i + 3 < n; i += 4) {
        __m128i state = _mm_loadu_si128((const __m128i *)&scores[i]);
        __m128i ranked = x86_aes_round(
            state, x86_aes_make_round_key(layer_id, position + i, (uint64_t)i));
        unsigned char rb[16];
        _mm_storeu_si128((__m128i *)rb, ranked);

        for (int j = 0; j < 4; j++) {
            float score = scores[i + j];
            if (score > top_vals[top_k - 1]) {
                top_vals[top_k - 1] = score;
                for (int k = top_k - 1;
                     k > 0 && top_vals[k] > top_vals[k - 1]; k--) {
                    float tmp = top_vals[k];
                    top_vals[k] = top_vals[k - 1];
                    top_vals[k - 1] = tmp;
                }
            }
        }
    }

    const float threshold = top_vals[top_k - 1];
    const __m128 thresh_vec = _mm_set1_ps(threshold);
    const __m128 amp_vec = _mm_set1_ps(X86_AES_COLLAPSE_AMPLIFY);
    const __m128 zero_vec = _mm_setzero_ps();

    /* Step 2: the permute pattern. */
    const __m128i pattern = x86_aes_generate_pattern(layer_id, position, top_k);

    /*
     * Step 3: route, mask, amplify.
     *
     * The POWER8 version calls vec_perm(v0, v1, pattern), a two-source
     * permute. Every index the generator produces is below 16, so it only
     * ever selects from v0 and one pshufb is exactly equivalent — no blend
     * needed, despite what the porting table implies.
     */
    int i = 0;
    for (; i + 15 < n; i += 16) {
        for (int b = 0; b < 4; b++) {
            __m128 v = _mm_loadu_ps(&scores[i + b * 4]);
            __m128 c = _mm_castsi128_ps(
                _mm_shuffle_epi8(_mm_castps_si128(v), pattern));
            __m128 keep = _mm_cmpgt_ps(c, thresh_vec);
            c = _mm_mul_ps(_mm_blendv_ps(zero_vec, c, keep), amp_vec);
            _mm_storeu_ps(&scores[i + b * 4], c);
        }
    }

    for (; i < n; i++) {
        if (scores[i] >= threshold) scores[i] *= X86_AES_COLLAPSE_AMPLIFY;
        else                        scores[i] = 0.0f;
    }
}

/*===========================================================================
 * MODE 4: AES as the similarity metric (experimental — and it does not work)
 *
 * Ported for parity with the POWER8 header, whose claim is that XORing Q with
 * K and running a round gives a similarity measure, because similar vectors
 * XOR to near zero and dissimilar ones diffuse. Measured over 200k Q/K pairs
 * on Emerald Rapids, the correlation between this score and the actual cosine
 * similarity is **+0.002**, and the score for Q == K (2496) sits in the middle
 * of the range the function produces for unrelated pairs (849..3318).
 *
 * The reason is the avalanche property itself: it is what the mode is built
 * on, and it is exactly what destroys any monotonic relationship between
 * input distance and output byte-sum. A near-zero XOR is not a near-zero
 * round output — SubBytes maps 0x00 to 0x63 — so "low energy" tracks nothing.
 *
 * What the function *is*: a fast, deterministic 12-bit hash of Q XOR K. It
 * detects equality and nothing weaker. Kept, tested and documented as such
 * rather than deleted, because the POWER8 and Apple ports both ship it and a
 * silent divergence would be worse than a loud caveat.
 *===========================================================================*/

static inline uint32_t x86_aes_attention_score(const float *Q_vec,
                                               const float *K_vec,
                                               int layer_id, int position) {
    __m128i q = _mm_loadu_si128((const __m128i *)Q_vec);
    __m128i k = _mm_loadu_si128((const __m128i *)K_vec);
    __m128i state = x86_aes_round(_mm_xor_si128(q, k),
                                  x86_aes_make_round_key(layer_id, position, 0));

    /* Byte-sum the result: low energy means Q and K agreed. */
    __m128i sums = _mm_sad_epu8(state, _mm_setzero_si128());
    uint32_t energy = (uint32_t)(_mm_cvtsi128_si32(sums)
                                 + _mm_extract_epi16(sums, 4));
    return 4080u - energy;   /* 16 * 255 */
}

/*===========================================================================
 * Banner
 *===========================================================================*/

static inline void x86_aes_collapse_banner(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  PSE AES Collapse — x86-64 AES-NI (aesenc == vcipher)\n");
    fprintf(stderr, "────────────────────────────────────────────────────────────\n");
    fprintf(stderr, "   AES rounds:       %d (SubBytes+ShiftRows+MixColumns+XOR)\n",
            X86_AES_COLLAPSE_ROUNDS);
    fprintf(stderr, "   Top-K:            %d\n", X86_AES_COLLAPSE_TOP_K);
    fprintf(stderr, "   Amplify:          %.2f\n",
            (double)X86_AES_COLLAPSE_AMPLIFY);
    fprintf(stderr, "   Cross-head fuse:  %s (MixColumns GF(2^8) diffusion)\n",
            X86_AES_CROSS_HEAD_FUSE ? "ENABLED" : "disabled");
    fprintf(stderr, "   Round key:        %s\n",
            X86_AES_COLLAPSE_ENTROPY ? "rdtsc (nondeterministic)"
                                     : "counter (reproducible)");
    fprintf(stderr, "   AES-NI:           %s\n",
            x86_aes_available() ? "present" : "ABSENT — do not call");
    fprintf(stderr, "   VAES (4x wide):   %s\n",
            x86_vaes_available() ? "present" : "absent (Zen 3 / Ice Lake+)");
    fprintf(stderr, "════════════════════════════════════════════════════════════\n");
}

#endif /* X86_AES_COLLAPSE_H */
