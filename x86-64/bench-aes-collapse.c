/*
 * bench-aes-collapse.c — correctness and cost of the x86 AES collapse.
 *
 * The correctness half matters more than the benchmark. The whole claim of
 * aes-collapse.h is that AESENC is the same function as POWER8's vcipher, so
 * that claim is checked against a scalar AES round written from FIPS-197
 * rather than against another intrinsic. The same scalar reference then shows
 * that ARM's aese+aesmc is *not* that function, which is the bug in the
 * porting table in apple-silicon/PAPER-architecture-general-pse.md.
 *
 *   cc -O2 -maes -msse4.1 -o bench-aes-collapse bench-aes-collapse.c
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "aes-collapse.h"

/*===========================================================================
 * Scalar AES round, FIPS-197, as the reference
 *===========================================================================*/

static const uint8_t SBOX[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
};

static uint8_t xtime(uint8_t x) {
    return (uint8_t)((x << 1) ^ ((x >> 7) * 0x1b));
}

/* One AES round on a 16-byte column-major state: SubBytes, ShiftRows,
 * MixColumns, AddRoundKey — in that order, key last. */
static void aes_round_ref(const uint8_t in[16], const uint8_t key[16],
                          uint8_t out[16]) {
    uint8_t s[16];
    for (int i = 0; i < 16; i++) s[i] = SBOX[in[i]];

    /* ShiftRows: row r rotates left by r. State is column-major, so the byte
     * at (row, col) lives at index col * 4 + row. */
    uint8_t t[16];
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
            t[c * 4 + r] = s[((c + r) % 4) * 4 + r];

    for (int c = 0; c < 4; c++) {
        const uint8_t *a = &t[c * 4];
        uint8_t b[4];
        b[0] = (uint8_t)(xtime(a[0]) ^ (xtime(a[1]) ^ a[1]) ^ a[2] ^ a[3]);
        b[1] = (uint8_t)(a[0] ^ xtime(a[1]) ^ (xtime(a[2]) ^ a[2]) ^ a[3]);
        b[2] = (uint8_t)(a[0] ^ a[1] ^ xtime(a[2]) ^ (xtime(a[3]) ^ a[3]));
        b[3] = (uint8_t)((xtime(a[0]) ^ a[0]) ^ a[1] ^ a[2] ^ xtime(a[3]));
        for (int r = 0; r < 4; r++) out[c * 4 + r] = (uint8_t)(b[r] ^ key[c * 4 + r]);
    }
}

/*===========================================================================
 * Tests
 *===========================================================================*/

static uint64_t rng_state = 0x243F6A8885A308D3ULL;

static uint64_t rng(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static void fill_random(uint8_t *p, int n) {
    for (int i = 0; i < n; i++) p[i] = (uint8_t)(rng() >> 24);
}

static int failures;

static void check(const char *what, int ok) {
    printf("  %-52s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) failures++;
}

/* aesenc must equal the scalar round, i.e. must equal vcipher. */
static int test_aesenc_is_vcipher(void) {
    for (int trial = 0; trial < 4096; trial++) {
        uint8_t s[16], k[16], want[16], got[16];
        fill_random(s, 16);
        fill_random(k, 16);
        aes_round_ref(s, k, want);
        _mm_storeu_si128((__m128i *)got,
                         x86_aes_round(_mm_loadu_si128((const __m128i *)s),
                                       _mm_loadu_si128((const __m128i *)k)));
        if (memcmp(want, got, 16) != 0) return 0;
    }
    return 1;
}

/* aesenclast must equal the same round without MixColumns, i.e. vcipherlast. */
static int test_aesenclast_is_vcipherlast(void) {
    for (int trial = 0; trial < 4096; trial++) {
        uint8_t s[16], k[16], got[16], want[16];
        fill_random(s, 16);
        fill_random(k, 16);

        uint8_t sub[16];
        for (int i = 0; i < 16; i++) sub[i] = SBOX[s[i]];
        for (int c = 0; c < 4; c++)
            for (int r = 0; r < 4; r++)
                want[c * 4 + r] = (uint8_t)(sub[((c + r) % 4) * 4 + r]
                                            ^ k[c * 4 + r]);

        _mm_storeu_si128((__m128i *)got,
                         x86_aes_round_last(_mm_loadu_si128((const __m128i *)s),
                                            _mm_loadu_si128((const __m128i *)k)));
        if (memcmp(want, got, 16) != 0) return 0;
    }
    return 1;
}

/*
 * ARM's aese+aesmc is aesenc(s ^ k, 0), and that is a different function of
 * (s, k). If this ever starts passing, the porting table was right after all.
 */
static int test_arm_order_differs(void) {
    int differed = 0;
    for (int trial = 0; trial < 4096; trial++) {
        uint8_t s[16], k[16];
        fill_random(s, 16);
        fill_random(k, 16);
        __m128i sv = _mm_loadu_si128((const __m128i *)s);
        __m128i kv = _mm_loadu_si128((const __m128i *)k);
        __m128i power8 = x86_aes_round(sv, kv);
        __m128i arm = x86_aes_arm_order(sv, kv);
        if (!_mm_test_all_zeros(_mm_xor_si128(power8, arm),
                                _mm_set1_epi8((char)0xff)))
            differed++;
    }
    /* A zero key is the one case where they agree, and random keys are not
     * zero, so every trial should differ. */
    return differed == 4096;
}

static int test_arm_order_agrees_on_zero_key(void) {
    uint8_t s[16];
    fill_random(s, 16);
    __m128i sv = _mm_loadu_si128((const __m128i *)s);
    __m128i z = _mm_setzero_si128();
    return _mm_test_all_zeros(
        _mm_xor_si128(x86_aes_round(sv, z), x86_aes_arm_order(sv, z)),
        _mm_set1_epi8((char)0xff));
}

/* With entropy off, the same inputs must give the same pattern every time. */
static int test_pattern_is_reproducible(void) {
    __m128i a = x86_aes_generate_pattern(7, 129, 8);
    __m128i b = x86_aes_generate_pattern(7, 129, 8);
    if (!_mm_test_all_zeros(_mm_xor_si128(a, b), _mm_set1_epi8((char)0xff)))
        return 0;
    /* And it must actually depend on where it is in the model. */
    __m128i c = x86_aes_generate_pattern(8, 129, 8);
    return !_mm_test_all_zeros(_mm_xor_si128(a, c), _mm_set1_epi8((char)0xff));
}

/* Every index must be a legal pshufb selector, or the collapse reads garbage. */
static int test_pattern_indices_are_in_range(void) {
    for (int top_k = 1; top_k <= 16; top_k++) {
        unsigned char p[16];
        _mm_storeu_si128((__m128i *)p,
                         x86_aes_generate_pattern(3, top_k * 11, top_k));
        for (int i = 0; i < 16; i++)
            if (p[i] > 15) return 0;
        for (int i = 0; i < top_k; i++)
            if (p[i] != (unsigned char)i) return 0;
    }
    return 1;
}

/*
 * The POWER8 8-way loop advances by eight and only ever writes vector 0, so
 * seven eighths of the input passes through untouched. Ours must touch all of
 * it: after a collapse, no score may still hold its original value unless the
 * collapse decided to amplify it.
 */
static int test_8way_touches_every_vector(void) {
    enum { N = 64 };
    float scores[N * 4], before[N * 4];
    for (int i = 0; i < N * 4; i++) scores[i] = (float)(i + 1) * 0.25f;
    memcpy(before, scores, sizeof(before));

    x86_aes_collapse_8way(scores, N, 2, 40);

    for (int i = 0; i < N * 4; i++) {
        int pruned = scores[i] == 0.0f;
        int amplified = scores[i] > before[i];
        if (!pruned && !amplified) return 0;
    }
    return 1;
}

static int test_hybrid_keeps_top_k(void) {
    enum { N = 64 };
    float scores[N];
    for (int i = 0; i < N; i++) scores[i] = (float)((i * 37) % N);
    x86_aes_hybrid_collapse(scores, N, 8, 1, 0);

    int survivors = 0;
    for (int i = 0; i < N; i++) if (scores[i] != 0.0f) survivors++;
    return survivors > 0 && survivors < N;
}

/* It is a hash, so it must at least be a stable one. */
static int test_attention_score_is_deterministic(void) {
    float q[4] = {1.5f, -2.25f, 0.125f, 9.0f};
    float k[4] = {1.5f, -2.25f, 0.125f, 9.5f};
    uint32_t a = x86_aes_attention_score(q, k, 3, 11);
    uint32_t b = x86_aes_attention_score(q, k, 3, 11);
    uint32_t c = x86_aes_attention_score(q, q, 3, 11);
    return a == b && a != c;
}

/*
 * Pins the defect described above mode 4, so that nobody reads its name and
 * assumes it ranks anything. If this ever fails because the correlation rose,
 * the mode became useful and the comment needs rewriting.
 */
static int test_attention_score_does_not_rank_similarity(void) {
    enum { N = 20000 };
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    const float q[4] = {1.5f, -2.25f, 0.125f, 9.0f};

    for (int i = 0; i < N; i++) {
        float k[4];
        double eps = (double)(rng() % 1000) / 1000.0;
        for (int j = 0; j < 4; j++)
            k[j] = (float)(q[j] * (1.0 - eps)
                           + eps * (double)((int64_t)(rng() % 200) - 100) * 0.1);

        double dot = 0, nq = 0, nk = 0;
        for (int j = 0; j < 4; j++) {
            dot += (double)q[j] * k[j];
            nq += (double)q[j] * q[j];
            nk += (double)k[j] * k[j];
        }
        double cos = dot / (sqrt(nq) * sqrt(nk) + 1e-12);
        double sc = (double)x86_aes_attention_score(q, k, 0, 0);
        sx += cos; sy += sc; sxx += cos * cos; syy += sc * sc; sxy += cos * sc;
    }

    double n = (double)N;
    double corr = (n * sxy - sx * sy)
                / (sqrt(n * sxx - sx * sx) * sqrt(n * syy - sy * sy));
    printf("      (correlation with cosine similarity: %+.4f)\n", corr);
    return corr < 0.05 && corr > -0.05;
}

/*===========================================================================
 * Cost
 *===========================================================================*/

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void benchmark(void) {
    const uint64_t n = 50000000ULL;
    __m128i a = _mm_set1_epi32(1), b = _mm_set1_epi32(2);
    __m128i c = _mm_set1_epi32(3), d = _mm_set1_epi32(4);
    const __m128i k = _mm_set1_epi32((int)0x9E3779B9);

    double t = now();
    for (uint64_t i = 0; i < n; i++) a = x86_aes_round(a, k);
    double dep = now() - t;

    t = now();
    for (uint64_t i = 0; i < n; i++) {
        a = x86_aes_round(a, k);
        b = x86_aes_round(b, k);
        c = x86_aes_round(c, k);
        d = x86_aes_round(d, k);
    }
    double par = now() - t;

    t = now();
    for (uint64_t i = 0; i < n; i++) b = _mm_shuffle_epi8(b, k);
    double shuf = now() - t;

    printf("\ncost per AES round\n");
    printf("  dependent chain (latency bound)     %5.2f ns\n",
           dep / (double)n * 1e9);
    printf("  4 chains in flight (throughput)     %5.2f ns  (%.1f Grounds/s)\n",
           par / (double)(n * 4) * 1e9, (double)(n * 4) / par / 1e9);
    printf("  pshufb, dependent, for scale        %5.2f ns\n",
           shuf / (double)n * 1e9);

    /* Keep the chains alive so none of that is optimised away. */
    volatile int sink = _mm_cvtsi128_si32(
        _mm_xor_si128(_mm_xor_si128(a, b), _mm_xor_si128(c, d)));
    (void)sink;
}

int main(void) {
    x86_aes_collapse_banner();

    if (!x86_aes_available()) {
        fprintf(stderr, "\nno AES-NI on this CPU; nothing to test\n");
        return 77;
    }

    printf("\nequivalence with POWER8\n");
    check("aesenc(s,k) == vcipher(s,k)", test_aesenc_is_vcipher());
    check("aesenclast(s,k) == vcipherlast(s,k)",
          test_aesenclast_is_vcipherlast());

    printf("\ndivergence of the ARM port\n");
    check("aese+aesmc differs from vcipher for a nonzero key",
          test_arm_order_differs());
    check("...and agrees only when the key is zero",
          test_arm_order_agrees_on_zero_key());

    printf("\ncollapse behaviour\n");
    check("patterns are reproducible and position dependent",
          test_pattern_is_reproducible());
    check("pattern indices are legal pshufb selectors",
          test_pattern_indices_are_in_range());
    check("8-way collapse touches all eight vectors",
          test_8way_touches_every_vector());
    check("hybrid collapse prunes some and keeps some",
          test_hybrid_keeps_top_k());
    check("mode 4 score is a deterministic hash of Q^K",
          test_attention_score_is_deterministic());
    check("mode 4 does NOT rank similarity (known defect)",
          test_attention_score_does_not_rank_similarity());

    benchmark();

    printf("\n%s\n", failures ? "FAILURES" : "all checks passed");
    return failures ? 1 : 0;
}
