/* Self-test for the CPU kernels: correctness first, then a throughput figure.
 *
 * The throughput number printed here is what `g9-probe` records as a node's
 * measured GEMV rate, and it is the number the planner should be given for any
 * node whose backend it cannot predict — every ROCm node, and any console with
 * an unusual downbin.
 */

#include "fp8.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void gen9_gemv_f32(const float *w, const float *x, float *out, int rows,
                   int cols);
void gen9_gemv_fp8(const uint8_t *w, const float *scales, const float *x,
                   float *out, int rows, int cols);
void gen9_expert_f32(const float *gate_w, const float *up_w,
                     const float *down_w, const float *x, float *out,
                     float *scratch, int hidden, int intermediate);

static int failures = 0;

static void check(const char *what, int ok)
{
    printf("%-46s %s\n", what, ok ? "ok" : "FAILED");
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

static void test_fp8_roundtrip(void)
{
    /* Exactly representable E4M3 values must survive a round trip. */
    const float exact[] = {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, 1.25f, 448.0f,
                           -448.0f, 0.015625f};
    int ok = 1;
    for (size_t i = 0; i < sizeof(exact) / sizeof(exact[0]); ++i) {
        const float back = gen9_fp8_to_f32(gen9_f32_to_fp8(exact[i]));
        if (fabsf(back - exact[i]) > 1e-6f * fmaxf(1.0f, fabsf(exact[i]))) {
            printf("  %g -> %g\n", (double)exact[i], (double)back);
            ok = 0;
        }
    }
    check("fp8 e4m3 round-trips exact values", ok);

    /* Saturation, not overflow to NaN or infinity. */
    check("fp8 saturates above the max normal",
          fabsf(gen9_fp8_to_f32(gen9_f32_to_fp8(1e6f)) - 448.0f) < 1e-3f);

    /* Relative error inside the normal range stays within the format's 3-bit
     * mantissa: 2^-4 = 6.25% worst case, half that typical. */
    float worst = 0.0f;
    for (float v = 0.02f; v < 400.0f; v *= 1.037f) {
        const float back = gen9_fp8_to_f32(gen9_f32_to_fp8(v));
        const float rel = fabsf(back - v) / v;
        if (rel > worst) {
            worst = rel;
        }
    }
    printf("  worst relative error over the normal range: %.4f\n",
           (double)worst);
    check("fp8 relative error within the format's bound", worst <= 0.0626f);
}

static void test_gemv(void)
{
    const int rows = 64;
    const int cols = 256;
    float *w = malloc(sizeof(float) * rows * cols);
    float *x = malloc(sizeof(float) * cols);
    float *y = malloc(sizeof(float) * rows);
    float *ref = malloc(sizeof(float) * rows);

    for (int i = 0; i < rows * cols; ++i) {
        w[i] = (float)((i * 37 % 101) - 50) / 50.0f;
    }
    for (int i = 0; i < cols; ++i) {
        x[i] = (float)((i * 17 % 61) - 30) / 30.0f;
    }
    for (int m = 0; m < rows; ++m) {
        double sum = 0.0;
        for (int k = 0; k < cols; ++k) {
            sum += (double)w[m * cols + k] * (double)x[k];
        }
        ref[m] = (float)sum;
    }

    gen9_gemv_f32(w, x, y, rows, cols);
    int ok = 1;
    for (int m = 0; m < rows; ++m) {
        if (fabsf(y[m] - ref[m]) > 1e-3f * fmaxf(1.0f, fabsf(ref[m]))) {
            ok = 0;
        }
    }
    check("gemv fp32 matches a scalar reference", ok);

    /* FP8: the same GEMV within the quantisation error the format allows. */
    uint8_t *wq = malloc((size_t)rows * cols);
    float *scales = malloc(sizeof(float) * rows * (cols / GEN9_FP8_BLOCK));
    for (int m = 0; m < rows; ++m) {
        for (int b = 0; b < cols / GEN9_FP8_BLOCK; ++b) {
            float peak = 0.0f;
            for (int i = 0; i < GEN9_FP8_BLOCK; ++i) {
                const float v = fabsf(w[m * cols + b * GEN9_FP8_BLOCK + i]);
                if (v > peak) {
                    peak = v;
                }
            }
            const float scale = peak > 0.0f ? peak / 448.0f : 1.0f;
            scales[m * (cols / GEN9_FP8_BLOCK) + b] = scale;
            for (int i = 0; i < GEN9_FP8_BLOCK; ++i) {
                const int index = m * cols + b * GEN9_FP8_BLOCK + i;
                wq[index] = gen9_f32_to_fp8(w[index] / scale);
            }
        }
    }
    gen9_gemv_fp8(wq, scales, x, y, rows, cols);

    /* Two separate questions, which a single comparison against the fp32
     * reference would conflate.
     *
     * First: does the kernel compute the right thing *given* the quantised
     * weights? That is exact arithmetic and is checked tightly, against a
     * scalar dot product over the dequantised values. */
    float *dequant = malloc(sizeof(float) * (size_t)rows * cols);
    for (int m = 0; m < rows; ++m) {
        gen9_fp8_dequant(wq + (size_t)m * cols,
                         scales + (size_t)m * (cols / GEN9_FP8_BLOCK),
                         dequant + (size_t)m * cols, (size_t)cols);
    }
    int exact_ok = 1;
    for (int m = 0; m < rows; ++m) {
        double sum = 0.0;
        for (int k = 0; k < cols; ++k) {
            sum += (double)dequant[m * cols + k] * (double)x[k];
        }
        if (fabs((double)y[m] - sum) > 1e-3 * fmax(1.0, fabs(sum))) {
            exact_ok = 0;
        }
    }
    check("gemv fp8 matches its own dequantised weights", exact_ok);

    /* Second: how much did quantisation itself cost? Measured against the
     * magnitude of the terms being summed, not against the dot product, which
     * is near-zero under cancellation and would make a 3% weight error look
     * like a 30% one. */
    double worst = 0.0;
    for (int m = 0; m < rows; ++m) {
        double magnitude = 0.0;
        for (int k = 0; k < cols; ++k) {
            magnitude += fabs((double)w[m * cols + k]) * fabs((double)x[k]);
        }
        const double err = fabs((double)y[m] - (double)ref[m])
                           / fmax(1e-9, magnitude);
        if (err > worst) {
            worst = err;
        }
    }
    printf("  fp8 quantisation error, relative to term magnitude: %.4f\n",
           worst);
    check("fp8 blockwise quantisation stays under 2% of magnitude",
          worst < 0.02);

    free(dequant);
    free(w);
    free(x);
    free(y);
    free(ref);
    free(wq);
    free(scales);
}

static void test_expert_and_throughput(void)
{
    /* DeepSeek's expert shape, at V3/V4-Pro width. */
    const int hidden = 7168;
    const int intermediate = 2048;
    float *gate = malloc(sizeof(float) * (size_t)intermediate * hidden);
    float *up = malloc(sizeof(float) * (size_t)intermediate * hidden);
    float *down = malloc(sizeof(float) * (size_t)hidden * intermediate);
    float *x = malloc(sizeof(float) * hidden);
    float *out = malloc(sizeof(float) * hidden);
    float *scratch = malloc(sizeof(float) * 2 * intermediate);

    if (!gate || !up || !down || !x || !out || !scratch) {
        printf("not enough memory for the expert benchmark; skipping\n");
        free(gate); free(up); free(down); free(x); free(out); free(scratch);
        return;
    }
    for (size_t i = 0; i < (size_t)intermediate * hidden; ++i) {
        gate[i] = 0.001f * (float)(i % 7);
        up[i] = 0.001f * (float)(i % 5);
    }
    for (size_t i = 0; i < (size_t)hidden * intermediate; ++i) {
        down[i] = 0.001f * (float)(i % 3);
    }
    for (int i = 0; i < hidden; ++i) {
        x[i] = 0.01f * (float)(i % 11);
    }

    gen9_expert_f32(gate, up, down, x, out, scratch, hidden, intermediate);
    int finite = 1;
    for (int i = 0; i < hidden; ++i) {
        if (!isfinite(out[i])) {
            finite = 0;
        }
    }
    check("expert fp32 produces finite output", finite);

    const int iterations = 3;
    const double started = now_seconds();
    for (int i = 0; i < iterations; ++i) {
        gen9_expert_f32(gate, up, down, x, out, scratch, hidden, intermediate);
    }
    const double elapsed = now_seconds() - started;
    /* Three GEMVs, two multiply-accumulates each. */
    const double flops = 2.0 * 3.0 * (double)hidden * intermediate * iterations;
    const double bytes = 3.0 * (double)hidden * intermediate * 4.0 * iterations;
    printf("\nmeasured on this host (fp32 weights):\n");
    printf("  %.1f GFLOP/s, %.1f GB/s effective, %.2f ms per expert\n",
           flops / elapsed / 1e9, bytes / elapsed / 1e9,
           elapsed / iterations * 1e3);
    printf("  (this is the figure g9-probe records as measured_gemv_gflops)\n");

    free(gate);
    free(up);
    free(down);
    free(x);
    free(out);
    free(scratch);
}

int main(void)
{
    gen9_fp8_init();
    test_fp8_roundtrip();
    test_gemv();
    test_expert_and_throughput();
    if (failures) {
        printf("\n%d check(s) FAILED\n", failures);
        return 1;
    }
    printf("\nall kernel checks passed\n");
    return 0;
}
