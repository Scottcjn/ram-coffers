#include "fp8.h"

#include <math.h>
#include <string.h>

static float g_table[256];
static int g_ready = 0;

/* Decode one E4M3FN byte the slow, obvious way. Called 256 times, ever. */
static float decode_e4m3(uint8_t v)
{
    const int sign = (v >> 7) & 0x1;
    const int exp = (v >> 3) & 0xF;
    const int mant = v & 0x7;
    float mag;

    if (exp == 0) {
        /* Subnormal: no implicit leading 1, exponent fixed at 1 - bias. */
        mag = (float)mant * 0.125f * ldexpf(1.0f, -6);
    } else if (exp == 0xF && mant == 0x7) {
        /* E4M3FN has no infinity; 0x7F/0xFF are the only NaNs. */
        return NAN;
    } else {
        mag = (1.0f + (float)mant * 0.125f) * ldexpf(1.0f, exp - 7);
    }
    return sign ? -mag : mag;
}

void gen9_fp8_init(void)
{
    if (g_ready) {
        return;
    }
    for (int i = 0; i < 256; ++i) {
        g_table[i] = decode_e4m3((uint8_t)i);
    }
    g_ready = 1;
}

float gen9_fp8_to_f32(uint8_t v)
{
    gen9_fp8_init();
    return g_table[v];
}

void gen9_fp8_dequant(const uint8_t *src, const float *scales, float *out,
                      size_t n)
{
    gen9_fp8_init();
    for (size_t base = 0; base < n; base += GEN9_FP8_BLOCK) {
        const float scale = scales[base / GEN9_FP8_BLOCK];
        const size_t end = (base + GEN9_FP8_BLOCK < n) ? base + GEN9_FP8_BLOCK
                                                       : n;
        for (size_t i = base; i < end; ++i) {
            out[i] = g_table[src[i]] * scale;
        }
    }
}

uint8_t gen9_f32_to_fp8(float v)
{
    if (isnan(v)) {
        return 0x7F;
    }
    const int sign = signbit(v) ? 0x80 : 0x00;
    float mag = fabsf(v);

    if (mag > 448.0f) {
        return (uint8_t)(sign | 0x7E);          /* saturate at max normal */
    }
    if (mag < ldexpf(1.0f, -9) * 0.5f) {
        return (uint8_t)sign;                   /* flush to zero */
    }
    if (mag < ldexpf(1.0f, -6)) {
        /* Subnormal: quantise to the 1/8 grid at 2^-6. */
        int mant = (int)lrintf(mag / (0.125f * ldexpf(1.0f, -6)));
        if (mant > 7) {
            mant = 7;
        }
        return (uint8_t)(sign | mant);
    }
    int exp;
    const float frac = frexpf(mag, &exp);       /* mag = frac * 2^exp, .5<=frac<1 */
    int e = exp - 1 + 7;                        /* unbiased -> biased */
    int mant = (int)lrintf((frac * 2.0f - 1.0f) * 8.0f);
    if (mant == 8) {
        mant = 0;
        e += 1;
    }
    if (e >= 0xF && mant >= 0x7) {
        return (uint8_t)(sign | 0x7E);
    }
    if (e > 0xF) {
        return (uint8_t)(sign | 0x7E);
    }
    return (uint8_t)(sign | (e << 3) | mant);
}
