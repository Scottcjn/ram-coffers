/* FP8 E4M3 with 128-element block scales — the format DeepSeek ships.
 *
 * A block of 128 weights shares one fp32 scale, so a weight is one byte and the
 * overhead is 4 bytes per 128, about 3%. Dequantisation is therefore a byte
 * load, a table lookup or bit shuffle, and a multiply — cheap enough to do
 * inside the GEMV loop, which is the whole point: the weights stay FP8 in
 * memory, and memory is what a console is short of.
 *
 * E4M3 layout (OCP / NVIDIA "E4M3FN" variant, which is what DeepSeek's
 * checkpoints use):
 *
 *     s eeee mmm     bias 7, no infinities, 0xFF/0x7F are NaN
 *     max normal     448.0     (0x7E / 0xFE)
 *     min normal     2^-6      (0x08)
 *     min subnormal  2^-9      (0x01)
 *
 * The conversion here is a 256-entry table built once at startup. A branchless
 * bit-twiddle version is possible and marginally faster on paper, but on Zen 2
 * the table lives in L1 and the loop is bound by the weight stream from GDDR6,
 * not by the conversion.
 */

#ifndef GEN9_FP8_H
#define GEN9_FP8_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Elements sharing one scale. Fixed by the checkpoint format, not a tunable. */
#define GEN9_FP8_BLOCK 128

/* Build the 256-entry E4M3 -> fp32 table. Idempotent and thread-safe to call
 * more than once before any conversion begins. */
void gen9_fp8_init(void);

/* One value, for tests and for code that is not in a loop. */
float gen9_fp8_to_f32(uint8_t v);

/* Dequantise `n` FP8 values into `out`, applying one scale per 128 elements.
 * `scales` must hold ceil(n / GEN9_FP8_BLOCK) entries. */
void gen9_fp8_dequant(const uint8_t *src, const float *scales, float *out,
                      size_t n);

/* Round-trip helper used by the tests to build fixtures; saturates at 448. */
uint8_t gen9_f32_to_fp8(float v);

#ifdef __cplusplus
}
#endif

#endif /* GEN9_FP8_H */
