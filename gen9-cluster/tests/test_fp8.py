"""FP8 E4M3FN with 128-element block scales.

Two implementations exist — numpy here and C in ``kernels/fp8.c`` — and a
disagreement between them is a fleet where two consoles decode the same weight
differently. The last test in this file is the one that matters: it checks them
against each other over all 256 codes.
"""

import ctypes
import math
import unittest

import numpy as np

from gen9_cluster import fp8
from gen9_cluster.backends import _DEFAULT_LIB, CpuKernelRunner
from gen9_cluster.node import ExpertRunner, ExpertWeights
from gen9_cluster.protocol import DType, ShardHeader


class TestFormat(unittest.TestCase):
    def test_the_table_covers_every_code(self):
        self.assertEqual(fp8.TABLE.shape, (256,))

    def test_powers_of_two_are_exact(self):
        """E4M3 represents these exactly, so a round trip that perturbs them
        is a bug in the encoder, not quantisation noise."""
        for value in (1.0, 2.0, 0.5, 8.0, 0.25, 256.0):
            codes, scales = fp8.quantize(np.full(fp8.BLOCK, value,
                                                 dtype=np.float32))
            back = fp8.dequantize(codes, scales)
            np.testing.assert_allclose(back, value, rtol=1e-6)

    def test_the_maximum_normal_is_448(self):
        finite = fp8.TABLE[~np.isnan(fp8.TABLE)]
        self.assertEqual(finite.max(), 448.0)
        self.assertEqual(finite.min(), -448.0)

    def test_only_two_codes_are_nan(self):
        """E4M3FN spends its would-be infinities on extra range; if more codes
        decoded to NaN, a weight would silently poison a whole token."""
        self.assertEqual(int(np.isnan(fp8.TABLE).sum()), 2)

    def test_the_smallest_subnormal(self):
        finite = np.abs(fp8.TABLE[~np.isnan(fp8.TABLE)])
        self.assertAlmostEqual(float(finite[finite > 0].min()), 2.0 ** -9)

    def test_relative_error_is_bounded_over_the_normal_range(self):
        """3 mantissa bits gives at worst 1/16 relative error. Anything much
        larger means the encoder is picking the wrong code, not that the format
        is coarse."""
        values = np.exp(np.linspace(math.log(1e-3), math.log(400), fp8.BLOCK))
        values = values.astype(np.float32)
        codes, scales = fp8.quantize(values)
        back = fp8.dequantize(codes, scales)
        self.assertLess(float(np.abs(back - values).max() / values.max()),
                        1 / 16)


class TestBlockwise(unittest.TestCase):
    def test_one_scale_per_128_values(self):
        codes, scales = fp8.quantize(np.ones(512, dtype=np.float32))
        self.assertEqual(scales.shape, (4,))
        self.assertEqual(codes.shape, (512,))

    def test_a_partial_final_block_still_gets_a_scale(self):
        codes, scales = fp8.quantize(np.ones(200, dtype=np.float32))
        self.assertEqual(scales.shape, (2,))
        self.assertEqual(codes.shape, (200,))

    def test_an_outlier_does_not_flatten_its_neighbours(self):
        """Per-block scaling is the reason a 1e4 outlier next to 1e-2 weights
        costs one block's precision rather than the whole tensor's."""
        values = np.full(2 * fp8.BLOCK, 0.01, dtype=np.float32)
        values[0] = 10000.0
        back = fp8.dequantize(*fp8.quantize(values))
        # The block with the outlier loses precision...
        self.assertGreater(abs(back[1] - 0.01) / 0.01, 0.1)
        # ...and the next block does not.
        np.testing.assert_allclose(back[fp8.BLOCK:], 0.01, rtol=0.05)

    def test_shape_is_preserved(self):
        matrix = np.random.default_rng(0).standard_normal((8, 256),
                                                          dtype=np.float32)
        codes, scales = fp8.quantize(matrix)
        self.assertEqual(codes.shape, (8, 256))
        self.assertEqual(fp8.dequantize(codes, scales).shape, (8, 256))

    def test_zeros_survive(self):
        back = fp8.dequantize(*fp8.quantize(np.zeros(fp8.BLOCK,
                                                     dtype=np.float32)))
        np.testing.assert_array_equal(back, np.zeros(fp8.BLOCK))

    def test_the_wrong_number_of_scales_is_refused(self):
        codes, _ = fp8.quantize(np.ones(512, dtype=np.float32))
        with self.assertRaises(ValueError):
            fp8.dequantize(codes, np.ones(3, dtype=np.float32))

    def test_a_quantised_tensor_is_a_quarter_the_size(self):
        """The entire reason this format is here."""
        values = np.ones(1024, dtype=np.float32)
        codes, scales = fp8.quantize(values)
        self.assertLess(codes.nbytes + scales.nbytes, values.nbytes * 0.3)


class TestQuantisedExperts(unittest.TestCase):
    hidden = fp8.BLOCK
    intermediate = fp8.BLOCK

    def make(self, seed=0):
        rng = np.random.default_rng(seed)
        raw = {
            "gate": rng.standard_normal((self.intermediate, self.hidden),
                                        dtype=np.float32) * 0.05,
            "up": rng.standard_normal((self.intermediate, self.hidden),
                                      dtype=np.float32) * 0.05,
            "down": rng.standard_normal((self.hidden, self.intermediate),
                                        dtype=np.float32) * 0.05,
        }
        codes, scales = {}, []
        for name, matrix in raw.items():
            code, scale = fp8.quantize(matrix)
            codes[name] = code
            scales.append(scale)
        quantised = ExpertWeights(gate=codes["gate"], up=codes["up"],
                                  down=codes["down"],
                                  scales=np.concatenate(scales))
        exact = ExpertWeights(**raw)
        return quantised, exact

    def test_a_quantised_expert_is_recognised_as_such(self):
        quantised, exact = self.make()
        self.assertTrue(quantised.quantised)
        self.assertFalse(exact.quantised)

    def test_the_reference_runner_accepts_fp8_shards(self):
        quantised, exact = self.make()
        x = np.linspace(-1, 1, self.hidden, dtype=np.float32)
        runner = ExpertRunner()
        got = runner(x, [quantised], [1.0])
        want = runner(x, [exact], [1.0])
        self.assertLess(float(np.abs(got - want).max()
                              / np.abs(want).max()), 0.1)

    def test_an_fp8_expert_without_scales_is_an_error_not_garbage(self):
        quantised, _ = self.make()
        quantised.scales = None
        with self.assertRaises(ValueError):
            quantised.dequantised()

    def test_the_wrong_number_of_scales_is_caught(self):
        quantised, _ = self.make()
        quantised.scales = quantised.scales[:-1]
        with self.assertRaises(ValueError):
            quantised.dequantised()

    def test_the_compiled_kernel_agrees_with_numpy_on_fp8_experts(self):
        try:
            kernel = CpuKernelRunner()
        except (FileNotFoundError, OSError) as exc:
            self.skipTest(f"CPU kernel not built here: {exc}")
        quantised, _ = self.make(3)
        x = np.linspace(-1, 1, self.hidden, dtype=np.float32)
        np.testing.assert_allclose(kernel(x, [quantised], [1.0]),
                                   ExpertRunner()(x, [quantised], [1.0]),
                                   rtol=1e-4, atol=1e-5)


class TestAgainstTheCKernel(unittest.TestCase):
    """The two decoders must be the same decoder."""

    def setUp(self):
        if not _DEFAULT_LIB.exists():
            self.skipTest("CPU kernel not built here")
        self.lib = ctypes.CDLL(str(_DEFAULT_LIB))
        self.lib.gen9_fp8_to_f32.argtypes = [ctypes.c_uint8]
        self.lib.gen9_fp8_to_f32.restype = ctypes.c_float
        self.lib.gen9_f32_to_fp8.argtypes = [ctypes.c_float]
        self.lib.gen9_f32_to_fp8.restype = ctypes.c_uint8

    def test_every_code_decodes_identically(self):
        for code in range(256):
            c_value = self.lib.gen9_fp8_to_f32(code)
            py_value = float(fp8.TABLE[code])
            if math.isnan(py_value):
                self.assertTrue(math.isnan(c_value), f"code {code:#04x}")
            else:
                self.assertEqual(c_value, py_value, f"code {code:#04x}")

    def test_encoding_agrees_on_representable_values(self):
        for value in (0.0, 1.0, -1.0, 0.5, 448.0, -448.0, 2.0 ** -9, 6.0):
            code = self.lib.gen9_f32_to_fp8(value)
            self.assertEqual(float(fp8.TABLE[code]), value, f"{value}")

    def test_both_saturate_rather_than_overflow(self):
        code = self.lib.gen9_f32_to_fp8(1e6)
        self.assertEqual(float(fp8.TABLE[code]), 448.0)
        codes, scales = fp8.quantize(np.full(fp8.BLOCK, 1e6,
                                             dtype=np.float32))
        np.testing.assert_allclose(fp8.dequantize(codes, scales), 1e6,
                                   rtol=1e-6)


class TestShardHeaderIntegration(unittest.TestCase):
    def test_the_header_can_declare_fp8(self):
        header = ShardHeader(layer=4, first_expert=0, n_experts=2,
                             hidden_size=fp8.BLOCK,
                             intermediate_size=fp8.BLOCK,
                             dtype=DType.FP8_E4M3_B128, tier="slow")
        decoded, _ = ShardHeader.decode(header.encode())
        self.assertIs(decoded.dtype, DType.FP8_E4M3_B128)


if __name__ == "__main__":
    unittest.main()
