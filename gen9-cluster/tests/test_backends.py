"""The compiled kernel, where it exists, must agree with the reference.

If ``libgen9_cpu.so`` has not been built on this machine the kernel tests are
skipped rather than faked: a green run that silently tested numpy against numpy
would be worse than no run at all.
"""

import unittest

import numpy as np

from gen9_cluster.backends import (CpuKernelRunner, describe_backends,
                                   select_runner)
from gen9_cluster.node import ExpertRunner, ExpertWeights

HIDDEN = 64
INTERMEDIATE = 32


def make_expert(seed):
    rng = np.random.default_rng(seed)
    return ExpertWeights(
        gate=rng.standard_normal((INTERMEDIATE, HIDDEN), dtype=np.float32)
        * 0.1,
        up=rng.standard_normal((INTERMEDIATE, HIDDEN), dtype=np.float32) * 0.1,
        down=rng.standard_normal((HIDDEN, INTERMEDIATE), dtype=np.float32)
        * 0.1)


def cpu_kernel_or_skip():
    try:
        return CpuKernelRunner()
    except (FileNotFoundError, OSError) as exc:
        raise unittest.SkipTest(f"CPU kernel not built here: {exc}")


class TestCpuKernel(unittest.TestCase):
    def setUp(self):
        self.kernel = cpu_kernel_or_skip()
        self.reference = ExpertRunner()

    def test_it_agrees_with_the_reference_runner(self):
        """The AVX2 kernel and the numpy path must produce the same token; the
        fleet mixes consoles running either one."""
        experts = [make_expert(i) for i in range(4)]
        gates = [0.4, 0.3, 0.2, 0.1]
        x = np.linspace(-1.5, 1.5, HIDDEN, dtype=np.float32)
        np.testing.assert_allclose(self.kernel(x, experts, gates),
                                   self.reference(x, experts, gates),
                                   rtol=1e-4, atol=1e-5)

    def test_a_single_expert_matches_the_silu_gate_definition(self):
        expert = make_expert(11)
        x = np.linspace(-1, 1, HIDDEN, dtype=np.float32)
        hidden = x @ expert.gate.T
        hidden = hidden * (1.0 / (1.0 + np.exp(-hidden)))
        expected = (hidden * (x @ expert.up.T)) @ expert.down.T
        np.testing.assert_allclose(self.kernel(x, [expert], [1.0]), expected,
                                   rtol=1e-4, atol=1e-5)

    def test_an_empty_batch_is_a_zero_vector(self):
        out = self.kernel(np.ones(HIDDEN, dtype=np.float32), [], [])
        np.testing.assert_array_equal(out, np.zeros(HIDDEN, dtype=np.float32))

    def test_non_contiguous_input_is_handled(self):
        """Activations arriving as a slice of a larger buffer must not be
        handed to the kernel as a stale or strided pointer."""
        expert = make_expert(3)
        buffer = np.zeros(HIDDEN * 2, dtype=np.float32)
        buffer[::2] = np.linspace(-1, 1, HIDDEN, dtype=np.float32)
        strided = buffer[::2]
        self.assertFalse(strided.flags["C_CONTIGUOUS"])
        np.testing.assert_allclose(self.kernel(strided, [expert], [1.0]),
                                   self.reference(strided, [expert], [1.0]),
                                   rtol=1e-4, atol=1e-5)

    def test_repeated_calls_do_not_accumulate(self):
        """The scratch buffer is reused between calls; if it is not cleared the
        second token of a conversation is wrong."""
        expert = make_expert(5)
        x = np.ones(HIDDEN, dtype=np.float32)
        first = self.kernel(x, [expert], [1.0]).copy()
        for _ in range(4):
            np.testing.assert_allclose(self.kernel(x, [expert], [1.0]), first,
                                       rtol=0, atol=0)


class TestSelection(unittest.TestCase):
    def test_a_runner_is_always_available(self):
        """A console with no kernel built must still serve, slowly, rather than
        refuse to start."""
        self.assertTrue(callable(select_runner()))

    def test_an_unavailable_gpu_backend_falls_back_visibly(self):
        runner = select_runner("vulkan")
        self.assertIn(runner.name, {"cpu-avx2", "numpy"})

    def test_probe_reports_each_backend(self):
        text = "\n".join(describe_backends())
        for expected in ("cpu kernel", "vulkan shader", "hip kernel",
                         "rocminfo"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
