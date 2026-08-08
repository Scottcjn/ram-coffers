"""Model sizing: the arithmetic that decides how many consoles are needed."""

import unittest

from gen9_cluster.model import (DEEPSEEK_TINY, DEEPSEEK_V3, DEEPSEEK_V4_PRO,
                                PROFILES, profile_for)

GB = 1024 ** 3


class TestV3AgainstPublishedFigures(unittest.TestCase):
    """V3 is the calibration case: its parameter counts are published, so if the
    sizing arithmetic is wrong it is wrong here, visibly."""

    def test_total_parameters_match_the_published_671b(self):
        total = DEEPSEEK_V3.total_params()
        self.assertAlmostEqual(total / 1e9, 671.0, delta=15.0)

    def test_activated_parameters_match_the_published_37b(self):
        self.assertAlmostEqual(DEEPSEEK_V3.activated_params() / 1e9, 37.0,
                               delta=2.0)

    def test_fp8_weights_are_about_a_byte_per_parameter(self):
        """The block scales add ~3%; anything far off that means the scale
        accounting is broken."""
        ratio = DEEPSEEK_V3.total_bytes() / DEEPSEEK_V3.total_params()
        self.assertGreater(ratio, 1.0)
        self.assertLess(ratio, 1.10)

    def test_routed_experts_dominate_the_checkpoint(self):
        """~95% of the weights are experts that a given token does not touch.
        This ratio is the entire reason a console fleet is viable."""
        cold = DEEPSEEK_V3.cold_bytes_per_moe_layer()
        hot = DEEPSEEK_V3.hot_bytes_per_moe_layer()
        self.assertGreater(cold / (cold + hot), 0.9)


class TestMLAKVCache(unittest.TestCase):
    def test_mla_cache_is_far_smaller_than_a_plain_kv_cache(self):
        """MLA caches one 512-wide latent per layer instead of 128 heads of
        keys and values; on a console that is the difference between a usable
        context and none."""
        mla = DEEPSEEK_V3.kv_cache_bytes(8192)
        mha = (2 * DEEPSEEK_V3.n_layers * 8192 * DEEPSEEK_V3.mla.n_heads
               * (DEEPSEEK_V3.mla.qk_nope_head_dim
                  + DEEPSEEK_V3.mla.qk_rope_head_dim) * 2)
        self.assertLess(mla * 20, mha)

    def test_the_cache_grows_linearly_with_context(self):
        self.assertAlmostEqual(DEEPSEEK_V3.kv_cache_bytes(16384)
                               / DEEPSEEK_V3.kv_cache_bytes(8192), 2.0,
                               places=3)

    def test_a_console_sized_context_fits_in_a_coffer(self):
        self.assertLess(DEEPSEEK_V3.kv_cache_bytes(8192), 2 * GB)


class TestV4ProAssumption(unittest.TestCase):
    """V4 Pro has no public configuration. The profile must say so everywhere
    it can, because every number downstream inherits the uncertainty."""

    def test_the_profile_is_flagged_as_assumed(self):
        self.assertTrue(DEEPSEEK_V4_PRO.assumed)
        self.assertFalse(DEEPSEEK_V3.assumed)

    def test_it_carries_its_reasoning(self):
        self.assertTrue(DEEPSEEK_V4_PRO.assumptions)
        joined = " ".join(DEEPSEEK_V4_PRO.assumptions).lower()
        self.assertIn("no deepseek v4 pro configuration is public", joined)

    def test_it_is_larger_than_v3_but_the_same_family(self):
        self.assertGreater(DEEPSEEK_V4_PRO.total_bytes(),
                           DEEPSEEK_V3.total_bytes())
        self.assertEqual(DEEPSEEK_V4_PRO.moe.top_k, 8)
        self.assertEqual(DEEPSEEK_V4_PRO.dtype, "fp8")

    def test_an_expert_stays_small_enough_to_place(self):
        """Placement granularity is one expert. If an expert did not fit in a
        console's fast coffer with room for anything else, the whole
        shelf-and-shard scheme would collapse to layer-level pipelining."""
        self.assertLess(DEEPSEEK_V4_PRO.expert_bytes(), 64 * 1024 * 1024)


class TestPlanningLayers(unittest.TestCase):
    def test_mtp_heads_are_schedulable_blocks(self):
        """An MTP head is a decoder block. Treating it as one lump of I/O sent
        it to NVMe as a unit; counting it as a block lets it be placed."""
        self.assertEqual(DEEPSEEK_V4_PRO.planning_layers,
                         DEEPSEEK_V4_PRO.n_layers + DEEPSEEK_V4_PRO.n_mtp_heads)

    def test_an_mtp_block_costs_about_what_a_layer_costs(self):
        layer = (DEEPSEEK_V4_PRO.hot_bytes_per_moe_layer()
                 + DEEPSEEK_V4_PRO.cold_bytes_per_moe_layer())
        mtp = DEEPSEEK_V4_PRO.mtp_bytes() / DEEPSEEK_V4_PRO.n_mtp_heads
        self.assertGreater(mtp, layer * 0.5)


class TestProfileRegistry(unittest.TestCase):
    def test_named_lookup(self):
        self.assertIs(profile_for("deepseek-v3"), DEEPSEEK_V3)
        self.assertIs(profile_for("deepseek-v4-pro"), DEEPSEEK_V4_PRO)

    def test_unknown_names_say_what_is_available(self):
        with self.assertRaises(KeyError) as caught:
            profile_for("gpt-5")
        self.assertIn("deepseek-v3", str(caught.exception))

    def test_the_tiny_profile_is_small_enough_for_tests(self):
        self.assertLess(DEEPSEEK_TINY.total_bytes(), GB)
        self.assertIn(DEEPSEEK_TINY.name, PROFILES)


if __name__ == "__main__":
    unittest.main()
