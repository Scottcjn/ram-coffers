"""Model sizing: the arithmetic that decides how many consoles are needed."""

import unittest

from gen9_cluster.model import (DEEPSEEK_TINY, DEEPSEEK_V3, DEEPSEEK_V4_FLASH,
                                DEEPSEEK_V4_PRO, MXFP4, PROFILES, profile_for)

GB = 1024 ** 3
MB = 1024 ** 2


class TestV3AgainstPublishedFigures(unittest.TestCase):
    """V3 is the calibration case: its parameter counts are published, so if the
    sizing arithmetic is wrong it is wrong here, visibly."""

    def test_total_parameters_match_the_published_671b(self):
        """The card's figure excludes the MTP head, which the fleet still has
        to store; both numbers have to come out right."""
        self.assertAlmostEqual(DEEPSEEK_V3.total_params(include_mtp=False) / 1e9,
                               671.0, delta=8.0)
        self.assertGreater(DEEPSEEK_V3.total_params(), 671e9)

    def test_activated_parameters_match_the_published_37b(self):
        self.assertAlmostEqual(DEEPSEEK_V3.activated_params() / 1e9, 37.0,
                               delta=2.0)

    def test_fp8_weights_are_about_a_byte_per_parameter(self):
        """An fp32 scale per 128x128 tile is 0.02%, not the 3% you get if you
        mistake the tile for a 128-element vector."""
        ratio = DEEPSEEK_V3.total_bytes() / DEEPSEEK_V3.total_params()
        self.assertGreater(ratio, 1.0)
        self.assertLess(ratio, 1.02)

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
        mha = (2 * DEEPSEEK_V3.n_layers * 8192 * DEEPSEEK_V3.attention.n_heads
               * (DEEPSEEK_V3.attention.qk_nope_head_dim
                  + DEEPSEEK_V3.attention.qk_rope_head_dim) * 2)
        self.assertLess(mla * 20, mha)

    def test_the_cache_grows_linearly_with_context(self):
        self.assertAlmostEqual(DEEPSEEK_V3.kv_cache_bytes(16384)
                               / DEEPSEEK_V3.kv_cache_bytes(8192), 2.0,
                               places=3)

    def test_a_console_sized_context_fits_in_a_coffer(self):
        self.assertLess(DEEPSEEK_V3.kv_cache_bytes(8192), 2 * GB)


class TestV4AgainstPublishedFigures(unittest.TestCase):
    """Both V4 configurations are public, so the profiles are checked against
    the cards rather than flagged as guesses."""

    def test_pro_matches_the_published_1_6t_and_49b(self):
        self.assertAlmostEqual(DEEPSEEK_V4_PRO.total_params() / 1e12, 1.6,
                               delta=0.05)
        self.assertAlmostEqual(DEEPSEEK_V4_PRO.activated_params() / 1e9, 49.0,
                               delta=2.0)

    def test_flash_matches_the_published_284b_and_13b(self):
        self.assertAlmostEqual(
            DEEPSEEK_V4_FLASH.total_params(include_mtp=False) / 1e9, 284.0,
            delta=6.0)
        self.assertAlmostEqual(DEEPSEEK_V4_FLASH.activated_params() / 1e9, 13.0,
                               delta=1.0)

    def test_neither_is_flagged_as_assumed(self):
        for profile in (DEEPSEEK_V4_PRO, DEEPSEEK_V4_FLASH):
            self.assertFalse(profile.assumed)
            self.assertFalse(profile.assumptions)
            self.assertIn("config.json", profile.source)

    def test_experts_are_fp4_and_everything_else_is_fp8(self):
        """Where the bytes went: dropping only the experts to MXFP4 takes the
        checkpoint to roughly half a byte per parameter, and it is the single
        biggest term in how many consoles a fleet needs."""
        for profile in (DEEPSEEK_V4_PRO, DEEPSEEK_V4_FLASH):
            self.assertTrue(profile.mixed_precision)
            self.assertEqual(profile.expert_quant, MXFP4)
            self.assertEqual(profile.dtype, "fp8")
            ratio = profile.total_bytes() / profile.total_params()
            self.assertGreater(ratio, 0.5)
            self.assertLess(ratio, 0.6)

    def test_mxfp4_carries_its_scales(self):
        """E2M1 is half a byte; the E8M0 scale per 32 values adds 6.25%, and
        forgetting it under-counts a fleet by a console per shelf."""
        self.assertAlmostEqual(MXFP4.bytes_per_param, 0.53125, places=5)

    def test_pro_is_smaller_on_disk_than_v3_despite_being_larger(self):
        """The headline result of FP4 experts, and the reason the fleet size
        for V4 Pro is not simply V3's scaled by parameter count."""
        self.assertGreater(DEEPSEEK_V4_PRO.total_params(),
                           2 * DEEPSEEK_V3.total_params())
        self.assertLess(DEEPSEEK_V4_PRO.total_bytes(),
                        1.5 * DEEPSEEK_V3.total_bytes())

    def test_an_expert_stays_small_enough_to_place(self):
        """Placement granularity is one expert. If an expert did not fit in a
        console's fast coffer with room for anything else, the whole
        shelf-and-shard scheme would collapse to layer-level pipelining."""
        self.assertLess(DEEPSEEK_V4_PRO.expert_bytes(), 64 * MB)
        self.assertLess(DEEPSEEK_V4_FLASH.expert_bytes(), 64 * MB)


class TestHybridAttention(unittest.TestCase):
    """CSA and HCA differ by a factor of 32 in cache size and by far more in
    what a decoded token reads. Modelling them as one average layer is what the
    planner used to do and it is wrong in both directions at once."""

    def test_the_schedule_covers_every_block_including_mtp(self):
        for profile in (DEEPSEEK_V4_PRO, DEEPSEEK_V4_FLASH):
            self.assertEqual(len(profile.attention.compress_ratios),
                             profile.planning_layers)

    def test_pro_starts_with_hca_and_flash_with_sliding_window(self):
        self.assertEqual(DEEPSEEK_V4_PRO.attention.kind(0), "hca")
        self.assertEqual(DEEPSEEK_V4_PRO.attention.kind(1), "hca")
        self.assertEqual(DEEPSEEK_V4_FLASH.attention.kind(0), "swa")
        self.assertEqual(DEEPSEEK_V4_FLASH.attention.kind(1), "swa")

    def test_layers_alternate_after_the_prologue(self):
        kinds = [DEEPSEEK_V4_PRO.attention.kind(i) for i in range(2, 20)]
        self.assertEqual(kinds, ["csa", "hca"] * 9)

    def test_an_hca_layer_caches_far_less_than_a_csa_layer(self):
        csa = DEEPSEEK_V4_PRO.kv_cache_bytes_for_layer(2, 1_000_000)
        hca = DEEPSEEK_V4_PRO.kv_cache_bytes_for_layer(3, 1_000_000)
        self.assertGreater(csa / hca, 20)

    def test_a_sliding_window_layer_does_not_grow_with_context(self):
        short = DEEPSEEK_V4_FLASH.kv_cache_bytes_for_layer(0, 4096)
        long = DEEPSEEK_V4_FLASH.kv_cache_bytes_for_layer(0, 1_000_000)
        self.assertEqual(short, long)

    def test_csa_reads_only_its_selected_entries(self):
        """Sparse selection is the whole point: at a million tokens a CSA layer
        holds far more cache than it reads, which is why long context costs
        capacity here rather than bandwidth."""
        held = DEEPSEEK_V4_PRO.kv_cache_bytes_for_layer(2, 1_000_000)
        read = DEEPSEEK_V4_PRO.kv_read_bytes_for_layer(2, 1_000_000)
        self.assertLess(read * 4, held)

    def test_short_contexts_read_everything_they_hold(self):
        """Below the indexer's top-k there is nothing to select away, so the
        sparse path must not claim a saving it is not making."""
        held = DEEPSEEK_V4_PRO.kv_cache_bytes_for_layer(2, 2048)
        read = DEEPSEEK_V4_PRO.kv_read_bytes_for_layer(2, 2048)
        self.assertGreaterEqual(read, held * 0.9)

    def test_the_whole_cache_is_a_tenth_of_v3s(self):
        """The paper's claim, at the context it is claimed for."""
        v4 = DEEPSEEK_V4_PRO.kv_cache_bytes(1_000_000)
        v3 = DEEPSEEK_V3.kv_cache_bytes(1_000_000)
        self.assertLess(v4 / v3, 0.15)


class TestHyperConnections(unittest.TestCase):
    def test_a_shelf_hop_carries_the_whole_residual_stream(self):
        """mHC widens the residual to 4x hidden. The layer input stays hidden-
        wide, but a pipeline boundary has to ship every branch, so the wire
        cost of a shelf hop is four times what the hidden size suggests."""
        self.assertEqual(DEEPSEEK_V4_PRO.hc_mult, 4)
        self.assertEqual(DEEPSEEK_V4_PRO.activation_bytes(),
                         4 * DEEPSEEK_V4_PRO.hidden_size * 2)

    def test_v3_has_a_plain_residual(self):
        self.assertEqual(DEEPSEEK_V3.activation_bytes(),
                         DEEPSEEK_V3.hidden_size * 2)


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
        self.assertIs(profile_for("deepseek-v4-flash"), DEEPSEEK_V4_FLASH)

    def test_unknown_names_say_what_is_available(self):
        with self.assertRaises(KeyError) as caught:
            profile_for("gpt-5")
        self.assertIn("deepseek-v3", str(caught.exception))

    def test_the_tiny_profile_is_small_enough_for_tests(self):
        self.assertLess(DEEPSEEK_TINY.total_bytes(), GB)
        self.assertIn(DEEPSEEK_TINY.name, PROFILES)


if __name__ == "__main__":
    unittest.main()
