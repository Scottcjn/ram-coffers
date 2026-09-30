"""Model sizing: the arithmetic that decides how many consoles are needed."""

import dataclasses
import unittest

from gen9_cluster.model import (DEEPSEEK_TINY, DEEPSEEK_V3, DEEPSEEK_V4_1_FLASH,
                                DEEPSEEK_V4_FLASH, DEEPSEEK_V4_PRO,
                                FP4_E4M3_16, FP8_UE8M0_32, MXFP4, PROFILES,
                                profile_for)

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


class TestV41FlashAgainstPublishedFigures(unittest.TestCase):
    """V4.1-Flash is a different architecture, not a V4 variant: a causal
    encoder-decoder with shared KV pools, Engram memory, and DSpark drafts.
    The published numbers it has to land on are 552 B backbone, 16 B
    activated, ~196 B of Engram, and ~890 B of KV per token."""

    def test_backbone_matches_the_published_552b(self):
        """The card's 552 B excludes the draft blocks and the auxiliary
        stores; so does total_params with include_mtp off."""
        self.assertAlmostEqual(
            DEEPSEEK_V4_1_FLASH.total_params(include_mtp=False) / 1e9,
            552.0, delta=4.0)
        self.assertGreater(DEEPSEEK_V4_1_FLASH.total_params(), 552e9)

    def test_activated_matches_the_published_16b_decode(self):
        """The card splits activation by phase: 8 B on prefill, 16 B on
        decode. The planner's figure counts every block a decode token
        touches, so it is the decode number it must match."""
        self.assertAlmostEqual(DEEPSEEK_V4_1_FLASH.activated_params() / 1e9,
                               16.0, delta=1.5)

    def test_engram_tables_match_the_published_196b(self):
        engram = DEEPSEEK_V4_1_FLASH.engram
        self.assertIsNotNone(engram)
        self.assertAlmostEqual(
            engram.params(DEEPSEEK_V4_1_FLASH.hidden_size) / 1e9, 196.0,
            delta=3.0)

    def test_global_kv_cache_is_about_890_bytes_per_token(self):
        """Four main-KV streams and eight indexer streams, all FP4 at m=2.
        The derivation reads 864; the card claims ~890 — inside 4%, which is
        the tolerance the whole figure is quoted to."""
        per_token = sum(
            DEEPSEEK_V4_1_FLASH.attention.kv_cache_bytes_per_token(layer=layer)
            for layer in range(DEEPSEEK_V4_1_FLASH.planning_layers))
        self.assertAlmostEqual(per_token / 890.0, 1.0, delta=0.04)

    def test_the_cache_is_a_quarter_of_v4_flashs(self):
        """The paper's headline claim for the architecture change."""
        v4 = DEEPSEEK_V4_FLASH.kv_cache_bytes(1_000_000)
        v41 = DEEPSEEK_V4_1_FLASH.kv_cache_bytes(1_000_000)
        self.assertLess(v41 / v4, 0.30)

    def test_not_flagged_as_assumed(self):
        self.assertFalse(DEEPSEEK_V4_1_FLASH.assumed)
        self.assertIn("config.json", DEEPSEEK_V4_1_FLASH.source)

    def test_weights_are_fp8_32x32_and_experts_fp4(self):
        self.assertTrue(DEEPSEEK_V4_1_FLASH.mixed_precision)
        self.assertEqual(DEEPSEEK_V4_1_FLASH.weights, FP8_UE8M0_32)
        self.assertEqual(DEEPSEEK_V4_1_FLASH.expert_quant, MXFP4)
        self.assertAlmostEqual(FP8_UE8M0_32.bytes_per_param,
                               1.0 + 1.0 / 1024, places=6)
        self.assertAlmostEqual(FP4_E4M3_16.bytes_per_param, 0.5625, places=6)


class TestCSA2LayerModes(unittest.TestCase):
    """The schedule the checkpoint ships: two SWA prologue blocks, Full
    source layers at 2/8/14/20, decoder Reindex layers every four blocks,
    Reuse everywhere else, and SWA draft blocks at the tail."""

    def test_schedule_covers_every_block_including_drafts(self):
        attention = DEEPSEEK_V4_1_FLASH.attention
        self.assertEqual(len(attention.compress_ratios),
                         DEEPSEEK_V4_1_FLASH.planning_layers)
        self.assertEqual(DEEPSEEK_V4_1_FLASH.planning_layers, 43)

    def test_the_mode_layout(self):
        kinds = [DEEPSEEK_V4_1_FLASH.attention.kind(i) for i in range(43)]
        self.assertEqual(kinds[0:2], ["swa", "swa"])
        for full in (2, 8, 14, 20):
            self.assertEqual(kinds[full], "full", f"layer {full}")
        for reindex in (24, 28, 32, 36):
            self.assertEqual(kinds[reindex], "reindex", f"layer {reindex}")
        self.assertEqual(kinds[40:], ["swa"] * 3)
        self.assertEqual(kinds.count("reuse"), 30)

    def test_only_source_layers_grow_the_cache(self):
        """Under CSA2 most layers store nothing: the growing cache belongs to
        the eight source streams, so a decoder shelf host holds a window and
        little else."""
        attention = DEEPSEEK_V4_1_FLASH.attention
        for layer in range(43):
            per_token = attention.kv_cache_bytes_per_token(layer=layer)
            if layer in attention.kv_source_layers \
                    or layer in attention.index_source_layers:
                self.assertGreater(per_token, 0, f"layer {layer}")
            else:
                self.assertEqual(per_token, 0, f"layer {layer}")

    def test_decoder_reindex_scan_is_bounded_by_the_candidate_pool(self):
        """The hierarchical indexer is the reason a million-token decode does
        not scan a million entries: past the pool size the read stops growing
        with context."""
        pool_entries = (DEEPSEEK_V4_1_FLASH.attention.candidate_topk_blocks
                        * DEEPSEEK_V4_1_FLASH.attention.candidate_block_size)
        shallow = DEEPSEEK_V4_1_FLASH.kv_read_bytes_for_layer(24, 4 * 32768)
        deep = DEEPSEEK_V4_1_FLASH.kv_read_bytes_for_layer(24, 1_000_000)
        self.assertEqual(shallow, deep)
        # and a Full layer keeps scanning its whole stream
        full_shallow = DEEPSEEK_V4_1_FLASH.kv_read_bytes_for_layer(2, 65536)
        full_deep = DEEPSEEK_V4_1_FLASH.kv_read_bytes_for_layer(2, 1_000_000)
        self.assertGreater(full_deep, full_shallow * 4)
        self.assertGreater(pool_entries, 0)

    def test_reuse_layers_read_only_the_selection(self):
        """A Reuse layer performs no scan of its own; at long context it
        reads far less than a Full layer."""
        reuse = DEEPSEEK_V4_1_FLASH.kv_read_bytes_for_layer(30, 1_000_000)
        full = DEEPSEEK_V4_1_FLASH.kv_read_bytes_for_layer(2, 1_000_000)
        self.assertLess(reuse * 4, full)

    def test_candidate_source_layer_gates_the_bounded_scan(self):
        """The candidate pool exists only once the first decoder layer's Full
        pass has built it: a Reindex layer below ``candidate_source_layer``
        scans its own stream, so moving the boundary un-bounds the scan."""
        attention = DEEPSEEK_V4_1_FLASH.attention
        self.assertGreaterEqual(24, attention.candidate_source_layer)
        shifted = dataclasses.replace(attention, candidate_source_layer=40)
        bounded = attention.kv_read_bytes(1_000_000, layer=24)
        unbounded = shifted.kv_read_bytes(1_000_000, layer=24)
        self.assertGreater(unbounded, bounded * 4)

    def test_engram_splits_into_nvme_rows_and_a_ram_projection(self):
        """The planner prices the two parts of a table separately: the row
        store (a drive resident) and the fusion projection (a RAM piece read
        every token)."""
        engram = DEEPSEEK_V4_1_FLASH.engram
        for table in range(len(engram.layer_ids)):
            whole = engram.table_bytes(table, DEEPSEEK_V4_1_FLASH.hidden_size,
                                       DEEPSEEK_V4_1_FLASH.weights)
            rows = engram.table_row_bytes(table, DEEPSEEK_V4_1_FLASH.weights)
            fuse = engram.fusion_bytes(DEEPSEEK_V4_1_FLASH.hidden_size,
                                       DEEPSEEK_V4_1_FLASH.weights)
            self.assertEqual(rows + fuse, whole)
            self.assertGreater(rows, 50 * GB)
            self.assertLess(fuse, 64 * MB)


class TestV41DraftAndAuxiliary(unittest.TestCase):
    def test_draft_blocks_use_their_own_moe(self):
        """DSpark drafts are 128-expert top-3 blocks, not shrunken copies of
        the backbone's 384-expert top-6 MoE."""
        draft = DEEPSEEK_V4_1_FLASH.moe_for_block(40)
        self.assertEqual(draft.n_routed_experts, 128)
        self.assertEqual(draft.top_k, 3)
        backbone = DEEPSEEK_V4_1_FLASH.moe_for_block(39)
        self.assertEqual(backbone.n_routed_experts, 384)
        self.assertEqual(
            DEEPSEEK_V4_1_FLASH.cold_bytes_per_moe_layer(40)
            / DEEPSEEK_V4_1_FLASH.cold_bytes_per_moe_layer(39),
            128 / 384)

    def test_engram_reads_are_lookup_sized(self):
        """Kilobytes per token is what makes a 183 GiB table an NVMe
        resident rather than a refusal."""
        engram = DEEPSEEK_V4_1_FLASH.engram
        self.assertLess(engram.read_bytes_per_token(
            DEEPSEEK_V4_1_FLASH.weights), 64 * 1024)
        self.assertGreater(DEEPSEEK_V4_1_FLASH.engram_bytes(), 150 * GB)

    def test_auxiliary_stores_count_toward_total_bytes(self):
        """The fleet stores the Engram tables and the vision tower too, so
        they are in total_bytes even though they are not in the card's
        backbone parameter count."""
        total = DEEPSEEK_V4_1_FLASH.total_bytes()
        self.assertGreater(total, DEEPSEEK_V4_1_FLASH.engram_bytes())
        self.assertGreater(DEEPSEEK_V4_1_FLASH.vision_bytes(), 0)


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
        self.assertIs(profile_for("deepseek-v4.1-flash"), DEEPSEEK_V4_1_FLASH)

    def test_unknown_names_say_what_is_available(self):
        with self.assertRaises(KeyError) as caught:
            profile_for("gpt-5")
        self.assertIn("deepseek-v3", str(caught.exception))

    def test_the_tiny_profile_is_small_enough_for_tests(self):
        self.assertLess(DEEPSEEK_TINY.total_bytes(), GB)
        self.assertIn(DEEPSEEK_TINY.name, PROFILES)


if __name__ == "__main__":
    unittest.main()
