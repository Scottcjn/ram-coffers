"""The hardware model: what a console is after reality gets to it."""

import unittest

from gen9_cluster.hardware import (GB, ComputeBackend, ConsoleUnit,
                                   Downbin, Runtime, SKUS, fleet_summary,
                                   sku_for)


class TestSKUs(unittest.TestCase):
    def test_consoles_ship_downbinned_by_design(self):
        """Every one of these parts has CUs fused off at the factory."""
        for name, expected_physical, expected_enabled in [
            ("ps5", 40, 36), ("ps5-slim", 40, 36),
            ("xbox-series-x", 56, 52), ("xbox-series-s", 24, 20),
        ]:
            spec = sku_for(name)
            self.assertEqual(spec.cu_physical, expected_physical, name)
            self.assertEqual(spec.cu_enabled, expected_enabled, name)
            self.assertLess(spec.cu_enabled, spec.cu_physical, name)

    def test_series_x_memory_is_split_into_two_speeds(self):
        """The Series X's 10 GB fast / 6 GB slow split is the whole reason
        MemoryTier exists rather than a single capacity number."""
        spec = sku_for("xbox-series-x")
        tiers = {tier.name: tier for tier in spec.tiers}
        self.assertIn("gddr6-fast", tiers)
        self.assertIn("gddr6-slow", tiers)
        self.assertGreater(tiers["gddr6-fast"].bandwidth_gbps,
                           tiers["gddr6-slow"].bandwidth_gbps)
        self.assertEqual(sum(t.total_bytes for t in spec.tiers), 16 * GB)

    def test_salvage_boards_are_first_class(self):
        for name in ("amd-4700s", "amd-4800s", "bc-250"):
            self.assertIn(name, SKUS)

    def test_4700s_has_no_usable_gpu(self):
        """The 4700S is a PS5 Ariel die sold as a desktop kit with the GPU fused
        off. Planning it as if it had 36 CUs would be a 40x error."""
        cap = ConsoleUnit("kit", "amd-4700s", Runtime.SALVAGE_LINUX).effective()
        self.assertEqual(cap.cu_active, 0)
        self.assertEqual(cap.backend, ComputeBackend.CPU_AVX2)

    def test_the_two_kits_are_different_harvests(self):
        """The 4800S is not a revision of the 4700S: PS5 Ariel silicon against
        Series X silicon, and a SATA-only board against one with an M.2. Both
        distinctions reach the planner, so neither may be collapsed."""
        kit47, kit48 = sku_for("amd-4700s"), sku_for("amd-4800s")
        self.assertEqual(kit47.cu_physical, 40)
        self.assertEqual(kit48.cu_physical, 56)
        self.assertLess(kit47.storage.effective_read_gbps,
                        kit48.storage.effective_read_gbps)

    def test_neither_kit_can_host_a_shelf(self):
        """Their GDDR6 is fast for system memory and far short of a console's,
        which is the entire reason they are capacity nodes and not hosts."""
        for name in ("amd-4700s", "amd-4800s"):
            self.assertLess(sku_for(name).fast_tier.bandwidth_gbps, 200.0, name)


class TestDownbin(unittest.TestCase):
    def test_disabled_cus_come_off_the_enabled_count(self):
        unit = ConsoleUnit("ps5-x", "ps5", Runtime.PS5_LINUX,
                           downbin=Downbin(cu_disabled=8,
                                           reasons=("dead shader array",)))
        self.assertEqual(unit.effective().cu_active, 36 - 8)

    def test_a_clock_cap_only_ever_lowers(self):
        """A cap above the stock clock is a typo, not an overclock."""
        fast = ConsoleUnit("a", "ps5", Runtime.PS5_LINUX,
                           downbin=Downbin(gpu_ghz_cap=9.9)).effective()
        slow = ConsoleUnit("b", "ps5", Runtime.PS5_LINUX,
                           downbin=Downbin(gpu_ghz_cap=1.2)).effective()
        self.assertEqual(fast.gpu_ghz, sku_for("ps5").gpu_ghz)
        self.assertEqual(slow.gpu_ghz, 1.2)

    def test_a_throttled_gpu_is_still_bandwidth_bound(self):
        """Decode is GEMV, so a downclocked PS5 loses almost nothing: it was
        waiting on GDDR6, not on the shader clock. Halving the clock only
        matters once the arithmetic ceiling drops below the memory one."""
        stock = ConsoleUnit("a", "ps5", Runtime.PS5_LINUX).effective()
        throttled = ConsoleUnit("b", "ps5", Runtime.PS5_LINUX,
                                downbin=Downbin(gpu_ghz_cap=1.2)).effective()
        self.assertEqual(throttled.gemv_gflops, stock.gemv_gflops)

        crippled = ConsoleUnit("c", "ps5", Runtime.PS5_LINUX,
                               downbin=Downbin(cu_disabled=32)).effective()
        self.assertLess(crippled.gemv_gflops, stock.gemv_gflops)

    def test_a_dead_memory_package_reduces_that_tier_only(self):
        unit = ConsoleUnit("ps5-y", "ps5", Runtime.PS5_LINUX,
                           downbin=Downbin(tier_losses={"gddr6": 2 * GB},
                                           reasons=("failed memtest",)))
        stock = ConsoleUnit("ps5-z", "ps5", Runtime.PS5_LINUX).effective()
        self.assertEqual(stock.weight_bytes - unit.effective().weight_bytes,
                         2 * GB)

    def test_losses_never_drive_capacity_negative(self):
        unit = ConsoleUnit("gone", "xbox-series-s", Runtime.XBOX_GDK,
                           downbin=Downbin(tier_losses={"gddr6-fast": 99 * GB}))
        for tier in unit.effective().coffers:
            self.assertGreaterEqual(tier.usable_bytes, 0)

    def test_reasons_survive_into_the_capability(self):
        """An operator reading a plan must be able to see *why* a console is
        small, or they will 'fix' it by re-planning with the wrong numbers."""
        unit = ConsoleUnit("ps5-w", "ps5", Runtime.PS5_LINUX,
                           downbin=Downbin(cu_disabled=4,
                                           reasons=("shader array 3 fused",)))
        self.assertTrue(any("shader array 3 fused" in w
                            for w in unit.effective().warnings))


class TestRuntimeSandboxes(unittest.TestCase):
    def test_xbox_devmode_is_capped_far_below_the_hardware(self):
        """Dev Mode hands a game-like app about 5 GB of a 16 GB console. The
        planner must see 5, not 16, or it will place four times too much."""
        cap = ConsoleUnit("xsx", "xbox-series-x",
                          Runtime.XBOX_DEVMODE).effective()
        self.assertLessEqual(cap.weight_bytes, 5 * GB)

    def test_devmode_app_budget_is_smaller_still(self):
        game = ConsoleUnit("g", "xbox-series-x", Runtime.XBOX_DEVMODE).effective()
        app = ConsoleUnit("a", "xbox-series-x", Runtime.XBOX_DEVMODE,
                          devmode_app=True).effective()
        self.assertLess(app.weight_bytes, game.weight_bytes)

    def test_gdk_gets_the_real_machine(self):
        gdk = ConsoleUnit("g", "xbox-series-x", Runtime.XBOX_GDK).effective()
        dev = ConsoleUnit("d", "xbox-series-x", Runtime.XBOX_DEVMODE).effective()
        self.assertGreater(gdk.weight_bytes, dev.weight_bytes * 2)
        self.assertEqual(gdk.backend, ComputeBackend.D3D12)


class TestBackendSelection(unittest.TestCase):
    def test_ps5_defaults_to_vulkan(self):
        """RADV drives gfx1013; ROCm's library layer does not, uniformly."""
        cap = ConsoleUnit("ps5", "ps5", Runtime.PS5_LINUX).effective()
        self.assertEqual(cap.backend, ComputeBackend.VULKAN)

    def test_rocm_is_available_but_warns_without_a_measurement(self):
        cap = ConsoleUnit("ps5", "ps5", Runtime.PS5_LINUX,
                          backend=ComputeBackend.ROCM).effective()
        self.assertEqual(cap.backend, ComputeBackend.ROCM)
        self.assertFalse(cap.throughput_measured)
        self.assertTrue(any("rocm" in w.lower() for w in cap.warnings))

    def test_a_measured_rocm_node_is_trusted_and_not_warned_about_speed(self):
        cap = ConsoleUnit("ps5", "ps5", Runtime.PS5_LINUX,
                          backend=ComputeBackend.ROCM,
                          measured_gemv_gflops=1234.0).effective()
        self.assertTrue(cap.throughput_measured)
        self.assertEqual(cap.gemv_gflops, 1234.0)

    def test_hen_runtime_has_no_gpu_path(self):
        cap = ConsoleUnit("ps5", "ps5", Runtime.PS5_HEN).effective()
        self.assertEqual(cap.backend, ComputeBackend.CPU_AVX2)

    def test_a_gpu_backend_with_every_cu_dead_falls_back_to_the_cpu(self):
        cap = ConsoleUnit("ps5", "ps5", Runtime.PS5_LINUX,
                          downbin=Downbin(cu_disabled=36)).effective()
        self.assertEqual(cap.backend, ComputeBackend.CPU_AVX2)


class TestBC250(unittest.TestCase):
    """The mining board: 40 CUs on the die, 24 lit by the stock driver."""

    def test_stock_driver_baseline(self):
        cap = ConsoleUnit("bc", "bc-250", Runtime.SALVAGE_LINUX).effective()
        self.assertEqual(cap.cu_active, 24)

    def test_the_community_unlock_is_allowed_but_flagged(self):
        cap = ConsoleUnit("bc", "bc-250", Runtime.SALVAGE_LINUX,
                          cu_enabled_override=40).effective()
        self.assertEqual(cap.cu_active, 40)
        self.assertTrue(any("40" in w or "unlock" in w.lower()
                            for w in cap.warnings))

    def test_the_override_cannot_exceed_the_silicon(self):
        cap = ConsoleUnit("bc", "bc-250", Runtime.SALVAGE_LINUX,
                          cu_enabled_override=64).effective()
        self.assertEqual(cap.cu_active, sku_for("bc-250").cu_physical)


class TestFleetSummary(unittest.TestCase):
    def test_summary_counts_what_an_operator_needs_to_worry_about(self):
        fleet = [
            ConsoleUnit("a", "ps5", Runtime.PS5_LINUX),
            ConsoleUnit("b", "ps5", Runtime.PS5_LINUX,
                        downbin=Downbin(cu_disabled=4, reasons=("dead CU",))),
            ConsoleUnit("c", "xbox-series-s", Runtime.XBOX_GDK),
            ConsoleUnit("d", "bc-250", Runtime.SALVAGE_LINUX,
                        measured_gemv_gflops=500.0),
        ]
        summary = fleet_summary(fleet)
        self.assertEqual(summary.units, 4)
        self.assertEqual(summary.by_sku["ps5"], 2)
        self.assertEqual(summary.downbinned_units, 1)
        self.assertEqual(summary.estimated_throughput_units, 3)
        self.assertGreater(summary.weight_bytes, 0)


if __name__ == "__main__":
    unittest.main()
