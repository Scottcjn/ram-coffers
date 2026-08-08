"""The splitter: which console holds what, and whether it fits at all."""

import unittest

from gen9_cluster.hardware import GB, ConsoleUnit, Downbin, Runtime
from gen9_cluster.model import DEEPSEEK_TINY, DEEPSEEK_V4_PRO
from gen9_cluster.planner import PlanningError, describe_plan, plan_split


def ps5_fleet(count, **kwargs):
    return [ConsoleUnit(f"ps5-{i:03d}", "ps5", Runtime.PS5_LINUX, **kwargs)
            for i in range(count)]


class TestFeasibility(unittest.TestCase):
    def test_one_console_cannot_hold_a_frontier_model(self):
        """16 GB against ~1.3 TiB. The planner must refuse rather than emit a
        plan that silently drops most of the weights."""
        with self.assertRaises(PlanningError):
            plan_split(DEEPSEEK_V4_PRO, ps5_fleet(1), allow_ssd_tier=False)

    def test_a_large_fleet_hosts_it(self):
        plan = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(160))
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.shortfall_bytes, 0)

    def test_slow_is_not_infeasible(self):
        """A deployment that has to stream experts off NVMe is bad, not
        impossible; the planner reports it and continues."""
        plan = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(100), allow_ssd_tier=True)
        self.assertTrue(plan.feasible)
        self.assertGreater(plan.ssd_expert_bytes, 0)
        self.assertTrue(any("nvme" in w.lower() or "ssd" in w.lower()
                            for w in plan.warnings))

    def test_without_ssd_the_same_fleet_is_rejected(self):
        with self.assertRaises(PlanningError):
            plan_split(DEEPSEEK_V4_PRO, ps5_fleet(100), allow_ssd_tier=False)


class TestShelves(unittest.TestCase):
    """A shelf is the expert fan-out group: a token's top-k should be answered
    without leaving it, because leaving it costs a network hop."""

    def test_a_small_fleet_is_one_shelf(self):
        plan = plan_split(DEEPSEEK_TINY, ps5_fleet(4), shelf_size=22)
        self.assertEqual(len(plan.stages), 1)

    def test_a_large_fleet_is_divided_by_shelf_size(self):
        plan = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(160), shelf_size=22)
        self.assertEqual(len(plan.stages), 8)

    def test_every_block_is_hosted_exactly_once(self):
        """The pipeline must be a partition: a missing block is a wrong answer
        and a duplicated one is wasted memory."""
        plan = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(160))
        hosted = [layer for stage in plan.stages for layer in stage.layers]
        self.assertEqual(sorted(hosted), list(range(len(hosted))))
        self.assertEqual(len(hosted), DEEPSEEK_V4_PRO.planning_layers)

    def test_ranges_are_contiguous_and_in_order(self):
        plan = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(160))
        cursor = 0
        for stage in plan.stages:
            self.assertEqual(stage.first_layer, cursor)
            cursor += stage.n_layers

    def test_crossing_shelves_is_counted_and_warned_about(self):
        plan = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(160), shelf_size=22)
        self.assertTrue(any("crosses the network" in w.lower()
                            for w in plan.warnings))


class TestPlacement(unittest.TestCase):
    def test_experts_stay_on_their_own_shelf(self):
        plan = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(160), shelf_size=22)
        for stage in plan.stages:
            members = set(stage.expert_units)
            for layer in stage.layers:
                holders = {shard.unit_id for unit in plan.units.values()
                           for shard in unit.shards if shard.layer == layer}
                self.assertTrue(holders <= members,
                                f"layer {layer} spilled off its shelf")

    def test_every_expert_of_every_moe_layer_is_placed_once(self):
        plan = plan_split(DEEPSEEK_TINY, ps5_fleet(4))
        for layer in range(DEEPSEEK_TINY.moe.n_dense_layers,
                           DEEPSEEK_TINY.n_layers):
            placed = sorted(e for unit in plan.units.values()
                            for shard in unit.shards if shard.layer == layer
                            for e in shard.expert_ids)
            self.assertEqual(placed,
                             list(range(DEEPSEEK_TINY.moe.n_routed_experts)),
                             f"layer {layer}")

    def test_nothing_is_placed_beyond_a_console_capacity(self):
        plan = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(160))
        for unit in plan.units.values():
            resident = unit.resident_bytes - unit.ssd_expert_bytes
            self.assertLessEqual(resident, unit.capacity_bytes, unit.unit_id)

    def test_a_faster_console_is_given_more_experts(self):
        """Allocation is throughput-weighted, not equal: giving a crippled
        console its 'fair share' makes it the tail latency of every token."""
        fleet = ps5_fleet(3) + [
            ConsoleUnit("ps5-slow", "ps5", Runtime.PS5_LINUX,
                        downbin=Downbin(cu_disabled=30,
                                        tier_bandwidth_scale={"gddr6": 0.25},
                                        reasons=("degraded",)))]
        plan = plan_split(DEEPSEEK_TINY, fleet)
        counts = {uid: sum(s.n_experts for s in u.shards)
                  for uid, u in plan.units.items()}
        self.assertLess(counts["ps5-slow"], max(counts.values()))

    def test_the_shelf_host_holds_the_hot_weights(self):
        plan = plan_split(DEEPSEEK_TINY, ps5_fleet(4))
        for stage in plan.stages:
            host = plan.units[stage.host_unit]
            self.assertEqual(sorted(host.hot_layers + host.dense_layers),
                             sorted(stage.layers))
            self.assertGreater(host.kv_bytes, 0)

    def test_a_devmode_xbox_is_not_asked_to_hold_more_than_its_sandbox(self):
        fleet = ps5_fleet(3) + [ConsoleUnit("xsx-dev", "xbox-series-x",
                                            Runtime.XBOX_DEVMODE)]
        plan = plan_split(DEEPSEEK_TINY, fleet)
        sandboxed = plan.units["xsx-dev"]
        self.assertLessEqual(sandboxed.resident_bytes - sandboxed.ssd_expert_bytes,
                             5 * GB)


class TestEstimates(unittest.TestCase):
    def test_the_estimate_is_positive_and_slow(self):
        """It is a bandwidth-and-hops arithmetic model, not a benchmark. It
        should land in single-digit tokens per second, which is the honest
        expectation for this hardware."""
        plan = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(160))
        self.assertGreater(plan.tokens_per_second, 0.0)
        self.assertLess(plan.tokens_per_second, 100.0)

    def test_only_a_fraction_of_the_fleet_is_lit_per_token(self):
        plan = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(160))
        self.assertLess(plan.active_units_per_token, len(plan.units))

    def test_ssd_streaming_shows_up_as_a_slowdown(self):
        tight = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(100))
        roomy = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(160))
        self.assertLess(tight.tokens_per_second, roomy.tokens_per_second)

    def test_an_assumed_model_taints_the_plan(self):
        plan = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(160))
        self.assertTrue(plan.model_assumed)
        self.assertTrue(any("assum" in w.lower() for w in plan.warnings))


class TestDescription(unittest.TestCase):
    def test_the_summary_names_the_things_an_operator_must_act_on(self):
        plan = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(160))
        text = "\n".join(describe_plan(plan))
        self.assertIn("shelves", text)
        self.assertIn("tok/s", text)
        self.assertIn("ASSUMED", text)


if __name__ == "__main__":
    unittest.main()
