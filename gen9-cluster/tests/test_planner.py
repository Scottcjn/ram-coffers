"""The splitter: which console holds what, and whether it fits at all."""

import dataclasses
import unittest

from gen9_cluster.hardware import GB, ConsoleUnit, Downbin, Runtime
from gen9_cluster.inventory import deployment_config
from gen9_cluster.model import (DEEPSEEK_TINY, DEEPSEEK_V4_1_FLASH,
                                DEEPSEEK_V4_PRO)
from gen9_cluster.planner import (DEFAULT_HOP_SECONDS, PlanningError,
                                  _estimate_decode, describe_plan, plan_split)

#: A hypothetical unpublished model, for the paths that have to keep warning
#: about extrapolated configurations now that both V4 profiles are published.
ASSUMED_MODEL = dataclasses.replace(
    DEEPSEEK_V4_PRO, name="deepseek-v5-guess", assumed=True,
    assumptions=("nothing about this model is published",))


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

    def test_without_ssd_a_tight_fleet_is_rejected(self):
        with self.assertRaises(PlanningError):
            plan_split(DEEPSEEK_V4_PRO, ps5_fleet(70), allow_ssd_tier=False)


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

    def test_engram_tables_sit_on_their_stage_host_nvme(self):
        """CSA2's conditional-memory tables are NVMe residents by design, and
        they belong to the shelf whose host runs their layer — a lookup must
        not cross a shelf boundary ahead of the layer that needs it."""
        plan = plan_split(DEEPSEEK_V4_1_FLASH, ps5_fleet(60))
        host_by_layer = {}
        for stage in plan.stages:
            for layer in stage.layers:
                host_by_layer[layer] = stage.host_unit
        for layer in DEEPSEEK_V4_1_FLASH.engram.layer_ids:
            host = plan.units[host_by_layer[layer]]
            self.assertIn(f"engram-{layer}@ssd", host.io_pieces)
            # ...while the every-token fusion projection sits in its RAM
            self.assertIn(f"engram-{layer}-fuse", host.io_pieces)

    def test_a_no_ssd_plan_shards_engram_across_ram(self):
        """With no drive to use, a hash-addressed table can still live in the
        fleet's RAM — split across units exactly like routed experts are."""
        plan = plan_split(DEEPSEEK_V4_1_FLASH, ps5_fleet(160),
                          allow_ssd_tier=False)
        engram = DEEPSEEK_V4_1_FLASH.engram
        quant = DEEPSEEK_V4_1_FLASH.weights
        for table, layer in enumerate(engram.layer_ids):
            want = engram.table_row_bytes(table, quant)
            held = sum(u.engram_rows.get(layer, 0) for u in plan.units.values())
            self.assertEqual(held, want)
            self.assertFalse(any(f"engram-{layer}@ssd" in u.io_pieces
                                 for u in plan.units.values()))
        nodes = deployment_config(plan, [])["nodes"]
        config_rows = sum(node["engram_ram_bytes"] for node in nodes.values())
        plan_rows = sum(sum(u.engram_rows.values())
                        for u in plan.units.values())
        self.assertEqual(config_rows, plan_rows)

    def test_the_row_map_is_serialized_for_routing(self):
        """A lookup routes by row id, so the deployment config has to carry
        each unit's contiguous range — without it a --no-ssd deploy has the
        rows but no map of where they went."""
        plan = plan_split(DEEPSEEK_V4_1_FLASH, ps5_fleet(160),
                          allow_ssd_tier=False)
        engram = DEEPSEEK_V4_1_FLASH.engram
        nodes = deployment_config(plan, [])["nodes"]
        for table, layer in enumerate(engram.layer_ids):
            records = sorted(
                (dict(s, unit_id=uid) for uid, node in nodes.items()
                 for s in node["engram_shards"] if s["layer"] == layer),
                key=lambda s: s["first_row"])
            self.assertTrue(records)
            self.assertTrue(all(s["tier"] == "ram" for s in records))
            # The ranges tile the whole table with no gaps and no overlap.
            self.assertEqual(records[0]["first_row"], 0)
            self.assertEqual(
                sum(s["n_rows"] for s in records),
                engram.num_embeddings[table])
            for left, right in zip(records, records[1:]):
                self.assertEqual(right["first_row"],
                                 left["first_row"] + left["n_rows"])
            self.assertEqual(
                sum(s["bytes"] for s in records),
                engram.table_row_bytes(table, DEEPSEEK_V4_1_FLASH.weights))
            # The records are the accounting: per-unit shard bytes equal the
            # RAM bytes the unit reports for that layer.
            for s in records:
                self.assertEqual(
                    s["bytes"],
                    plan.units[s["unit_id"]].engram_rows[layer])

    def test_an_nvme_table_is_one_whole_shard_record(self):
        """Same routing map, degenerate case: with a drive to use, a table is
        a single record naming its holder and the whole row range."""
        plan = plan_split(DEEPSEEK_V4_1_FLASH, ps5_fleet(60))
        engram = DEEPSEEK_V4_1_FLASH.engram
        nodes = deployment_config(plan, [])["nodes"]
        for table, layer in enumerate(engram.layer_ids):
            records = [s for node in nodes.values()
                       for s in node["engram_shards"] if s["layer"] == layer]
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["tier"], "ssd")
            self.assertEqual(record["first_row"], 0)
            self.assertEqual(record["n_rows"], engram.num_embeddings[table])

    def test_ram_sharded_engram_counts_its_holders_per_token(self):
        """A RAM-sharded lookup fans out to every holder, so the per-token
        console count has to include them — hash addressing means the rows a
        token needs live everywhere the table does."""
        sharded = plan_split(DEEPSEEK_V4_1_FLASH, ps5_fleet(160),
                             allow_ssd_tier=False)
        engram = DEEPSEEK_V4_1_FLASH.engram
        max_holders = max(
            sum(1 for u in sharded.units.values() if layer in u.engram_rows)
            for layer in engram.layer_ids)
        self.assertGreaterEqual(sharded.active_units_per_token, max_holders)

    def test_a_tiny_no_ssd_fleet_still_cannot_hold_v41(self):
        """Sharding needs RAM to shard into — a fleet without the ~183 GiB of
        headroom the tables need refuses rather than drop rows."""
        with self.assertRaises(PlanningError):
            plan_split(DEEPSEEK_V4_1_FLASH, ps5_fleet(30),
                       allow_ssd_tier=False)

    def test_the_vision_tower_sits_where_images_enter(self):
        """The vision encoder produces embeddings consumed at the front of the
        pipeline, so it rides the first stage's host, not wherever headroom
        happens to be."""
        plan = plan_split(DEEPSEEK_V4_1_FLASH, ps5_fleet(60))
        first_host = plan.units[plan.stages[0].host_unit]
        self.assertIn("vision-encoder", first_host.io_pieces)

    def test_engram_reads_are_priced_into_decode(self):
        """A plan that carries the tables is slower than the same fleet
        without them: row lookups off NVMe plus the fusion read are part of
        every token's cost, not free."""
        with_tables = plan_split(DEEPSEEK_V4_1_FLASH, ps5_fleet(60))
        without = plan_split(dataclasses.replace(DEEPSEEK_V4_1_FLASH,
                                                 engram=None), ps5_fleet(60))
        self.assertGreater(with_tables.seconds_per_token,
                           without.seconds_per_token)

    def test_an_ssd_io_piece_is_priced_into_decode(self):
        """An embedding or lm_head that spilled to NVMe is read once per
        token — that streamed read (and the hop to its consumer) is part of
        the decode estimate, not free."""
        fleet = ps5_fleet(4)
        caps = {u.unit_id: u.effective() for u in fleet}
        plan = plan_split(DEEPSEEK_TINY, fleet)
        base, _ = _estimate_decode(DEEPSEEK_TINY, caps, plan.units,
                                   plan.stages, DEFAULT_HOP_SECONDS, 8192)
        # The vision tower is per-image: spilling it must not change the
        # per-token estimate, while embedding@ssd must.
        holder = plan.units[fleet[-1].unit_id]
        holder.io_pieces.append("vision-encoder@ssd")
        vision, _ = _estimate_decode(DEEPSEEK_TINY, caps, plan.units,
                                     plan.stages, DEFAULT_HOP_SECONDS, 8192)
        self.assertEqual(vision, base)
        holder.io_pieces.append("embedding@ssd")
        with_ssd, _ = _estimate_decode(DEEPSEEK_TINY, caps, plan.units,
                                       plan.stages, DEFAULT_HOP_SECONDS,
                                       8192)
        self.assertGreater(with_ssd, base)

    def test_decoder_stage_hosts_hold_little_growing_cache(self):
        """Only the source layers' hosts see a cache that grows with context —
        a host of pure Reuse layers is almost cache-free, which is the CED
        split's whole residency consequence."""
        plan = plan_split(DEEPSEEK_V4_1_FLASH, ps5_fleet(60),
                          context_tokens=1_000_000)
        attention = DEEPSEEK_V4_1_FLASH.attention
        sources = (set(attention.kv_source_layers)
                   | set(attention.index_source_layers))
        source_hosts = {stage.host_unit for stage in plan.stages
                        if any(layer in sources for layer in stage.layers)}
        for stage in plan.stages:
            host = plan.units[stage.host_unit]
            if stage.host_unit in source_hosts:
                # ~36 B/token for a decoder index stream, ~144 B for a main one
                self.assertGreater(host.kv_bytes, 50 * 1024 * 1024)
            else:
                self.assertLess(host.kv_bytes, 10 * 1024 * 1024)

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
        plan = plan_split(ASSUMED_MODEL, ps5_fleet(160))
        self.assertTrue(plan.model_assumed)
        self.assertTrue(any("assum" in w.lower() for w in plan.warnings))

    def test_a_published_model_does_not(self):
        plan = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(160))
        self.assertFalse(plan.model_assumed)
        self.assertFalse(any("assum" in w.lower() for w in plan.warnings))


class TestDescription(unittest.TestCase):
    def test_the_summary_names_the_things_an_operator_must_act_on(self):
        plan = plan_split(DEEPSEEK_V4_PRO, ps5_fleet(160))
        text = "\n".join(describe_plan(plan))
        self.assertIn("shelves", text)
        self.assertIn("tok/s", text)

    def test_an_assumed_model_says_so_in_the_summary(self):
        plan = plan_split(ASSUMED_MODEL, ps5_fleet(160))
        self.assertIn("ASSUMED", "\n".join(describe_plan(plan)))


if __name__ == "__main__":
    unittest.main()
