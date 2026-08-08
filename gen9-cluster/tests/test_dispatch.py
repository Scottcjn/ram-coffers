"""Expert fan-out, reduction, failover, and a whole fleet end to end.

Everything here runs against real node servers on loopback, because the
properties being checked — that a token's experts are grouped into one request
per console, that the reduction is order-independent, that a dead console is
routed around — are properties of the distributed system, not of the functions
in isolation.
"""

import socket
import unittest

import numpy as np

from gen9_cluster.coordinator import FleetCoordinator
from gen9_cluster.dispatch import (ExpertDispatcher, NodeAddress,
                                   placement_from_plan)
from gen9_cluster.errors import Gen9Error, ShardMissing
from gen9_cluster.hardware import ConsoleUnit, Runtime
from gen9_cluster.model import DEEPSEEK_TINY
from gen9_cluster.node import ExpertWeights, NodeServer, ShardStore
from gen9_cluster.planner import plan_split
from gen9_cluster.protocol import (ExpertRowsPayload, Flags, Frame, MsgType,
                                   accumulate, encode_vector)

HIDDEN = 16
INTERMEDIATE = 8


def make_expert(seed):
    rng = np.random.default_rng(seed)
    return ExpertWeights(
        gate=rng.standard_normal((INTERMEDIATE, HIDDEN), dtype=np.float32)
        * 0.05,
        up=rng.standard_normal((INTERMEDIATE, HIDDEN), dtype=np.float32) * 0.05,
        down=rng.standard_normal((HIDDEN, INTERMEDIATE), dtype=np.float32)
        * 0.05)


def _closed_port():
    """A port nothing is listening on, for simulating a console that died."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def reference_expert(weights, x, gate):
    hidden = x @ weights.gate.T
    hidden = hidden * (1.0 / (1.0 + np.exp(-hidden)))
    hidden = hidden * (x @ weights.up.T)
    return gate * (hidden @ weights.down.T)


class ClusterFixture(unittest.TestCase):
    """Three consoles, each holding two of six experts in layer 0."""

    placement_map = {"n0": [0, 1], "n1": [2, 3], "n2": [4, 5]}

    def setUp(self):
        self.experts = {e: make_expert(e) for e in range(6)}
        self.nodes = {}
        self.addresses = {}
        for unit_id, expert_ids in self.placement_map.items():
            store = ShardStore()
            for expert in expert_ids:
                store.put(0, expert, self.experts[expert])
            node = NodeServer(store, unit_id=unit_id, host="127.0.0.1", port=0,
                              block_fn=lambda layer, x: x)
            port = node.start()
            self.nodes[unit_id] = node
            self.addresses[unit_id] = NodeAddress(unit_id, "127.0.0.1", port)
            self.addCleanup(node.stop)
        self.placement = {(0, e): [u]
                          for u, ids in self.placement_map.items()
                          for e in ids}

    def dispatcher(self, **kwargs):
        disp = ExpertDispatcher(self.placement, self.addresses, **kwargs)
        self.addCleanup(disp.close)
        return disp


class TestGrouping(ClusterFixture):
    def test_a_token_is_grouped_into_one_request_per_console(self):
        """Six experts across three consoles is three requests, not six. On a
        fleet this is the difference between 8 round trips per layer and 3."""
        grouped = self.dispatcher().group_by_unit(0, [0, 1, 2, 3, 4, 5],
                                                  [1 / 6] * 6)
        self.assertEqual(set(grouped), {"n0", "n1", "n2"})
        self.assertEqual(grouped["n0"][0], [0, 1])
        self.assertEqual(grouped["n1"][0], [2, 3])

    def test_gates_stay_with_their_experts(self):
        grouped = self.dispatcher().group_by_unit(0, [5, 0, 3], [0.6, 0.3, 0.1])
        self.assertEqual(grouped["n2"], ([5], [0.6]))
        self.assertEqual(grouped["n0"], ([0], [0.3]))
        self.assertEqual(grouped["n1"], ([3], [0.1]))

    def test_an_unplaced_expert_is_refused_rather_than_dropped(self):
        """Silently omitting an expert produces a plausible wrong answer, which
        is worse than a failure."""
        with self.assertRaises(ShardMissing):
            self.dispatcher().group_by_unit(0, [42], [1.0])


class TestReduction(ClusterFixture):
    def test_the_layer_output_matches_a_local_computation(self):
        disp = self.dispatcher()
        x = np.linspace(-1, 1, HIDDEN, dtype=np.float32)
        ids, gates = [0, 2, 4, 5], [0.4, 0.3, 0.2, 0.1]
        got, stats = disp.run_layer(0, x, ids, gates)

        expected = np.zeros(HIDDEN, dtype=np.float32)
        for expert, gate in zip(ids, gates):
            expected += reference_expert(self.experts[expert], x, gate)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(stats.consoles_contacted, 3)
        self.assertEqual(stats.experts_run, 4)

    def test_the_result_is_bit_identical_across_runs(self):
        """Replies arrive in whatever order the network chooses, so the sum is
        taken in a fixed order instead. Without that, the same prompt can
        produce different tokens on different runs."""
        disp = self.dispatcher()
        x = np.linspace(-2, 2, HIDDEN, dtype=np.float32)
        ids, gates = [0, 1, 2, 3, 4, 5], [1 / 6] * 6
        first, _ = disp.run_layer(0, x, ids, gates)
        for _ in range(6):
            again, _ = disp.run_layer(0, x, ids, gates)
            np.testing.assert_array_equal(first, again)

    def test_the_result_does_not_depend_on_where_the_experts_live(self):
        """The property the whole per-expert reply shape exists to provide.

        This is the failure mode that matters on a fleet of second-hand
        consoles: a unit dies, the planner reshuffles the experts, and the
        model starts emitting different tokens for the same prompt with no
        error anywhere. Reducing per console would do exactly that, because
        fp32 addition is not associative. Here the same six experts are served
        by three consoles and then by one, and the bits must match.
        """
        x = np.linspace(-2, 2, HIDDEN, dtype=np.float32)
        ids = [3, 0, 5, 1, 4, 2]
        gates = [0.3, 0.25, 0.2, 0.1, 0.1, 0.05]
        spread, _ = self.dispatcher().run_layer(0, x, ids, gates)

        store = ShardStore()
        for expert, weights in self.experts.items():
            store.put(0, expert, weights)
        node = NodeServer(store, unit_id="all", host="127.0.0.1", port=0)
        port = node.start()
        self.addCleanup(node.stop)
        one = ExpertDispatcher(
            {(0, e): ["all"] for e in self.experts},
            {"all": NodeAddress("all", "127.0.0.1", port)})
        self.addCleanup(one.close)

        together, stats = one.run_layer(0, x, ids, gates)
        self.assertEqual(stats.consoles_contacted, 1)
        np.testing.assert_array_equal(spread, together)

    def test_the_reduction_follows_top_k_order_not_expert_id(self):
        """Rank order is the router's order, and it is what gets summed."""
        disp = self.dispatcher()
        x = np.linspace(-1, 1, HIDDEN, dtype=np.float32)
        ids, gates = [5, 1, 3, 0], [0.4, 0.3, 0.2, 0.1]
        got, _ = disp.run_layer(0, x, ids, gates)

        rows = [reference_expert(self.experts[e], x, np.float32(g))
                for e, g in zip(ids, gates)]
        want = rows[0].astype(np.float32).copy()
        for row in rows[1:]:
            want += row
        np.testing.assert_array_equal(got, want)

    def test_fast_mode_is_opt_in_and_re_associates_the_sum(self):
        """FAST is allowed to differ from exact — that is the trade — but it
        must only happen when it was asked for."""
        disp = self.dispatcher()
        x = np.linspace(-3, 3, HIDDEN, dtype=np.float32)
        ids, gates = [0, 1, 2, 3, 4, 5], [0.3, 0.25, 0.2, 0.1, 0.1, 0.05]
        exact, _ = disp.run_layer(0, x, ids, gates)
        fast, _ = disp.run_layer(0, x, ids, gates, fast=True)
        np.testing.assert_allclose(fast, exact, rtol=1e-5, atol=1e-6)

    def test_a_console_that_collapses_without_being_asked_is_refused(self):
        """A stale node answering a v2 coordinator must not be silently
        accepted: its sum is arithmetically unplaceable."""
        node = self.nodes["n0"]
        original = node._on_expert_batch

        def collapse(frame):
            reply = original(frame)
            rows = ExpertRowsPayload.decode(reply.payload).rows
            return Frame(MsgType.EXPERT_RESULT, frame.request_id,
                         encode_vector(accumulate(list(rows))),
                         layer=frame.layer, flags=Flags.PARTIAL)

        node._on_expert_batch = collapse
        try:
            with self.assertRaises(Gen9Error) as caught:
                self.dispatcher(max_retries=0).run_layer(
                    0, np.ones(HIDDEN, dtype=np.float32), [0], [1.0])
        finally:
            node._on_expert_batch = original
        self.assertIn("did not ask for FAST", str(caught.exception))

    def test_routing_all_experts_to_one_console_still_works(self):
        disp = self.dispatcher()
        x = np.ones(HIDDEN, dtype=np.float32)
        got, stats = disp.run_layer(0, x, [0, 1], [0.5, 0.5])
        self.assertEqual(stats.consoles_contacted, 1)
        expected = (reference_expert(self.experts[0], x, 0.5)
                    + reference_expert(self.experts[1], x, 0.5))
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


class TestFailover(ClusterFixture):
    def test_a_dead_console_with_no_replica_fails_the_token(self):
        disp = self.dispatcher(request_timeout=2.0)
        self.nodes["n1"].stop()
        with self.assertRaises(Gen9Error):
            disp.run_layer(0, np.ones(HIDDEN, dtype=np.float32), [2], [1.0])

    def test_a_replica_takes_over(self):
        """Experts are stateless, so a second console holding the same shard is
        a complete substitute — the one kind of failure that can be handled
        inside a token rather than by re-planning."""
        store = ShardStore()
        for expert in (2, 3):
            store.put(0, expert, self.experts[expert])
        replica = NodeServer(store, unit_id="n1-replica", host="127.0.0.1",
                             port=0)
        port = replica.start()
        self.addCleanup(replica.stop)
        self.addresses["n1-replica"] = NodeAddress("n1-replica", "127.0.0.1",
                                                   port)
        for expert in (2, 3):
            self.placement[(0, expert)].append("n1-replica")

        disp = self.dispatcher(request_timeout=2.0, max_retries=2)
        self.nodes["n1"].stop()
        x = np.ones(HIDDEN, dtype=np.float32)
        got, stats = disp.run_layer(0, x, [2, 3], [0.5, 0.5])
        expected = (reference_expert(self.experts[2], x, 0.5)
                    + reference_expert(self.experts[3], x, 0.5))
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
        self.assertGreaterEqual(stats.retries, 1)

    def test_a_console_with_no_address_is_named_in_the_error(self):
        placement = dict(self.placement)
        placement[(0, 7)] = ["n9"]
        disp = ExpertDispatcher(placement, self.addresses)
        self.addCleanup(disp.close)
        with self.assertRaises(ShardMissing) as caught:
            disp.run_layer(0, np.ones(HIDDEN, dtype=np.float32), [7], [1.0])
        self.assertIn("n9", str(caught.exception))


class TestPlacementFromPlan(unittest.TestCase):
    def test_the_index_covers_every_expert_the_plan_placed(self):
        fleet = [ConsoleUnit(f"ps5-{i}", "ps5", Runtime.PS5_LINUX)
                 for i in range(4)]
        plan = plan_split(DEEPSEEK_TINY, fleet, context_tokens=1024)
        placement = placement_from_plan(plan)
        for layer in range(DEEPSEEK_TINY.moe.n_dense_layers,
                           DEEPSEEK_TINY.n_layers):
            for expert in range(DEEPSEEK_TINY.moe.n_routed_experts):
                self.assertIn((layer, expert), placement)
                self.assertTrue(placement[(layer, expert)])


class TestEndToEnd(unittest.TestCase):
    """A planned fleet, served and driven, from plan to hidden state."""

    def setUp(self):
        self.profile = DEEPSEEK_TINY
        fleet = [ConsoleUnit(f"ps5-{i}", "ps5", Runtime.PS5_LINUX)
                 for i in range(3)]
        self.plan = plan_split(self.profile, fleet, context_tokens=512,
                               shelf_size=3)
        self.addresses = {}
        self.stores = {}
        rng = np.random.default_rng(7)
        for unit_id, unit_plan in self.plan.units.items():
            store = ShardStore()
            for shard in unit_plan.shards:
                for expert in shard.expert_ids:
                    store.put(shard.layer, expert,
                              ExpertWeights(
                                  gate=rng.standard_normal(
                                      (INTERMEDIATE, HIDDEN),
                                      dtype=np.float32) * 0.01,
                                  up=rng.standard_normal((INTERMEDIATE, HIDDEN),
                                                         dtype=np.float32)
                                  * 0.01,
                                  down=rng.standard_normal((HIDDEN,
                                                            INTERMEDIATE),
                                                           dtype=np.float32)
                                  * 0.01,
                                  tier=shard.tier))
            self.stores[unit_id] = store
            node = NodeServer(store, unit_id=unit_id, host="127.0.0.1", port=0,
                              sku=unit_plan.sku, backend=unit_plan.backend,
                              block_fn=lambda layer, x: x * 1.01)
            port = node.start()
            self.addCleanup(node.stop)
            self.addresses[unit_id] = NodeAddress(unit_id, "127.0.0.1", port)

    def router(self, layer, state):
        k = self.profile.moe.top_k
        return list(range(k)), [1.0 / k] * k

    def test_a_token_traverses_every_block(self):
        with FleetCoordinator(self.plan, self.addresses,
                              router=self.router) as fleet:
            x = np.ones(HIDDEN, dtype=np.float32)
            out, trace = fleet.forward(x)
            self.assertEqual(out.shape, (HIDDEN,))
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertEqual(len(trace.layers), self.profile.planning_layers)
            self.assertGreater(trace.seconds, 0.0)

    def test_the_trace_identifies_the_slowest_block(self):
        with FleetCoordinator(self.plan, self.addresses,
                              router=self.router) as fleet:
            _, trace = fleet.forward(np.ones(HIDDEN, dtype=np.float32))
            self.assertIsNotNone(trace.slowest)
            self.assertIn("ms/token", trace.summary())

    def test_health_reports_every_console(self):
        with FleetCoordinator(self.plan, self.addresses,
                              router=self.router) as fleet:
            status = fleet.health()
            self.assertEqual(set(status), set(self.addresses))
            for text in status.values():
                self.assertIn("shards=", text)

    def test_repeated_tokens_are_deterministic(self):
        with FleetCoordinator(self.plan, self.addresses,
                              router=self.router) as fleet:
            x = np.full(HIDDEN, 0.5, dtype=np.float32)
            first, _ = fleet.forward(x)
            second, _ = fleet.forward(x)
            np.testing.assert_array_equal(first, second)

    def test_losing_the_shelf_host_is_reported_as_unrecoverable(self):
        """The host holds the KV cache for its layers; failing over would mean
        continuing a different conversation."""
        host = self.plan.stages[0].host_unit
        with FleetCoordinator(self.plan, self.addresses, router=self.router,
                              request_timeout=2.0) as fleet:
            fleet.forward(np.ones(HIDDEN, dtype=np.float32))

            # Point the host at a closed port and drop the live connection:
            # the console is gone as far as the fleet is concerned.
            fleet.pool.drop(host)
            self.addresses[host] = NodeAddress(host, "127.0.0.1",
                                               _closed_port())

            with self.assertRaises(Gen9Error) as caught:
                fleet.forward(np.ones(HIDDEN, dtype=np.float32))
            self.assertIn("restarted", str(caught.exception))
            self.assertEqual(caught.exception.unit_id, host)


if __name__ == "__main__":
    unittest.main()
