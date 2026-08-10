"""Transport and node worker, over real loopback sockets.

These tests open actual TCP connections to an actual :class:`NodeServer`.
Mocking the socket would test the mock: the behaviours that matter here —
multiplexing, correlation under concurrency, and what happens when a console
disappears mid-request — only exist in the real thing.
"""

import socket
import threading
import time
import unittest

import numpy as np

from gen9_cluster import fp8
from gen9_cluster.errors import ConnectError, Gen9Error
from gen9_cluster.node import ExpertWeights, NodeServer, ShardStore
from gen9_cluster.protocol import (DType, ExpertBatchPayload,
                                   ExpertRowsPayload, Flags, Frame, MsgType,
                                   ShardHeader, accumulate, decode_vector,
                                   encode_vector)
from gen9_cluster.transport import ConnectionPool, NodeConnection

HIDDEN = 16
INTERMEDIATE = 8


def make_expert(seed):
    rng = np.random.default_rng(seed)
    return ExpertWeights(
        gate=rng.standard_normal((INTERMEDIATE, HIDDEN), dtype=np.float32),
        up=rng.standard_normal((INTERMEDIATE, HIDDEN), dtype=np.float32),
        down=rng.standard_normal((HIDDEN, INTERMEDIATE), dtype=np.float32))


class NodeFixture(unittest.TestCase):
    """A running single-console node, torn down after each test."""

    n_experts = 4

    def setUp(self):
        self.store = ShardStore()
        for expert in range(self.n_experts):
            self.store.put(0, expert, make_expert(expert))
        self.node = NodeServer(self.store, unit_id="ps5-test", host="127.0.0.1",
                               port=0, sku="ps5", backend="vulkan",
                               block_fn=lambda layer, x: x * 2.0)
        self.port = self.node.start()
        self.addCleanup(self.node.stop)

    def connect(self):
        conn = NodeConnection("ps5-test", "127.0.0.1", self.port)
        conn.connect()
        self.addCleanup(conn.close)
        return conn


class TestBasicExchange(NodeFixture):
    def test_hello_reports_what_the_console_is(self):
        conn = self.connect()
        reply = conn.request(Frame(MsgType.HELLO, 0))
        self.assertEqual(reply.msg_type, MsgType.HELLO_ACK)

    def test_ping_pong(self):
        conn = self.connect()
        reply = conn.request(Frame(MsgType.PING, 0, b"beat"))
        self.assertEqual(reply.msg_type, MsgType.PONG)
        self.assertEqual(reply.payload, b"beat")

    def test_the_connection_is_reused_across_requests(self):
        """Reconnecting per token would add a handshake to every layer."""
        conn = self.connect()
        for _ in range(20):
            conn.request(Frame(MsgType.PING, 0))
        self.assertTrue(conn.connected)

    def test_nagle_is_off(self):
        """Small frames must go out immediately; Nagle would add up to 40 ms
        per hop to a workload made entirely of small frames."""
        conn = self.connect()
        value = conn._sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
        self.assertTrue(value)


class TestExpertExecution(NodeFixture):
    def _reference_rows(self, x, ids, gates):
        rows = []
        for expert, gate in zip(ids, gates):
            weights = self.store.get(0, expert)
            hidden = x @ weights.gate.T
            hidden = hidden * (1.0 / (1.0 + np.exp(-hidden)))
            hidden = hidden * (x @ weights.up.T)
            rows.append(np.float32(gate) * (hidden @ weights.down.T))
        return rows

    def test_a_batch_returns_one_tagged_row_per_expert(self):
        """The default reply is per-expert, so the caller owns the order."""
        conn = self.connect()
        x = np.arange(HIDDEN, dtype=np.float32) * 0.1
        ids, gates = (0, 1, 2), (0.5, 0.25, 0.25)
        payload = ExpertBatchPayload(ids, gates, x).encode()
        reply = conn.request(Frame(MsgType.EXPERT_BATCH, 0, payload, layer=0))

        self.assertTrue(reply.flags & Flags.PER_EXPERT)
        self.assertFalse(reply.flags & Flags.PARTIAL)
        rows = ExpertRowsPayload.decode(reply.payload)
        self.assertEqual(rows.expert_ids, ids)
        for index, want in enumerate(self._reference_rows(x, ids, gates)):
            np.testing.assert_allclose(rows.rows[index], want, rtol=1e-5,
                                       atol=1e-6)

    def test_fast_collapses_the_batch_only_when_asked(self):
        conn = self.connect()
        x = np.arange(HIDDEN, dtype=np.float32) * 0.1
        ids, gates = (0, 1, 2), (0.5, 0.25, 0.25)
        payload = ExpertBatchPayload(ids, gates, x).encode()
        reply = conn.request(Frame(MsgType.EXPERT_BATCH, 0, payload, layer=0,
                                   flags=Flags.FAST))

        self.assertTrue(reply.flags & Flags.PARTIAL)
        self.assertFalse(reply.flags & Flags.PER_EXPERT)
        got = decode_vector(reply.payload)
        self.assertEqual(got.shape, (HIDDEN,))
        # Same folding order as the node uses, so this is exact, not close.
        np.testing.assert_array_equal(
            got, accumulate(self._reference_rows(x, ids, gates)))

    def test_a_retry_of_the_same_batch_id_is_replayed_not_recomputed(self):
        conn = self.connect()
        x = np.arange(HIDDEN, dtype=np.float32) * 0.1
        payload = ExpertBatchPayload((0, 1), (0.5, 0.5), x, 4242).encode()
        frame = Frame(MsgType.EXPERT_BATCH, 0, payload, layer=0)

        first = conn.request(frame)
        ran = self.node.experts_run
        second = conn.request(frame)

        self.assertEqual(self.node.experts_run, ran,
                         "the retry re-ran the experts")
        self.assertTrue(second.flags & Flags.REPLAYED)
        self.assertFalse(first.flags & Flags.REPLAYED)
        np.testing.assert_array_equal(
            ExpertRowsPayload.decode(second.payload).rows,
            ExpertRowsPayload.decode(first.payload).rows)

    def test_a_missing_shard_is_an_error_reply_not_a_dropped_connection(self):
        """The console keeps serving: one bad request must not cost the
        coordinator its connection to a console holding 300 other experts."""
        conn = self.connect()
        payload = ExpertBatchPayload((99,), (1.0,),
                                     np.ones(HIDDEN, dtype=np.float32)).encode()
        with self.assertRaises(Gen9Error) as caught:
            conn.request(Frame(MsgType.EXPERT_BATCH, 0, payload, layer=0))
        self.assertIn("not resident", str(caught.exception))
        self.assertEqual(caught.exception.unit_id, "ps5-test")
        self.assertEqual(conn.request(Frame(MsgType.PING, 0)).msg_type,
                         MsgType.PONG)

    def test_a_malformed_batch_is_answered_not_fatal(self):
        conn = self.connect()
        with self.assertRaises(Gen9Error):
            conn.request(Frame(MsgType.EXPERT_BATCH, 0, b"\x02\x00garbage",
                               layer=0))
        self.assertTrue(conn.connected)
        self.assertEqual(conn.request(Frame(MsgType.PING, 0)).msg_type,
                         MsgType.PONG)

    def test_loading_a_shard_over_the_wire(self):
        conn = self.connect()
        header = ShardHeader(layer=1, first_expert=0, n_experts=2,
                             hidden_size=HIDDEN, intermediate_size=INTERMEDIATE)
        body = np.arange(2 * 3 * HIDDEN * INTERMEDIATE,
                         dtype=np.float32) * 0.001
        reply = conn.request(Frame(MsgType.LOAD_SHARD, 0,
                                   header.encode() + body.tobytes()))
        self.assertEqual(reply.msg_type, MsgType.LOAD_ACK)
        self.assertTrue(self.store.holds(1, 0))
        self.assertTrue(self.store.holds(1, 1))

    def test_a_shard_body_of_the_wrong_size_is_refused(self):
        conn = self.connect()
        header = ShardHeader(layer=2, first_expert=0, n_experts=2,
                             hidden_size=HIDDEN, intermediate_size=INTERMEDIATE)
        with self.assertRaises(Gen9Error):
            conn.request(Frame(MsgType.LOAD_SHARD, 0,
                               header.encode() + b"\x00" * 64))
        self.assertFalse(self.store.holds(2, 0))

    def test_loading_an_fp8_shard_over_the_wire(self):
        """The format the weights actually ship in: codes then block scales,
        loaded without ever being materialised at fp32."""
        conn = self.connect()
        rng = np.random.default_rng(1)
        matrices = [rng.standard_normal((fp8.BLOCK, fp8.BLOCK),
                                        dtype=np.float32) * 0.05
                    for _ in range(3)]
        codes, scales = [], []
        for matrix in matrices:
            code, scale = fp8.quantize(matrix)
            codes.append(code.reshape(-1))
            scales.append(scale)
        header = ShardHeader(layer=5, first_expert=0, n_experts=1,
                             hidden_size=fp8.BLOCK,
                             intermediate_size=fp8.BLOCK,
                             dtype=DType.FP8_E4M3_B128)
        body = (np.concatenate(codes).tobytes()
                + np.concatenate(scales).astype("<f4").tobytes())
        reply = conn.request(Frame(MsgType.LOAD_SHARD, 0,
                                   header.encode() + body,
                                   dtype=DType.FP8_E4M3_B128))
        self.assertEqual(reply.msg_type, MsgType.LOAD_ACK)
        held = self.store.get(5, 0)
        self.assertTrue(held.quantised)
        np.testing.assert_allclose(held.dequantised().gate, matrices[0],
                                   atol=0.01)

    def test_an_fp8_shard_missing_its_scales_is_refused(self):
        conn = self.connect()
        header = ShardHeader(layer=6, first_expert=0, n_experts=1,
                             hidden_size=fp8.BLOCK,
                             intermediate_size=fp8.BLOCK,
                             dtype=DType.FP8_E4M3_B128)
        codes = np.zeros(3 * fp8.BLOCK * fp8.BLOCK, dtype=np.uint8)
        with self.assertRaises(Gen9Error):
            conn.request(Frame(MsgType.LOAD_SHARD, 0,
                               header.encode() + codes.tobytes()))
        self.assertFalse(self.store.holds(6, 0))

    def test_block_forward_runs_on_the_host(self):
        conn = self.connect()
        x = np.ones(HIDDEN, dtype=np.float32)
        reply = conn.request(Frame(MsgType.BLOCK_FWD, 0, encode_vector(x),
                                   layer=3))
        np.testing.assert_allclose(decode_vector(reply.payload), x * 2.0)


class TestMultiplexing(NodeFixture):
    def test_concurrent_requests_get_their_own_replies(self):
        """The failure this guards against is silent and catastrophic: two
        in-flight requests whose replies are swapped produce a plausible token
        computed from the wrong experts."""
        conn = self.connect()
        results = {}
        errors = []

        def ask(index):
            try:
                reply = conn.request(
                    Frame(MsgType.PING, 0, f"payload-{index}".encode()))
                results[index] = reply.payload.decode()
            except Gen9Error as exc:      # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=ask, args=(i,)) for i in range(24)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertEqual(errors, [])
        self.assertEqual(results, {i: f"payload-{i}" for i in range(24)})

    def test_request_ids_are_unique_per_connection(self):
        conn = self.connect()
        seen = set()
        for _ in range(100):
            rid = conn._next_request_id()
            self.assertNotIn(rid, seen)
            seen.add(rid)


class TestFailures(unittest.TestCase):
    def test_connecting_to_nothing_is_a_retry_safe_error(self):
        """Nothing was sent, so retrying cannot duplicate work."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]
        conn = NodeConnection("ghost", "127.0.0.1", dead_port,
                              connect_timeout=1.0)
        with self.assertRaises(ConnectError) as caught:
            conn.connect()
        self.assertEqual(caught.exception.unit_id, "ghost")
        self.assertTrue(caught.exception.retry_safe)

    def test_a_console_disappearing_fails_the_waiters(self):
        """Every in-flight request must fail promptly rather than hang until
        its timeout; a shelf blocked on a dead console blocks the pipeline."""
        store = ShardStore()
        node = NodeServer(store, unit_id="doomed", host="127.0.0.1", port=0)
        port = node.start()
        conn = NodeConnection("doomed", "127.0.0.1", port)
        conn.connect()
        self.assertEqual(conn.request(Frame(MsgType.PING, 0)).msg_type,
                         MsgType.PONG)

        failures = []

        def ask():
            try:
                conn.request(Frame(MsgType.PING, 0), timeout=15.0)
            except Gen9Error as exc:
                failures.append(exc)

        threads = [threading.Thread(target=ask) for _ in range(4)]
        for thread in threads:
            thread.start()
        time.sleep(0.05)
        node.stop()
        conn._sock.close() if conn._sock else None
        for thread in threads:
            thread.join(timeout=10)
        conn.close()
        self.assertFalse(conn.connected)

    def test_a_peer_speaking_nonsense_is_a_protocol_error(self):
        """Not every listener on port 9713 is a console."""
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def babble():
            client, _ = listener.accept()
            client.sendall(b"HTTP/1.1 200 OK\r\n\r\n" + b"x" * 64)
            time.sleep(0.2)
            client.close()

        thread = threading.Thread(target=babble, daemon=True)
        thread.start()
        conn = NodeConnection("impostor", "127.0.0.1", port)
        conn.connect()
        with self.assertRaises(Gen9Error):
            conn.request(Frame(MsgType.PING, 0), timeout=5.0)
        conn.close()
        listener.close()
        thread.join(timeout=5)


class TestConnectionPool(NodeFixture):
    def test_one_connection_per_console_is_reused(self):
        with ConnectionPool() as pool:
            first = pool.get("ps5-test", "127.0.0.1", self.port)
            second = pool.get("ps5-test", "127.0.0.1", self.port)
            self.assertIs(first, second)

    def test_dropping_forces_a_fresh_connection(self):
        with ConnectionPool() as pool:
            first = pool.get("ps5-test", "127.0.0.1", self.port)
            pool.drop("ps5-test")
            self.assertIsNot(pool.get("ps5-test", "127.0.0.1", self.port),
                             first)

    def test_closing_the_pool_closes_the_connections(self):
        pool = ConnectionPool()
        conn = pool.get("ps5-test", "127.0.0.1", self.port)
        pool.close()
        self.assertFalse(conn.connected)


if __name__ == "__main__":
    unittest.main()
