"""G9XC framing: round trips, and the malformed frames a network will produce."""

import struct
import unittest

import numpy as np

from gen9_cluster.protocol import (HEADER_SIZE, MAX_PAYLOAD, VERSION, _U16,
                                   _U32, DType, ExpertBatchPayload,
                                   ExpertRowsPayload, Flags, Frame,
                                   HelloPayload, MsgType, ShardHeader,
                                   accumulate, decode_error, decode_vector,
                                   encode_error, encode_vector)


class TestFrame(unittest.TestCase):
    def test_header_is_fixed_width(self):
        """A fixed header is what lets the reader do exactly two recvs per
        frame instead of parsing a stream."""
        self.assertEqual(HEADER_SIZE, 32)
        raw = Frame(MsgType.PING, 1).encode()
        self.assertEqual(len(raw), HEADER_SIZE)

    def test_round_trip_preserves_every_field(self):
        frame = Frame(MsgType.EXPERT_BATCH, request_id=0xDEADBEEF,
                      payload=b"body", layer=37, expert=291, token=1234,
                      dtype=DType.FP8_E4M3_B128, flags=Flags.FROM_STORAGE,
                      rank=5)
        decoded, length = Frame.decode_header(frame.encode()[:HEADER_SIZE])
        self.assertEqual(length, 4)
        self.assertEqual(decoded.msg_type, MsgType.EXPERT_BATCH)
        self.assertEqual(decoded.request_id, 0xDEADBEEF)
        self.assertEqual(decoded.layer, 37)
        self.assertEqual(decoded.expert, 291)
        self.assertEqual(decoded.token, 1234)
        self.assertEqual(decoded.dtype, DType.FP8_E4M3_B128)
        self.assertEqual(decoded.flags, Flags.FROM_STORAGE)
        self.assertEqual(decoded.rank, 5)

    def test_request_ids_survive_the_full_32_bit_range(self):
        """Correlation breaks silently if the id is truncated, and a truncated
        id means a reply delivered to the wrong caller."""
        for rid in (0, 1, 0x7FFFFFFF, 0xFFFFFFFF):
            decoded, _ = Frame.decode_header(
                Frame(MsgType.PONG, rid).encode()[:HEADER_SIZE])
            self.assertEqual(decoded.request_id, rid)

    def test_a_wrong_magic_is_rejected(self):
        raw = bytearray(Frame(MsgType.PING, 1).encode())
        raw[0:4] = b"XXXX"
        with self.assertRaises(ValueError):
            Frame.decode_header(bytes(raw))

    def test_a_wrong_version_is_rejected(self):
        raw = bytearray(Frame(MsgType.PING, 1).encode())
        raw[4] = VERSION + 7
        with self.assertRaises(ValueError):
            Frame.decode_header(bytes(raw))

    def test_a_short_header_is_rejected(self):
        with self.assertRaises(ValueError):
            Frame.decode_header(Frame(MsgType.PING, 1).encode()[:20])

    def test_an_oversized_payload_is_rejected_rather_than_allocated(self):
        """Trusting a length field is how a peer turns a bad frame into an OOM
        on every console at once."""
        raw = bytearray(Frame(MsgType.PING, 1).encode())
        struct.pack_into("<I", raw, 24, MAX_PAYLOAD + 1)
        with self.assertRaises(ValueError):
            Frame.decode_header(bytes(raw))

    def test_an_unknown_message_type_is_rejected(self):
        raw = bytearray(Frame(MsgType.PING, 1).encode())
        raw[5] = 200
        with self.assertRaises(ValueError):
            Frame.decode_header(bytes(raw))


class TestVectors(unittest.TestCase):
    def test_activation_round_trip(self):
        vec = np.arange(128, dtype=np.float32) * 0.25
        np.testing.assert_array_equal(decode_vector(encode_vector(vec)), vec)

    def test_non_float32_input_is_converted(self):
        vec = np.arange(8, dtype=np.float64)
        np.testing.assert_allclose(decode_vector(encode_vector(vec)), vec)


class TestExpertBatch(unittest.TestCase):
    def test_round_trip(self):
        payload = ExpertBatchPayload(
            expert_ids=(3, 17, 256, 4095),
            gates=(0.5, 0.25, 0.125, 0.125),
            activation=np.linspace(-1, 1, 64, dtype=np.float32))
        decoded = ExpertBatchPayload.decode(payload.encode())
        self.assertEqual(decoded.expert_ids, payload.expert_ids)
        np.testing.assert_allclose(decoded.gates, payload.gates, rtol=1e-6)
        np.testing.assert_allclose(decoded.activation, payload.activation)

    def test_one_message_carries_the_whole_top_k(self):
        """Eight experts on one console is one round trip, not eight."""
        payload = ExpertBatchPayload(tuple(range(8)), (0.125,) * 8,
                                     np.zeros(7168, dtype=np.float32))
        decoded = ExpertBatchPayload.decode(payload.encode())
        self.assertEqual(len(decoded.expert_ids), 8)

    def test_truncated_payloads_are_rejected(self):
        raw = ExpertBatchPayload((1, 2), (0.5, 0.5),
                                 np.zeros(16, dtype=np.float32)).encode()
        for cut in (4, len(raw) // 2, len(raw) - 4):
            with self.assertRaises(ValueError):
                ExpertBatchPayload.decode(raw[:cut])

    def test_mismatched_gates_are_refused_at_construction(self):
        with self.assertRaises(ValueError):
            ExpertBatchPayload((1, 2, 3), (0.5, 0.5),
                               np.zeros(4, dtype=np.float32)).encode()

    def test_the_batch_id_survives_the_round_trip(self):
        x = np.zeros(8, dtype=np.float32)
        with_id = ExpertBatchPayload((1,), (1.0,), x, 2 ** 63 + 5)
        self.assertEqual(ExpertBatchPayload.decode(with_id.encode()).batch_id,
                         2 ** 63 + 5)
        without = ExpertBatchPayload((1,), (1.0,), x)
        self.assertIsNone(ExpertBatchPayload.decode(without.encode()).batch_id)

    def test_a_batch_id_that_is_not_a_u64_is_refused(self):
        x = np.zeros(4, dtype=np.float32)
        for bad in (-1, 2 ** 64):
            with self.assertRaises(ValueError):
                ExpertBatchPayload((1,), (1.0,), x, bad).encode()

    def test_a_batch_truncated_inside_its_id_is_rejected(self):
        raw = ExpertBatchPayload((1,), (1.0,), np.zeros(4, dtype=np.float32),
                                 99).encode()
        with self.assertRaises(ValueError):
            ExpertBatchPayload.decode(raw[:11])


class TestExpertRows(unittest.TestCase):
    def test_round_trip(self):
        rows = np.arange(12, dtype=np.float32).reshape(3, 4)
        payload = ExpertRowsPayload((7, 8, 9), rows)
        decoded = ExpertRowsPayload.decode(payload.encode())
        self.assertEqual(decoded.expert_ids, (7, 8, 9))
        np.testing.assert_array_equal(decoded.rows, rows)

    def test_a_row_stays_with_its_expert(self):
        """The tag is the whole point: the coordinator reorders these, so a
        row that loses its expert id becomes a term in the wrong place."""
        rows = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        decoded = ExpertRowsPayload.decode(
            ExpertRowsPayload((41, 5), rows).encode())
        by_expert = dict(zip(decoded.expert_ids, decoded.rows))
        np.testing.assert_array_equal(by_expert[41], [1.0, 1.0])
        np.testing.assert_array_equal(by_expert[5], [2.0, 2.0])

    def test_truncated_payloads_are_rejected(self):
        raw = ExpertRowsPayload(
            (1, 2), np.zeros((2, 8), dtype=np.float32)).encode()
        for cut in (2, len(raw) // 2, len(raw) - 4):
            with self.assertRaises(ValueError):
                ExpertRowsPayload.decode(raw[:cut])

    def test_trailing_bytes_are_rejected(self):
        raw = ExpertRowsPayload(
            (1,), np.zeros((1, 4), dtype=np.float32)).encode()
        with self.assertRaises(ValueError):
            ExpertRowsPayload.decode(raw + b"\x00\x00\x00\x00")

    def test_ids_and_rows_must_agree(self):
        with self.assertRaises(ValueError):
            ExpertRowsPayload((1, 2), np.zeros((3, 4), dtype=np.float32))

    def test_an_empty_reply_is_rejected(self):
        """A batch always names at least one expert, so no rows means the
        reply is malformed rather than legitimately empty."""
        with self.assertRaises(ValueError):
            ExpertRowsPayload.decode(_U16.pack(0) + _U32.pack(4))


class TestAccumulate(unittest.TestCase):
    def test_it_folds_left_to_right(self):
        """Not np.sum: numpy sums pairwise, which is a different number."""
        rows = [np.float32([1.0]), np.float32([1e8]), np.float32([-1e8])]
        # 1 + 1e8 rounds the 1 away, and it never comes back.
        np.testing.assert_array_equal(accumulate(rows), np.float32([0.0]))
        np.testing.assert_array_equal(
            accumulate(list(reversed(rows))), np.float32([1.0]))

    def test_it_does_not_alias_its_first_row(self):
        first = np.ones(4, dtype=np.float32)
        accumulate([first, np.ones(4, dtype=np.float32)])
        np.testing.assert_array_equal(first, np.ones(4, dtype=np.float32))

    def test_nothing_to_add_is_an_error_not_a_zero(self):
        with self.assertRaises(ValueError):
            accumulate([])


class TestShardHeader(unittest.TestCase):
    def test_fp8_shard_round_trip(self):
        header = ShardHeader(layer=12, first_expert=64, n_experts=8,
                             hidden_size=7168, intermediate_size=2048,
                             dtype=DType.FP8_E4M3_B128, tier="ssd")
        decoded, offset = ShardHeader.decode(header.encode())
        self.assertEqual(offset, len(header.encode()))
        self.assertEqual(decoded.layer, 12)
        self.assertEqual(decoded.first_expert, 64)
        self.assertEqual(decoded.n_experts, 8)
        self.assertEqual(decoded.hidden_size, 7168)
        self.assertEqual(decoded.intermediate_size, 2048)
        self.assertEqual(decoded.dtype, DType.FP8_E4M3_B128)
        self.assertEqual(decoded.tier, "ssd")

    def test_the_body_follows_the_header(self):
        header = ShardHeader(layer=0, first_expert=0, n_experts=1,
                             hidden_size=8, intermediate_size=4)
        blob = header.encode() + b"weights"
        _, offset = ShardHeader.decode(blob)
        self.assertEqual(blob[offset:], b"weights")


class TestHello(unittest.TestCase):
    def test_round_trip(self):
        hello = HelloPayload(unit_id="bc-250-07", sku="bc-250", backend="rocm",
                             runtime="salvage-linux",
                             weight_bytes=14 * 1024 ** 3,
                             fast_bytes=13 * 1024 ** 3, gemv_gflops=940.5)
        decoded = HelloPayload.decode(hello.encode())
        self.assertEqual(decoded.unit_id, "bc-250-07")
        self.assertEqual(decoded.sku, "bc-250")
        self.assertEqual(decoded.backend, "rocm")
        self.assertEqual(decoded.runtime, "salvage-linux")
        self.assertEqual(decoded.weight_bytes, 14 * 1024 ** 3)
        self.assertAlmostEqual(decoded.gemv_gflops, 940.5, places=1)


class TestErrors(unittest.TestCase):
    def test_round_trip(self):
        self.assertEqual(decode_error(encode_error("shard 3/17 missing")),
                         "shard 3/17 missing")

    def test_non_ascii_survives(self):
        self.assertEqual(decode_error(encode_error("düşük bellek")),
                         "düşük bellek")


if __name__ == "__main__":
    unittest.main()
