"""The dedup cache: a retried batch runs once, and the state stays bounded.

The concurrency tests here use real threads rather than a fake clock, because
the case that matters — a coordinator retrying while the first attempt is still
on the console's GPU — is a race, and a test that cannot lose the race cannot
show that the cache wins it.
"""

import threading
import time
import unittest

import numpy as np

from gen9_cluster.dedup import (DedupCache, DedupCapacityError,
                                MismatchedBatchError, batch_fingerprint)

FP_A = b"a" * 32
FP_B = b"b" * 32


class TestSingleExecution(unittest.TestCase):
    def test_a_repeated_batch_id_runs_once(self):
        cache = DedupCache()
        calls = []

        def compute():
            calls.append(1)
            return b"reply"

        first, replayed = cache.run(7, FP_A, compute)
        second, again = cache.run(7, FP_A, compute)

        self.assertEqual(len(calls), 1)
        self.assertEqual(first, second)
        self.assertFalse(replayed)
        self.assertTrue(again)
        self.assertEqual((cache.hits, cache.misses), (1, 1))

    def test_no_batch_id_means_no_deduplication(self):
        """An unnamed batch has nothing to match on, and must not be matched
        against some other unnamed batch."""
        cache = DedupCache()
        calls = []
        for _ in range(3):
            cache.run(None, FP_A, lambda: (calls.append(1), b"x")[1])
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(cache), 0)

    def test_a_reused_id_with_different_content_is_refused(self):
        """Answering this from cache would be a wrong activation's output
        returned as if it were this one's."""
        cache = DedupCache()
        cache.run(7, FP_A, lambda: b"reply")
        with self.assertRaises(MismatchedBatchError):
            cache.run(7, FP_B, lambda: b"other")

    def test_a_failed_batch_is_not_cached_as_an_answer(self):
        cache = DedupCache()
        with self.assertRaises(ZeroDivisionError):
            cache.run(7, FP_A, lambda: 1 // 0)
        self.assertIsNone(cache.replay(7))
        # And the id is free to be retried properly.
        reply, replayed = cache.run(7, FP_A, lambda: b"second time")
        self.assertEqual(reply, b"second time")
        self.assertFalse(replayed)

    def test_fast_and_exact_are_not_the_same_batch(self):
        """Same experts, same activation, different arithmetic and different
        reply shape: a FAST retry must not be served an exact reply."""
        x = np.arange(8, dtype=np.float32)
        exact = batch_fingerprint(3, [1, 2], [0.5, 0.5], x, False)
        fast = batch_fingerprint(3, [1, 2], [0.5, 0.5], x, True)
        self.assertNotEqual(exact, fast)

    def test_the_fingerprint_covers_everything_that_changes_the_answer(self):
        x = np.arange(8, dtype=np.float32)
        base = batch_fingerprint(3, [1, 2], [0.5, 0.5], x, False)
        self.assertNotEqual(base, batch_fingerprint(4, [1, 2], [0.5, 0.5], x,
                                                    False))
        self.assertNotEqual(base, batch_fingerprint(3, [1, 9], [0.5, 0.5], x,
                                                    False))
        self.assertNotEqual(base, batch_fingerprint(3, [1, 2], [0.6, 0.4], x,
                                                    False))
        self.assertNotEqual(base, batch_fingerprint(3, [1, 2], [0.5, 0.5],
                                                    x + 1, False))
        self.assertEqual(base, batch_fingerprint(3, [1, 2], [0.5, 0.5],
                                                 x.copy(), False))


class TestConcurrentRetries(unittest.TestCase):
    def test_a_retry_waits_for_the_attempt_already_running(self):
        """The expensive case: a coordinator whose timeout fired early. The
        console must not start a second pass over the same weights."""
        cache = DedupCache()
        started = threading.Event()
        release = threading.Event()
        calls = []

        def compute():
            calls.append(1)
            started.set()
            release.wait(5.0)
            return b"reply"

        results = {}

        def attempt(name):
            results[name] = cache.run(7, FP_A, compute)

        first = threading.Thread(target=attempt, args=("first",))
        first.start()
        self.assertTrue(started.wait(5.0))

        second = threading.Thread(target=attempt, args=("second",))
        second.start()
        time.sleep(0.05)                # long enough to have raced, if it did
        self.assertEqual(len(calls), 1)

        release.set()
        first.join(5.0)
        second.join(5.0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(results["first"], (b"reply", False))
        self.assertEqual(results["second"], (b"reply", True))

    def test_waiting_on_one_batch_does_not_stall_the_others(self):
        """A node runs several batches at once. If a waiter held the cache
        lock, one slow expert would serialise the whole console."""
        cache = DedupCache()
        started = threading.Event()
        release = threading.Event()

        def slow():
            started.set()
            release.wait(30.0)
            return b"slow"

        blocked = threading.Thread(target=lambda: cache.run(1, FP_A, slow))
        blocked.start()
        self.assertTrue(started.wait(5.0))
        waiter = threading.Thread(target=lambda: cache.run(1, FP_A, slow))
        waiter.start()

        done = threading.Event()

        def unrelated():
            cache.run(2, FP_B, lambda: b"quick")
            done.set()

        threading.Thread(target=unrelated).start()
        # Short, and unrelated to the slow batch's own timeout: the point is
        # that this must not wait on it at all.
        self.assertTrue(done.wait(2.0), "an unrelated batch was blocked")

        release.set()
        blocked.join(5.0)
        waiter.join(5.0)

    def test_a_waiter_sees_the_failure_rather_than_hanging(self):
        cache = DedupCache()
        started = threading.Event()
        release = threading.Event()

        def compute():
            started.set()
            release.wait(5.0)
            raise ValueError("the shard is missing")

        errors = []

        def attempt():
            try:
                cache.run(7, FP_A, compute)
            except BaseException as exc:        # noqa: BLE001
                errors.append(exc)

        first = threading.Thread(target=attempt)
        first.start()
        self.assertTrue(started.wait(5.0))
        second = threading.Thread(target=attempt)
        second.start()
        release.set()
        first.join(5.0)
        second.join(5.0)

        self.assertEqual(len(errors), 2)
        self.assertTrue(all(isinstance(e, ValueError) for e in errors))


class TestBounds(unittest.TestCase):
    def test_entries_are_capped_and_the_oldest_completed_goes_first(self):
        cache = DedupCache(max_entries=3)
        for batch in range(5):
            cache.run(batch, FP_A, lambda: b"x")
        self.assertEqual(len(cache), 3)
        self.assertIsNone(cache.replay(0))
        self.assertIsNotNone(cache.replay(4))
        self.assertEqual(cache.evictions, 2)

    def test_in_flight_batches_are_never_evicted(self):
        """Evicting a running batch would leave its waiters orphaned and let a
        second attempt start, which is the whole thing this prevents."""
        cache = DedupCache(max_entries=1)
        started = threading.Event()
        release = threading.Event()

        def slow():
            started.set()
            release.wait(5.0)
            return b"slow"

        worker = threading.Thread(target=lambda: cache.run(1, FP_A, slow))
        worker.start()
        self.assertTrue(started.wait(5.0))

        with self.assertRaises(DedupCapacityError):
            cache.run(2, FP_B, lambda: b"fast")
        self.assertEqual(cache.rejected, 1)

        release.set()
        worker.join(5.0)
        self.assertEqual(cache.replay(1), b"slow")

    def test_the_byte_budget_is_enforced(self):
        cache = DedupCache(max_entries=100, max_bytes=300)
        for batch in range(4):
            cache.run(batch, FP_A, lambda: b"x" * 100)
        self.assertLessEqual(cache.bytes_used, 300)
        self.assertIsNone(cache.replay(0))
        self.assertIsNotNone(cache.replay(3))

    def test_a_reply_too_large_to_cache_is_returned_but_not_kept(self):
        cache = DedupCache(max_bytes=10)
        reply, replayed = cache.run(1, FP_A, lambda: b"x" * 100)
        self.assertEqual(len(reply), 100)
        self.assertFalse(replayed)
        self.assertIsNone(cache.replay(1))
        self.assertEqual(cache.oversized, 1)
        self.assertEqual(cache.bytes_used, 0)

    def test_a_reply_older_than_the_ttl_is_dropped(self):
        cache = DedupCache(ttl=0.05)
        cache.run(1, FP_A, lambda: b"x")
        time.sleep(0.1)
        calls = []
        _, replayed = cache.run(1, FP_A, lambda: (calls.append(1), b"x")[1])
        self.assertFalse(replayed, "an expired reply was replayed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.expiries, 1)

    def test_the_byte_count_does_not_drift(self):
        cache = DedupCache(max_entries=4, max_bytes=1000)
        for batch in range(12):
            cache.run(batch, FP_A, lambda: b"x" * 50)
        self.assertEqual(cache.bytes_used, 4 * 50)
        cache.clear()
        self.assertEqual(cache.bytes_used, 0)


if __name__ == "__main__":
    unittest.main()
