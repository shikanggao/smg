"""Counter/clock contract tests; native completion must be supplied by adapters."""
import unittest

from smg_grpc_servicer.engine_work_progress import CompletedWorkProgress


class WorkProgressTests(unittest.TestCase):
    def recorder(self):
        return CompletedWorkProgress("epoch", "profile", 0, "pin", "native_result_return")

    def test_cache_excluded_chunk_overlap_and_distinct_reexecution(self):
        p = self.recorder()
        first = p.begin([(40, 80, 100)], 1.0)
        second = p.begin([(80, 100, 100), (100, 101, 100)], 1.1)
        self.assertEqual(p.snapshot(1.2)["cumulative"]["completed_uncached_prefill_tokens"], 0)
        p.complete(first, 2.0)
        p.complete(first, 2.1)
        p.complete(second, 2.2)
        # A NEW native execution may recompute the same prompt range. It is
        # actual executed input work again, not duplicate useful progress.
        replay = p.begin([(40, 80, 100)], 3.0)
        p.complete(replay, 4.0)
        out = p.snapshot(4.1)
        self.assertEqual(out["cumulative"], dict(
            completed_batches=3, completed_uncached_prefill_tokens=100,
            completed_generation_context_tokens=1, completed_generation_context_attention_tokens=101))
        self.assertEqual(out["completion_sequence"], 3)
        self.assertEqual(out["last_completion"]["host_observation_span_seconds"], 1.0)

    def test_original_prompt_split_in_replayed_generation_context(self):
        p = self.recorder()
        p.complete(p.begin([(98, 103, 100)], 1.0), 2.0)
        self.assertEqual(p.cumulative["completed_uncached_prefill_tokens"], 2)
        self.assertEqual(p.cumulative["completed_generation_context_tokens"], 3)
        self.assertEqual(p.cumulative["completed_generation_context_attention_tokens"], 101 + 102 + 103)

    def test_malformed_metadata_is_sticky_unavailable_never_zero_service(self):
        for rows in [None, [None], [()], [(1, 1, 100)], [(0, 2, None)], [(True, 2, 100)]]:
            p = self.recorder()
            p.complete(p.begin(rows, 1.0), 2.0)
            p.complete(p.begin([(0, 10, 100)], 3.0), 4.0)
            self.assertFalse(p.snapshot(4.1)["counters_available"])
            self.assertEqual(p.cumulative["completed_uncached_prefill_tokens"], 10)

    def test_completion_clock_or_missing_submission_is_unavailable(self):
        for when in [None, float("nan"), True, -1.0]:
            p = self.recorder()
            batch = p.begin([(0, 10, 100)], 1.0)
            p.complete(batch, when if when is not None else .5)
            self.assertFalse(p.available)
            self.assertEqual(p.cumulative["completed_batches"], 0)
        p = self.recorder()
        p.complete(None, 2.0)
        self.assertFalse(p.available)


if __name__ == "__main__":
    unittest.main()
