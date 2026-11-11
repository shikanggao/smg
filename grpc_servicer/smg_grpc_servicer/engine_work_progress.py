"""Bounded, opt-in host completion observations; no GPU imports or timing.

Rows are frozen logical input ranges (start, end, original prompt length).
Only the engine adapter can establish the native completion witness. Scheduling
alone never credits work. Re-executed ranges count again on a new completed
batch; this is executed work, not unique useful progress or accepted output.
"""

import math
import time
from dataclasses import dataclass


@dataclass
class WorkBatch:
    sequence: int
    scheduled_at: float
    request_count: int
    prefill: int
    generation: int
    attention: int
    valid: bool
    completed: bool = False


class CompletedWorkProgress:
    def __init__(
        self, engine_epoch, profile_id, dp_rank, source_commit, completion_boundary, supported=True
    ):
        self.identity = dict(
            schema_version=1,
            engine_epoch=engine_epoch,
            profile_id=profile_id,
            dp_rank=dp_rank,
            source_commit=source_commit,
            completion_boundary=completion_boundary,
        )
        self.available = supported
        self.reason = None if supported else "unsupported_engine_configuration"
        self.sequence = 0
        self.completion_sequence = 0
        self.cumulative = dict(
            completed_batches=0,
            completed_uncached_prefill_tokens=0,
            completed_generation_context_tokens=0,
            completed_generation_context_attention_tokens=0,
        )
        self.last = None

    def invalidate(self, reason):
        self.available = False
        self.reason = self.reason or reason

    def begin(self, rows, observed_at=None):
        """Freeze work at submission; do not credit it until native completion."""
        now = time.monotonic() if observed_at is None else observed_at
        self.sequence += 1
        count = prefill = generation = attention = 0
        valid = type(now) in (int, float) and math.isfinite(now) and now >= 0
        if not isinstance(rows, (list, tuple)):
            valid = False
            rows = ()
        for row in rows:
            if (
                not isinstance(row, (list, tuple))
                or len(row) != 3
                or any(type(v) is not int for v in row)
                or not 0 <= row[0] < row[1]
                or row[2] <= 0
            ):
                valid = False
                continue
            start, end, prompt = row
            count += 1
            prompt_end = min(end, prompt)
            prefill += max(0, prompt_end - start)
            generation_start = max(start, prompt)
            n = max(0, end - generation_start)
            generation += n
            # Sum logical causal attention lengths, including the current input
            # position. This is a work proxy, never physical KV occupancy.
            attention += n * (generation_start + 1 + end) // 2
        if not valid:
            self.invalidate("invalid_or_missing_scheduled_input_ranges")
        return WorkBatch(self.sequence, now, count, prefill, generation, attention, valid)

    def complete(self, batch, observed_at=None):
        """Called only after the adapter's existing host completion witness."""
        if batch is None:
            self.invalidate("completion_without_submission_record")
            return
        if batch.completed:
            return
        batch.completed = True
        now = time.monotonic() if observed_at is None else observed_at
        if (
            type(now) not in (int, float)
            or not math.isfinite(now)
            or type(batch.scheduled_at) not in (int, float)
            or not math.isfinite(batch.scheduled_at)
            or now < batch.scheduled_at
        ):
            self.invalidate("invalid_completion_observation_clock")
            return
        self.completion_sequence += 1
        if not batch.valid:
            return
        self.cumulative["completed_batches"] += 1
        self.cumulative["completed_uncached_prefill_tokens"] += batch.prefill
        self.cumulative["completed_generation_context_tokens"] += batch.generation
        self.cumulative["completed_generation_context_attention_tokens"] += batch.attention
        self.last = dict(
            batch_sequence=batch.sequence,
            schedule_observed_monotonic_secs=batch.scheduled_at,
            completion_observed_monotonic_secs=now,
            host_observation_span_seconds=now - batch.scheduled_at,
            request_count=batch.request_count,
            uncached_prefill_tokens=batch.prefill,
            generation_context_tokens=batch.generation,
            generation_context_attention_tokens=batch.attention,
        )

    def snapshot(self, observed_at=None):
        now = time.monotonic() if observed_at is None else observed_at
        if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
            self.invalidate("invalid_snapshot_observation_clock")
            now = time.monotonic()
        return dict(
            **self.identity,
            snapshot_monotonic_secs=now,
            counters_available=self.available,
            unavailable_reason=self.reason,
            completion_sequence=self.completion_sequence,
            cumulative=dict(self.cumulative),
            last_completion=None if self.last is None else dict(self.last),
        )
