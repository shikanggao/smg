# Completed input-work diagnostics

Set `SMG_NATIVE_ENGINE_WORK_DIAGNOSTICS=1` in an already enabled native
deadline-signals process. It is off by default. The authenticated raw native
snapshot adds `deadline_recovery.engine_work_progress`; no predictor, admission
threshold, native deadline, cancellation ACK, or first-nonempty-text rule changes.
The raw capture must preserve this dictionary even if a typed consumer ignores it.

This diagnostic counts completed **scheduled input work** at a host completion
witness. It does not count accepted output tokens or unique useful progress,
measure physical GPU time, or establish how much work remains until visible text.
Unknown no-text decode work stays unknown in the existing native inventory.

## Native completion witnesses

* vLLM `2a69949bdadf0e8942b7a1619b229cb475beef20`: freeze input ranges when
  `Scheduler.schedule` returns, after `_update_after_schedule` has advanced
  `num_computed_tokens`. Keep the exact `SchedulerOutput` strongly until its
  `update_from_output` returns normally. `EngineCore.step` and its batch-queue
  variant call this only after the existing model future/result path. A complete
  request mapping is required; pruned/nonstandard outputs cannot certify work.
* SGLang `c9e5799bd12fd5fefb0fb375ac537a6b7d53892e`: freeze CPU
  `seq_lens_cpu`, matched prefixes and explicit mixed-batch decode membership
  before `run_batch`. Carry the record on the result, because overlap's
  `ScheduleBatch.copy` drops prefix metadata. Commit after normal
  `process_batch_result`, whose pinned output processing already waits for
  completion/copy. No additional device synchronization is introduced.

Mutable request fields at completion cannot redefine a submitted range. Cached
prefixes are excluded. A duplicate result callback credits once; a distinct
completed execution of a range counts again, including after retraction or
preemption. No separate recomputation total is claimed because its lineage can
be ambiguous. Work canceled after submission counts only when the normal
completion witness and required result provenance still exist. Native failures
propagate; diagnostic metadata failures only make counters unavailable.

## Schema version 1

Identity fields are `engine_epoch`, `profile_id`, `dp_rank`, `source_commit`,
`completion_boundary`, `schema_version`. Enabling the diagnostic changes the
profile so its counters cannot be mixed with an earlier profile. Counters reset
with the engine observer epoch. `snapshot_monotonic_secs` is the snapshot's host
clock, and `completion_sequence` counts distinct observed result completions.

`cumulative` contains:

| Field | Meaning |
|---|---|
| `completed_batches` | Recognized valid completion callbacks, including zero-input cleanup callbacks where the engine produces them; not a count of GPU forwards |
| `completed_uncached_prefill_tokens` | Executed logical input positions inside the **original prompt**, excluding the matched/cached prefix |
| `completed_generation_context_tokens` | Executed input positions beyond the original prompt, including decode and replayed generated context; not newly accepted output tokens |
| `completed_generation_context_attention_tokens` | Sum of logical causal context lengths (`input_position + 1`) for generation-context input positions; a work proxy, not physical KV allocation |

`last_completion` is null before the first valid completion. Afterwards it
contains `batch_sequence`, `schedule_observed_monotonic_secs`,
`completion_observed_monotonic_secs`, `host_observation_span_seconds`,
`request_count`, `uncached_prefill_tokens`, `generation_context_tokens`, and
`generation_context_attention_tokens`. Host spans include asynchronous queueing,
overlap and result processing. **Do not sum them as elapsed capacity time.**
Use cumulative work deltas over compatible monotonic snapshot windows. A zero
delta does not establish GPU idle or no in-flight/chunk progress. The last record
drops intermediate batches; it cannot reconstruct a full per-iteration latency
model.

`counters_available` becomes sticky false on unsupported configurations, missing
or malformed metadata, nonstandard outputs or failed execution/result processing.
`unavailable_reason` identifies the first failure. Known partial totals may remain
for audit; they are not complete capacity measurements. Missing, unavailable,
stale, rank/profile/epoch changes and counter resets must not become zero progress.
Speculation, external KV/encoder transfer, multimodal, embedding and
encoder-decoder coordinates are unsupported where exact work semantics are not
established. Existing single-rank standalone native-support gates still apply.

## Per-request observation state

Under the same switch, existing active-request rows add `engine_work_state`.
It is diagnostic only and associated with that row's native request/attempt.
It reports `schema_version`, `diagnostic_only`, `available`,
`unavailable_reason`, `output_token_count`, `retraction_count`,
`preemption_count`, `abort_pending`, `abort_pending_source`, and
`cache_release_observed`. Unsupported or missing values are null, never zero.
Unavailable or faulty state diagnostics do not change the native inventory,
remaining work, first text, cancellation or serving result.

Output count is a CPU-observed sampled-token count, not token content or future
batch placeholders; it may remain unchanged while an old request reference is
retained. SGLang exposes `Req.retraction_count` and a `FINISH_ABORT` value in
`Req.to_finish`; vLLM exposes its preemption count and explicit retained aborted
status / pending native ACK. vLLM reports `abort_pending=null` without such an
explicit witness because an EngineCore abort may not yet have reached Scheduler.
An absent client response or disconnected HTTP client is not an abort witness.
The source label states the engine-specific observation boundary.

Cache release is a **historical callback/observer witness**, not current physical
KV allocation, usable capacity or proof that all native references are gone. In
SGLang its flag can survive resumption/retraction. In vLLM the existing
`_free_blocks` observer retains the request while native ownership still exists;
other free paths such as preemption do not imply this callback was observed.
These diagnostics neither mutate nor reset request fields. Their purpose is to
separate observed token-count growth from pending aborts and retained references,
including long no-text episodes; a flat count is not proof of GPU idle.

## Packaging and validation

vLLM requires both `native_deadline_vllm.py` and the package helper
`smg_grpc_servicer/engine_work_progress.py`. The SGLang patch installer copies the
same helper byte-for-byte as `sglang/srt/managers/smg_engine_work_progress.py` and
includes it in its output hash manifest. An immutable Python overlay must carry
both files and verify their hashes/imports before engine launch.

CPU tests execute exact pinned scheduler advancement, core future completion,
overlap copy and result-processing bodies with controlled native boundary
objects. They cover chunking/cache, in-flight mutation, re-execution, duplicate
callbacks, cancellation, failures, unsupported metadata and no-text decode.
These establish hook arithmetic and source compatibility. Actual GPU execution,
telemetry overhead and predictive benefit still need fresh DEV evidence.
