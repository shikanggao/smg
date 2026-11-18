# Completed input-work diagnostics

Set `SMG_NATIVE_ENGINE_WORK_DIAGNOSTICS=1` in an already enabled native
deadline-signals process. It is off by default. The authenticated raw native
snapshot adds `deadline_recovery.engine_work_progress`. The router uses compatible
cumulative uncached-prefill work deltas to estimate prefill service capacity.
This does not change native deadlines, cancellation ACKs, or the first-nonempty-text
rule. Router-side research capture and response hashing are not required.

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

The recorder emits cumulative work and completion sequence only. Service is
calculated from compatible monotonic snapshot intervals; no last-batch timing or
per-request debug records are collected. A zero delta does not establish GPU
idle or absence of in-flight progress. Generation-context counters participate
only in snapshot consistency checks, not the prefill-rate divisor.

`counters_available` becomes sticky false on unsupported configurations, missing
or malformed metadata, nonstandard outputs or failed execution/result processing.
`unavailable_reason` identifies the first failure. Known partial totals may remain
for audit; they are not complete capacity measurements. Missing, unavailable,
stale, rank/profile/epoch changes and counter resets must not become zero progress.
Speculation, external KV/encoder transfer, multimodal, embedding and
encoder-decoder coordinates are unsupported where exact work semantics are not
established. Existing single-rank standalone native-support gates still apply.

## Packaging and validation

vLLM requires both `native_deadline_vllm.py` and the package helper
`smg_grpc_servicer/engine_work_progress.py`. SGLang does not install this recorder: its selected estimator uses the original-prefill
request witness and qualified started-request rates. See its native-patch README.
An immutable vLLM overlay must carry both files and verify their hashes/imports
before engine launch.

CPU tests execute exact pinned scheduler advancement, core future completion,
result-processing bodies with controlled native boundary
objects. They cover chunking/cache, in-flight mutation, re-execution, duplicate
callbacks, cancellation, failures, unsupported metadata and no-text decode.
These establish hook arithmetic and source compatibility. Actual GPU execution,
telemetry overhead and predictive benefit still need fresh DEV evidence.
