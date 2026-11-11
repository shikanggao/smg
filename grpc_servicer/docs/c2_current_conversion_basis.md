# Experimental C2 request-start conversion basis

This research candidate changes only an already selected C2 raw fallback's
divisor. It expresses the existing historical request-start rates in the single
previous-conversion mean known at that raw poll, then keeps the existing lower
quartile and configured throughput floor. Request starts are a work proxy, not
completed engine service. This candidate has no new public configuration or
protocol fields, and does not change shared monitor capacity, queue work,
calibration/own-fit learning, native guards, first-visible-text semantics or
admission enforcement.

The private state retains at most 32 positive eligible intervals plus one
counter/time anchor and one latest poll. A valid zero or ineligible interval
advances continuity without adding or aging a sample. Interval duration, rate,
counter/time adjacency, source identity, published sample count and the original
converted lower quartile must agree. An explicit source-zero window establishes
an empty history; late enable, missed/conflicting intervals and ambiguous resets
stay unavailable until 32 fully witnessed replacement samples or a proven empty
source window. A C2 fit/basis reset or queue drain does not reset this history.

`CalibrationForecast` freezes work, original divisor, observed-delay floor,
measured-only branch, raw poll start and source key. Selection requires that exact
poll/source; a later poll cannot enrich an earlier forecast. Unavailable history
retains the unchanged v1 outer fallback and records the current-basis sub-arm's
unavailability. Native-supported and LocalCompute predictions bypass rebasing.

## Diagnostic ABI

The outer method is
`stable_work_native_service_composition_C2_raw_fallback_current_basis_v2`.
An old v1 adapter must reject this version rather than interpret a recomputed
extra as published raw extra. The version label changes for all C2 admissions;
Qwen/native parity means unchanged prediction values, selected arithmetic,
native state and learning, apart from this version metadata.

On selected StartedRequests fallbacks, `projection.selected_raw_fallback` retains
baseline/source/extra/total and adds `capacity_basis`:

- `method = started_requests_current_conversion_basis_v1`;
- `selected_composition` is `recomputed_raw_work_over_current_conversion_basis`
  when available, or `unchanged_v1_raw_fallback` when unavailable;
- `capacity` reports status/static reason, captured raw poll/source, completeness,
  sample count, current prior conversion mean, request-start lower quartile,
  rebased learned/used capacity and actual configured floor;
- original raw extra/divisor, frozen token work, observed-delay floor and
  measured-only branch remain explicit.

A new adapter must dispatch on both versions and reject unknown or contradictory
compositions. Validate the original extra as `measured_only ? floor :
max(work/original_divisor, floor)`. For an available sub-arm, validate the divisor
as `max(request_start_quantile * current_prior_mean, configured_floor)`, recompute
the same extra branch and add the same current calibration baseline. For an
unavailable sub-arm, require unchanged original extra/total and an explicit
unavailable status. Do not infer zero service, fill gaps, or silently treat the
recomputed extra as a gateway-published raw extra. Ordinary native projections
and LocalCompute fallbacks keep their original numeric compositions; the outer
v2 version still requires explicit adapter support.

## Local verification

Synthetic unit tests cover window startup/recovery, zero/ineligible intervals,
adjacency/rate/duration faults, reset/identity transitions, exact-poll selection,
floor branches and state equivalence. A separately ignored bounded replay test
accepts an explicitly supplied local numeric fixture and expected SHA256 through
environment variables. Full capture-derived fixtures, source mappings and
attestation hashes remain in local research artifacts, not in this repository.
No test fixture contains prompt or response text.

Compile and run the ordinary tests after the coordinated build hold is released.
Run the ignored replay separately with its source-bound fixture receipt; a skipped
replay is not a parity pass. CPU replay does not qualify live engine integration,
GPU overhead, future accuracy or production admission behavior.
