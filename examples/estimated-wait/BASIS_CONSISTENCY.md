# Size-basis calibration experiment

`size_basis_consistent_calibration_shadow_two_arm_v2` runs under the existing calibration
and feature shadow flags. The original predictions and admission policy retain
their arithmetic. No accuracy gain is established before a complete capture.

The monitor publishes optional per-rank `prefill_metric_provenance`: raw native
counter and prior counter, exact interval endpoints on the gateway diagnostic
clock, unconverted rate, previous conversion mean, current unshrunk size mean,
explicit observed sample count usable by the queue blend, native size-counter
count, counter epoch/reset, and saturated capacity eligibility. An explicit
usable count of zero identifies a prior-only queue basis. Missing is unknown;
a mean equal to 1024 does not identify the basis.

The router adds configured prior tokens/sample weight, effective prior weight
and blended queue basis, freezing them in `snapshot.size_basis` at admission.
Labels retain that stamp. A verified same-source/profile prior-only to observed
transition resets only the experimental fit and tracked labels. Delayed prior
labels remain available to the original arm and cannot enter the new epoch.
Same observed-basis labels retain their original stamp as counts mature.
Missing provenance does not invent a transition. Source, profile and counter
changes reset the experimental state.

Forecasts use current v3 selected X, preserving source gate, native drain and
dispatch debt. Training retains original admission X, matching the existing
fit. Both snapshots are exported. Known but untrained basis uses raw B+X with
gain 1. Unsupported basis copies the paired current control. Availability and
fallback remain explicit; an unavailable numeric value is not a valid forecast.

The event contract is:

- `basis_decision_admission`, JSON `basis_decision_json`: `{id, tracked,
  started, native_attempt_id, recorded_scalar_fallback, admission}`. Admission
  contains `method`, `basis_epoch`, `basis_reason`,
  `paired_fallback_to_original`, `training_snapshot`, `prediction_snapshot`,
  `paired_control_method`, `paired_control_predictions`, and `predictions`.
  Arrays use `bound_queue` and `bound_early`, in that order. Exact-only
  comparison arms are excluded from this branch; readers must select by arm name.
  Current control is `pending_work_admission.predictions`.
- `basis_decision_label`: frozen label, accepted status, current basis epoch
  and acceptance reason. Rejected old-era labels remain visible.
- `basis_decision_poll`: experimental state, raw poll identity and profile.
- `decision_native_poll`: same-poll per-rank metric provenance and existing
  queue, progress and capacity observations.
- `native_deadline_poll`: `native_inventory_json.wire` is the existing bounded
  complete native frame. Worker, calibration epoch, raw poll start, native poll
  start and receipt time support causal joins. Admissions carry the exact
  upstream attempt ID. Use only frames received before each admission.

Native absolute monotonic clocks remain separate from gateway/controller
clocks. Age and measured transport bounds determine freshness. Identical native
snapshots retain original expiry and do not replay work or ACKs. An idle engine
without a native heartbeat can still age out.

For explicitly configured native TTL, a complete `native_ttft_deadline` SSE
error followed by DONE before first text is a right-censored observation with
measured elapsed time. No rounding to 300 or ACK creation occurs. Fragmented
events work; missing DONE, malformed lookalikes and ordinary interrupted
responses stay unknown. First text stays exact after later terminal events.

The existing `--api-key` accepts `SMG_API_KEY`. Explicit CLI takes precedence;
help hides environment values. A Secret environment reference can supply it
without command arguments. Native TTL remains separate and defaults to none.
