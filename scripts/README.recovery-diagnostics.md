# Recovery diagnostic capture

This instrumentation observes the current admission policy without changing
enforcement. Calibration fitting, history expiry, readiness, and the 0.5/second
gain movement cap are unchanged. Enable the existing `smg::estimated_wait::observation=debug`
tracing target for a bounded experiment. No prompt, response text, credential,
per-token event, or per-request metric label is added by the router diagnostics.

## Router JSON events

The existing `calibration_update`, `calibration_forecast` and
`calibration_feedback` schemas remain compatible. Forecast and feedback events
also include `diagnostic_time_seconds`.

`diagnostic_time_seconds` uses one monotonic clock per router process, shared by
workers and calibration epochs. Its origin is the first diagnostic clock use;
an earlier poll start can have a small negative value. Do not compare this clock
directly with a client or another router process. Match owned request IDs and
existing timestamp fields to align captures.

Each load publication emits two `calibration_poll` records, one per calibration
state (`capacity_unready` / `capacity_ready`), including successful publications
with unchanged feedback history. Join them by worker and
`poll_started_time_seconds`; do not count them as two separate native scrapes.

| Field | Meaning |
| --- | --- |
| `outcome` | `success`, `failed` (no load), `invalid_load`, `stale`, `duplicate`, `out_of_order`, `generation_changed`, `worker_removed`, or `untracked` |
| `fit_executed` | Whether this publication called the fitter and changed its fitted revision for this state |
| `poll_started_time_seconds`, `poll_age_seconds` | Load poll start on the shared clock, and elapsed time to diagnostic logging |
| `calibration_epoch`, `calibration_time_seconds`, `calibration_bank_index`, `bank_age_seconds` | Epoch and bank alignment |
| `sample_count`, `observation_sequence`, `fit_sequence`, `history_revision`, `fitted_revision` | Exact state after the existing poll fit/rotation checks |
| `scale`, `target_scale` | Applied gain at diagnostic time and current target |
| `last_accepted_feedback_age_seconds`, `last_accepted_admission_age_seconds` | Ages of the most recently accepted feedback and its saved admission; not the oldest retained sample |
| `native_waiting`, `native_running`, `pending_engine` | Prepared report inputs, before publication applies local observation corrections |
| `pending_first_text`, `total_dispatched` | Current router-local counts |

Unknown optional values are omitted by tracing; absence means unavailable, not
zero. Last-accepted ages can remain after samples expire; inspect `sample_count`
and reconstruct retained sample ages from accepted feedback, bank rotations and
sample caps. The outcome label is diagnostic. Duplicate and older publications are ignored
before replacing a newer snapshot; failed or stale new polls invalidate recovery
evidence.

`request_transport` records bounded per-request transitions with worker,
`request_id`, `request_elapsed_seconds`, and the shared diagnostic clock:

- `upstream_send_started`, `upstream_headers_received`, `upstream_send_failed`.
- `downstream_receiver_closed`: the response receiver closed or a send failed.
- `upstream_eof`, `upstream_stream_error`.
- `upstream_stream_released`: the local reqwest response stream was dropped.
- `observer_dropped`: the TTFT observer was released; this can also occur while
  waiting for response headers and does not by itself prove client disconnect.

The response relay retains request identity after first text. A local stream
release does **not** prove a native scheduler cancellation acknowledgement or a
completed TCP shutdown. Router records therefore say
`engine_cancellation_acknowledgement="unavailable"` and
`engine_ack_observable=false`. Whole-capture isolation audits include these
request events, including events after first text.

## Client and engine evidence

The mixed-workload driver writes `client-events.jsonl` live, containing only
owned IDs, phases and client monotonic timestamps. Phases are `request_started`,
`response_headers_received`, `first_text`, `cancellation_delivered`,
`intentional_response_release` (legacy abort-after-text mode only), and
`request_context_exited`. Cancellation delivery is captured before aiohttp
response-context cleanup when headers have arrived; a second observer covers
cancellation while waiting for headers. It is not a scheduler acknowledgement
or the exact instant the event loop's timeout callback executed.

`requests.jsonl` adds the scheduled TTFT and observation deadline times,
cancellation delivery/context exit times and reason. The 300-second TTFT
threshold remains an evaluation threshold; the configured observation horizon
still determines actual client cancellation. `engine_cancellation_acknowledged`
is null. Normal completions have no cancellation time.

The driver sends `X-Request-ID` equal to the existing owned `opc-request-id`;
the router forwards this allowlisted header. The driver saves the first SSE
response `id` and the returned `x-request-id` (or `opc-request-id`) header,
limited to 256 characters each, when present. Backend request-ID correlation
must be verified against installed engine behavior; a returned ID alone does
not prove native cancellation.

The existing metrics sampler continues throughout pressure, recovery and drain.
In addition to raw Prometheus and waiting/running/KV, it extracts native
completion, prompt-token, generation-token, first-token and preemption counters.
Aliases use the first available metric family, not a sum across aliases. Missing
families remain null. These engine-wide counters show progress and must not be
reported as owned-request cancellation acknowledgements. Preserve each raw
scrape, sampling interval, scrape errors and backend identity.

For backend logs, capture only existing lifecycle events with request ID,
source timestamp, matched event kind and declared scope. Do not enable prompt
logging or change backend configuration. For example, a native frontend log
emitted after submitting an abort to the engine proves abort submission at
that layer, not scheduler completion. If no acknowledgement is exposed, record
that limitation and use queue/token progress only as aggregate evidence.


## Live recovery candidate (`live_arrival_bounded_depletion_v2`)

Calibration shadow also computes a separate, untrained recovery comparator from
fresh native waiting (`W`) and running (`R`) gauges. Let `P=W+R`. Over both a
10-second and a 30-second window, require a strict decline in **both** `W` and `P`,
at least five polls, and no unsupported intermediate rebound. Flat individual polls are
allowed because request counts are discrete, but a flat short window is not.
A positive per-poll change in either `W` or `P` is tolerated only when each
increase is at most `min(1, new_request_credits)`. Credits count router admissions,
not prompt tokens, and are captured atomically at poll start. Each interval uses
only its watermark delta: old, duplicate, future or previously consumed credits
cannot fund another bump. Counter rollback clears history; saturation supplies
no new allowance. This bounded continuity rule does not prove native acceptance
or attribute a native arrival to an owned request. External arrivals and delayed
native acceptance can still reset history. It is deliberately conservative.

Use the smaller of the two **unadjusted** net population-depletion rates `r`;
never add arrivals to this rate or subtract them from native work:

```
Nlive = min(Nqueue_without_dispatch_debt, P/r) + dispatch_debt
Tlive = B + Nlive
Tlegacy_gain_ablation = B + g*Nlive
```

The existing `g` still trains only on the original admission-time queue estimate.
Applying it to `Nlive` is an **untrained component ablation**, not a separately
calibrated model. It retains the existing gain trajectory and 0.5/second cap.
No candidate observation changes the original fitter or raw overload bit.

`P/r` extrapolates aggregate population clearance assuming the observed decline
continues. It is not native remaining-token work, a guaranteed TTFT upper bound,
or measured service throughput. Native departures include completions and
cancellations; local client cancellation is never subtracted from native work.
A sustained cancellation burst can qualify even if service remains slow.
Request-specific native cancellation acknowledgement remains unavailable.

History is bounded to 121 polls / 120 seconds. Missing, malformed, negative,
fractional or overflowing count gauges invalidate the aggregate, including one
bad series among otherwise valid ranks. Source-series/rank profile changes,
detected native counter resets, capacity-profile resets, fresh failed/stale polls,
poll gaps beyond the configured snapshot age, unsupported rebounds and zero population clear
history. Worker invalidation clears evidence. Duplicate and older publications
cannot replace or invalidate newer evidence. Unobservable native restarts or
model changes cannot be detected magically; the observed signals bound this claim.

Every admission diagnostic `recovery_forecast` includes request ID, worker and
shared monotonic time; original and candidate queue extra delay (both include
dispatch debt), `dispatch_debt_seconds`, `native_population`, `native_waiting`,
net depletion rate, window endpoints/count, opaque source profile, applied
`legacy_calibration_scale`, calibration epoch/history revision/fit sequence, and
baseline provenance from the companion original forecast. `available=false`
means no recovery evidence; the numeric candidate fields then explicitly fall
back to the original queue-only estimate. Missing baseline leaves total fields
unknown. Do not score unavailable candidates as successful recovery predictions.

`would_reject` and `legacy_gain_ablation_would_reject` compare **extra delay**
against the configured budget and are absent when recovery evidence or native
capacity readiness is missing. The `total_would_reject` pair compares **total
TTFT** against that numeric threshold for analysis and requires a known baseline.
These are distinct scorecards. `recovery_poll` adds the once-per-publication
state, exact poll times and opaque profile needed to reconstruct eligibility;
original `calibration_poll` events remain available for failed/stale/ignored polls.
Both recovery events expose `poll_request_watermark`, current
`total_dispatched_requests`, optional `requests_since_previous_poll`,
`rebound_allowance_requests` and `rebound_retained`. The existing
`total_dispatched` counts tokens and must never be used as an arrival count.
Request counters are worker-generation scoped, reset on invalidation/replacement,
and saturate without wrapping. Diagnostic comparisons may replay strict v1 on
the same polls; v1 is not a second runtime policy.


Recovery estimates are shadow diagnostics only. No admission bypass or probe
exception exists. Customer requests use the unchanged normal admission path.

### Experimental feature-conditioned gain (shadow only)

`--estimated-wait-feature-shadow` requires `--estimated-wait-calibration-shadow` and is off by default. Method `feature_context_deadline_ridge_v2` leaves the original scalar gain, recovery candidate, and all admission enforcement unchanged. It predicts `B + g_context * N` using the original queue-only `N`; it does not use the observed-delay floor or recovery `N`. When `N=0`, the gain cannot invent an additive KV delay.

The features, in fixed order, are native KV usage, deadline-or-actual-cutoff rate, and actual rejection rate. KV needs explicit valid provenance; all selected native series must be finite in `[0,1]`, and all ranks must be present and unique. Backend aggregation remains vLLM maximum / SGLang mean, followed by mean across ranks. Defaulted or malformed KV is unknown. Metric/rank profile changes invalidate only the feature history; original scalar learning remains unchanged.

Rates use the current one-second bucket and the preceding 59 buckets. The v2 second dimension is `deadline_or_cutoff_adverse / (deadline_timely_first_text + deadline_or_cutoff_adverse)`. A request resolves once: first text before 300 seconds is timely; no observed first text at or after 300 seconds is a deadline breach and adverse; an actual timeout/pretext disconnect before that boundary is adverse without claiming a deadline breach; an earlier unknown observation ending is unknown. Late first text, later actual cutoff, or unknown ending after a breach cannot reclassify or double-count it. Actual terminal counters remain separately recorded and do not supply fabricated exact TTFT or native-cancellation acknowledgements. In particular, a client 300-second cancellation arriving at server age 299.99 counts as actual cutoff, not breach.

Pending deadlines use the existing server TTFT start, before upstream send on the supported typed path. The existing worker poll resolves due requests even without a first-text callback; terminals also check elapsed time. Poll observation can lag 300 by the polling interval. An atomic scan claim prevents a poll from overtaking timely text while its terminal callback waits for the ledger lock. Metadata/no-text scans release the claim. Storage is bounded to 8192 pending observations per worker. Overflow remains terminal-observable but lacks a poll deadline; `deadline_untracked_pending` explicitly suppresses dimension 2 until overflow observations resolve. Worker/calibration epoch resets clear pending storage and counters; old callbacks are ignored. KV-profile-only changes clear learning contexts while preserving pending deadline observations and rates.

The union rate requires at least 16 known resolutions, strictly less than 10% union-unknown observations, no recent `untracked_admitted`, and no outstanding `deadline_untracked_pending`. The guard describes tracked resolved observations rather than all arrivals. An admission without a saved identity remains `outcome_tracking_supported=false` and does not invent a terminal class. Rejection remains `actual_rejected / (admitted + actual_rejected)`, at least 16 decisions, only for supported typed regular HTTP streaming single-worker paths. Shadow would-reject never means actual rejection. In-flight requests before their deadline remain unresolved, not silently successful.

Each accepted scalar first-text label carries its admission-time feature values. There are no TTFT labels for rejected, timed-out, or cancelled requests. Features use the same two 300-second banks and up to 64 samples as the existing scalar fit. For each supported subset, normalize weights proportional to `N_i²`, compute feature means, and solve a three-dimensional centered ridge system against `max(TTFT_i-B_i,0)/N_i - g_scalar_target`. Fixed prospective constants are ridge `0.1`, neutral prior `[0,0,0]`, per-slope clamp `[-2,2]`, and context gain correction clamp `[-0.5,0.5]`. Clipping the ridge result is a bounded approximation, not the exact solution to constrained ridge regression. Require 12 common known samples, weighted effective sample size at least 8, each feature variance at least `1e-4`, and range at least `0.05`. Choose the largest supported common-row subset, in deterministic mask order `[7,3,5,6,1,2,4]`; missing dimensions are never imputed. Unknown or out-of-range current dimensions contribute zero. All unsupported dimensions yield the exact legacy prediction.

New labels or bank removal trigger slope fitting on the next fresh successful poll; every fresh successful poll can retarget the context gain as current KV/rates change. The effective target is `max(0, g_scalar_target + clamp(beta dot (context-mean), -.5, .5))`. A separate trajectory approaches it at most `0.5` gain units/second, starting at the current legacy gain when context becomes supported. Availability transitions back to exact legacy fallback can jump; the movement cap describes the active context trajectory, not a fallback transition. Unsupported dimensions or stale/failed inputs must not continue an old context effect. Learned slopes are predictive associations; correlated workload phases and missing outcome support do not establish causal effects.

Deadline diagnostics add `feature_deadline_started`, `feature_deadline_resolution`, and `feature_deadline_ignored`. Started/resolution events include request start, 300-second deadline, monotonic observation/publication times, epoch and request ID. Resolution source distinguishes poll from actual terminal callbacks. Counts add `deadline_timely_first_text`, `deadline_breach`, `deadline_or_cutoff_adverse`, and `deadline_or_cutoff_unknown`; a breach increments both breach and adverse, once. All event activity snapshots include `deadline_untracked_pending`.

Diagnostics use `feature_context_forecast`, `feature_context_poll`, `feature_context_feedback`, `feature_outcome`, `feature_outcome_ignored`, and `feature_admission_decision` on the existing debug observation target. Structured data is JSON in `feature_json`. Forecast and poll include exact feature values/counters, coefficients, weighted center, support ranges, ESS/sample count, fit revision, target/applied gain/trajectory anchor, original N/B/scalar gain, threshold, and separate EXTRA/TOTAL verdicts. Forecasts share the original pre-credit forecast clock. Feedback includes saved admission features, source profile, baseline/N, observed first text, and acceptance reason. The actual admission decision/outcome events carry request ID, worker, epoch and monotonic diagnostic time. Unknown inputs remain JSON null. Compare the feature candidate against the unchanged scalar forecast on the same admissions and keep false-positive/false-negative denominators and unknown/censored rows explicit. Actual hard rejections have no counterfactual TTFT truth.
