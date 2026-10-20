# Recovery diagnostic capture

With recovery probes disabled (the default), this instrumentation observes the
current admission policy without changing enforcement. Calibration fitting,
history expiry, readiness, and the 0.5/second gain movement cap are unchanged.
The separate opt-in recovery probe exception is described below. Enable the existing `smg::estimated_wait::observation=debug`
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


## Live recovery candidate (`live_net_depletion_v1`)

Calibration shadow also computes a separate, untrained recovery comparator from
fresh native waiting (`W`) and running (`R`) gauges. Let `P=W+R`. Over both a
10-second and a 30-second window, require a strict decline in **both** `W` and `P`,
at least five polls, and no intermediate rebound. Flat individual polls are
allowed because request counts are discrete, but a flat short window is not.
Use the smaller of the two net population-depletion rates `r`:

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
poll gaps beyond the configured snapshot age, rebounds and zero population clear
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

## Experimental recovery admission probes

`--estimated-wait-recovery-probes` defaults to **off** and requires calibration
shadow. With hard raw enforcement enabled, it permits at most **one request per
worker incarnation every ten seconds and one full response in flight**. Raw
shadow mode never consumes an actual permit. There is no automatic allowance
ramp in this first implementation.

Only the typed regular HTTP streaming `/v1/chat/completions` and `/v1/completions`
paths support the response-lifetime handoff. Non-streaming, raw-body passthrough,
multipart, disaggregated and gRPC paths do not acquire exceptions. Exactly one
available worker must remain in the candidate cohort, and the permit is bound
to the actual selected worker. Health/circuit-breaker/static overload checks
remain in place. The original raw sample must be fresh, ready and over budget;
both `B+Nlive` and `B+g*Nlive` must be below the numeric budget. This deliberately
conservative probe condition does not turn either predictor into a guarantee.

Selection and reservation are atomic under the admission mutex; a policy that
selects no worker consumes no permit or cooldown. The full response body owns the
permit until EOF, stream error or client drop. First text does not release it.
Pre-response failures/drop release through RAII. Invalidation clears evidence
but retains occupied permits and cooldown within the same worker incarnation;
replacement workers start independent state. Logs `recovery_probe_attempt`,
`recovery_probe_reserved`, `recovery_probe_response` and `recovery_probe_released`
include owned request correlation and the shared clock. Local release never
claims that native engine work has stopped.

The DEV comparison keeps probes **off** and evaluates the candidate in shadow.
It cannot establish the performance of an enforced reopening policy. Local
integration tests cover the bounded exception separately.
