# Recovery diagnostic capture

This instrumentation observes the current admission policy. It does not change
calibration fitting, history expiry, readiness, enforcement, or the 0.5/second
gain movement cap. Enable the existing `smg::estimated_wait::observation=debug`
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
sample caps. The outcome is diagnostic only and does not change the handling of
late/failed/duplicate publications.

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
