# Router capture of native work diagnostics

Set `SMG_NATIVE_ENGINE_WORK_DIAGNOSTICS=1` explicitly in the **router process**
to retain the existing work diagnostics from each actual native HTTP side poll.
It is off unless the value is exactly `1`. The native process must separately
enable the same flag to produce these fields. This adds no endpoint calls,
counters, estimator inputs, native capability rules, or admission behavior.

The router buffers its existing `/v1/native_deadline_snapshot` response and
performs the original `WorkerLoadResponse` parse unchanged. An independent
counts-only companion copies a strict whitelist from those same bytes. It is
not added to the shared native protocol structs and never enters native
validation, the ledger, capacity calculations, or predictions. Malformed work
metadata cannot turn a valid typed response into a failed native observation.

## Event and causal binding

The opt-in `native_engine_work_capture` event uses method
`router_native_response_counts_v1`. It carries `worker`, the existing gateway
`native_poll_started_time_seconds`, `native_receipt_time_seconds`,
`raw_poll_started_time_seconds`, and calibration epoch. An accepted publication
shares these exact values with its existing `native_deadline_poll` event, whose
`native_inventory_json.wire` remains the typed base evidence.
The receipt remains the existing publisher/native-contract observation time
after lock acquisition, not an HTTP-body-completion timestamp. Only small
publication-context values are copied under the lock; companion serialization
and logging run after the admission and ledger guards are released.

`publication_reason` distinguishes an accepted raw-poll watermark from a
missing watermark, a stale worker/raw-poll generation, or disabled native
telemetry. `accepted_by_adapter` and `native_contract_reason` retain their
existing meanings: availability after observing a response does not alone
prove that an older or invalid frame was accepted. Keep the reason and require
an active/cached current frame for causal feature use. No discarded response
receives native progress or deadline credit from this companion.
Completed responses discarded after worker retirement or monitor disappearance
also emit an explicit terminal record without typed publication. An opt-in task
guard records an aborted/dropped task that never reached a terminal observation;
its unreturned prefix size and timeout classification stay unknown. This
distinguishes cancellation from a classified transport timeout and does not
claim a completed native response.

`native_engine_work_json` is a serialized capture envelope:

| Field | Meaning |
|---|---|
| `schema_version`, `diagnostic_only` | Capture schema 1; always diagnostic only |
| `http_status` | Observed status, or null for transport failure |
| `response_complete` | Whether the complete bounded body was available |
| `response_sha256` | SHA256 of those exact complete response bytes; null for partial, transport, or size-limit paths |
| `response_bytes`, `bytes_observed` | Complete body size when known, and observed prefix byte count; an unreturned cancelled-task prefix stays null |
| `available`, `unavailable_reason` | Diagnostic capture completeness and first static failure reason; not native capability or physical capacity |
| `failure_reason_counts` | Bounded enum-key histogram of each observed failure, including unavailable diagnostics on retained rows; saturating counts, never arbitrary error strings |
| `ranks_seen/retained/dropped`, `requests_seen/retained/dropped` | Explicit coverage counts; unknowable counts are null |
| `ranks` | Retained rank observations in original response order |

Each rank retains epoch, profile, rank, `snapshot_seq`, native capture time,
per-rank coverage/availability, `engine_work_progress`, and associated active
request rows. Each request retains native/attempt IDs, `state_seq`, and the
existing `engine_work_state`. Nested identity/association must match its base
row. Whitelisting preserves cumulative work, completion sequence and last
completion, and the existing output count, retraction/preemption count, native
abort witness and historical cache-release witness. Unknown values remain
null. Dropped/malformed/source-unavailable records cannot certify full coverage.

The full-body digest is a source-attested binding to bytes seen by this fetch.
**It is not an archive of the full raw body.** Sanitized counts plus typed base
cannot reconstruct unknown fields or recover byte-for-byte HTTP evidence.
Keep source/binary/config/capture attestations and the existing independent raw
collector; the collector's faster sampling is not router consumption.

## Bounds and interpretation

The existing HTTP input limit remains 8 MiB. The companion has a hard 2 MiB
serialized limit, at most 256 ranks, at most 8,192 retained request rows, and
bounded 256-byte identifiers. Unrecognized content fields, token IDs, prompt
or response text, headers and arbitrary exception strings are not copied.
HTTP, transport, malformed, row/rank/byte-limit, and unavailable-source cases
remain explicit diagnostics. Partial prefixes are never labeled complete.
Body-read errors/timeouts retain the known HTTP status and buffered prefix size
while keeping complete-body size and SHA null. Failure categories can coexist;
the histogram counts each failed observation or unit rejected by a cap and does
not treat those categories as mutually exclusive rows.

Work semantics remain those in [engine_work_progress.md](engine_work_progress.md):
completed scheduled input at host completion witnesses, not GPU time or accepted
output. Do not sum overlapping host spans, infer GPU idle from zero deltas, call
historical cache release current free capacity, or replace unknown visible-text
decode work with zero. Fitting requires same-identity monotonic windows and
receipt at or before the admission cutoff; retain missing coverage and censoring.

The opt-in path adds bounded parsing, hashing and logging work. Its live cost
and complete capture coverage require measurement; local source/unit checks do
not establish live GPU integration or prediction accuracy.
