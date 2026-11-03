# Native cancellation and deadline telemetry

The [size-basis experiment](BASIS_CONSISTENCY.md) documents the separate paired
calibration arm and native-rich capture event schema.

The gateway consumes opt-in native scheduler evidence without changing the raw
estimated-wait formula, calibration control, or admission policy. Install the
version-pinned [vLLM adapter](../../grpc_servicer/docs/vllm_native_deadline.md) or
[SGLang patch](../../grpc_servicer/native_patches/sglang/README.md) first. Their
supported request shapes and scheduler configurations are narrower than the
engines' general APIs. CPU component tests do not qualify a GPU deployment.

## Observe native cancellation

Enable the existing experimental feature diagnostics, for example:

```sh
smg --worker-urls http://ENGINE:8000 \
  --max-estimated-wait-secs 300 \
  --estimated-wait-shadow \
  --estimated-wait-calibration-shadow \
  --estimated-wait-feature-shadow
```

Feature shadow enables a separate authenticated
`GET /v1/native_deadline_snapshot` poll. With feature shadow off, no requests
are made to that endpoint. Its missing, malformed, slow, or failed responses
do not replace `/metrics` or existing load responses, delay their publication,
or change raw estimator readiness. Side polls use the configured worker API key,
a two-second timeout, an eight-MiB response limit, one in-flight job per worker,
and at most 32 jobs globally.

For supported regular HTTP feature streams, each dispatch receives an internal
`x-smg-attempt-id` containing a gateway-process UUID and a dispatch UUID. Caller
values for this header are not forwarded. The gateway retains native uncertainty
after a timeout, disconnect, or observer drop. Only an exact native scheduler ACK
removes canceled work; first text independently resolves pre-first-text work.
Repeated ACKs and later decode-abort ACKs cannot release that work again.

Engine epoch, profile, rank, sequence, complete inventory, and the bounded ACK
journal are checked. Missing rows are not cancellation ACKs. A fresh native
snapshot can attest an older retained terminal event; its original engine-clock
occurrence time stays separate from the fresh observation time. Cross-host
absolute monotonic clocks are never compared.

An identical repeated snapshot retains the previously accepted sample only
until its original freshness expiry; it does not refresh age or replay work or
ACKs. Changed content at the same sequence fails closed. An idle engine without
a native heartbeat can still age out and make deadline capability unavailable.

## Explicit native deadline enforcement

The optional gateway flag is:

```text
--estimated-wait-native-hard-ttft-timeout-secs 10
```

Its default is **unset (`None`)**, including the Python binding and
`RouterArgs.estimated_wait_native_hard_ttft_timeout_secs`. Python launchers also
accept the `--router-` prefix. It requires feature shadow and a finite value in
`(0, 86400]`. It is independent of `--max-estimated-wait-secs`; the example
300-second admission budget never creates an engine deadline.

Both the gateway option and the backend's documented enforcement opt-in are
needed. At send time, the gateway subtracts elapsed time since the original typed
route started, including retries, and sends only the remaining whole milliseconds
in `x-smg-hard-ttft-timeout-ms`. Caller values are discarded. An already exhausted
budget returns a gateway timeout before dispatch. The current observation scope
is an isolated single-worker pool of regular streaming chat/text completions;
nonstreaming, gRPC, PD, and other routes do not receive this option. Missing
forecast/lease context in an intended streaming route fails before dispatch;
it never silently drops a configured deadline. Without fresh verified native
enforcement capability, a supported instrumented dispatch returns
`native_deadline_unavailable` rather than silently assuming support.

The native timer starts at **engine HTTP receipt** and stops at the backend's
advertised first-output boundary. The current pinned adapters use first nonempty
text. Network transit remains additional time; this is not proof of an identical
client-observed wall-clock deadline. A cached worker capability or sent TTL is
also not proof that a particular request received a deadline, for example across
an engine restart. Per-request confirmation requires its matched native active
inventory record, engine epoch, exact native request ID, and actual native
deadline. Requested and confirmed deadlines are reported separately.

## Results and current limit

Debug observation events expose the distinction:

- `native_deadline_dispatch_requested`: exact attempt and requested remaining TTL;
  per-request confirmation is still pending.
- `native_deadline_poll`: matched native deadline attestations and ACKs, preserving
  engine occurrence timestamps and gateway observation transport duration.
- `deadline_recovery_forecast`: accepted-work shadow counts, actual ACK releases,
  native support/fallback reasons, and the unchanged raw-control status.

Native ACK accounting is functional when a supported adapter reports actual
scheduler cleanup. This does not switch raw admission to a new estimate.

The unused strict-FIFO predictor is excluded from this branch. Continuous
batching is not a single FIFO work server, and a known deadline does not prove
immediate cleanup. The accepted-work ledger and native acknowledgement checks
remain because C2 requires them.

### Experimental service/deadline shadow (C2)

`--estimated-wait-feature-shadow` also emits `survival_decision_admission` at
`smg::estimated_wait::observation=debug`. This diagnostic does not change routing,
raw estimates, admission thresholds, or hard rejection policy. Its method is
`stable_work_native_service_composition_C2_guarded`; the candidate total is
`survival_json.admission.projection.total_ttft_seconds`. Missing totals remain
missing. Unsupported native projections use the paired stable-work forecast,
with `supported=false` and an explicit reason.

The independent fit stores its selected admission X in fixed work units
`z = X * C_admission / C_reference`. The first verified positive admission
capacity fixes `C_reference` for that basis/counter/source epoch. Labels retain
that immutable admission z. The dimensionless gain `k` changes the service rate
`C_effective = C_reference / k` **before** the predecessor queue projection.

The projection uses a fresh complete native inventory, exact gateway attempt
identity, gateway admission order, native remaining prefill work when known,
and explicitly counted full-prompt proxies for waiting/prefill rows otherwise.
Unknown no-text decode work falls back. Actual first text and native ACKs remain
separate from predicted expiration. Overdue active rows retain their full work
and receive no expiration credit. The target itself is never expired or clipped.

Cleanup lag is `max(1 second, past observed hard-deadline ACK lag)` within the
current 600-second history and native/source scope. The 1-second value is an
**experimental engineering prior**, including before any observed ACK; it is
not a measured cleanup bound or cancellation guarantee. The event exports
observed sample counts/max/age separately from this prior. Repeated snapshots
retain their original expiry; timeout or malformed inventory supplies no C2
release, service, or learning evidence. Histories are bounded and overflow
falls back explicitly.

Continuous batching, gateway order, full-prompt proxies, a fixed effective
service rate, and future arrivals limit this statistical approximation. The unguarded C2 replay showed that recovery exact labels could make
cancellation-assisted completions look like faster service. The selected guarded
arm freezes whether its own admission projection actually used an expiry branch
before full service. It excludes only subsequent exact labels with that flag
from the unadjusted service fit, retaining all censor evidence and any earlier
censor row. Floating subtraction residuals alone never set the flag. Local
rebound tests cover both the original risk and this guard. Development replay success does not establish accuracy on
a later rebound or constitute a fresh DEV confirmation.
