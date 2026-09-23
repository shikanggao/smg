# Estimated-wait admission

An opt-in admission guard for vLLM and SGLang workers. It runs before worker
selection in regular HTTP (including streamed request bodies), gRPC/ZMQ, and
prefill/decode routing. EPD checks its prefill and decode pools; encoder load is
not modeled. The configured routing policy still chooses the worker.

For each healthy, circuit-breaker-eligible worker that passes the independent
static overload guard:

```
wait_seconds = (queued_uncached_tokens + since_poll_dispatch_tokens) / throughput
             + kv_weight * token_usage / (1 - token_usage)
```

Each worker's score is compared inclusively to its effective budget. When every
eligible worker in the pool has a budget and is at or above it,
enforcement mode returns HTTP 503 with `worker_overload_protection_shed` in both the
JSON error code and the `X-SMG-Error-Code` header, matching static overload
protection. The message identifies the estimated-wait budget. The shared shed
helper marks it non-retryable inside SMG and sets `Retry-After` to the configured
load-monitor polling interval (at least one second); clients should back off
and jitter. Static overload thresholds remain independent.

## Configuration

The Rust and Python CLIs accept the same flags. Python embedding also supports
the `--router-` prefix. JSON config uses the corresponding snake_case names at
the top level. Rust callers can set `RouterConfig.estimated_wait` or call the
builder's `estimated_wait(EstimatedWaitConfig)` method.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--max-estimated-wait-secs` | unset | Gateway wait budget; disabled unless a worker sets its own |
| `--estimated-wait-shadow` | false | Record would-reject decisions and continue routing |
| `--estimated-wait-queue-tokens-per-request` | 0 | Explicit waiting-request proxy; 0 disables it |
| `--estimated-wait-fallback-prefill-throughput` | 2000 | Cold-start prefill capacity until a qualified value is learned; `--estimated-wait-min-prefill-throughput` and `--estimated-wait-default-throughput` remain aliases |
| `--estimated-wait-prompt-size-prior-samples` | 32 | Effective sample count assigned to the configured prompt-size prior |
| `--estimated-wait-mean-prefill-tokens` | 1024 | Dispatch credit when tokenized input is unavailable |
| `--estimated-wait-kv-pressure-weight` | 0.15 | KV pressure weight in seconds; 0 disables the penalty |
| `--estimated-wait-kv-pressure-threshold` | 0 | KV usage below this ratio adds no pressure penalty |
| `--estimated-wait-base-overhead-secs` | 0 | Fixed wait overhead |
| `--estimated-wait-queue-work-correction` | 1 | Correction applied to queued token work |
| `--estimated-wait-dispatch-blocking-factor` | 1 | Fraction of newly dispatched work that blocks an arrival |
| `--estimated-wait-max-snapshot-age-secs` | 30 | Age after which a sample is unusable |

Defaults are calibration starting points, not measured model capacity. Positive
budgets, throughput, dispatch estimates, and maximum age are required. Floating
point values must be finite. Coefficients and the KV weight may be zero.

A worker can override the gateway budget using the existing overload block:

```json
{
  "url": "http://worker:8000",
  "overload": { "max_estimated_wait_secs": 2.5 }
}
```

The worker value takes precedence, and a worker budget alone enables admission
for that worker even without `--max-estimated-wait-secs`. It is validated at
registration and preserved through worker updates. Estimator parameters remain
gateway-wide. A worker with no effective budget stays unprotected and prevents
a pool-wide estimated-wait rejection, just as a worker below its budget does.
The static waiting-request and KV ceilings are resolved independently.

SGLang's native HTTP/gRPC loads preserve exact queued uncached tokens, including
zero, aggregate throughput across ranks, and average rank KV usage. The numeric
queue-token field retains wire compatibility; an explicit
`num_waiting_uncached_tokens_available: false` marks it unavailable. An omitted
availability flag keeps the legacy interpretation that the numeric field is
usable when the numeric field is present. Native HTTP reports that omit the numeric
field are marked unavailable before schema defaults are applied. Unavailable tokens are converted from waiting requests only with an explicit proxy;
partial rank reports use that proxy only for ranks lacking token data. vLLM's
Prometheus path requires waiting-request and KV gauges and uses maximum KV usage
across samples. Histogram deltas provide a rolling P50 uncached prompt size,
which is blended with the configured prior. Prefill capacity is learned as P25
over a bounded window only while requests are waiting; at least eight qualified
intervals are required. Cold start and sparse traffic use the configured fallback
prefill throughput. Both vLLM KV metric names are supported. SGLang Prometheus
fallback and native load endpoints retain their existing behavior.

The calibrated formula is
`base + alpha * (queued + rho * dispatched) / capacity + weight * max(0, kv - threshold) / (1 - kv)`.
The default `base=0`, `alpha=1`, `rho=1`, and `threshold=0` reproduce the legacy
arithmetic.

The load monitor runs even with `--disable-load-monitoring` when this guard is
enabled, including when only a worker override enables it. No network queries
occur on admission. Rank aggregation, validity checks and queue/KV arithmetic
run during load ingestion using the same arithmetic as least-load routing.
Admission retains a compact prepared estimate, not another copy of the backend
report. Ingestion and dispatch-credit updates latch the verdict; pool checks
read the verdict and its age, stopping at the first permissive eligible worker.
A mutex still orders the pool check, selection, and credit; this preserves
concurrent admission semantics. Tokenized input is credited when supplied by routing;
otherwise the mean prefill estimate is used. Selection reserves that credit
until polling even if dispatch subsequently fails. Completions do not refund
credit because the next snapshot reconciles engine state.

Polls capture a dispatch watermark before fetching load. Publication clears
only credit preceding that watermark, preserving admissions during the poll.
Sample age starts before polling I/O. Failed polls invalidate old samples.
Eviction generations prevent late responses from recreating removed state or
overwriting replacement workers. Accounting is local to one SMG instance;
multiple independent routers do not share dispatch credit.

## Missing and stale data

The initial policy is fail open. An empty eligible pool does not emit an
estimated-wait 503; existing availability handling applies. If any eligible
worker has missing, stale, or unusable data, the pool cannot prove universal
saturation and this guard admits. Keep engine limits and independently chosen
static guardrails enabled as appropriate. A stale low score is never used as a
current capacity measurement. Fail-closed policy and hysteresis remain future
decisions requiring availability and calibration evidence.

Worker gauges are updated on load publication and dispatch credit; snapshot
age is the age at the latest such update, not a continuously advancing clock.
Pool checks mark stale/unknown data unusable and emit an unknown counter when
that is why they fail open. Early exit means the counter counts admission
decisions, not every unknown worker in a pool. Rejections retain their separate
counter and share the existing overload-shed response metrics.

The threshold gauge now uses a worker label to reflect per-worker overrides:

| Metric | Labels |
| --- | --- |
| `smg_estimated_wait_seconds` | worker |
| `smg_estimated_wait_threshold_seconds` | worker |
| `smg_estimated_wait_snapshot_age_seconds` | worker |
| `smg_estimated_wait_data_usable` | worker; 1 usable, 0 unknown/stale |
| `smg_estimated_wait_queue_proxy` | worker; 1 proxy used |
| `smg_estimated_wait_throughput_fallback` | worker; 1 fallback used |
| `smg_estimated_wait_blended_prompt_tokens` | worker; prior-blended vLLM P50 or compatibility proxy |
| `smg_estimated_wait_effective_prefill_capacity` | worker; learned or configured effective capacity |
| `smg_estimated_wait_unknown_total` | model, reason: missing/stale/unusable |
| `smg_estimated_wait_rejections_total` | model; enforced pool rejections |
| `smg_estimated_wait_shadow_rejections_total` | model; would-reject pool checks in shadow mode |

Gate dashboards' last-known score/proxy/fallback gauges on `data_usable == 1`.
Snapshot age is sampled on publication and credit; use poll health alongside
these gauges when monitoring idle pools.

## Start in shadow mode

Set `--estimated-wait-shadow` together with a realistic gateway wait budget or
per-worker budgets to observe the estimator before enforcing it. For example,
append these options to your existing launch command (values are illustrative):

```sh
--max-estimated-wait-secs 2.5 --estimated-wait-shadow
```

For vLLM, also configure a calibrated nonzero
`--estimated-wait-queue-tokens-per-request`; missing queued tokens with the
zero default remain unknown. Check `smg_estimated_wait_data_usable` and
`smg_estimated_wait_unknown_total` so missing data is not mistaken for spare
capacity.

Shadow mode uses the same polling, threshold resolution, sample freshness,
ledger lock and dispatch credit as enforcement. When an eligible pool would
be rejected, it increments `smg_estimated_wait_shadow_rejections_total` and
continues normal worker selection. It never emits an estimated-wait 503 or
increments `smg_estimated_wait_rejections_total`. Independent static overload,
health, circuit-breaker and other admission checks still apply. A lower
per-worker budget affects the shadow decision but cannot turn on enforcement.
The flag alone, without any gateway or worker wait budget, leaves the estimator
disabled.

The shadow counter counts pool checks, not unique requests: PD can record both
prefill, decode and compatibility-cohort checks for one request. Because shadow traffic keeps flowing,
its estimates reflect the admitted workload, not a simulation of queues after
hypothetical rejections. Shadow mode has the same estimator overhead as
enforcement and does not protect the backend from overload.

After calibration, remove `--estimated-wait-shadow` (or set
`estimated_wait_shadow` to `false` in configuration) and restart SMG to enforce
the same budgets. Re-enable shadow mode to stop estimated-wait rejection while
keeping observations. Remove all gateway and worker budgets to disable this
estimator and its ledger lock entirely.

## Repeatable calibration procedure

Use the accompanying profile template for each model, engine image, accelerator,
quantization, TP/DP shape, and scheduler configuration. Do not copy a threshold
between profiles without rerunning this procedure.

1. Record engine image and metric names. Confirm token-field presence, queue
   counts, throughput units, rank aggregation, and KV utilization under idle and
   loaded conditions. Validate both HTTP and gRPC if both are deployed.
2. Measure matched no-queue TTFT and TPOT at low load. Exercise prompt/output
   shapes 1024/16, 1024/64, 2048/16, 1024/256, plus the production distribution,
   cache misses/hits, and multimodal traffic where applicable. Pin the seed and
   workload generator version. Request rate and concurrency are test controls.
3. With a candidate wait budget and `--estimated-wait-shadow`, sweep offered load above
   stable capacity on an isolated test deployment. Scrape SMG and engine metrics
   at or faster than load polling cadence; retain per-request queue delay, TTFT,
   TPOT, errors, and timestamps. A simulator can validate control flow but cannot
   provide model/accelerator calibration.
4. Fit conservative fallback throughput, waiting-request token proxy, mean
   dispatch tokens, and KV weight against measured delay. Report the upper-tail
   underprediction `max(0, observed_queue_wait - estimated_wait)` for each shape.
5. Derive `W_max <= max(0, TTFT_SLO - no_queue_TTFT - polling_margin -
   underprediction_margin - operational_margin)`. A zero remaining budget means
   the profile has no feasible positive threshold. Do not derive it from the
   request timeout alone. Keep maximum snapshot age consistent with poll cadence
   and the measured staleness margin.
6. Remove `--estimated-wait-shadow` and enforce the fitted budget. Verify rejection at equality, no rejection while
   an eligible worker remains below budget, stable queues during overload,
   bounded TTFT/TPOT, no memory failure, and recovery after load drops. Include
   bursts, concurrent admissions, delayed/failed polls, worker replacement,
   multi-rank skew, and multi-router traffic.
7. Save raw observations, exact launch configuration, profile, and release
   criteria. Approve a profile only with measured SLO attainment and bounded
   underprediction. This repository change supplies no production calibration.

For functional tests without GPUs, run `cargo test -p smg --lib estimated_wait`.
The concurrent placement test verifies that 16 simultaneous requests sharing a
2-second budget admit exactly two 100-token requests at 100 tokens/s before 503.
