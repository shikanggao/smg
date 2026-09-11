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

The minimum score across the candidate pool is compared inclusively to the
budget. When every eligible worker in that pool is at or above the budget,
this guard returns HTTP 503 with `worker_overload_protection_shed` in both the
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
| `--max-estimated-wait-secs` | unset (disabled) | Calibrated wait budget in seconds |
| `--estimated-wait-queue-tokens-per-request` | 0 | Explicit waiting-request proxy; 0 disables it |
| `--estimated-wait-default-throughput` | 2000 | Fallback aggregate generation tokens/s |
| `--estimated-wait-mean-prefill-tokens` | 1024 | Dispatch credit when tokenized input is unavailable |
| `--estimated-wait-kv-pressure-weight` | 0.15 | KV pressure weight in seconds; 0 disables the penalty |
| `--estimated-wait-max-snapshot-age-secs` | 30 | Age after which a sample is unusable |

Defaults are calibration starting points, not measured model capacity. Positive
budgets, throughput, dispatch estimates, and maximum age are required. Floating
point values must be finite. The KV weight may be zero.

SGLang's native HTTP/gRPC loads preserve exact queued uncached tokens, including
zero, aggregate throughput across ranks, and average rank KV usage. Missing
queued tokens are converted from waiting requests only with an explicit proxy;
partial rank reports use that proxy only for ranks lacking token data. vLLM's
Prometheus path requires waiting-request and KV gauges, uses maximum KV usage
across samples, and uses the configured throughput fallback. Both vLLM KV metric
names are supported. SGLang Prometheus fallback also requires a queue proxy.

The load monitor runs even with `--disable-load-monitoring` when this guard is
enabled. No network queries occur on admission. A mutex orders the pool check,
selection, and credit. Tokenized input is credited when supplied by routing;
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

Metrics, emitted when admission evaluates the pool:

| Metric | Labels |
| --- | --- |
| `smg_estimated_wait_seconds` | worker |
| `smg_estimated_wait_threshold_seconds` | model |
| `smg_estimated_wait_snapshot_age_seconds` | worker |
| `smg_estimated_wait_data_usable` | worker; 1 usable, 0 unknown/stale |
| `smg_estimated_wait_queue_proxy` | worker; 1 proxy used |
| `smg_estimated_wait_throughput_fallback` | worker; 1 fallback used |
| `smg_estimated_wait_unknown_total` | model, reason: missing/stale/unusable |
| `smg_estimated_wait_rejections_total` | model |

Gate dashboards' last-known score/proxy/fallback gauges on `data_usable == 1`.
Gauges are request-evaluation samples, not a background estimate for idle pools.

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
3. With an intentionally high non-binding wait budget, sweep offered load above
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
6. Enforce the fitted budget. Verify rejection at equality, no rejection while
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
