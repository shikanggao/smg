# Estimated-wait admission

An opt-in admission guard for vLLM and SGLang workers. It runs before worker
selection in regular HTTP (including streamed request bodies), gRPC/ZMQ, and
prefill/decode routing. EPD checks its prefill and decode pools; encoder load is
not modeled. The configured routing policy still chooses the worker.

For each healthy, circuit-breaker-eligible worker that passes the independent
static overload guard:

```
dispatch_work = dispatch_blocking_factor * since_poll_dispatch_tokens
work_delay = (queued_uncached_tokens + dispatch_work) / prefill_capacity
extra_delay = max(work_delay, observed_delay_floor)
wait_seconds = learned_low_load_baseline + extra_delay
```

Each worker's extra delay is compared inclusively to its effective budget. When every
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
| `--estimated-wait-calibration-shadow` | false | Experimental multiplicative TTFT diagnostics only; never affects admission |
| `--estimated-wait-feature-shadow` | false | Learn KV/deadline-or-cutoff/rejection context adjustments in shadow; requires calibration shadow |
| `--estimated-wait-provisional-baseline-secs` | unset | Positive, finite same-model representative baseline for calibration startup only; measured baseline takes precedence |
| `--estimated-wait-queue-tokens-per-request` | 0 | Explicit waiting-request proxy; 0 disables it |
| `--estimated-wait-fallback-prefill-throughput` | 2000 | Cold-start prefill capacity until a qualified value is learned; `--estimated-wait-min-prefill-throughput` and `--estimated-wait-default-throughput` remain aliases |
| `--estimated-wait-prompt-size-prior-samples` | 32 | Effective sample count assigned to the configured prompt-size prior |
| `--estimated-wait-mean-prefill-tokens` | 1024 | Dispatch credit when tokenized input is unavailable |
| `--estimated-wait-kv-pressure-weight` | 0 | Deprecated compatibility option; only zero is accepted because the KV term was removed |
| `--estimated-wait-dispatch-blocking-factor` | 0.05 | Fraction of newly dispatched work that blocks an arrival; range 0–1 |
| `--estimated-wait-max-snapshot-age-secs` | 30 | Age after which a sample is unusable |

Defaults are calibration starting points, not measured model capacity. Positive
budgets, throughput, dispatch estimates, and maximum age are required. Floating
point values must be finite. The dispatch factor may be zero.
The dispatch factor cannot exceed 1. Static overload protection remains the
separate memory-safety guard near full KV-cache utilization.

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
which is blended with the configured prior. Prefill rates prefer
`vllm:prompt_tokens_by_source_total{source="local_compute"}`, published with
prefill outputs before decode completes. Cached and transferred token series
are excluded. The request-completion uncached-token counter remains a fallback
for older engines, followed by the legacy prompt-token counter. A zero or
invalid preferred counter never falls back to another rate; counter resets or
source changes clear learned capacity and establish a new baseline.
Prefill capacity is learned as P25
over a bounded window only while requests are waiting; at least eight qualified
intervals are required. Cold start and sparse traffic use the configured fallback
prefill throughput. Both vLLM KV metric names are supported. SGLang Prometheus
fallback and native load endpoints retain their existing behavior.

The manual additive overhead and work multiplier options have been removed.
Their former defaults (`b=0`, `alpha=1`) are fixed; the learned baseline remains.
Remove these options from launch commands and configurations. Python constructor
arguments for these two options were also removed; use keyword arguments when
migrating positional callers. The dispatch factor remains configurable (`rho=0.05`).

The optional correction `g` defaults off in Rust and Python. When enabled, it
remains shadow-only. Eligible first-text callbacks populate two 300-second banks,
each retaining at most 32 observations (64 combined). The first accepted feedback
starts the bank clock; membership uses feedback acceptance time, not admission time.
Each fresh, distinct load poll fits `g = sum(x*y) / sum(x*x)` when feedback arrives
or rotation removes observations, using saved queue-only extra delay `x=(Q+rho*D)/C` and observed
extra delay `y = max(TTFT - saved_baseline, 0)`. Four combined observations are
required. The calibrated prediction is `B_cal + g*x`; the observed-delay floor
and legacy visibility diagnostics are excluded from both training input and
prediction. The raw floor-inclusive path remains the admission comparator.
Failed/stale/duplicate polls do not fit. The applied scale approaches
the target at at most **0.5 absolute units per second**, independent of polling
frequency; retargeting starts from the current applied value without idle credit.

### Experimental context gain

`--estimated-wait-feature-shadow` adds a separate prediction alongside the scalar
calibration and recovery diagnostics. Method `feature_context_deadline_ridge_v2`
never replaces the admission score; its second input differs from v1.
With queue-only extra delay `N`, its prediction is `B_cal + g_context * N`:

```
z = [native_KV_usage, deadline_or_cutoff_rate, actual_rejection_rate]
gain_target = max(0, scalar_gain_target + clip(beta · (z - weighted_mean), -0.5, 0.5))
```

Each fresh load poll updates the target using current context. The applied
context gain approaches it at at most 0.5 absolute units per second. Positive
and negative feature coefficients can raise or lower the prediction; they do
not add delay when `N` is zero. The raw observed-delay floor is excluded.

The learner retains the same accepted first-text samples and two 300-second
banks as scalar calibration, with each request's immutable admission context.
It fits three weighted-centered residual slopes using normalized `N²` weights
and ridge regularization of 0.1 toward initial coefficients `[0, 0, 0]`. The
residual is `max(TTFT - saved_baseline, 0) / saved_N - scalar_gain_target`.
Coefficients are clipped to `[-2, 2]`, making this a bounded approximation to
unconstrained ridge. The scalar fit remains unchanged. These fixed versioned
constants are experimental starting values, not optimized model profiles.

Rates use bounded one-second buckets over the preceding 60 seconds:

- Deadline-or-cutoff rate is `adverse / (timely_first_text + adverse)`.
  A request is adverse once when no first text has been observed by 300 seconds,
  or when an actual upstream timeout/pre-text downstream disconnect occurs.
  The existing worker poll resolves a pending deadline at its first observation
  at or after 300 seconds; terminal callbacks also check the deadline. This does
  not wait for a 420-second test observation horizon. The clock starts at the
  existing server TTFT start before upstream send, not at response headers.
  Client/server clock offsets can make a client 300-second cancellation arrive
  before server age 300: that is an actual cutoff, not an invented deadline breach.
  Late first text or later cancellation cannot reverse or double-count a breach.
  Actual first-text, timeout and disconnect diagnostics remain separate.
  The rate requires at least 16 resolved observations and less than 10% unknown
  outcomes. Early backpressure-disabled observation/unclassified endings are
  unknown; an unknown ending after a breach does not change its resolved class.
  It describes recent resolved observations, not a matched arrival cohort.
  Unsupported admissions suppress the feature for their 60-second window.
  Pending deadline storage is capped at 8192 per worker; overflow is explicit
  and suppresses this feature until all untracked pending observations resolve.
  Resetting an epoch clears pending deadlines and invalidates older callbacks.
- Rejection rate is `actual estimated-wait rejections / (selected admission attempts +
  actual estimated-wait rejections)`, with at least 16 decisions. Shadow
  would-reject checks and upstream errors are not actual gateway rejections.
  These are local placement decisions; retried upstream attempts are not
  deduplicated into a customer-level request rate.

The initial supported path is typed streaming HTTP chat/completions with a
single-worker pool whose worker is available. Other paths and multiworker pools
are unsupported, even when only one member is currently healthy.
KV input requires explicitly present, finite native measurements in `[0, 1]`;
missing metrics and default numeric zeros cannot establish availability. A KV
source/rank-profile change invalidates feature contexts while preserving the
independent scalar fit.

A fit needs at least 12 complete observations, weighted effective sample size
of 8, and variation in each fitted feature (weighted variance at least `1e-4`
and range at least `0.05`). The largest supported subset is fitted using common
observations; missing dimensions are not imputed. An unsupported or currently
out-of-range dimension contributes zero. With no currently supported dimensions,
the candidate falls back explicitly to scalar calibration. This availability
transition is distinct from the movement cap on an active context trajectory.
Worker/baseline lifecycle, stale-data and history-expiry protections still apply.

Debug observation events `feature_context_poll`, `feature_context_forecast`,
`feature_context_feedback`, `feature_admission_decision`, and `feature_outcome`
include context, counts, coefficients, support and prediction provenance for
replay. Rejected requests have no TTFT label. No-text timeouts do not supply an
invented numeric training label. Surviving first-text labels and policy-dependent
activity rates can be biased; evaluate decision errors and coverage by phase
before considering enforcement.

### Baseline and history

For startup under an already busy workload, explicitly supply a representative
same-model, same-deployment baseline using `--estimated-wait-provisional-baseline-secs`.
There is no universal default and no zero substitution for an unknown baseline.
This provisional value affects calibration only: raw baseline availability,
observed-delay floor and readiness are unchanged. Calibrated total TTFT is
unavailable (`-1` in telemetry) when neither baseline is available. Calibrated
would-reject diagnostics require fresh native-capacity readiness and a baseline;
a floor-promoted raw readiness signal cannot enable them.

The measured baseline takes precedence once the existing low-load learner has
four eligible observations (local first-text samples or qualified native monitor
intervals). Every admission saves its baseline value and source (`unknown`,
`provisional`, or `measured`). A provisional-to-measured change is material when
`abs(measured-provisional) > max(0.1 seconds, 0.2*provisional)`. Material promotion
starts a new calibration epoch and discards old-generation feedback; smaller
changes keep history and use each observation's immutable saved baseline. These
thresholds are explicit initial engineering choices, not measured accuracy bounds.
The configured value is immutable after startup; removing/changing an effective
fallback also resets learning if that situation arises. No baseline is fitted
jointly with gain from the same busy-queue feedback.

At 300 seconds the first bank becomes the previous bank and a new current bank
starts. At 600 seconds discard the first bank, retain feedback accepted during
300–600 seconds and start another current bank. Continue rotating every 300
seconds. Thus retained observations are at most 600 seconds old and can be removed
after 300–600 seconds, or earlier by the per-bank count limit. Ordinary rotation
preserves the calibration generation and current gain trajectory when at least
four observations remain. Removal triggers a fresh-poll refit even without new
feedback. Fewer than four retained observations cause immediate raw fallback
(`g=1`), including on forecast/feedback processing after idle; this fallback is
an explicit exception to gradual gain movement. Delayed feedback across ordinary
rotations remains eligible if its admission snapshot is less than 600 seconds old.

Two consecutive fresh queue-empty polls, with no local pending first-text requests
or admissions during either poll, still reset history, scale, target and generation.
Decoding may continue: this confirms queue drain, not engine idle. A failed, stale
or nonempty poll breaks drain confirmation. Worker invalidation and detected native
capacity profile changes also reset the generation and reject prior-generation
feedback. Raw admission, capacity, baseline and observed-delay floor are unaffected.

Logs identify this algorithm as `mse_poll_two_wave_queue_v2`. `calibration_rotation`
records bank index, previous/current counts, history revision, and before/after
gain. Forecasts save history and fitted revisions because a removal-only poll fit
can reuse the accepted-observation sequence. The method is distinct from historical
`mse_poll` (32 observations, absolute 600-second epoch, 0.1/s gain movement) and
`mse_poll_two_wave_v1` (two banks, floor-inclusive input); older
analyzers must reject the new method rather than silently replay it as the old one.
`queue_excess_seconds` and `calibration_input_excess_seconds` record the queue-only
input; `calibration_baseline_seconds/source/available` describe its baseline.
Existing `baseline_seconds/available` and raw prediction fields retain their raw
meaning, including a missing raw baseline when only the calibration seed is known.
The baseline, epoch, accepted-observation sequence, history/fitted revisions and
input excess also have `smg_estimated_wait_calibration_*` gauges. Baseline-source
gauges `baseline_provisional` and `baseline_measured` are one-hot; both zero mean
unknown. `calibration_baseline` events record promotions; `calibration_reset`
identifies `baseline_promoted`, `baseline_fallback_changed`, and existing lifecycle
or `queue_drained` resets for audit and test isolation.

Admission discounts newly dispatched work and does not read KV usage in its
wait calculation. The retired KV-weight option accepts only zero for old
configuration and Python positional-argument compatibility. Least-load
routing keeps its existing full-dispatch and KV arithmetic.

The load monitor runs even with `--disable-load-monitoring` when this guard is
enabled, including when only a worker override enables it. No network queries
occur on admission. Rank aggregation, validity checks and token-work arithmetic
run during load ingestion. Least-load routing retains its independent KV index.
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
4. After the rolling prompt-size and saturated-capacity windows are warm, validate
   the dispatch-blocking factor (`rho`) against request-level wait evidence.
   Manual `b` and `alpha` tuning is removed; experimental `g` remains off unless
   explicitly enabled for shadow diagnostics. The wait formula has no
   KV-pressure term. Report the upper-tail underprediction
   `max(0, observed_queue_wait - estimated_wait)` for each shape.
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
