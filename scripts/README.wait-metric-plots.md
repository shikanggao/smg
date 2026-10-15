# Engine metrics, calculated estimates and actual TTFT over time

For the sustained KV-pressure / completion-deadline experiment, use the companion
`plot_wait_pressure.py` described at the end of this document.

`plot_wait_metrics.py` reads recorded files only. It produces three aligned views
per run (formula, engine inputs/learning, diagnostics). Every view includes the
same request's calculated estimate and observed first-text TTFT at its start time.
Both baseline and test requests are retained. Rejected requests have no actual
TTFT. `train` in older filenames means calibration inference, not model training.

Install Python 3.10+ and Matplotlib. Supply one analyzed run directory, or a root
containing `MODEL/SUITE` directories. Each run needs:

- `metrics.tsv`: the classifier harness's router/native metric sample columns,
  `elapsed_seconds` and `sample_completed_elapsed_seconds` on the harness clock.
- `requests.tsv`: `opc_request_id`, `case`, `status`, `started_elapsed_seconds`.
- `router-events.jsonl`: exact pre-admission `forecast` events from the
  admission observation schema, with formula components and readiness flags.
- `paired-requests.json`: request-ID joins of forecasts and SMG first-text events.
- `summary.json`: `engine`, `image`, `suite` and test configuration.

```sh
python scripts/plot_wait_metrics.py \
  --input recorded-tests/final \
  --output-dir metric-graphs \
  --zoom-suite holdout
```

Use `--title` for one run; `--zoom-suite` can be repeated. `--overwrite` explicitly
replaces existing PNGs. `--capacity-floor` changes only the reference line; the
recalculation always uses logged effective capacity. The manifest records graph
names, numbered phase starts, sampling intervals, missing metrics and validation.

## Interpretation and validation

Orange filled triangles are ready estimates; hollow triangles are unready.
Gray crosses have no baseline, so their logged total is incomplete.
Blue dots are actual **SMG first-text** TTFT, including nonempty reasoning or
content, not metadata-only SSE. Gray steps are scraped estimates, which may
change between admissions. Estimates and actuals are joined by request ID, never
by nearest timestamp. Durations use the SMG upstream boundary; placement on the
common axis uses the harness request start (a small route/transport offset).
Native-engine and SMG scrapes are fetched concurrently, but SMG's cached engine
snapshot comes from its separate one-second poll and need not match the native
scrape at that instant. Steps do not
imply an uninterrupted observation between polls. The zoom includes the last
request's TTFT tail; dots remain at admission/start, not at completion.

This script replays the tested shared formula with alpha=1, b=0:

`T = B + max(W / C, F)`, where `W = Q + 0.05 D`.

Before capacity calibration, when only a fallback rate exists and measured delay
is available, the follow-up branch uses `T = B + F`.
Otherwise the numerical estimate remains diagnostic until ready. Older traces
do not explicitly log floor availability; for those fresh queue-proxy runs it is
equivalent to native waiting > 0 and at least four observed samples. New traces
log floor availability and learned-capacity readiness explicitly; the measured
rate follow-up also logs `measured_delay_only`, so replay uses that exact branch.
Delay-only warm-up suppresses an unmeasured fallback rate, while an active native
prefill measurement remains visible with its unready flag. The loader
rejects stale (>30 s) snapshots, non-proxy queues, duplicate IDs, missing actuals
and any formula mismatch >1e-8 s. This schema/branch assumption must be revisited
if future serving arithmetic or trace fields change. A missing B is displayed
as a gap in the B metric; a logged total without B is not a complete TTFT forecast.
The request-level baseline-availability strip identifies these cases.

The extra-delay graph subtracts **B at admission** from actual TTFT. This is a
formula diagnostic, not the test classifier's prompt-specific low-load reference.
Generation TPS, KV usage, pending-stream counts and visibility delay are
diagnostics in the retained implementation. Rejected intermediate traces may
record pending-stream counts contributing to F scaling; their exact logged
branches remain supported for comparison. No version converts pending count to
full remaining prompt work W. Native TTFT histogram means
may have a different first-token boundary from first text. All native histogram
means use counter delta sum / delta count, never cumulative sum / count.
Counter resets and unavailable/negative sentinel values become gaps, not zeros.

## Recorded-data limits

Older traces log only the combined `W`; do not derive Q or D by matching
asynchronous scrapes. New traces log Q, post-poll D, the backlog used to scale F,
pending first-text count and branch availability. The generator plots exact Q
and 0.05D arithmetic inputs when available; Q and D can themselves be estimated
token work. New traces also show snapshot age at admission beside the scraped
age gauge (which holds its value from the last SMG refresh), plus the explicit
delay-only fallback branch in a separate lane.
SGLang `cached_tokens_total`, `num_requests_total`, and vLLM histogram buckets /
prefill-size count were not captured continuously in the TSV. The recorded SMG
derived mean, blended size, prefill rate, learned capacity and sample counts
preserve their effects, but cannot reconstruct every raw-input counter history.
SGLang's TSV `prompt_compute_tokens` alias is its cache-inclusive prompt counter;
it must not be treated as the uncached capacity divisor. Its prompt median and
prefill-time histograms are absent in these recordings and explicitly labelled.

The collector paused 0.25 seconds after fetching; actual start-to-start intervals
include fetch time and are reported per run. SMG's engine poll interval was one
second in the recorded experiment. No interpolation can recover sub-poll spikes.

## Sustained KV pressure and full-response deadlines

`plot_wait_pressure.py` reads `MODEL/MODE` directories containing `metrics.tsv`,
`requests.tsv`, exact `router-events.jsonl`, and `summary.json` from the pressure
harness. It supports completed probes, planned background cancellation, deadline
misses with incomplete responses, and enforced rejections with unknown outcomes.

```sh
python scripts/plot_wait_pressure.py \
  --input recorded-pressure-tests/results \
  --prior earlier-tests/results/candidate \
  --output-dir pressure-graphs
```

The optional prior is earlier calibration inference data, not model training.
Its final metric timestamp must precede the new experiment. The script replays
every serving forecast, joins SMG first-text events by request ID, and produces
three figures per run: pressure/latency timeline, accuracy by traffic phase,
and learning evidence. JSON exports contain phase outcomes and per-probe
predictions. Metrics are placed at collection completion; native-rate inputs to
experimental forecasts must have been available before client dispatch.
The learning-context view also places completed-request input usage at dispatch
for a diagnostic comparison with the historical queue-size proxy. That usage is
known only after completion, is not an admission input, and includes prompt tokens
that may have been cached; it is not exact uncached work remaining in the queue.

The 300-second **full-response completion deadline** and the serving router's
300-second **extra-TTFT budget** are different quantities. Both are recorded;
setting the latter cannot establish an end-to-end deadline guarantee. Background
traffic holds KV blocks resident while probes test actual new-arrival latency.
Observation continues beyond the deadline to identify late completion. A response
still open after the deadline is a known miss even when its exact completion is
unknown. Rejected requests receive no invented completion-time label.

The analysis-only learner uses the same 32-completion window, four-sample minimum,
and 2,000-token/s capacity floor for every model. It learns a queue-delay ratio
only from prior completed requests with a ready native queue term dominating F,
then corrects effective capacity while preserving B, F, the delay-only branch and
readiness gates. A request's result cannot affect an earlier prediction. This
learned scale is a diagnostic hypothesis, not a measured physical GPU capacity.
It can worsen false positives when traffic changes; it is not a serving change.

For each eligible completed observation, the learned ratio is
`max(0, actual_TTFT - B) / (W / C)`. Its rolling median `k` selects
`C_corrected = max(2000, C / k)` and the diagnostic estimate remains
`B + max(W / C_corrected, F)`. A zero queue contributes zero after correction;
there is no added global offset that would inflate idle traffic. The fixed
delay-only branch is preserved. These properties do not prove safety under a new
traffic mix: the tests can still show an increased false-positive rate.

A second diagnostic adds an estimated decode duration using aggregate generated
tokens/s divided by the sampled native running count. That fair-share approximation
does not account exactly for prefill stages, heterogeneous remaining outputs,
preemptions, future arrivals or scheduling. It is labelled experimental and must
not be interpreted as a reliable deadline bound. The script deliberately reports
regressions as well as improvements. No model-specific constants are assigned.
