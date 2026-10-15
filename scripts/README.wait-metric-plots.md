# Engine metrics, calculated estimates and actual TTFT over time

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
Generation TPS, KV usage and visibility delay are diagnostics, not additional
terms in this formula. The first-text backlog fix uses the pending-stream count
to preserve/scale measured F until first-text work clears; it never converts that
count into remaining prompt work W. Native TTFT histogram means
may have a different first-token boundary from first text. All native histogram
means use counter delta sum / delta count, never cumulative sum / count.
Counter resets and unavailable/negative sentinel values become gaps, not zeros.

## Recorded-data limits

Older traces log only the combined `W`; do not derive Q or D by matching
asynchronous scrapes. New traces log Q, post-poll D, the backlog used to scale F,
pending first-text count and branch availability. The generator plots exact Q
and 0.05D when available.
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
