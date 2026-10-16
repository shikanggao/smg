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
The pressure timeline includes a gray step line from every recorded SMG estimate
sample, including periods without scored probe arrivals. Colored TTFT points and
Q/W points remain exact request-at-admission observations. Do not connect them
across a period without requests and treat that interpolation as measured TTFT.
The vLLM rate panel also shows local-compute counter deltas including valid zeros.
SMG's positive-prefill-rate signal has gaps when no positive rate is available;
those gaps are not metric-scrape failures. SGLang's captured prompt counter has
different semantics and is not plotted as a local-compute rate.

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

## Live shadow calibration from first text

`analyze_wait_calibration.py` and `plot_wait_calibration.py` reproduce the online
shadow experiment. For each suite, retain `summary.json`, `requests.tsv`,
`metrics.tsv` and `router-events.jsonl`. Keep the full router log history since
startup: if a later suite inherits learning state, its trace must include the
previous suite's feedback. The analyzer refuses to replace missing exact events
with nearby metric samples.

```sh
for suite in train holdout cold hard; do
  python scripts/analyze_wait_calibration.py "results/llama/$suite"
done
python scripts/plot_wait_calibration.py \
  --input results/llama --title 'vLLM Llama 3.1 8B'
```

Enable serving diagnostics with `--estimated-wait-calibration-shadow` and
`RUST_LOG=info,smg::estimated_wait::observation=debug`. The flag defaults off.
Raw readiness and admission stay unchanged, including when raw hard admission is
enabled. The experimental estimate is `B + g * E_raw`, scaling the entire extra
delay (including F) while preserving B. It does not replace physical capacity C.
This differs from the older pressure script's offline capacity-correction
hypothesis described above.

At worker selection, a snapshot saves B, E_raw, capacity-readiness state, epoch,
observation sequence and g before dispatch credit. The first nonempty supported
OpenAI-compatible SSE reasoning/content text contributes
`max(actual_TTFT - saved_B, 0) / saved_E_raw`, weighted by saved E_raw. Each native
capacity state uses its own rolling 32-observation weighted-median target, with
initial g=1 and four valid observations required for a different target. Those shared policies apply to every engine/model.
Only supported regular HTTP streaming responses supply learning feedback.

The audit rebuilds each live history by worker, epoch, state and accepted
observation sequence. It checks window weights, before/after targets and applied
scales, saved inputs,
exact forecast arithmetic and that forecasts did not use later feedback. It
separately audits the original pre-admission raw gate. Selected snapshots can
slightly differ from that earlier gate during concurrent dispatch; accuracy
compares raw/calibrated predictions from the same selected snapshot.

Tables/scatterplots join those exact forecasts to each request's first-text TTFT
on SMG's clock. Client TTFT is retained separately. Healthy/over-budget truth uses
prompt-size/cache/output-limit matched serial baseline probes and the shared
five-second extra-delay budget. Rejected requests have unknown counterfactual
TTFT. They receive no invented healthy/over-budget label.

Timeline curves are continuously sampled estimate gauges; actual dots are
individual measured TTFTs placed at request start. The learning panel colors g
by the saved native-capacity state. Other panels show queue/running/pending text,
KV usage, B, F and C. These samples cannot recover sub-poll engine changes.
Cold means a fresh router with an already warm engine. Holdout is new traffic
with online learning still enabled, rather than a frozen fitted model. The
`train` suite contains calibration inference requests, not model training.

Accuracy axes are linear from zero to one second and logarithmic above one
second; both use the same scale, so the diagonal still means perfect agreement.
TTFT and scale timelines use the same compression above one. All outliers remain
in the plots and error scores.

A typical-error fit does not optimize admission classification. A tiny raw work
estimate can teach a huge ratio before the queue becomes visible, then amplify
an accurate later raw estimate. Sparse observations can also teach g near zero.
Keep this diagnostic until transition, false-positive and false-negative checks
pass. This flag provides no calibrated enforcement mode.

## Elapsed-time scale limit

The live shadow scale now approaches the weighted-median target at a shared
maximum of **0.1 absolute scale units per second**, in either direction. This is
an absolute rate, not a percentage. For one native-capacity history/epoch, with
applied anchor `a`, target `h` and monotonic anchor time `t0`:

```
g(t) = clamp(h, a - 0.1 * (t - t0), a + 0.1 * (t - t0))
T_cal = B + g(t) * E_raw
```

On accepted feedback, first evaluate the old trajectory at the feedback time,
then anchor the new target at that current applied scale/time. Earlier idle time
cannot be spent on a new target. Callback bursts do not increase the movement
budget. Read-only snapshots evaluate the trajectory even with no new feedback.
Separate native-ready/unready histories and epoch resets are unchanged: switching
histories or resetting to initial g=1 can still cause a displayed transition.
Changes in B, F or raw token work can still change the final TTFT prediction fast.

The `calibration_target_scale` and `calibration_time_seconds` fields accompany
selected/general forecasts; time is monotonic seconds since the calibration epoch.
Feedback records `target_scale_before/after`, `calibration_time_seconds` and the
saved `admission_calibration_time_seconds`. Existing scale fields refer to the
applied scale. The existing readiness/sample-count diagnostic means the history has at least
four valid observations; it does not mean the applied scale has reached its
target. The additional gauge is
`smg_estimated_wait_calibration_target_scale`.

The analyzer reconstructs each old-target trajectory, checks continuity at
accepted updates and the 0.1/s bound, then evaluates saved selected snapshots
at their exact monotonic times. It also reports `target` scores: the unrestricted
median from that same prior live history, applied to the same raw snapshot.
That comparison isolates the limiter without a future-data fit or another live
execution. It does not simulate queues from calibrated enforcement. Old traces
without elapsed-time fields retain their original unrestricted replay behavior.

Graphs show raw (orange) and current time-limited (teal) predictions against
actual TTFT, plus the applied scale over time. Unrestricted targets are omitted
from the figures, legends, MAE annotations and axis limits to keep the current
version readable. Their values remain in the analysis tables and retained data.
All actual/raw/current outliers remain visible. Inspect false positives and
false negatives as well as typical error. A time limit controls movement speed,
not whether a learned target is valid for the next traffic regime. The existing
observation eligibility, baseline, whole-extra-delay scaling including F,
readiness gates and default-off shadow setting are preserved.


## Admission-snapshot eligibility and real 300-second waits

The current learner accepts a scale observation only when the **saved admission
snapshot** has both mature native prefill capacity and engine-reported
`native_waiting > 0`. Earlier freshness, baseline and finite-positive-extra checks
still apply. Dispatch credit and SMG pending-first-text counts cannot substitute
for a visible native queue. Completion-time queue changes cannot make an earlier
request eligible. Healthy low-load observations still update the baseline B;
they no longer teach the queue multiplier. The shared 32-observation window,
four-observation minimum, initial scale 1 and 0.1 absolute units/second limiter
are unchanged for all models and engines. Native-unready history remains at 1.

Selected forecasts save `native_waiting`, `native_queue_visible` and `native_ready`.
Feedback saves `admission_native_waiting`, `admission_native_queue_visible` and
`admission_native_capacity_ready`. New skip reasons are
`native_capacity_unready` and `native_queue_not_visible`; original guard reasons
retain precedence. The analyzer verifies saved inputs, accepted eligibility and
that skips do not change the accepted sequence, count or target. It can still
replay older traces that lack these fields.

The stronger deadline pressure experiment uses short 100-word/8-token sentinels
behind unique 4,096-word/8,192-token background inference, with common bounded
adaptive load across all engines. It keeps a 1,200-second observation window while
scoring the production 300-second **client full-completion deadline**. Shadow
coverage requires at least three completed sentinel IDs with client first text
≥330s and independently joined SMG first text ≥300s. Age alone is not measured
TTFT. Full completion requires both SSE `[DONE]` and a finish reason. The separate
router budget is 300 seconds of **extra TTFT above B**; it cannot guarantee full
completion for arbitrary long outputs. Both boundaries are reported separately.

Retain `requests.tsv`, `metrics.tsv`, `summary.json`, `stages.json` and the complete
`router-events.jsonl` for each mode, including capacity warm-up feedback. Then:

```sh
python scripts/analyze_wait_deadline.py results/llama/shadow-attempt1
python scripts/plot_wait_deadline.py \
  --input results/llama/shadow-attempt1 --title 'Llama — 300-second pressure'
```

The analyzer separates measured SMG first text, client first text, full completion,
known deadline misses with censored exact times, and unknown rejected
counterfactuals. It reports low/recovery false positives, sentinel late-request
misses, raw-ready/unready misses, and observed 240–300s healthy / 300–360s late
boundary buckets. Empty buckets mean insufficient coverage in that slice, not
zero risk. The timeline preserves continuous metric samples through pressure,
unload, drain and recovery, with a 300-second reference. Exact outcomes remain
individual dots; they are never interpolated. Graphs show only raw and the current
calibrated estimate; all outliers are retained.

### Freeze deadline truth before background intervention

Planned unload can accelerate queued probes. A client timeout, cancellation,
error or incomplete background stream can do the same **earlier**. Use the
earliest such background termination or planned unload as the pressure cutoff.
Natural background completion with `[DONE]` and `finish_reason` does not create
an artificial cutoff. Controlled HTTP503 admission rejection does not withdraw
admitted work and remains an unknown latency counterfactual.

```sh
python scripts/stratify_wait_deadline.py results/llama/shadow-attempt1
python scripts/plot_wait_deadline_stratified.py \
  --input results/llama/shadow-attempt1 --title 'Llama — pressure cutoff'
python scripts/test_stratify_wait_deadline.py
```

Schema version 2 in `deadline-strata.json` provides `cutoff_elapsed_seconds`,
`cutoff_reason`, `completion_deadline_at_cutoff`,
`first_text_deadline_at_cutoff`, `completed_paired_proof_before_cutoff`,
`healthy_boundary_completed_before_cutoff`, and the corresponding request IDs.
Primary truth includes completed outcomes observed before cutoff and requests
whose own 300-second deadline already passed while still observed before cutoff.
Other pending outcomes are unknown. A request canceled before its deadline stays
unknown even if the clock passes 300 seconds later. First-text deadline truth
uses the client clock; measured SMG first-text proof remains separate.

`paired-deadline-stratified.json` keeps every original paired row and its later
observed latency. `requests-deadline-stratified.json` additionally retains every
raw request, including rejected and unselected canceled requests, with its exact
original TSV fields under `raw_request`. Original requests, analyses and paired
data remain untouched. The stratifier records its own SHA256 and hashes of
`requests.tsv`, `paired-deadline.json` and `stages.json` so stale derived scores
can be detected. Compatibility fields named `*_unload` keep their literal
planned-unload meaning; they are not aliases for an earlier cutoff.

Plots retain all probe observations and outliers. Crosses mark probes pending
at cutoff; circles mark probes not pending then, including later recovery
controls. Metric curves remain sampled values rather than interpolated actual
latencies. The cutoff and planned unload are marked separately when they differ.
Healthy outcomes completed only after cutoff cannot establish specificity under
sustained pressure. An empty 240–300-second healthy bucket before cutoff means
that boundary remains untested, rather than proving zero false positives.
