# Recovery diagnostic capture

This instrumentation observes the current admission policy without changing
enforcement. Calibration fitting, history expiry, readiness, and the 0.5/second
gain movement cap are unchanged. Enable the existing `smg::estimated_wait::observation=debug`
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


## Live recovery candidate (`live_arrival_bounded_depletion_v2`)

Calibration shadow also computes a separate, untrained recovery comparator from
fresh native waiting (`W`) and running (`R`) gauges. Let `P=W+R`. Over both a
10-second and a 30-second window, require a strict decline in **both** `W` and `P`,
at least five polls, and no unsupported intermediate rebound. Flat individual polls are
allowed because request counts are discrete, but a flat short window is not.
A positive per-poll change in either `W` or `P` is tolerated only when each
increase is at most `min(1, new_request_credits)`. Credits count router admissions,
not prompt tokens, and are captured atomically at poll start. Each interval uses
only its watermark delta: old, duplicate, future or previously consumed credits
cannot fund another bump. Counter rollback clears history; saturation supplies
no new allowance. This bounded continuity rule does not prove native acceptance
or attribute a native arrival to an owned request. External arrivals and delayed
native acceptance can still reset history. It is deliberately conservative.

Use the smaller of the two **unadjusted** net population-depletion rates `r`;
never add arrivals to this rate or subtract them from native work:

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
poll gaps beyond the configured snapshot age, unsupported rebounds and zero population clear
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
Both recovery events expose `poll_request_watermark`, current
`total_dispatched_requests`, optional `requests_since_previous_poll`,
`rebound_allowance_requests` and `rebound_retained`. The existing
`total_dispatched` counts tokens and must never be used as an arrival count.
Request counters are worker-generation scoped, reset on invalidation/replacement,
and saturate without wrapping. Diagnostic comparisons may replay strict v1 on
the same polls; v1 is not a second runtime policy.


Recovery estimates are shadow diagnostics only. No admission bypass or probe
exception exists. Customer requests use the unchanged normal admission path.

### Experimental joint feature gains (shadow only)

`--estimated-wait-feature-shadow` requires `--estimated-wait-calibration-shadow` and is off by default. Method `feature_joint_gains_deadline_ridge_v3` leaves the original scalar gain, recovery candidate, and all admission enforcement unchanged. It predicts `B + N*(gQ + gKV*KV + gC*C + gR*R)` using the original queue-only `N`; it does not use the observed-delay floor or recovery `N`. Each coefficient is jointly learned and separately reported. When `N=0`, the model cannot invent an additive feature delay. The live-remaining-work comparator continues to use its own `Nlive` and the original scalar gain; it is not combined with joint feature gains.

The features, in fixed order, are native KV usage, deadline-or-actual-cutoff rate, and actual rejection rate. KV needs explicit valid provenance; all selected native series must be finite in `[0,1]`, and all ranks must be present and unique. Backend aggregation remains vLLM maximum / SGLang mean, followed by mean across ranks. Defaulted or malformed KV is unknown. Metric/rank profile changes invalidate only the feature history; original scalar learning remains unchanged.

Rates use the current one-second bucket and the preceding 59 buckets. The v2 second dimension is `deadline_or_cutoff_adverse / (deadline_timely_first_text + deadline_or_cutoff_adverse)`. A request resolves once: first text before 300 seconds is timely; no observed first text at or after 300 seconds is a deadline breach and adverse; an actual timeout/pretext disconnect before that boundary is adverse without claiming a deadline breach; an earlier unknown observation ending is unknown. Late first text, later actual cutoff, or unknown ending after a breach cannot reclassify or double-count it. Actual terminal counters remain separately recorded and do not supply fabricated exact TTFT or native-cancellation acknowledgements. In particular, a client 300-second cancellation arriving at server age 299.99 counts as actual cutoff, not breach.

Pending deadlines use the existing server TTFT start, before upstream send on the supported typed path. The existing worker poll resolves due requests even without a first-text callback; terminals also check elapsed time. Poll observation can lag 300 by the polling interval. An atomic scan claim prevents a poll from overtaking timely text while its terminal callback waits for the ledger lock. Metadata/no-text scans release the claim. Storage is bounded to 8192 pending observations per worker. Overflow remains terminal-observable but lacks a poll deadline; `deadline_untracked_pending` explicitly suppresses dimension 2 until overflow observations resolve. Worker/calibration epoch resets clear pending storage and counters; old callbacks are ignored. KV-profile-only changes clear learning contexts while preserving pending deadline observations and rates.

The union rate requires at least 16 known resolutions, strictly less than 10% union-unknown observations, no recent `untracked_admitted`, and no outstanding `deadline_untracked_pending`. The guard describes tracked resolved observations rather than all arrivals. An admission without a saved identity remains `outcome_tracking_supported=false` and does not invent a terminal class. Rejection remains `actual_rejected / (admitted + actual_rejected)`, at least 16 decisions, only for supported typed regular HTTP streaming single-worker paths. Shadow would-reject never means actual rejection. In-flight requests before their deadline remain unresolved, not silently successful.

Each accepted scalar first-text label carries its admission-time feature values. There are no numeric TTFT labels for rejected requests or requests that time out/cancel before first text; an observed first-text label survives a later completion timeout. Features use the same two 300-second banks and up to 64 samples as the scalar fit. For a supported common-row subset, let `x_i=[1,KV_i,C_i,R_i]`, `y_i=max(TTFT_i-B_i,0)/N_i`, and normalize weights proportional to `N_i²`. Solve the joint system `(sum_i w_i*x_i*x_iᵀ + 0.1*I)*g = sum_i w_i*x_i*y_i + 0.1*[1,0,0,0]`. The queue coefficient is nonnegative, contextual slopes are clamped to `[-2,2]`, and unsupported dimensions have zero slope. Clipping is a bounded approximation, not an exact constrained-ridge optimizer. The means remain diagnostics for support checks; prediction is uncentered.

Require 12 common known samples, weighted effective sample size at least 8, each selected feature variance at least `1e-4`, and range at least `0.05`. Choose the largest supported common-row subset in deterministic mask order `[7,3,5,6,1,2,4]`. Missing training dimensions are never imputed. Every feature in the fitted subset must currently be known and in range, otherwise use exact scalar fallback. Dropping only one slope would leave the jointly fitted queue coefficient biased for a different subset. A poll with incomplete support resets the trajectory; failed/stale inputs, profile changes, bank removal, and epoch resets retain their existing invalidation behavior.

New labels or bank removal trigger joint fitting on the next successful poll. The four applied coefficients move together from their current vector toward the target vector, with progress `min(1,0.5*elapsed/sum(abs(target-anchor)))`. This bounds the sum of coefficient movements by 0.5/second, so no individual coefficient or combined gain at fixed feature inputs can move faster. First activation or a fitted support-mask change anchors at `[current_scalar_gain,0,0,0]`. Live feature inputs and `N` still change immediately; the coefficient movement cap does not bound changes caused by those inputs, fallback, or lifecycle reset.

At evaluation, the combined contextual term `gKV*KV+gC*C+gR*R` is capped to `[-0.5,0.5]` and cannot make total gain negative. Contextual coefficients are proportionally rescaled to enforce this cap, preserving a signed decomposition. Diagnostics distinguish target/applied coefficients before this projection from effective coefficients after it. This amplitude limit is separate from the 0.5/second movement limit. Learned slopes are predictive associations; correlated workload phases, selected uncensored labels and missing rejection support do not establish causal effects.

Deadline diagnostics add `feature_deadline_started`, `feature_deadline_resolution`, and `feature_deadline_ignored`. Started/resolution events include request start, 300-second deadline, monotonic observation/publication times, epoch and request ID. Resolution source distinguishes poll from actual terminal callbacks. Counts add `deadline_timely_first_text`, `deadline_breach`, `deadline_or_cutoff_adverse`, and `deadline_or_cutoff_unknown`; a breach increments both breach and adverse, once. All event activity snapshots include `deadline_untracked_pending`.

Diagnostics use `feature_context_forecast`, `feature_context_poll`, `feature_context_feedback`, `feature_outcome`, `feature_outcome_ignored`, and `feature_admission_decision` on the existing debug observation target. Structured data is JSON in `feature_json`. Forecast and poll include exact feature values/counters, four fitted coefficients, support means/ranges, ESS/sample count, fit revision, target/applied/effective coefficient vectors and vector trajectory anchor, original N/B/scalar gain, threshold, and separate EXTRA/TOTAL verdicts. Forecasts share the original pre-credit forecast clock. Feedback includes saved admission features, source profile, baseline/N, observed first text, and acceptance reason. The actual admission decision/outcome events carry request ID, worker, epoch and monotonic diagnostic time. Unknown inputs remain JSON null. Compare the feature candidate against the unchanged scalar forecast on the same admissions and keep false-positive/false-negative denominators and unknown/censored rows explicit. Actual hard rejections have no counterfactual TTFT truth.

The v3 prediction object includes `applied_coefficients`, `target_coefficients`, `effective_coefficients`, `target_effective_coefficients`, `trajectory_anchor_coefficients`, and `contribution_seconds`, in `[queue,kv,deadline_or_cutoff,rejection]` order. Signed contributions sum to predicted extra delay; baseline is separate. Legacy `beta` remains an alias for the three uncentered fitted contextual slopes. The payload also records `neutral_prior` and `coefficient_movement_l1_per_second`. Historical v1/v2 parsing remains explicitly versioned; a frozen fresh run rejects mixed methods.

Local verification: `python3 -m unittest scripts.test_wait_feature_joint scripts.test_wait_feature_context scripts.test_wait_feature_deadline`. Exact current capture replay: `python3 scripts/replay_wait_joint_gains.py --router /absolute/canonical/router-events.jsonl --state /absolute/final/state.json --out /new/audit/directory`. The state must be valid and every delivered file must match its hash. The replay independently reconstructs retained exact labels, joint fitting and vector movement; native parsing and original scalar acceptance are recorded prerequisites. Historical comparisons use `--v2-helper /path/to/frozen/replay_feature_fit.py --receipt /path/to/fit-replay.json` (repeat receipt per valid arm), and verify the frozen helper and canonical source hashes before reconstructing the v2 history. Historical v3 counterfactuals update at captured feature-poll publication because an exact v3 internal retarget instant did not exist. Scoring preserves exact arrival predictions, phase denominators, unknowns and binary censored late labels; MAE/MSE use exact labels only. Offline comparisons cannot infer changed outcomes under enforced admission.

For linear fresh-capture figures: `python3 scripts/plot_wait_joint_gains.py --admissions /absolute/audit/admissions.jsonl --out /absolute/figure.png --title "Model / arm" --mode fresh`. Use `--mode historical` only for counterfactual replay. The four panels show total TTFT, queue-only contribution, signed contextual contributions, and four applied coefficients with dashed targets. Unknown predictions break lines; censor markers use verified per-request lower bounds, not a fixed nominal timeout.

### Decision-focused early/censored shadow comparison

Method `decision_live_queue_context_continuity_hinge_v4` runs under the existing feature-shadow switch. It does not change admission enforcement or replace the preceding joint-v3 diagnostics. The fitted input `X` is uncalibrated `Nlive` when eligible live remaining-work evidence is available, otherwise uncalibrated `Noriginal`. The existing scalar gain is excluded from fitted inputs and supported predictions. The primary comparator remains `B + existing_scalar_gain * X`. Historical `decision_live_scalar_early_hinge_v1` multiplied the scalar gain into X before fitting an additional correction; explicit recorded-method dispatch preserves that audit without mixing the two bases.

The four fixed candidates are `exact_queue`, `exact_early`, `bound_queue`, and `bound_early`. Queue controls use `B+X*q`; early candidates use `B+X*(q+aG*G+aA*A+aP*P)`. The neutral coefficient vector is `[1,0,0,0]`. `G` measures positive pending-first-text population growth against the newest complete same-profile poll 30–60 seconds old, divided by the current pending count (at least one). `A=min(oldest_pending_age/300,1)`. `P=clip(1-current_native_prefill_progress/learned_capacity,0,1)` requires positive pending population and native backlog. Progress accepts explicit zero, requires all unique ranks, valid native request-count provenance, `local_compute_tokens` or `prefill_started_requests` sources, and an interval no longer than the configured snapshot age. Completed-request prompt counters cannot substitute. Missing evidence is unknown, not zero.

Labels retain their original admission snapshot. Exact first text uses observed extra delay `max(T-B,0)`; clean pretext timeout/disconnect uses its actual elapsed lower bound, including 299.x seconds. Poll deadline observations supply a 300-second lower bound only after that age has actually elapsed. Bounds use one-sided squared error: predictions already above the bound receive no downward gradient. They never become exact TTFT labels or exact MAE observations. Bound-to-exact upgrades retain their first-label bank and original position; evicted rows cannot resurrect. Two 300-second banks hold at most 32 rows each, and original admission age must remain below 600 seconds at both acceptance and every poll. Unknown terminals supply no numeric label.

Each model minimizes half squared ratio error plus ridge `0.1` on the three contextual slopes only. The base gain q has no ridge penalty; its initialization remains 1. The published `coefficient_ridge=[0,0.1,0.1,0.1]` makes this distinction explicit. Historical direct-basis v2 regularized all four coefficients, shrinking a queue-only gain toward 1 as `(q_unpenalized+0.1)/1.1`. Explicit method dispatch preserves v1 and v2 replay without changing their solvers. V3 retains normalized `X²` weights. Bounds use only the negative residual. The fixed solver uses 256 projected-gradient steps starting at the prior; step size is `1/(0.1+weighted_sum(norm(z)^2))`, `q` is boxed to `[0,4]`, and slopes to `[-2,2]`. Support requires 12 rows, ESS 8, and each selected early feature range at least 0.05 and weighted variance at least 0.0001. Subsets are tried in order `7,6,5,3,4,2,1,0`; unavailable fitted inputs fall back to the complete matching queue control. V4 additionally requires every selected early input to lie within its closed fitted minimum/maximum interval (no tolerance). Missing inputs, unsupported early fits, or out-of-range inputs select the matching independent queue control, then scalar fallback if that queue control is unsupported. V3’s bounded extrapolation remains available only through explicit historical-method replay.

If the selected queue model has no fitted support, or the snapshot is unavailable, v2/v3 explicitly report `scalar_fallback=true` and `source_arm=live_n_scalar`; effective diagnostic coefficients are `[existing_scalar_gain,0,0,0]` and contributions reproduce the existing scalar prediction. Unavailable predictions stay unavailable, and scoring uses the original recorded scalar fallback. A missing early input uses the matching supported queue model when possible. Fitted vectors/targets remain separate; fallback effective coefficients may exceed 4 and transitions are exempt from fitted-vector slew. The unchanged fitted q cap of 4 now bounds an absolute queue gain, whereas v1 bounded an additional correction. Report saturation without retuning.

Polls refit only after label/history changes. Applied vectors move toward their targets with total absolute coefficient movement at most 0.5/second. Support changes anchor at the neutral prior. Freshness gaps clear growth evidence and trajectories, then may refit still-age-valid labels; source profile changes clear learned history while pending requests remain until terminal. At prediction the contextual sum is capped to ±0.5 and cannot make total gain negative. This post-fit cap differs from the raw linear fitting objective: diagnostics expose raw/bounded training objectives, clipped-row count, unbounded prediction, and clipping status. Selection scores the actual capped and slewed output, not the solver's uncapped line.

`decision_admission` contains the immutable selected-admission snapshot and four predictions and is authoritative for scoring. `decision_forecast` is earlier diagnostic context only. `decision_label` publishes accepted training labels; `decision_terminal` binds observed end category/time. `decision_poll` always publishes state, including missing-baseline polls; `decision_native_poll` binds the same poll clock to per-rank progress/capacity/provenance evidence. Preserve the complete router log. Fresh test profiles require `decision_calibration_required=true` and `expected_decision_method=decision_live_queue_context_continuity_hinge_v4`; readiness and analysis reject missing/mixed schemas.

Audit a closed canonical capture with `python3 scripts/replay_wait_decision.py --router /absolute/canonical/router-events.jsonl --state /absolute/final/state.json --out /new/audit/directory`. It verifies source hashes before and after, deterministic independent fitting/lifecycle replay, native feature eligibility, selected-clock snapshots, and terminal/deadline provenance. The host separately verifies closure/restoration. `wait_decision_selection.py` requires identical known identities and no FP or FN increase in any capture/phase, plus a strict improvement somewhere; zero denominators remain undefined. EXTRA admission flags and total-TTFT diagnostics are distinct. Rejected traffic has no observed counterfactual outcome. Historical forecast-clock comparisons are development evidence; their missing native progress remains unknown, and reused historical captures are not an unseen holdout.

For a direct-basis development counterfactual, use `python3 scripts/replay_wait_decision_direct.py --method decision_live_queue_slope_ridge_hinge_v3 --audit /absolute/prior/v1/audit --out /new/replay/directory`. It verifies the immutable canonical sources, reconstructs every terminal and poll-deadline label opportunity at original availability, first proves unchanged-basis v1 parity, then changes only the saved input basis/eligibility and explicit unsupported fallback. Reusing the old accepted-label set would miss changes at the minimum-X eligibility boundary. The output preserves selected targets, applied vectors, contributions and regenerated labels. Previously inspected captures remain development evidence. Promotion is gated on a fixed variant having zero FP/FN regressions in every model/phase under both recorded EXTRA and TOTAL>=300 contracts, identical known coverage, and at least one strict improvement.

V3 development replay changes only q regularization relative to v2; all four ablations, original label opportunities, exact cohorts, caps, fallback and causal timing are unchanged. Authorized fresh shadow tests may evaluate a failed promotion candidate without enabling enforcement. A ridge change cannot supply labels before their actual availability or fix a cold first ramp by itself.

For v4, supported-to-supported mask changes preserve the currently applied queue coefficient and coefficients in the intersection of the old/new masks. Newly enabled and removed slopes start at zero. Same-mask updates are unchanged. Unsupported transitions, stale-poll gaps, profile changes, and genuine epoch resets keep their previous reset behavior. Removing a slope and changing prediction source are basis switches, so their output discontinuities are not bounded by the learned-vector slew limit. No compensating queue offset is introduced.

Predictions publish `fallback_reason` (`snapshot_unavailable`, `early_fit_unsupported`, `missing_context`, `outside_fitted_ranges`, `fit_unsupported`, or null). `requested_within_fitted_ranges` is null for unsupported/missing inputs, false outside the requested model’s fitted ranges, and true for valid inputs (including the vacuous queue-only mask). Existing `within_fitted_ranges` describes the actual selected source; it can be true for queue fallback while the requested model’s flag is false. `source_arm` and `scalar_fallback` identify the actual calculation. Saved training snapshots and label eligibility are unchanged by this prediction guard.

Use `replay_wait_context_continuity.py --audit /absolute/closed-v3-audit --out /new/directory` for the fixed development ablations (previous v3, transition only, range guard only, combined). It verifies original-source bindings and v3 prediction/state parity, reconstructs original label opportunities, and requires byte-equivalent queue controls and identical label lifecycle/cohorts. An optional `--historical-planner` accepts only an exact hash-matching archived planner when the receipt’s old repository path has advanced; it never bypasses canonical data bindings. All reused traces remain development evidence. No-regression ties can qualify a regression repair for fresh shadow validation; claims of accuracy improvement still require a strict gain somewhere, without a regression in any model/phase under either decision contract.
