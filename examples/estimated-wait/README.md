# TTFT prediction and admission

This feature exposes one TTFT predictor. The coherent native work/service estimate
(C2) uses current pending work, source-qualified throughput, acknowledged native
releases, and immutable raw/native evidence cutoffs. When native evidence is
unavailable it uses a fresh baseline plus queue-work/capacity fallback. Unknown or
stale evidence never proves overload. A configured same-model provisional baseline
is superseded by qualified quiet-traffic measurements.

## Configuration

TTFT admission is disabled without a gateway or worker budget. Set
`--max-estimated-wait-secs 300` for a **total** first-text deadline, not an extra-delay
allowance. This retains the existing research field name; the decision compares
the complete predicted TTFT, including baseline, to the budget.

The budget is also the computation opt-in. Without it, SMG keeps the upstream
load-monitoring path: no estimator histogram/provenance processing, learning
history, request ledger, native side poll, prediction fitting, or additional
TTFT stream observer. Normal load-aware routing, static overload protection and
their existing polls continue. A worker-specific budget opts in that worker;
it does not enable estimator processing for unconfigured peers. Configured
shadow mode intentionally performs estimation so its decisions can be evaluated.

`--estimated-wait-shadow=true` is the default. Rust CLI enforcement requires the
explicit `--estimated-wait-shadow=false`; the Python launcher uses
`--no-estimated-wait-shadow`. There is no operator switch selecting raw, scalar, or
C2 algorithms. The native per-request deadline remains independently opt-in via
`--estimated-wait-native-hard-ttft-timeout-secs`, and requires a verified enforcing
engine. A timeout setting does not qualify the prediction or authorize enforcement.

C2 currently applies to regular HTTP streaming chat/completions in a single-worker
pool. Other paths use the same qualified baseline/work fallback, without native
context projection. Static worker protection remains independent. The fallback
fails open when its baseline, capacity, or freshness is unavailable.

## Lifecycle and evidence

Polls capture dispatch watermarks before I/O. Request work admitted during a poll
survives its publication. Request IDs match native lifecycle acknowledgements; they are never metric
labels. Metadata-only SSE chunks are not first text. Nonempty content or reasoning
text supplies TTFT. Native cancellation intent is not a release acknowledgement.
Stale admissions cannot train a new worker generation; operational outcome metrics
remain attached to the immutable request prediction across a learning reset.

Source identity, rank, profile, original expiry, and coherent publication are part
of the estimator. A later publication cannot renew old evidence.
Known rebound and original-expiry protections ship with the estimator itself.

## Rollout and validation

Keep shadow mode for qualification. Existing DEV evidence concerns earlier research
builds; the reorganized stack and total-TTFT admission integration require a new
controlled DEV qualification before enforcement. Targets are observed FP < 2%,
FN < 3%, and no exact overestimate above 100 seconds, with adequate measurable
coverage and no invalid-evidence decision. A low error rate with missing/censored
outcomes is not a pass. See [essential operational metrics](METRICS.md).

Automated Rust and engine-adapter regression tests are maintained in the repository.
Manual replay, capture, and graph-generation scripts remain local.

## Minimal producer scope

The router retains one selected predictor and its required fallback, feedback,
coherent sampling and lifecycle guards. The vLLM producer retains completed
uncached-prefill work and its source/clock/counter consistency contract. The
SGLang producer retains the per-request original-prefill completion witness;
its service rate comes from qualified started-request measurements.

SGLang scheduler-decision receipts, aggregate completed-work diagnostics and
duplicate per-request debug records are excluded. The engine flag
`SMG_NATIVE_ENGINE_WORK_DIAGNOSTICS` keeps its existing deployment spelling,
but on SGLang now enables only the required original-prefill witness. The reduced
SGLang profile changes identity, causing the existing profile reset on rollout.
Manual capture, replay, benchmark and plot scripts are not shipped.

Only the selected native service learner is fitted. The legacy scalar learner,
independently fitted comparison arms, rolling research activity counters, and
per-request research JSON dumps are excluded. Pending/context trackers remain
because they supply causal labels and inputs to that learner; the learner's
queue-only fallback remains necessary when contextual features lack support.
Captured experiment fixtures and their exporters stay outside the source tree.
