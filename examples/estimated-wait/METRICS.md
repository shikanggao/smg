# Essential TTFT operational metrics

All five families use `model`, `engine` (`vllm`, `sglang`, `other`, `unknown`), and
`mode` (`shadow`, `enforce`). They cover regular HTTP streaming chat/completion
attempts while a TTFT budget is enabled. A retry is another attempt. They are not
fleet-wide counters for every protocol. Request IDs belong in traces only.

| Metric | Additional label / unit | Purpose |
|---|---|---|
| `smg_ttft_predictions_total` | `availability=usable,unavailable` | Prediction coverage, including missing forecasts |
| `smg_ttft_outcomes_total` | `outcome=exact,late_censored,early_interrupted,rejected,unknown` | Outcome coverage and censoring; once per attempt |
| `smg_ttft_classifications_total` | `result=tp,fp,tn,fn` | Observable deadline classification |
| `smg_ttft_overestimate_seconds` | histogram, seconds | Positive exact error; explicit 100-second bucket |
| `smg_ttft_invalid_evidence_used_total` | bounded `reason` | Selected-prediction evidence/number invariant violations; must remain zero |

The same prediction and total-TTFT threshold are frozen before dispatch and used
for admission and scoring. First text measures elapsed time from router request
start. Later model updates do not rewrite the saved prediction. Worker reset does
not erase its outcome. Concurrent timeout/first-text/drop paths resolve at most
once. Unfinished observations resolve as unknown rather than fabricated latency.
A validity violation is counted at the boundary and excluded from enforcement and
scoring; normal lack of fresh evidence is merely unavailable.

Positive means predicted/observed TTFT **at or above** the configured deadline.
Exact first text supports all four labels. A continuously observed interruption
at or beyond the deadline proves late TTFT and can support TP/FN, but never an
exact error. Early interruption, lost observation, and rejected requests cannot
supply counterfactual TTFT or FP/TN. Rejected requests never train exact latency.

Use increases over a stable evaluation window, grouped by model/engine/mode:

- FPR = FP / (FP + TN), target **< 2%**.
- FNR = FN / (FN + TP), target **< 3%**.
- Exact overestimate violations = histogram count minus `le="100"` bucket;
  target **zero**, not merely p99 < 100 seconds.
- Invalid-evidence counter increase: **zero**.
- Show available predictions / all predictions and all outcome counts next to
  error rates. Do not declare success with a zero denominator or insufficient
  coverage. Require a predeclared sample minimum and confidence interval for rollout.

An enforced run hides the actual TTFT of rejected requests. Consequently these
metrics describe the observable sample, not unbiased full-policy FPR/FNR. Continue
scheduled, isolated shadow qualification or paired controlled tests; rejection
ratio is not FPR and late/admitted ratio is not FNR. No probing bypass is introduced.

Three per-worker operational gauges remain: `smg_estimated_wait_data_usable`,
`smg_estimated_wait_capacity_ready`, and `smg_estimated_wait_snapshot_age_seconds`.
Existing rejection/unknown counters and unrelated SMG service metrics are retained.
Scalar-fit, duplicate raw-estimate, and derived prefill-research gauges are removed;
bounded structured traces retain diagnostic detail without permanent metric series.
