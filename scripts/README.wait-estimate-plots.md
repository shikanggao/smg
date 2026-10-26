# Wait-estimate comparison graphs

Use `plot_wait_estimates.py` after each wait-estimation test suite. It creates the same four views used in the DEV FRA results: estimated and actual first-text TTFT over time, a focused phase, signed prediction error, and actual-versus-estimated TTFT colored by request start time.

The script uses the test data provided to it. Model names, suites, dates, deployment revisions, and Confluence pages are not hardcoded. It performs no deployment or publication.

## Run after a test suite

Requires Python and Matplotlib. Validated with Python 3.14.5 and Matplotlib 3.11.2; no other third-party packages are used.

```bash
python3 -m pip install matplotlib
python3 scripts/plot_wait_estimates.py \
  --input /path/to/results/model/holdout \
  --output-dir /path/to/results/model/holdout/graphs
```

The input directory must contain `requests.tsv`, `paired-requests.json`, `router-events.jsonl`, and `summary.json` produced by the existing test harness and request-pair analyzer. Generate the paired-request analysis first. One exact forecast and one successful first-text observation must correspond to each successful request; ambiguous or missing pairs fail validation.

Add this step immediately after analysis in future test runners:

```python
subprocess.run(
    [
        python_executable,
        str(smg_repo / "scripts" / "plot_wait_estimates.py"),
        "--input",
        str(run_directory),
        "--output-dir",
        str(run_directory / "graphs"),
    ],
    check=True,
)
```

The existing DEV FRA `work/run_classifier_suite.py` runner already invokes this step. For later runners, use the same script path and analyzed run-directory contract.

## Replot an exported CSV

```bash
python3 scripts/plot_wait_estimates.py \
  --input /path/to/smg-ttft-timeline-vllm-pairs.csv \
  --output-dir /path/to/replotted-results
```

CSV input can contain several models and suites. The script creates separate graphs for each `(model, suite)` group. Required fields are:

| Field | Meaning |
| --- | --- |
| `model`, `suite`, `case` | Group and phase names |
| `request_id` | Unique request identifier within the group |
| `status` | HTTP status; only `200` has observed first-text TTFT |
| `request_start_seconds` | Client request start on a common monotonic clock within the group |
| `estimated_ttft_seconds` | Exact estimate recorded before admission |
| `actual_ttft_seconds` | SMG first nonempty reasoning/content duration; blank for rejected/error requests |
| `forecast_ready` | `true`/`false` or `1`/`0` |
| `baseline_available` | `true`/`false` or `1`/`0` |

Other columns are retained in the output CSV. The script recomputes phase-relative time, error, and estimate/actual ratio rather than trusting stale derived fields. Baseline phases named `baseline-*` are excluded consistently for both input formats.

## Options and outputs

`--focus-case PHASE` selects the zoom and parity phase. By default, the script chooses the largest mixed/mixture phase, or the largest phase when none has that name. `--title TEXT` overrides the display title for a single group. `--source-revision REVISION` records the tested revision in the summary; raw run inputs also retain their image from `summary.json`.

For each group, outputs are `ttft-MODEL-SUITE.png`, `ttft-MODEL-SUITE-pairs.csv`, and `ttft-MODEL-SUITE-summary.json`. Existing outputs are preserved unless `--overwrite` is explicitly supplied. Importing the module generates nothing. Matplotlib and font caches use the output directory unless `MPLCONFIGDIR` or `XDG_CACHE_HOME` are already configured.

## Interpretation

Both durations are placed at the same original request start; actual duration is associated retrospectively by request ID, never shifted to response completion time. The time coordinate uses the client clock, while both comparable TTFT durations use the SMG clock.

Signed error is `estimate - actual`: positive means overestimation and negative means underestimation. MAE is the mean absolute per-request error. The parity diagonal means equality, and color adds request start time. Hollow markers denote unready estimates; crosses denote missing baselines and incomplete total-TTFT reporting.

Rejected/error requests have no actual first-text counterfactual and appear as start-time ticks only. Their actual TTFT, error, and ratio remain blank. If no successful request exists, the graph states that no actual first-text observations are available. Error summaries describe successful requests only and separately report ready predictions with an available baseline. Client completion time includes generation and transport and remains a separate measurement.
