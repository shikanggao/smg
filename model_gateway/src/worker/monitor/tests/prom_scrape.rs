//! The `/metrics` scrape read for a worker's load: the engine families' gauges, a body with none of
//! them, and the rank loads a data-parallel engine reports.

use super::super::*;

// Trimmed sample mirroring the fields observed on a live vLLM v0.7.3
// `/metrics` endpoint (dev ORD cluster).
const VLLM_METRICS: &str = r#"
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="llama"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="llama"} 5.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="llama"} 0.75
vllm:generation_tokens_total{model_name="llama"} 123456.0
"#;

const SGLANG_METRICS: &str = r#"
sglang:num_running_reqs{model="llama"} 2.0
sglang:num_queue_reqs{model="llama"} 4.0
sglang:token_usage{model="llama"} 0.42
sglang:utilization{model="llama"} 0.9
"#;

#[test]
fn parses_gauges_ignoring_comments_and_labels() {
    let m = PromScrape::parse(VLLM_METRICS);
    assert!(m.has("vllm:num_requests_running"));
    assert_eq!(m.sum("vllm:num_requests_running"), 3.0);
    assert_eq!(m.sum("vllm:num_requests_waiting"), 5.0);
    assert_eq!(m.mean("vllm:gpu_cache_usage_perc"), 0.75);
    assert!(!m.has("vllm:does_not_exist"));
    assert_eq!(m.sum("vllm:does_not_exist"), 0.0);
    assert_eq!(m.mean("vllm:does_not_exist"), 0.0);
}

#[test]
fn sum_adds_ranks_mean_averages_ratios() {
    // Two DP-rank series for the same gauge name.
    let text = "g{dp=\"0\"} 10\ng{dp=\"1\"} 30\nr{dp=\"0\"} 0.2\nr{dp=\"1\"} 0.8\n";
    let m = PromScrape::parse(text);
    assert_eq!(m.sum("g"), 40.0); // counts add
    assert_eq!(m.mean("r"), 0.5); // ratios average
}

#[test]
fn vllm_metrics_map_onto_token_usage_snapshot() {
    let m = PromScrape::parse(VLLM_METRICS);
    let snap = SchedulerLoadSnapshot {
        num_running_reqs: m.sum("vllm:num_requests_running") as i32,
        num_waiting_reqs: m.sum("vllm:num_requests_waiting") as i32,
        token_usage: m.mean("vllm:gpu_cache_usage_perc"),
        ..Default::default()
    };
    let resp = WorkerMonitor::single_rank(snap);
    assert_eq!(resp.dp_rank_count, 1);
    assert_eq!(resp.effective_token_usage(), 0.75);
    assert_eq!(resp.loads[0].num_running_reqs, 3);
    assert_eq!(resp.loads[0].num_waiting_reqs, 5);
}

#[test]
fn sglang_metrics_map_onto_token_usage_snapshot() {
    let m = PromScrape::parse(SGLANG_METRICS);
    assert_eq!(m.mean("sglang:token_usage"), 0.42);
    assert_eq!(m.sum("sglang:num_running_reqs"), 2.0);
    assert_eq!(m.sum("sglang:num_queue_reqs"), 4.0);
}

#[test]
fn sglang_v054_underscore_prefix_is_recognized() {
    // SGLang v0.5.4+ renamed the metric prefix `sglang:` -> `sglang_`.
    let v054 = "sglang_token_usage{model=\"llama\"} 0.5\n\
                sglang_num_running_reqs{model=\"llama\"} 7\n\
                sglang_num_queue_reqs{model=\"llama\"} 1\n";
    let m = PromScrape::parse(v054);
    assert!(!m.has("sglang:token_usage"));
    let prefix = ["sglang:", "sglang_"]
        .into_iter()
        .find(|p| m.has(&format!("{p}token_usage")));
    assert_eq!(prefix, Some("sglang_"));
    assert_eq!(m.mean("sglang_token_usage"), 0.5);
    assert_eq!(m.sum("sglang_num_running_reqs"), 7.0);
}

#[test]
fn parse_ignores_trailing_timestamp_and_label_spaces() {
    // Optional trailing timestamp must not be read as the value, and a
    // space inside a quoted label value must not split the head early.
    let text = "vllm:gpu_cache_usage_perc{model_name=\"my model\"} 0.75 1719849600000\n";
    let m = PromScrape::parse(text);
    assert_eq!(m.mean("vllm:gpu_cache_usage_perc"), 0.75);
}

#[test]
fn vllm_v1_kv_cache_usage_metric_is_recognized() {
    // vLLM v1 renamed the gauge from `gpu_cache_usage_perc` to
    // `kv_cache_usage_perc`; the fetcher accepts either name.
    let v1 = "vllm:kv_cache_usage_perc{model_name=\"llama\"} 0.6\n";
    let m = PromScrape::parse(v1);
    assert!(!m.has("vllm:gpu_cache_usage_perc"));
    let kv_usage = ["vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"]
        .into_iter()
        .find(|name| m.has(name));
    assert_eq!(kv_usage, Some("vllm:kv_cache_usage_perc"));
    assert_eq!(m.mean("vllm:kv_cache_usage_perc"), 0.6);
}

#[test]
fn metric_derived_snapshot_has_no_absolute_token_data() {
    // A ratio-only snapshot (from `/metrics`) must not be treated as
    // carrying absolute token counts: it stays out of the DP cache,
    // even at high KV usage.
    let resp = WorkerMonitor::single_rank(SchedulerLoadSnapshot {
        token_usage: 0.75,
        num_running_reqs: 3,
        ..Default::default()
    });
    assert!(!resp.has_absolute_token_data());
    assert_eq!(resp.total_used_tokens(), 0);
    assert_eq!(resp.effective_token_usage(), 0.75);
}

#[test]
fn real_snapshot_with_capacity_has_absolute_token_data() {
    // A snapshot from gRPC/`/v1/loads` reports KV capacity, so its
    // absolute token fields are meaningful (even when idle).
    let resp = WorkerMonitor::single_rank(SchedulerLoadSnapshot {
        max_total_num_tokens: 8192,
        num_used_tokens: 1024,
        ..Default::default()
    });
    assert!(resp.has_absolute_token_data());
    assert_eq!(resp.total_used_tokens(), 1024);
}

#[test]
fn fleet_rollup_is_not_dp_ranks() {
    // A gateway worker answers `/loads` with one entry per downstream
    // worker, all at `dp_rank: 0`. Keying those by rank collapses them
    // onto a single entry, so the DP cache must reject the response.
    let fleet = WorkerLoadResponse {
        dp_rank_count: 2,
        loads: vec![
            SchedulerLoadSnapshot {
                worker: Some("http://a:8000".to_string()),
                max_total_num_tokens: 8192,
                num_used_tokens: 1024,
                ..Default::default()
            },
            SchedulerLoadSnapshot {
                worker: Some("http://b:8000".to_string()),
                max_total_num_tokens: 8192,
                num_used_tokens: 4096,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    assert!(fleet.has_absolute_token_data());
    assert!(!fleet.ranks_are_dp_ranks());
    assert_eq!(fleet.dp_rank_loads().len(), 1);

    // One engine's own ranks are distinct and stay eligible.
    let engine = WorkerLoadResponse {
        dp_rank_count: 2,
        loads: vec![
            SchedulerLoadSnapshot {
                dp_rank: 0,
                max_total_num_tokens: 8192,
                num_used_tokens: 1024,
                ..Default::default()
            },
            SchedulerLoadSnapshot {
                dp_rank: 1,
                max_total_num_tokens: 8192,
                num_used_tokens: 4096,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    assert!(engine.ranks_are_dp_ranks());
    assert_eq!(engine.dp_rank_loads().len(), 2);
}

#[test]
fn histogram_parser_requires_the_exact_le_label() {
    let scrape = PromScrape::parse(
        "vllm:request_prefill_kv_computed_tokens_bucket{fake_le=\"1\",le=\"32\"} 4\n",
    );
    assert_eq!(
        scrape.histogram("vllm:request_prefill_kv_computed_tokens_bucket"),
        Some(vec![(32.0, 4.0)])
    );
}

#[test]
fn vllm_counter_deltas_derive_work_rates_and_recent_means() {
    let first = PromScrape::parse(
        "vllm:prompt_tokens_total 1000\n\
             vllm:generation_tokens_total 500\n\
             vllm:request_prefill_kv_computed_tokens_sum 10240\n\
             vllm:request_prefill_kv_computed_tokens_count 10\n\
             vllm:prefix_cache_hits_total 100\n\
             vllm:prefix_cache_queries_total 200\n",
    );
    let second = PromScrape::parse(
        "vllm:prompt_tokens_total 5000\n\
             vllm:generation_tokens_total 1500\n\
             vllm:request_prefill_kv_computed_tokens_sum 30720\n\
             vllm:request_prefill_kv_computed_tokens_count 20\n\
             vllm:prefix_cache_hits_total 180\n\
             vllm:prefix_cache_queries_total 300\n",
    );
    let observed = Instant::now();
    let (cold, history) =
        derive_engine_metrics(&first, vllm_counter_sample(&first, observed), None);
    assert_eq!(cold.prefill_throughput, None);
    assert_eq!(cold.gen_throughput, 0.0);
    assert_eq!(cold.avg_request_prefill_kv_computed_tokens, Some(1024.0));
    assert_eq!(cold.cache_hit_rate, 0.5);

    let (derived, _) = derive_engine_metrics(
        &second,
        vllm_counter_sample(&second, observed + Duration::from_secs(10)),
        Some(history),
    );
    assert_eq!(derived.prefill_throughput, Some(2048.0));
    assert_eq!(derived.gen_throughput, 100.0);
    assert_eq!(derived.avg_request_prefill_kv_computed_tokens, Some(2048.0));
    assert!((derived.cache_hit_rate - 0.8).abs() < f64::EPSILON);
}

#[test]
fn low_load_ttft_excludes_congestion_and_clears_on_reset() {
    let now = Instant::now();
    let mut history = None;
    let mut count = 0.0;
    let mut sum = 0.0;
    for second in 0..=8 {
        let waiting = if second == 5 { 1 } else { 0 };
        if second > 0 {
            count += 1.0;
            sum += if second == 5 || second == 6 {
                30.0
            } else {
                0.2
            };
        }
        let scrape = PromScrape::parse(&format!(
                "vllm:num_requests_waiting {waiting}\nvllm:num_requests_running 1\nvllm:time_to_first_token_seconds_sum {sum}\nvllm:time_to_first_token_seconds_count {count}\n"
            ));
        let (derived, next) = derive_engine_metrics(
            &scrape,
            vllm_counter_sample(&scrape, now + Duration::from_secs(second)),
            history,
        );
        if second >= 4 {
            assert!((derived.low_load_ttft_seconds.unwrap() - 0.2).abs() < 1e-9);
        }
        history = Some(next);
    }
    let reset = PromScrape::parse(
        "vllm:time_to_first_token_seconds_sum 0\nvllm:time_to_first_token_seconds_count 0\n",
    );
    let (derived, _) = derive_engine_metrics(
        &reset,
        vllm_counter_sample(&reset, now + Duration::from_secs(9)),
        history,
    );
    assert_eq!(derived.low_load_ttft_seconds, None);
}

#[test]
fn sglang_learns_uncached_work_with_the_shared_capacity_rules() {
    for prefix in ["sglang:", "sglang_"] {
        let now = Instant::now();
        let mut history = None;
        for second in 0..=8 {
            let body = format!(
                    "{prefix}token_usage 0.1\n{prefix}num_queue_reqs 1\n{prefix}num_running_reqs 2\n{prefix}prompt_tokens_total {}\n{prefix}cached_tokens_total {}\n{prefix}num_requests_total {}\n{prefix}gen_throughput 1\n",
                    second * 8000, second * 2000, second * 2,
                );
            let (snapshot, next) = WorkerMonitor::decode_sglang_metrics_with_history(
                &body,
                now + Duration::from_secs(second),
                history,
            )
            .unwrap();
            if second > 0 {
                assert_eq!(snapshot.prefill_throughput, Some(6000.0));
                assert_eq!(
                    snapshot.avg_request_prefill_kv_computed_tokens,
                    Some(3000.0)
                );
                assert_eq!(snapshot.prefill_size_sample_count, Some(second * 2));
            }
            assert_eq!(
                snapshot.learned_prefill_capacity,
                (second >= 8).then_some(6000.0)
            );
            assert_eq!(snapshot.low_load_ttft_seconds, None);
            history = Some(next);
        }
    }
}

#[test]
fn sglang_capacity_uses_prefill_starts_before_decode_completion() {
    let now = Instant::now();
    let mut history = None;
    for second in 0..=8 {
        let body = format!(
                "sglang:token_usage 0.1\nsglang:num_queue_reqs 1\nsglang:num_running_reqs 10\nsglang:prompt_tokens_total 80000\nsglang:cached_tokens_total 20000\nsglang:num_requests_total 20\nsglang:queue_time_seconds_count {}\n",
                20 + second * 2,
            );
        let (snapshot, next) = WorkerMonitor::decode_sglang_metrics_with_history(
            &body,
            now + Duration::from_secs(second),
            history,
        )
        .unwrap();
        if second > 0 {
            assert_eq!(snapshot.prefill_throughput, Some(6000.0));
        }
        assert_eq!(
            snapshot.learned_prefill_capacity,
            (second >= 8).then_some(6000.0)
        );
        history = Some(next);
    }
}

#[test]
fn zero_uncached_delta_does_not_fall_back_to_prompt_rate_or_train_capacity() {
    let first = PromScrape::parse(
        "vllm:num_requests_waiting 1\n\
             vllm:prompt_tokens_total 100\n\
             vllm:request_prefill_kv_computed_tokens_sum 50\n",
    );
    let second = PromScrape::parse(
        "vllm:num_requests_waiting 1\n\
             vllm:prompt_tokens_total 1100\n\
             vllm:request_prefill_kv_computed_tokens_sum 50\n",
    );
    let observed = Instant::now();
    let (_, history) = derive_engine_metrics(&first, vllm_counter_sample(&first, observed), None);
    let (derived, _) = derive_engine_metrics(
        &second,
        vllm_counter_sample(&second, observed + Duration::from_secs(1)),
        Some(history),
    );

    assert_eq!(derived.prefill_throughput, None);
    assert_eq!(derived.prefill_capacity_learning_active, Some(false));
    assert_eq!(derived.prefill_capacity_sample_count, Some(0));
    assert_eq!(derived.learned_prefill_capacity, None);
}

#[test]
fn local_compute_parser_selects_exact_source_and_sums_engines() {
    let scrape = PromScrape::parse(
        "vllm:prompt_tokens_by_source_total{engine=\"0\",source=\"local_compute\"} 100\n\
             vllm:prompt_tokens_by_source_total{source=\"local_compute\",engine=\"1\"} 200 1234\n\
             vllm:prompt_tokens_by_source_total{source=\"local_cache_hit\"} 10000\n\
             vllm:prompt_tokens_by_source_total{source=\"external_kv_transfer\"} 10000\n\
             vllm:prompt_tokens_by_source_total{fake_source=\"local_compute\"} 10000\n\
             vllm:prompt_tokens_by_source_total{source=\"local_compute_extra\"} 10000\n",
    );
    assert_eq!(scrape.local_compute_tokens, Some(300.0));
    for invalid in ["NaN", "-1", "Inf", "invalid"] {
        let scrape = PromScrape::parse(&format!(
            "vllm:prompt_tokens_by_source_total{{source=\"local_compute\"}} {invalid}\n"
        ));
        assert!(scrape.local_compute_tokens.unwrap().is_nan());
    }
}

fn vllm_compute_scrape(computed: f64, completed: f64) -> PromScrape {
    PromScrape::parse(&format!(
        "vllm:num_requests_waiting 1\n\
             vllm:prompt_tokens_by_source_total{{source=\"local_compute\"}} {computed}\n\
             vllm:prompt_tokens_by_source_total{{source=\"local_cache_hit\"}} 100000\n\
             vllm:request_prefill_kv_computed_tokens_sum {completed}\n\
             vllm:prompt_tokens_total 100000\n"
    ))
}

#[test]
fn local_compute_trains_capacity_before_requests_finish() {
    let observed = Instant::now();
    let first = vllm_compute_scrape(0.0, 0.0);
    let (_, mut history) =
        derive_engine_metrics(&first, vllm_counter_sample(&first, observed), None);
    for second in 1..=8 {
        let scrape = vllm_compute_scrape(second as f64 * 1000.0, 0.0);
        let (derived, next) = derive_engine_metrics(
            &scrape,
            vllm_counter_sample(&scrape, observed + Duration::from_secs(second)),
            Some(history),
        );
        assert_eq!(derived.prefill_throughput, Some(1000.0));
        assert_eq!(derived.prefill_capacity_sample_count, Some(second as u32));
        assert_eq!(
            derived.learned_prefill_capacity,
            (second == 8).then_some(1000.0)
        );
        history = next;
    }
    // Decode finishes later in a burst. This is not new prefill work.
    let completed = vllm_compute_scrape(8000.0, 8000.0);
    let (derived, _) = derive_engine_metrics(
        &completed,
        vllm_counter_sample(&completed, observed + Duration::from_secs(9)),
        Some(history),
    );
    assert_eq!(derived.prefill_throughput, None);
    assert_eq!(derived.prefill_capacity_sample_count, Some(8));
    assert_eq!(derived.learned_prefill_capacity, Some(1000.0));
}

#[test]
fn zero_local_compute_does_not_use_completion_or_cached_prompt_tokens() {
    let observed = Instant::now();
    let first = vllm_compute_scrape(1000.0, 0.0);
    let (_, history) = derive_engine_metrics(&first, vllm_counter_sample(&first, observed), None);
    let second = vllm_compute_scrape(1000.0, 100000.0);
    let (derived, _) = derive_engine_metrics(
        &second,
        vllm_counter_sample(&second, observed + Duration::from_secs(1)),
        Some(history),
    );
    assert_eq!(derived.prefill_throughput, None);
    assert_eq!(derived.timely_prefill_progress, Some(0.0));
    assert_eq!(
        derived.timely_prefill_progress_source,
        Some("local_compute_tokens")
    );
    assert_eq!(derived.timely_prefill_progress_interval, Some(1.0));
    assert_eq!(derived.prefill_capacity_sample_count, Some(0));
    assert_eq!(derived.learned_prefill_capacity, None);
}

#[test]
fn local_compute_reset_invalid_data_and_source_changes_clear_capacity() {
    let observed = Instant::now();
    let baseline = vllm_compute_scrape(10000.0, 10000.0);
    let (_, mut history) =
        derive_engine_metrics(&baseline, vllm_counter_sample(&baseline, observed), None);
    history.saturated_prefill_rates = VecDeque::from(vec![1000.0; 8]);
    for scrape in [
        vllm_compute_scrape(1.0, 10000.0),
        vllm_compute_scrape(f64::NAN, 10000.0),
        PromScrape::parse(
            "vllm:num_requests_waiting 1\n\
                 vllm:request_prefill_kv_computed_tokens_sum 20000\n",
        ),
    ] {
        let (derived, _) = derive_engine_metrics(
            &scrape,
            vllm_counter_sample(&scrape, observed + Duration::from_secs(1)),
            Some(history.clone()),
        );
        assert_eq!(derived.prefill_throughput, None);
        assert_eq!(derived.timely_prefill_progress, None);
        assert_eq!(derived.prefill_capacity_sample_count, Some(0));
        assert_eq!(derived.learned_prefill_capacity, None);
    }
    let legacy = PromScrape::parse(
        "vllm:num_requests_waiting 1\n\
             vllm:request_prefill_kv_computed_tokens_sum 10000\n",
    );
    let (_, mut legacy_history) =
        derive_engine_metrics(&legacy, vllm_counter_sample(&legacy, observed), None);
    legacy_history.saturated_prefill_rates = VecDeque::from(vec![1000.0; 8]);
    let (derived, _) = derive_engine_metrics(
        &baseline,
        vllm_counter_sample(&baseline, observed + Duration::from_secs(1)),
        Some(legacy_history),
    );
    assert_eq!(derived.prefill_throughput, None);
    assert_eq!(derived.learned_prefill_capacity, None);
}

#[test]
fn vllm_counter_reset_does_not_publish_negative_rates() {
    let before =
        PromScrape::parse("vllm:prompt_tokens_total 1000\nvllm:generation_tokens_total 500\n");
    let after = PromScrape::parse("vllm:prompt_tokens_total 10\nvllm:generation_tokens_total 5\n");
    let observed = Instant::now();
    let (_, history) = derive_engine_metrics(&before, vllm_counter_sample(&before, observed), None);
    let (derived, _) = derive_engine_metrics(
        &after,
        vllm_counter_sample(&after, observed + Duration::from_secs(10)),
        Some(history),
    );
    assert_eq!(derived.prefill_throughput, None);
    assert_eq!(derived.gen_throughput, 0.0);
}

fn vllm_histogram_scrape(sum: f64, count: u64, waiting: u64, buckets: &[u64]) -> PromScrape {
    PromScrape::parse(&format!(
        "vllm:num_requests_waiting {waiting}\n\
             vllm:request_prefill_kv_computed_tokens_sum {sum}\n\
             vllm:request_prefill_kv_computed_tokens_count {count}\n\
             vllm:request_prefill_kv_computed_tokens_bucket{{le=\"32\"}} {}\n\
             vllm:request_prefill_kv_computed_tokens_bucket{{le=\"1024\"}} {}\n\
             vllm:request_prefill_kv_computed_tokens_bucket{{le=\"+Inf\"}} {}\n",
        buckets[0], buckets[1], buckets[2]
    ))
}

#[test]
fn vllm_histogram_delta_derives_rolling_median() {
    let observed = Instant::now();
    let first = vllm_histogram_scrape(0.0, 0, 0, &[0, 0, 0]);
    let (_, history) = derive_engine_metrics(&first, vllm_counter_sample(&first, observed), None);
    let tiny = vllm_histogram_scrape(20.0, 1, 0, &[1, 1, 1]);
    let (tiny_metrics, history) = derive_engine_metrics(
        &tiny,
        vllm_counter_sample(&tiny, observed + Duration::from_secs(1)),
        Some(history),
    );
    assert_eq!(
        tiny_metrics.median_request_prefill_kv_computed_tokens,
        Some(32.0)
    );
    assert_eq!(tiny_metrics.prefill_size_sample_count, Some(1));

    let representative = vllm_histogram_scrape(10_260.0, 11, 0, &[1, 11, 11]);
    let (representative_metrics, _) = derive_engine_metrics(
        &representative,
        vllm_counter_sample(&representative, observed + Duration::from_secs(2)),
        Some(history),
    );
    assert_eq!(
        representative_metrics.median_request_prefill_kv_computed_tokens,
        Some(1024.0)
    );
    assert_eq!(representative_metrics.prefill_size_sample_count, Some(11));
}

#[test]
fn vllm_cold_histogram_establishes_baseline_without_lifetime_samples() {
    let scrape = vllm_histogram_scrape(20.0, 1, 0, &[1, 1, 1]);
    let (metrics, _) =
        derive_engine_metrics(&scrape, vllm_counter_sample(&scrape, Instant::now()), None);
    assert_eq!(metrics.median_request_prefill_kv_computed_tokens, None);
    assert_eq!(metrics.prefill_size_sample_count, None);
}

#[test]
fn malformed_missing_and_reset_histograms_are_safe() {
    assert!(matches!(
        histogram_delta(
            Some(&[(32.0, 2.0), (1024.0, 1.0)]),
            Some(&[(32.0, 0.0), (1024.0, 0.0)])
        ),
        HistogramDelta::Unavailable
    ));
    let observed = Instant::now();
    let first = vllm_histogram_scrape(0.0, 0, 0, &[0, 0, 0]);
    let (_, history) = derive_engine_metrics(&first, vllm_counter_sample(&first, observed), None);
    let populated = vllm_histogram_scrape(100.0, 1, 0, &[0, 1, 1]);
    let (_, history) = derive_engine_metrics(
        &populated,
        vllm_counter_sample(&populated, observed + Duration::from_secs(1)),
        Some(history),
    );

    let missing = PromScrape::parse(
        "vllm:num_requests_waiting 0\n\
             vllm:request_prefill_kv_computed_tokens_sum 100\n\
             vllm:request_prefill_kv_computed_tokens_count 1\n",
    );
    let (missing_metrics, history) = derive_engine_metrics(
        &missing,
        vllm_counter_sample(&missing, observed + Duration::from_secs(2)),
        Some(history),
    );
    assert_eq!(
        missing_metrics.median_request_prefill_kv_computed_tokens,
        Some(1024.0)
    );

    let reset = vllm_histogram_scrape(0.0, 0, 0, &[0, 0, 0]);
    let (reset_metrics, _) = derive_engine_metrics(
        &reset,
        vllm_counter_sample(&reset, observed + Duration::from_secs(3)),
        Some(history),
    );
    assert_eq!(
        reset_metrics.median_request_prefill_kv_computed_tokens,
        None
    );
    assert_eq!(reset_metrics.prefill_size_sample_count, None);
}

#[test]
fn vllm_capacity_learns_only_from_saturated_intervals_and_uses_p25() {
    let observed = Instant::now();
    let first = vllm_histogram_scrape(0.0, 0, 0, &[0, 0, 0]);
    let (_, mut history) =
        derive_engine_metrics(&first, vllm_counter_sample(&first, observed), None);
    let mut sum = 0.0;

    for second in 1..=4 {
        sum += 50.0;
        let idle = vllm_histogram_scrape(sum, second, 0, &[0, second, second]);
        let (metrics, next) = derive_engine_metrics(
            &idle,
            vllm_counter_sample(&idle, observed + Duration::from_secs(second)),
            Some(history),
        );
        assert_eq!(metrics.prefill_capacity_sample_count, Some(0));
        assert_eq!(metrics.learned_prefill_capacity, None);
        history = next;
    }

    for (index, rate) in [800.0, 200.0, 700.0, 100.0, 600.0, 300.0, 500.0, 400.0]
        .into_iter()
        .enumerate()
    {
        sum += rate;
        let second = index as u64 + 5;
        let saturated = vllm_histogram_scrape(sum, second, 1, &[0, second, second]);
        let (metrics, next) = derive_engine_metrics(
            &saturated,
            vllm_counter_sample(&saturated, observed + Duration::from_secs(second)),
            Some(history),
        );
        if index < PREFILL_CAPACITY_MIN_SAMPLES - 1 {
            assert_eq!(metrics.learned_prefill_capacity, None);
        } else {
            assert_eq!(metrics.learned_prefill_capacity, Some(200.0));
        }
        history = next;
    }

    let idle = vllm_histogram_scrape(sum, 12, 0, &[0, 12, 12]);
    let (metrics, _) = derive_engine_metrics(
        &idle,
        vllm_counter_sample(&idle, observed + Duration::from_secs(13)),
        Some(history),
    );
    assert_eq!(metrics.learned_prefill_capacity, Some(200.0));
    assert_eq!(metrics.prefill_capacity_sample_count, Some(8));
    assert_eq!(metrics.prefill_capacity_learning_active, Some(false));
}

#[test]
fn sglang_generation_rates_are_aggregated_and_missing_rates_stay_unavailable() {
    let body = "sglang_token_usage 0.5\nsglang_num_queue_reqs 1\n";
    assert_eq!(
        WorkerMonitor::decode_sglang_metrics(body)
            .unwrap()
            .gen_throughput,
        -1.0
    );
    let body = format!(
        "{body}sglang_gen_throughput{{dp_rank=\"0\"}} 1500\n\
             sglang_gen_throughput{{dp_rank=\"1\"}} 2500\n"
    );
    assert_eq!(
        WorkerMonitor::decode_sglang_metrics(&body)
            .unwrap()
            .gen_throughput,
        4000.0
    );
}
