use openai_protocol::worker::{PrefillMetricProvenance, SchedulerLoadSnapshot};

use super::*;

struct Frame {
    started: f64,
    load: WorkerLoadResponse,
}

#[derive(Default)]
struct Source {
    ended: f64,
    counter: f64,
    epoch: u64,
    profile: u64,
    window: VecDeque<(f64, f64)>,
}

impl Source {
    fn frame(
        &self,
        previous: Option<f64>,
        began: Option<f64>,
        rate: Option<f64>,
        size: Option<f64>,
        eligible: bool,
        reset: bool,
    ) -> Frame {
        let learned = (self.window.len() >= MIN_SAMPLES).then(|| {
            let mut values: Vec<_> = self.window.iter().map(|(rate, size)| rate * size).collect();
            values.sort_by(f64::total_cmp);
            values[(values.len() - 1) / 4]
        });
        Frame {
            // Relative source clocks may legitimately start below zero.
            started: self.ended - 0.1,
            load: WorkerLoadResponse {
                dp_rank_count: 1,
                loads: vec![SchedulerLoadSnapshot {
                    dp_rank: 0,
                    native_request_counts_available: Some(true),
                    native_request_counts_profile: Some(self.profile),
                    timely_prefill_progress_source: Some("prefill_started_requests".into()),
                    prefill_capacity_learning_active: Some(eligible),
                    prefill_capacity_sample_count: Some(self.window.len() as u32),
                    learned_prefill_capacity: learned,
                    prefill_metric_provenance: Some(PrefillMetricProvenance {
                        schema_version: 1,
                        counter_epoch: self.epoch,
                        progress_source: Some("prefill_started_requests".into()),
                        raw_counter: Some(self.counter),
                        previous_raw_counter: previous,
                        interval_seconds: began.map(|began| self.ended - began),
                        interval_started_time_seconds: began,
                        interval_ended_time_seconds: Some(self.ended),
                        raw_rate: rate,
                        previous_conversion_size_mean: size,
                        capacity_interval_eligible: eligible,
                        counter_reset: reset,
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            },
        }
    }

    fn initial(&self) -> Frame {
        self.frame(None, None, None, None, false, false)
    }

    fn next(&mut self, rate: f64, size: Option<f64>, eligible: bool) -> Frame {
        let (counter, ended) = (self.counter, self.ended);
        self.counter += rate;
        self.ended += 1.0;
        if eligible {
            assert!(positive(rate) && size.is_some_and(positive));
            self.window.push_back((rate, size.unwrap()));
            if self.window.len() > WINDOW {
                self.window.pop_front();
            }
        }
        self.frame(
            Some(counter),
            Some(ended),
            Some(rate),
            size,
            eligible,
            false,
        )
    }

    fn reset_empty(&mut self) -> Frame {
        let (counter, ended) = (self.counter, self.ended);
        self.counter = 0.0;
        self.ended += 1.0;
        self.epoch += 1;
        self.window.clear();
        self.frame(Some(counter), Some(ended), None, None, false, true)
    }
}

fn provenance(frame: &mut Frame) -> &mut PrefillMetricProvenance {
    frame.load.loads[0]
        .prefill_metric_provenance
        .as_mut()
        .unwrap()
}

fn observe(history: &mut History, frame: &Frame) {
    history.observe(Some(&frame.load), frame.started, true);
}

fn selected(history: &History, frame: &Frame) -> Selection {
    history.selection(frame.started, source_key(&frame.load), 2000.0)
}

fn unavailable(value: Selection) {
    assert_eq!(value.status, "unavailable");
    assert_eq!(value.used_capacity, None);
    assert_eq!(value.lower_quartile_request_start_rate, None);
    assert_eq!(value.rebased_learned_capacity, None);
}

fn warmed() -> (Source, History, Frame) {
    let mut source = Source::default();
    let mut history = History::default();
    observe(&mut history, &source.initial());
    let mut frame = source.initial();
    for _ in 0..MIN_SAMPLES {
        frame = source.next(1.0, Some(1000.0), true);
        observe(&mut history, &frame);
    }
    assert_eq!(selected(&history, &frame).status, "available");
    (source, history, frame)
}

fn current_size(frame: &mut Frame, mean: f64, count: u64) {
    let rank = &mut frame.load.loads[0];
    rank.avg_request_prefill_kv_computed_tokens = Some(mean);
    rank.prefill_size_sample_count = Some(count);
    let p = rank.prefill_metric_provenance.as_mut().unwrap();
    p.current_unshrunk_size_mean = Some(mean);
    p.queue_size_observed_sample_count = Some(count);
}

#[test]
fn same_poll_size_is_optional_and_does_not_change_previous_window_or_reference() {
    let (mut source, mut history, _) = warmed();
    let mut frame = source.next(3.0, Some(1000.0), true);
    current_size(&mut frame, 20000.0, 9);
    observe(&mut history, &frame);
    let before: Vec<_> = history
        .samples
        .iter()
        .map(|s| (s.raw_rate, s.original_size))
        .collect();
    let value = selected(&history, &frame);
    assert_eq!(value.status, "available");
    assert_eq!(value.previous_conversion_size_mean, Some(1000.0));
    assert_eq!(value.lower_quartile_request_start_rate, Some(1.0));
    assert_eq!(value.used_capacity, Some(2000.0)); // Eight rate1 samples keep the floor.
    let current = value.same_poll_observed_size.unwrap();
    assert_eq!(current.status, "available");
    assert_eq!(current.current_unshrunk_size_mean, Some(20000.0));
    assert_eq!(current.current_observed_sample_count, Some(9));
    assert_eq!(current.used_capacity, Some(20000.0));
    assert_eq!(
        before,
        history
            .samples
            .iter()
            .map(|s| (s.raw_rate, s.original_size))
            .collect::<Vec<_>>()
    );
    assert_eq!(frame.load.loads[0].learned_prefill_capacity, Some(1000.0));
}

#[test]
fn missing_invalid_or_mismatched_current_size_retains_previous_selection() {
    for bad in 0..5 {
        let (mut source, mut history, _) = warmed();
        let mut frame = source.next(3.0, Some(1000.0), true);
        current_size(&mut frame, 20000.0, 9);
        match bad {
            0 => provenance(&mut frame).current_unshrunk_size_mean = None,
            1 => provenance(&mut frame).current_unshrunk_size_mean = Some(0.0),
            2 => provenance(&mut frame).queue_size_observed_sample_count = Some(0),
            3 => frame.load.loads[0].prefill_size_sample_count = Some(10),
            _ => frame.load.loads[0].avg_request_prefill_kv_computed_tokens = Some(19000.0),
        }
        observe(&mut history, &frame);
        let value = selected(&history, &frame);
        assert_eq!(value.status, "available");
        assert_eq!(value.used_capacity, Some(2000.0));
        let current = value.same_poll_observed_size.unwrap();
        assert_eq!(current.status, "unavailable");
        assert_eq!(current.used_capacity, None);
        assert_eq!(history.samples.len(), 9);
    }
}

#[test]
fn current_size_cannot_activate_missing_previous_conversion() {
    let (mut source, mut history, _) = warmed();
    let mut frame = source.next(0.0, None, false);
    current_size(&mut frame, 20000.0, 9);
    observe(&mut history, &frame);
    let value = selected(&history, &frame);
    assert_eq!(value.status, "unavailable");
    assert_eq!(value.reason, "current_prior_conversion_unavailable");
    assert_eq!(value.used_capacity, None);
    assert!(value.same_poll_observed_size.is_none());
}

#[test]
fn changed_current_size_on_duplicate_invalidates_only_new_optional_basis() {
    let (mut source, mut history, _) = warmed();
    let mut frame = source.next(3.0, Some(1000.0), true);
    current_size(&mut frame, 20000.0, 9);
    observe(&mut history, &frame);
    let before: Vec<_> = history
        .samples
        .iter()
        .map(|s| (s.raw_rate, s.original_size))
        .collect();
    current_size(&mut frame, 21000.0, 9);
    observe(&mut history, &frame);
    let value = selected(&history, &frame);
    assert_eq!(value.status, "available");
    assert_eq!(value.used_capacity, Some(2000.0));
    let current = value.same_poll_observed_size.unwrap();
    assert_eq!(current.reason, "duplicate_current_observed_size_changed");
    assert_eq!(current.used_capacity, None);
    assert_eq!(
        before,
        history
            .samples
            .iter()
            .map(|s| (s.raw_rate, s.original_size))
            .collect::<Vec<_>>()
    );
}

#[test]
fn startup_empty_window_requires_eight_positive_intervals_and_keeps_floor() {
    let mut source = Source::default();
    let mut history = History::default();
    let initial = source.initial();
    assert!(initial.started < 0.0);
    observe(&mut history, &initial);
    assert!(history.complete);
    unavailable(selected(&history, &initial));
    for n in 1..=MIN_SAMPLES {
        let frame = source.next(1.0, Some(1000.0), true);
        observe(&mut history, &frame);
        assert_eq!(history.samples.len(), n);
        let value = selected(&history, &frame);
        if n < MIN_SAMPLES {
            unavailable(value);
        } else {
            assert_eq!(value.status, "available");
            assert_eq!(value.lower_quartile_request_start_rate, Some(1.0));
            assert_eq!(value.rebased_learned_capacity, Some(1000.0));
            assert_eq!(value.used_capacity, Some(2000.0));
        }
    }
}

#[test]
fn ineligible_zero_and_positive_intervals_do_not_enter_the_positive_window() {
    let (mut source, mut history, _) = warmed();
    for rate in [0.0, 4.0] {
        let frame = source.next(rate, Some(3000.0), false);
        observe(&mut history, &frame);
        assert_eq!(history.samples.len(), MIN_SAMPLES);
        assert_eq!(selected(&history, &frame).used_capacity, Some(3000.0));
        assert!(history
            .samples
            .iter()
            .all(|sample| sample.original_size == 1000.0));
    }
}

#[test]
fn current_mean_rebases_raw_quantile_without_reweighting_or_mutating_membership() {
    let mut source = Source::default();
    let mut history = History::default();
    observe(&mut history, &source.initial());
    for (rate, size) in (1..=8).zip((1..=8).rev()) {
        let frame = source.next(rate as f64, Some(size as f64 * 1000.0), true);
        observe(&mut history, &frame);
    }
    // The original converted lower quartile is 8000, while the raw-rate
    // lower quartile is 2. One current size applies to every existing member.
    let frame = source.next(0.0, Some(3000.0), false);
    observe(&mut history, &frame);
    assert_eq!(frame.load.loads[0].learned_prefill_capacity, Some(8000.0));
    let value = selected(&history, &frame);
    assert_eq!(value.lower_quartile_request_start_rate, Some(2.0));
    assert_eq!(value.rebased_learned_capacity, Some(6000.0));
    assert_eq!(value.used_capacity, Some(6000.0));
    assert_eq!(history.samples.len(), 8);
    assert_eq!(history.samples.front().unwrap().original_size, 8000.0);
    assert_eq!(history.samples.back().unwrap().original_size, 1000.0);
    assert_eq!(
        history
            .selection(frame.started, source_key(&frame.load), 7000.0)
            .used_capacity,
        Some(7000.0)
    );
}

#[test]
fn late_enable_stays_unknown_until_thirty_two_witnessed_positive_intervals() {
    let mut source = Source::default();
    for _ in 0..12 {
        source.next(1.0, Some(1000.0), true);
    }
    let mut history = History::default();
    for n in 1..=WINDOW {
        let frame = source.next(1.0, Some(1000.0), true);
        observe(&mut history, &frame);
        if n < WINDOW {
            assert!(!history.complete);
            unavailable(selected(&history, &frame));
        } else {
            assert!(history.complete);
            assert_eq!(selected(&history, &frame).used_capacity, Some(2000.0));
        }
    }
}

#[test]
fn missed_positive_interval_drops_prefix_and_recovers_only_after_full_window() {
    let (mut source, mut history, _) = warmed();
    source.next(2.0, Some(1000.0), true); // Deliberately unobserved.
    for n in 1..=WINDOW {
        let frame = source.next(1.0, Some(1000.0), true);
        observe(&mut history, &frame);
        if n < WINDOW {
            assert!(!history.complete);
            unavailable(selected(&history, &frame));
        } else {
            assert_eq!(selected(&history, &frame).status, "available");
        }
    }
}

#[test]
fn missed_zero_interval_is_a_time_gap_even_when_counter_is_unchanged() {
    let (mut source, mut history, _) = warmed();
    source.next(0.0, Some(1000.0), false); // No counter increment to reveal this gap.
    let frame = source.next(0.0, Some(1000.0), false);
    observe(&mut history, &frame);
    assert_eq!(history.reason, "counter_or_time_adjacency_gap");
    assert!(history.samples.is_empty());
    unavailable(selected(&history, &frame));
}

#[test]
fn counter_or_interval_start_mismatch_cannot_retain_a_complete_window() {
    for counter_gap in [true, false] {
        let (mut source, mut history, _) = warmed();
        let mut frame = source.next(1.0, Some(1000.0), true);
        let p = provenance(&mut frame);
        if counter_gap {
            p.previous_raw_counter = p.previous_raw_counter.map(|value| value - 1.0);
            p.raw_rate = Some(2.0);
        } else {
            p.interval_started_time_seconds =
                p.interval_started_time_seconds.map(|value| value - 1.0);
            p.interval_seconds = Some(2.0);
            p.raw_rate = Some(0.5);
        }
        observe(&mut history, &frame);
        assert!(!history.complete);
        unavailable(selected(&history, &frame));
    }
}

#[test]
fn exact_duplicate_is_idempotent_but_conflict_and_out_of_order_invalidate() {
    let (mut source, mut history, frame) = warmed();
    let count = history.samples.len();
    observe(&mut history, &frame);
    assert_eq!(history.samples.len(), count);
    assert_eq!(selected(&history, &frame).status, "available");
    let mut conflict = Frame {
        started: frame.started,
        load: frame.load.clone(),
    };
    provenance(&mut conflict).previous_conversion_size_mean = Some(2000.0);
    observe(&mut history, &conflict);
    assert!(history.samples.is_empty());
    unavailable(selected(&history, &conflict));

    let (_, mut history, frame) = warmed();
    history.observe(Some(&frame.load), frame.started - 1.0, true);
    assert!(!history.complete);
    let later = source.next(1.0, Some(1000.0), true);
    unavailable(selected(&history, &later));
}

#[test]
fn failed_or_missing_poll_invalidates_history_without_zero_credit() {
    for failed in [true, false] {
        let (mut source, mut history, _) = warmed();
        let frame = source.next(0.0, Some(1000.0), false);
        history.observe(
            if failed { Some(&frame.load) } else { None },
            frame.started,
            !failed,
        );
        assert!(history.samples.is_empty());
        assert!(!history.complete);
        unavailable(selected(&history, &frame));
    }
}

#[test]
fn explicit_empty_count_proves_reset_but_ambiguous_full_count_does_not() {
    let (mut source, mut history, _) = warmed();
    let empty = source.reset_empty();
    observe(&mut history, &empty);
    assert!(history.complete && history.samples.is_empty());
    unavailable(selected(&history, &empty));
    for n in 1..=MIN_SAMPLES {
        let frame = source.next(1.0, Some(1000.0), true);
        observe(&mut history, &frame);
        if n == MIN_SAMPLES {
            assert_eq!(selected(&history, &frame).status, "available");
        }
    }

    let (mut source, mut history, _) = warmed();
    for _ in MIN_SAMPLES..WINDOW {
        let frame = source.next(1.0, Some(1000.0), true);
        observe(&mut history, &frame);
    }
    source.epoch += 1;
    let mut ambiguous = source.next(0.0, Some(1000.0), false);
    let p = provenance(&mut ambiguous);
    p.counter_reset = true;
    p.raw_rate = None;
    observe(&mut history, &ambiguous);
    assert!(!history.complete && history.samples.is_empty());
    unavailable(selected(&history, &ambiguous));
    for n in 1..=WINDOW {
        let frame = source.next(1.0, Some(1000.0), true);
        observe(&mut history, &frame);
        if n < WINDOW {
            unavailable(selected(&history, &frame));
        } else {
            assert_eq!(selected(&history, &frame).status, "available");
        }
    }
}

#[test]
fn profile_changes_and_nonstarted_or_multirank_sources_fail_closed() {
    let (mut source, mut history, _) = warmed();
    source.profile += 1;
    let frame = source.next(1.0, Some(1000.0), true);
    observe(&mut history, &frame);
    assert!(!history.complete);
    unavailable(selected(&history, &frame));
    for kind in ["local_compute_tokens", "completed_uncached_tokens"] {
        let (mut source, mut history, _) = warmed();
        let mut frame = source.next(1.0, Some(1000.0), true);
        frame.load.loads[0].timely_prefill_progress_source = Some(kind.into());
        provenance(&mut frame).progress_source = Some(kind.into());
        assert!(source_key(&frame.load).is_none());
        observe(&mut history, &frame);
        unavailable(selected(&history, &frame));
    }
    let (_, _, mut frame) = warmed();
    frame.load.dp_rank_count = 2;
    frame.load.loads.push(frame.load.loads[0].clone());
    assert!(source_key(&frame.load).is_none());
}

#[test]
fn source_availability_schema_and_interval_eligibility_must_be_explicit() {
    for case in 0..4 {
        let (mut source, mut history, _) = warmed();
        let mut frame = source.next(1.0, Some(1000.0), true);
        match case {
            0 => frame.load.loads[0].native_request_counts_available = None,
            1 => provenance(&mut frame).schema_version = 2,
            2 => provenance(&mut frame).progress_source = None,
            _ => provenance(&mut frame).capacity_interval_eligible = false,
        }
        observe(&mut history, &frame);
        assert!(!history.complete);
        unavailable(selected(&history, &frame));
    }
}

#[test]
fn current_mean_missing_zero_negative_or_nan_is_unavailable_not_imputed() {
    for size in [None, Some(0.0), Some(-1.0), Some(f64::NAN)] {
        let (mut source, mut history, _) = warmed();
        let frame = source.next(0.0, size, false);
        observe(&mut history, &frame);
        assert!(history.complete);
        let value = selected(&history, &frame);
        assert_eq!(value.reason, "current_prior_conversion_unavailable");
        unavailable(value);
    }
}

#[test]
fn eligible_original_mean_and_declared_old_capacity_must_be_valid() {
    for invalid_mean in [true, false] {
        let (mut source, mut history, _) = warmed();
        let mut frame = source.next(1.0, Some(1000.0), true);
        if invalid_mean {
            provenance(&mut frame).previous_conversion_size_mean = None;
        } else {
            frame.load.loads[0].learned_prefill_capacity = Some(999.0);
        }
        observe(&mut history, &frame);
        assert!(!history.complete);
        unavailable(selected(&history, &frame));
    }
}

#[test]
fn malformed_raw_rate_dt_and_interval_endpoints_cannot_credit_samples() {
    for case in 0..6 {
        let (mut source, mut history, _) = warmed();
        let mut frame = source.next(1.0, Some(1000.0), true);
        let p = provenance(&mut frame);
        match case {
            0 => p.raw_rate = None,
            1 => p.raw_rate = Some(2.0),
            2 => p.interval_seconds = Some(0.0),
            3 => p.interval_seconds = Some(f64::NAN),
            4 => p.interval_ended_time_seconds = p.interval_ended_time_seconds.map(|v| v + 1.0),
            _ => p.interval_ended_time_seconds = Some(f64::INFINITY),
        }
        observe(&mut history, &frame);
        assert!(!history.complete);
        unavailable(selected(&history, &frame));
    }
}

#[test]
fn invalid_poll_clock_and_configured_floor_are_explicitly_unavailable() {
    let (_, mut history, frame) = warmed();
    for floor in [0.0, -1.0, f64::NAN] {
        let value = history.selection(frame.started, source_key(&frame.load), floor);
        assert_eq!(value.reason, "rebased_capacity_or_configured_floor_invalid");
        unavailable(value);
    }
    history.observe(Some(&frame.load), f64::NAN, true);
    assert!(!history.complete);
    unavailable(selected(&history, &frame));
}

#[test]
fn captured_poll_mismatch_never_exports_newer_history_or_conversion_mean() {
    let (_, history, frame) = warmed();
    let value = history.selection(frame.started - 1.0, source_key(&frame.load), 2000.0);
    unavailable(value);
    assert!(!value.history_complete);
    assert_eq!(value.history_samples, 0);
    assert_eq!(value.previous_conversion_size_mean, None);
}
