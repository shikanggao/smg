//! Experimental net-population drain extrapolation. This is not measured
//! remaining token work, service throughput, or a native cancellation ACK.
use openai_protocol::worker::WorkerLoadResponse;
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

pub(super) const METHOD: &str = "live_arrival_bounded_depletion_v2";
const SHORT: Duration = Duration::from_secs(10);
const LONG: Duration = Duration::from_secs(30);
const HISTORY: Duration = Duration::from_secs(120);
const MAX_POLLS: usize = 121;

#[derive(Clone, Copy, Debug)]
struct Point {
    at: Instant,
    waiting: f64,
    population: f64,
    request_watermark: u64,
    new_requests: Option<u64>,
    rebound_retained: bool,
}

#[derive(Debug)]
pub(super) struct RecoveryTracker {
    points: VecDeque<Point>,
    ranks: Vec<(i32, Option<u64>)>,
    pub(super) reason: &'static str,
}
impl Default for RecoveryTracker {
    fn default() -> Self {
        Self {
            points: VecDeque::new(),
            ranks: Vec::new(),
            reason: "insufficient_history",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RecoveryForecast {
    pub available: bool,
    pub reason: &'static str,
    pub queue_extra: f64,
    pub total: Option<f64>,
    pub legacy_gain_total: Option<f64>,
    pub rate: Option<f64>,
    pub horizon: f64,
    pub points: usize,
    pub debt: f64,
    pub population: Option<f64>,
    pub waiting: Option<f64>,
    pub gain: f64,
    pub profile: Option<u64>,
    pub window_started: Option<Instant>,
    pub latest_poll: Option<Instant>,
    pub poll_request_watermark: Option<u64>,
    pub requests_since_previous_poll: Option<u64>,
    pub rebound_allowance: u64,
    pub rebound_retained: bool,
}
impl RecoveryTracker {
    pub(super) fn reset(&mut self, reason: &'static str) {
        self.points.clear();
        self.ranks.clear();
        self.reason = reason;
    }
    pub(super) fn observe(
        &mut self,
        load: &WorkerLoadResponse,
        at: Instant,
        max_gap: f64,
        request_watermark: u64,
    ) {
        if self.points.back().is_some_and(|p| at <= p.at) {
            return;
        }
        let mut ranks: Vec<_> = load
            .loads
            .iter()
            .map(|r| (r.dp_rank, r.native_request_counts_profile))
            .collect();
        ranks.sort_unstable();
        if ranks.is_empty()
            || ranks.windows(2).any(|w| w[0].0 == w[1].0)
            || (load.dp_rank_count > 0 && load.dp_rank_count as usize != ranks.len())
            || load.loads.iter().any(|r| {
                r.native_request_counts_available != Some(true)
                    || r.num_waiting_reqs < 0
                    || r.num_running_reqs < 0
            })
        {
            self.reset("native_counts_unavailable");
            return;
        }
        let waiting = load
            .loads
            .iter()
            .map(|r| f64::from(r.num_waiting_reqs))
            .sum();
        let population = load
            .loads
            .iter()
            .map(|r| f64::from(r.num_waiting_reqs) + f64::from(r.num_running_reqs))
            .sum();
        if !self.ranks.is_empty() && self.ranks != ranks {
            self.reset("rank_profile_changed");
        }
        let mut new_requests = None;
        let mut rebound_retained = false;
        if let Some(last) = self.points.back() {
            if at.duration_since(last.at).as_secs_f64() > max_gap {
                self.reset("poll_gap");
            } else if let Some(arrivals) = request_watermark.checked_sub(last.request_watermark) {
                new_requests = Some(arrivals);
                if waiting > last.waiting || population > last.population {
                    // Local admissions support continuity only. They are NOT
                    // added to the measured net-depletion rate or subtracted
                    // from native work; native acceptance is unobservable.
                    let allowance = arrivals.min(1) as f64;
                    if waiting - last.waiting > allowance
                        || population - last.population > allowance
                    {
                        self.reset("population_rebound");
                    } else {
                        rebound_retained = true;
                    }
                }
            } else {
                self.reset("arrival_counter_reset");
            }
        }
        let point = Point {
            at,
            waiting,
            population,
            request_watermark,
            new_requests,
            rebound_retained,
        };
        // Zero may mean a genuine drain or a native process reset. Never use a
        // jump to zero to project the speed of a subsequent population.
        if population == 0.0 {
            self.reset("native_empty");
            return;
        }
        self.ranks = ranks;
        self.points.push_back(point);
        while self.points.len() > MAX_POLLS
            || self
                .points
                .front()
                .is_some_and(|p| at.duration_since(p.at) > HISTORY)
        {
            self.points.pop_front();
        }
    }
    pub(super) fn forecast(
        &self,
        now: Instant,
        max_age: f64,
        queue: f64,
        debt: f64,
        baseline: Option<f64>,
        gain: f64,
    ) -> RecoveryForecast {
        use std::hash::{Hash, Hasher};
        let profile = (!self.ranks.is_empty()).then(|| {
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            self.ranks.hash(&mut hash);
            hash.finish()
        });
        let fallback = |reason| RecoveryForecast {
            available: false,
            reason,
            queue_extra: queue + debt,
            total: baseline.map(|b| b + queue + debt),
            legacy_gain_total: baseline.map(|b| b + gain * (queue + debt)),
            rate: None,
            horizon: 0.0,
            points: self.points.len(),
            debt,
            population: self.points.back().map(|p| p.population),
            waiting: self.points.back().map(|p| p.waiting),
            gain,
            profile,
            window_started: self.points.front().map(|p| p.at),
            latest_poll: self.points.back().map(|p| p.at),
            poll_request_watermark: self.points.back().map(|p| p.request_watermark),
            requests_since_previous_poll: self.points.back().and_then(|p| p.new_requests),
            rebound_allowance: self
                .points
                .back()
                .and_then(|p| p.new_requests)
                .unwrap_or(0)
                .min(1),
            rebound_retained: self.points.back().is_some_and(|p| p.rebound_retained),
        };
        let Some(last) = self.points.back() else {
            return fallback(self.reason);
        };
        if now.saturating_duration_since(last.at).as_secs_f64() > max_age {
            return fallback("stale");
        }
        if self.points.len() < 5 {
            return fallback("insufficient_history");
        }
        let endpoint = |span| {
            self.points
                .iter()
                .rev()
                .find(|p| last.at.duration_since(p.at) >= span)
        };
        let (Some(short), Some(long)) = (endpoint(SHORT), endpoint(LONG)) else {
            return fallback("insufficient_history");
        };
        let rate = |p: &Point| {
            if last.waiting >= p.waiting || last.population >= p.population {
                None
            } else {
                Some((p.population - last.population) / last.at.duration_since(p.at).as_secs_f64())
            }
        };
        let (Some(short_rate), Some(long_rate)) = (rate(short), rate(long)) else {
            return fallback("no_sustained_decline");
        };
        let rate = short_rate.min(long_rate);
        let extra = queue.min(last.population / rate) + debt;
        RecoveryForecast {
            available: true,
            reason: "sustained_decline",
            queue_extra: extra,
            total: baseline.map(|b| b + extra),
            legacy_gain_total: baseline.map(|b| b + gain * extra),
            rate: Some(rate),
            horizon: last.at.duration_since(long.at).as_secs_f64(),
            points: self.points.len(),
            debt,
            population: Some(last.population),
            waiting: Some(last.waiting),
            gain,
            profile,
            window_started: Some(long.at),
            latest_poll: Some(last.at),
            poll_request_watermark: Some(last.request_watermark),
            requests_since_previous_poll: last.new_requests,
            rebound_allowance: last.new_requests.unwrap_or(0).min(1),
            rebound_retained: last.rebound_retained,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openai_protocol::worker::SchedulerLoadSnapshot;
    fn load(waiting: i32, running: i32) -> WorkerLoadResponse {
        WorkerLoadResponse {
            loads: vec![SchedulerLoadSnapshot {
                num_waiting_reqs: waiting,
                num_running_reqs: running,
                native_request_counts_available: Some(true),
                ..Default::default()
            }],
            ..Default::default()
        }
    }
    fn declining(start: Instant) -> RecoveryTracker {
        let mut history = RecoveryTracker::default();
        for i in 0..7 {
            history.observe(
                &load(100 - i * 10, 10),
                start + Duration::from_secs(i as u64 * 5),
                30.0,
                0,
            );
        }
        history
    }
    fn forecast(history: &RecoveryTracker, at: Instant) -> RecoveryForecast {
        history.forecast(at, 30.0, 1000.0, 7.0, Some(2.0), 2.0)
    }

    #[test]
    fn recovery_uses_population_decline_without_any_first_text_and_preserves_dispatch_debt() {
        let start = Instant::now();
        let history = declining(start);
        let f = forecast(&history, start + LONG);
        assert!(f.available);
        assert_eq!(f.rate, Some(2.0));
        assert_eq!(f.queue_extra, 25.0 + 7.0);
        assert_eq!(f.total, Some(34.0));
        assert_eq!(f.legacy_gain_total, Some(66.0));
        // A small queue estimate remains unchanged; debt is not discounted.
        assert_eq!(
            history
                .forecast(start + LONG, 30.0, 3.0, 7.0, Some(2.0), 2.0)
                .queue_extra,
            10.0
        );
    }
    #[test]
    fn recovery_rejects_waiting_to_running_flat_population_and_stall() {
        let start = Instant::now();
        for transfer in [false, true] {
            let mut history = RecoveryTracker::default();
            for i in 0..7 {
                history.observe(
                    &load(100 - i * 10, if transfer { i * 10 } else { 100 }),
                    start + Duration::from_secs(i as u64 * 5),
                    30.0,
                    0,
                );
            }
            assert_eq!(forecast(&history, start + LONG).available, !transfer);
        }
        let mut history = declining(start);
        for i in 7..=9 {
            history.observe(&load(40, 10), start + Duration::from_secs(i * 5), 30.0, 0);
        }
        assert!(!forecast(&history, start + Duration::from_secs(45)).available);
    }
    #[test]
    fn recovery_cancel_burst_does_not_establish_a_sustained_short_window() {
        let start = Instant::now();
        let mut history = RecoveryTracker::default();
        history.observe(&load(1000, 10), start, 30.0, 0);
        for i in 1..=6 {
            history.observe(&load(10, 10), start + Duration::from_secs(i * 5), 30.0, 0);
        }
        assert!(!forecast(&history, start + LONG).available);
        // No assertion of completion: a sustained cancellation drain can qualify;
        // this extrapolation is deliberately shadow and not a service-rate fact.
    }
    #[test]
    fn recovery_rebound_gap_empty_missing_and_profile_change_clear_history() {
        let start = Instant::now();
        for (mut report, time) in [
            (load(50, 10), 35),
            (load(30, 10), 61),
            (load(0, 0), 35),
            (load(30, 10), 35),
            (load(30, 10), 35),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (mut l, t))| {
            if i == 3 {
                l.loads[0].native_request_counts_available = None;
            }
            if i == 4 {
                l.loads[0].dp_rank = 1;
            }
            (l, t)
        }) {
            let mut history = declining(start);
            history.observe(&report, start + Duration::from_secs(time), 30.0, 0);
            assert!(!forecast(&history, start + Duration::from_secs(time)).available);
            report.loads[0].num_running_reqs = -1;
            history.observe(&report, start + Duration::from_secs(time + 1), 30.0, 0);
            assert_eq!(history.reason, "native_counts_unavailable");
        }
    }
    #[test]
    fn recovery_stale_and_duplicate_or_older_polls_do_not_resurrect_or_erase_evidence() {
        let start = Instant::now();
        let mut history = declining(start);
        history.observe(&load(999, 999), start + LONG, 30.0, 0);
        history.observe(&load(999, 999), start, 30.0, 0);
        assert!(forecast(&history, start + LONG).available);
        assert_eq!(
            forecast(&history, start + Duration::from_secs(61)).reason,
            "stale"
        );
        history.reset("failed");
        assert_eq!(forecast(&history, start + LONG).reason, "failed");
    }
    #[test]
    fn arrival_supported_single_bump_preserves_actual_net_rate_only() {
        let start = Instant::now();
        let mut history = declining(start);
        history.observe(&load(41, 10), start + Duration::from_secs(35), 30.0, 1);
        let f = forecast(&history, start + Duration::from_secs(35));
        assert!(f.available && f.rebound_retained);
        assert_eq!(f.requests_since_previous_poll, Some(1));
        assert_eq!(f.rate, Some(0.9)); // (P25 - P35)/10; arrival NOT added.
        assert_eq!(f.queue_extra, 51.0 / 0.9 + 7.0);
        // A second bump cannot reuse the already consumed credit.
        history.observe(&load(42, 10), start + Duration::from_secs(40), 30.0, 1);
        assert_eq!(history.reason, "population_rebound");
        assert_eq!(history.points.len(), 1);
    }

    #[test]
    fn arrivals_do_not_allow_large_unaccounted_or_net_growing_population() {
        let start = Instant::now();
        for (waiting, running, arrivals) in [(41, 10, 0), (42, 10, 100), (41, 11, 100)] {
            let mut history = declining(start);
            history.observe(
                &load(waiting, running),
                start + Duration::from_secs(35),
                30.0,
                arrivals,
            );
            assert_eq!(history.reason, "population_rebound");
            assert!(!forecast(&history, start + Duration::from_secs(35)).available);
        }
        let mut history = RecoveryTracker::default();
        for i in 0..10 {
            history.observe(
                &load(10 + i, 10),
                start + Duration::from_secs(i as u64 * 5),
                30.0,
                i as u64,
            );
        }
        assert_eq!(
            forecast(&history, start + Duration::from_secs(45)).reason,
            "no_sustained_decline"
        );
    }

    #[test]
    fn arrival_watermarks_ignore_older_polls_and_fail_closed_on_reset_or_saturation() {
        let start = Instant::now();
        let mut history = declining(start);
        history.observe(&load(999, 10), start + LONG, 30.0, 100);
        history.observe(&load(999, 10), start, 30.0, 100);
        history.observe(&load(41, 10), start + Duration::from_secs(35), 30.0, 1);
        assert!(forecast(&history, start + Duration::from_secs(35)).available);
        history.observe(&load(30, 10), start + Duration::from_secs(40), 30.0, 0);
        assert_eq!(history.reason, "arrival_counter_reset");
        assert_eq!(history.points.len(), 1);
        history.reset("failed");
        history.observe(
            &load(20, 10),
            start + Duration::from_secs(45),
            30.0,
            u64::MAX,
        );
        history.observe(
            &load(21, 10),
            start + Duration::from_secs(50),
            30.0,
            u64::MAX,
        );
        assert_eq!(history.reason, "population_rebound");
        assert_eq!(history.points.len(), 1);
    }

    #[test]
    fn recovery_history_memory_is_bounded_and_unknown_baseline_stays_unknown() {
        let start = Instant::now();
        let mut history = RecoveryTracker::default();
        for i in 0..500 {
            history.observe(
                &load(1000 - i, 1),
                start + Duration::from_secs(i as u64),
                30.0,
                0,
            );
        }
        assert!(history.points.len() <= MAX_POLLS);
        let f = history.forecast(
            start + Duration::from_secs(499),
            30.0,
            1000.0,
            0.0,
            None,
            1.0,
        );
        assert!(f.available);
        assert!(f.total.is_none());
        assert!(f.legacy_gain_total.is_none());
    }
}
