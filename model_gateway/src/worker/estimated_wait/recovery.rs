//! Experimental net-population drain extrapolation. This is not measured
//! remaining token work, service throughput, or a native cancellation ACK.
use axum::{body::Body, response::Response};
use openai_protocol::worker::WorkerLoadResponse;
use std::{
    collections::VecDeque,
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

pub(super) const METHOD: &str = "live_net_depletion_v1";
pub(super) const PROBE_INTERVAL: Duration = Duration::from_secs(10);
const SHORT: Duration = Duration::from_secs(10);
const LONG: Duration = Duration::from_secs(30);
const HISTORY: Duration = Duration::from_secs(120);
const MAX_POLLS: usize = 121;

#[derive(Clone, Copy, Debug)]
struct Point {
    at: Instant,
    waiting: f64,
    population: f64,
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
}
impl RecoveryTracker {
    pub(super) fn reset(&mut self, reason: &'static str) {
        self.points.clear();
        self.ranks.clear();
        self.reason = reason;
    }
    pub(super) fn observe(&mut self, load: &WorkerLoadResponse, at: Instant, max_gap: f64) {
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
        let point = Point {
            at,
            waiting,
            population,
        };
        if !self.ranks.is_empty() && self.ranks != ranks {
            self.reset("rank_profile_changed");
        }
        if let Some(last) = self.points.back() {
            if at.duration_since(last.at).as_secs_f64() > max_gap {
                self.reset("poll_gap");
            } else if waiting > last.waiting || population > last.population {
                self.reset("population_rebound");
            }
        }
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
        }
    }
}

/// One actual admission exception. The counter survives worker generations so
/// stale evidence resets cannot silently free an occupied probe slot within
/// this worker incarnation. A replacement worker has independent state.
#[derive(Debug)]
pub(crate) struct RecoveryProbePermit {
    inflight: Arc<AtomicUsize>,
    worker: String,
    request_id: String,
}
impl RecoveryProbePermit {
    pub(super) fn reserve(
        inflight: &Arc<AtomicUsize>,
        worker: &str,
        request_id: Option<&str>,
    ) -> Option<Self> {
        inflight
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .ok()?;
        tracing::debug!(target: "smg::estimated_wait::observation", event = "recovery_probe_reserved", diagnostic_time_seconds = super::diagnostic_time(Instant::now()), worker, request_id = request_id.unwrap_or(""), max_inflight = 1, min_interval_seconds = PROBE_INTERVAL.as_secs());
        Some(Self {
            inflight: Arc::clone(inflight),
            worker: worker.to_owned(),
            request_id: request_id.unwrap_or("").to_owned(),
        })
    }
    pub(crate) fn attach(self, response: Response) -> Response {
        tracing::debug!(target: "smg::estimated_wait::observation", event = "recovery_probe_response", diagnostic_time_seconds = super::diagnostic_time(Instant::now()), worker = self.worker, request_id = self.request_id, status = response.status().as_u16());
        let (parts, body) = response.into_parts();
        Response::from_parts(
            parts,
            Body::new(ProbeBody {
                body,
                permit: Some(self),
            }),
        )
    }
}
impl Drop for RecoveryProbePermit {
    fn drop(&mut self) {
        self.inflight.fetch_sub(1, Ordering::AcqRel);
        tracing::debug!(target: "smg::estimated_wait::observation", event = "recovery_probe_released", diagnostic_time_seconds = super::diagnostic_time(Instant::now()), worker = self.worker, request_id = self.request_id, native_cancellation_acknowledgement = "unavailable");
    }
}
struct ProbeBody {
    body: Body,
    permit: Option<RecoveryProbePermit>,
}
impl http_body::Body for ProbeBody {
    type Data = bytes::Bytes;
    type Error = axum::Error;
    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.body).poll_frame(cx);
        if matches!(polled, Poll::Ready(None) | Poll::Ready(Some(Err(_)))) {
            this.permit.take();
        }
        polled
    }
    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }
    fn size_hint(&self) -> http_body::SizeHint {
        self.body.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;
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
                );
            }
            assert_eq!(forecast(&history, start + LONG).available, !transfer);
        }
        let mut history = declining(start);
        for i in 7..=9 {
            history.observe(&load(40, 10), start + Duration::from_secs(i * 5), 30.0);
        }
        assert!(!forecast(&history, start + Duration::from_secs(45)).available);
    }
    #[test]
    fn recovery_cancel_burst_does_not_establish_a_sustained_short_window() {
        let start = Instant::now();
        let mut history = RecoveryTracker::default();
        history.observe(&load(1000, 10), start, 30.0);
        for i in 1..=6 {
            history.observe(&load(10, 10), start + Duration::from_secs(i * 5), 30.0);
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
            history.observe(&report, start + Duration::from_secs(time), 30.0);
            assert!(!forecast(&history, start + Duration::from_secs(time)).available);
            report.loads[0].num_running_reqs = -1;
            history.observe(&report, start + Duration::from_secs(time + 1), 30.0);
            assert_eq!(history.reason, "native_counts_unavailable");
        }
    }
    #[test]
    fn recovery_stale_and_duplicate_or_older_polls_do_not_resurrect_or_erase_evidence() {
        let start = Instant::now();
        let mut history = declining(start);
        history.observe(&load(999, 999), start + LONG, 30.0);
        history.observe(&load(999, 999), start, 30.0);
        assert!(forecast(&history, start + LONG).available);
        assert_eq!(
            forecast(&history, start + Duration::from_secs(61)).reason,
            "stale"
        );
        history.reset("failed");
        assert_eq!(forecast(&history, start + LONG).reason, "failed");
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
    #[tokio::test]
    async fn recovery_probe_permit_is_atomic_and_released_on_eof_error_or_client_drop() {
        let count = Arc::new(AtomicUsize::new(0));
        let first = RecoveryProbePermit::reserve(&count, "w", Some("r")).unwrap();
        assert!(RecoveryProbePermit::reserve(&count, "w", Some("other")).is_none());
        let mut body = first
            .attach(Response::new(Body::from("first text is not completion")))
            .into_body();
        assert!(body.frame().await.unwrap().is_ok());
        assert_eq!(count.load(Ordering::Acquire), 1);
        assert!(body.frame().await.is_none());
        assert_eq!(count.load(Ordering::Acquire), 0);
        let permit = RecoveryProbePermit::reserve(&count, "w", None).unwrap();
        let stream = futures::stream::iter([Err::<bytes::Bytes, _>(std::io::Error::other(
            "upstream failed",
        ))]);
        let mut body = permit
            .attach(Response::new(Body::from_stream(stream)))
            .into_body();
        assert!(body.frame().await.unwrap().is_err());
        assert_eq!(count.load(Ordering::Acquire), 0);
        let permit = RecoveryProbePermit::reserve(&count, "w", None).unwrap();
        drop(permit.attach(Response::new(Body::from("cancelled"))));
        assert_eq!(count.load(Ordering::Acquire), 0);
    }
}
