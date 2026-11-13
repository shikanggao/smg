//! Local activity and first-text deadline observations for the C2 shadow candidate.
//! Rates describe resolved local observations, not native cancellation acknowledgements.
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use serde::Serialize;

pub const METHOD: &str = "local_ttft_activity_v1";

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Context {
    pub values: [Option<f64>; 3],
    pub counts: Counts,
    pub bucket_second: u64,
    pub activity_origin_time_seconds: f64,
    pub deadline_untracked_pending: u64,
}
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Counts {
    pub first_text: u64,
    pub upstream_timeout: u64,
    pub native_deadline_cutoff: u64,
    pub pretext_downstream_disconnect: u64,
    pub unknown_pretext_end: u64,
    pub admitted: u64,
    pub actual_rejected: u64,
    pub untracked_admitted: u64,
    pub deadline_timely_first_text: u64,
    pub deadline_breach: u64,
    pub deadline_or_cutoff_adverse: u64,
    pub deadline_or_cutoff_unknown: u64,
}
#[derive(Clone, Copy, Debug)]
pub enum Outcome {
    FirstText,
    Timeout,
    NativeDeadline,
    Disconnect,
    Unknown,
    Admitted,
    Rejected,
    UntrackedAdmitted,
    DeadlineTimely,
    DeadlineBreach,
    ActualCutoff,
    DeadlineUnknown,
}
impl Outcome {
    pub fn name(self) -> &'static str {
        match self {
            Self::FirstText => "first_text",
            Self::Timeout => "upstream_timeout",
            Self::NativeDeadline => "native_hard_ttft_deadline",
            Self::Disconnect => "pretext_downstream_disconnect",
            Self::Unknown => "unknown_pretext_end",
            Self::Admitted | Self::UntrackedAdmitted => "admitted",
            Self::Rejected => "actual_rejected",
            Self::DeadlineTimely => "timely_first_text",
            Self::DeadlineBreach => "deadline_breach",
            Self::ActualCutoff => "actual_cutoff",
            Self::DeadlineUnknown => "unknown_pretext_end",
        }
    }
}
impl Counts {
    fn add(&mut self, other: Self) {
        self.first_text = self.first_text.saturating_add(other.first_text);
        self.upstream_timeout = self.upstream_timeout.saturating_add(other.upstream_timeout);
        self.native_deadline_cutoff = self
            .native_deadline_cutoff
            .saturating_add(other.native_deadline_cutoff);
        self.pretext_downstream_disconnect = self
            .pretext_downstream_disconnect
            .saturating_add(other.pretext_downstream_disconnect);
        self.unknown_pretext_end = self
            .unknown_pretext_end
            .saturating_add(other.unknown_pretext_end);
        self.deadline_timely_first_text = self
            .deadline_timely_first_text
            .saturating_add(other.deadline_timely_first_text);
        self.deadline_breach = self.deadline_breach.saturating_add(other.deadline_breach);
        self.deadline_or_cutoff_adverse = self
            .deadline_or_cutoff_adverse
            .saturating_add(other.deadline_or_cutoff_adverse);
        self.deadline_or_cutoff_unknown = self
            .deadline_or_cutoff_unknown
            .saturating_add(other.deadline_or_cutoff_unknown);
        self.admitted = self.admitted.saturating_add(other.admitted);
        self.actual_rejected = self.actual_rejected.saturating_add(other.actual_rejected);
        self.untracked_admitted = self
            .untracked_admitted
            .saturating_add(other.untracked_admitted);
    }
    fn record(&mut self, outcome: Outcome) {
        if matches!(outcome, Outcome::UntrackedAdmitted) {
            self.untracked_admitted = self.untracked_admitted.saturating_add(1);
        }
        if matches!(outcome, Outcome::DeadlineBreach) {
            self.deadline_breach = self.deadline_breach.saturating_add(1);
        }
        let value = match outcome {
            Outcome::FirstText => &mut self.first_text,
            Outcome::Timeout => &mut self.upstream_timeout,
            Outcome::NativeDeadline => &mut self.native_deadline_cutoff,
            Outcome::Disconnect => &mut self.pretext_downstream_disconnect,
            Outcome::Unknown => &mut self.unknown_pretext_end,
            Outcome::Admitted | Outcome::UntrackedAdmitted => &mut self.admitted,
            Outcome::Rejected => &mut self.actual_rejected,
            Outcome::DeadlineTimely => &mut self.deadline_timely_first_text,
            Outcome::DeadlineBreach | Outcome::ActualCutoff => &mut self.deadline_or_cutoff_adverse,
            Outcome::DeadlineUnknown => &mut self.deadline_or_cutoff_unknown,
        };
        *value = value.saturating_add(1);
    }
}
#[derive(Debug)]
pub struct Activity {
    origin: Instant,
    buckets: VecDeque<(u64, Counts)>,
}
impl Activity {
    pub fn new(now: Instant) -> Self {
        Self {
            origin: now,
            buckets: VecDeque::new(),
        }
    }
    pub fn record(&mut self, now: Instant, outcome: Outcome) {
        let second = now.saturating_duration_since(self.origin).as_secs();
        // Callbacks are serialized by the ledger lock. Never insert an older bucket.
        if self.buckets.back().is_some_and(|(time, _)| *time > second) {
            return;
        }
        while self
            .buckets
            .front()
            .is_some_and(|(time, _)| time.saturating_add(60) <= second)
        {
            self.buckets.pop_front();
        }
        if self.buckets.back().is_none_or(|(time, _)| *time != second) {
            self.buckets.push_back((second, Counts::default()));
        }
        #[expect(
            clippy::unwrap_used,
            reason = "the current bucket is inserted immediately above when absent"
        )]
        self.buckets.back_mut().unwrap().1.record(outcome);
    }
    pub fn context(&self, now: Instant, kv: Option<f64>) -> Context {
        let second = now.saturating_duration_since(self.origin).as_secs();
        let mut counts = Counts::default();
        for (time, bucket) in &self.buckets {
            if *time <= second && time.saturating_add(60) > second {
                counts.add(*bucket);
            }
        }
        let adverse = counts.deadline_or_cutoff_adverse;
        let known = counts.deadline_timely_first_text.saturating_add(adverse);
        let resolved = known.saturating_add(counts.deadline_or_cutoff_unknown);
        let attempts = counts.admitted.saturating_add(counts.actual_rejected);
        Context {
            bucket_second: second,
            deadline_untracked_pending: 0,
            activity_origin_time_seconds: super::diagnostic_time(self.origin),
            values: [
                kv.filter(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                (known >= 16
                    && counts.untracked_admitted == 0
                    && (counts.deadline_or_cutoff_unknown as f64) < 0.1 * resolved as f64)
                    .then(|| adverse as f64 / known as f64),
                (attempts >= 16).then(|| counts.actual_rejected as f64 / attempts as f64),
            ],
            counts,
        }
    }
}

/// A bounded local first-text observation, independent of upstream cancellation.
pub const TTFT_DEADLINE: Duration = Duration::from_secs(300);
const MAX_PENDING_DEADLINES: usize = 8192;
#[derive(Clone, Debug)]
pub enum DeadlineHandle {
    Tracked(u64, Arc<AtomicU8>),
    Untracked,
}
// 0 idle, 1 scanning a received chunk, 2 terminal claimed, 3 poll breach.
// Claim before the observation timestamp so a poll cannot overtake timely text
// while its callback waits for the ledger lock. Polling never blocks on scans.
pub struct ObservationClaim(Arc<AtomicU8>);
impl Drop for ObservationClaim {
    fn drop(&mut self) {
        let _ = self
            .0
            .compare_exchange(1, 0, Ordering::SeqCst, Ordering::SeqCst);
    }
}
impl DeadlineHandle {
    pub fn observing(&self) -> Option<ObservationClaim> {
        if let Self::Tracked(_, state) = self {
            state
                .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
                .ok()
                .map(|_| ObservationClaim(Arc::clone(state)))
        } else {
            None
        }
    }
    pub fn claim_terminal(&self) {
        if let Self::Tracked(_, state) = self {
            let _ =
                state.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |s| (s < 2).then_some(2));
        }
    }
}
#[derive(Debug)]
pub struct DeadlineResolution {
    pub tracking_id: Option<u64>,
    pub started: Instant,
    pub request_id: Option<String>,
    pub outcome: Outcome,
}
#[derive(Debug, Default)]
pub struct Deadlines {
    next: u64,
    pending: HashMap<u64, (Instant, Option<String>, Arc<AtomicU8>)>,
    pub untracked: u64,
}
impl Deadlines {
    pub fn register(&mut self, started: Instant, request_id: Option<&str>) -> DeadlineHandle {
        if self.pending.len() >= MAX_PENDING_DEADLINES || self.next == u64::MAX {
            self.untracked = self.untracked.saturating_add(1);
            return DeadlineHandle::Untracked;
        }
        self.next += 1;
        let state = Arc::new(AtomicU8::new(0));
        self.pending.insert(
            self.next,
            (started, request_id.map(str::to_owned), Arc::clone(&state)),
        );
        DeadlineHandle::Tracked(self.next, state)
    }
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
    pub fn advance(&mut self, now: Instant) -> Vec<DeadlineResolution> {
        let mut resolved = Vec::new();
        self.pending.retain(|id, (started, request_id, state)| {
            if now.saturating_duration_since(*started) < TTFT_DEADLINE {
                return true;
            }
            if state
                .compare_exchange(0, 3, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                return true;
            }
            resolved.push(DeadlineResolution {
                tracking_id: Some(*id),
                started: *started,
                request_id: request_id.clone(),
                outcome: Outcome::DeadlineBreach,
            });
            false
        });
        resolved
    }
    pub fn resolve(
        &mut self,
        handle: DeadlineHandle,
        started: Instant,
        request_id: Option<&str>,
        now: Instant,
        terminal: Outcome,
    ) -> Option<DeadlineResolution> {
        let tracking_id = match &handle {
            DeadlineHandle::Tracked(id, _) => Some(*id),
            DeadlineHandle::Untracked => None,
        };
        let (started, request_id) = match handle {
            DeadlineHandle::Tracked(id, _) => {
                let (started, request_id, _) = self.pending.remove(&id)?;
                (started, request_id)
            }
            DeadlineHandle::Untracked => {
                self.untracked = self.untracked.saturating_sub(1);
                (started, request_id.map(str::to_owned))
            }
        };
        let outcome = if now.saturating_duration_since(started) >= TTFT_DEADLINE {
            Outcome::DeadlineBreach
        } else {
            match terminal {
                Outcome::FirstText => Outcome::DeadlineTimely,
                Outcome::Timeout | Outcome::Disconnect | Outcome::NativeDeadline => {
                    Outcome::ActualCutoff
                }
                _ => Outcome::DeadlineUnknown,
            }
        };
        Some(DeadlineResolution {
            tracking_id,
            started,
            request_id,
            outcome,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    #[test]
    fn deadline_poll_is_once_and_late_terminals_do_not_reclassify() {
        let start = Instant::now();
        for terminal in [
            Outcome::FirstText,
            Outcome::Timeout,
            Outcome::Disconnect,
            Outcome::Unknown,
        ] {
            let mut d = Deadlines::default();
            let h = d.register(start, Some("request"));
            assert!(d.advance(start + Duration::from_millis(299_999)).is_empty());
            let resolved = d.advance(start + TTFT_DEADLINE);
            assert_eq!(resolved.len(), 1);
            assert!(matches!(resolved[0].outcome, Outcome::DeadlineBreach));
            assert!(d.advance(start + Duration::from_secs(420)).is_empty());
            assert!(d
                .resolve(
                    h,
                    start,
                    Some("request"),
                    start + Duration::from_secs(420),
                    terminal
                )
                .is_none());
        }
    }
    #[test]
    fn terminal_resolves_deadline_without_a_poll_and_early_cutoff_is_not_breach() {
        let start = Instant::now();
        for (elapsed, terminal, expected) in [
            (299.99, Outcome::Disconnect, "actual_cutoff"),
            (299.99, Outcome::Timeout, "actual_cutoff"),
            (299.99, Outcome::Unknown, "unknown_pretext_end"),
            (299.99, Outcome::FirstText, "timely_first_text"),
            (300.0, Outcome::FirstText, "deadline_breach"),
        ] {
            let mut d = Deadlines::default();
            let h = d.register(start, None);
            h.claim_terminal();
            let r = d
                .resolve(
                    h,
                    start,
                    None,
                    start + Duration::from_secs_f64(elapsed),
                    terminal,
                )
                .unwrap();
            assert_eq!(r.outcome.name(), expected);
            assert!(d.advance(start + Duration::from_secs(420)).is_empty());
        }
    }
    #[test]
    fn timely_scan_claim_cannot_be_overtaken_by_poll_waiting_on_ledger() {
        let start = Instant::now();
        let mut d = Deadlines::default();
        let h = d.register(start, None);
        let scan = h.observing().unwrap();
        // The bytes were observed at299.99; the ledger callback arrives after300.
        let observed = start + Duration::from_millis(299_990);
        assert!(d.advance(start + TTFT_DEADLINE).is_empty());
        h.claim_terminal();
        drop(scan);
        assert!(d.advance(start + Duration::from_secs(301)).is_empty());
        assert!(matches!(
            d.resolve(h, start, None, observed, Outcome::FirstText)
                .unwrap()
                .outcome,
            Outcome::DeadlineTimely
        ));
        let h = d.register(start, None);
        let scan = h.observing().unwrap();
        drop(scan);
        // Metadata/malformed/no-text scans release the claim; next poll can resolve.
        assert_eq!(d.advance(start + TTFT_DEADLINE).len(), 1);
        assert!(h.observing().is_none());
    }
    #[test]
    fn deadline_tracking_is_bounded_and_overflow_drains() {
        let start = Instant::now();
        let mut d = Deadlines::default();
        for _ in 0..MAX_PENDING_DEADLINES {
            assert!(matches!(
                d.register(start, None),
                DeadlineHandle::Tracked(_, _)
            ));
        }
        let h = d.register(start, None);
        assert!(matches!(h, DeadlineHandle::Untracked));
        assert_eq!(d.pending_count(), MAX_PENDING_DEADLINES);
        assert_eq!(d.untracked, 1);
        assert!(matches!(
            d.resolve(
                h,
                start,
                None,
                start + Duration::from_secs(1),
                Outcome::Unknown
            )
            .unwrap()
            .outcome,
            Outcome::DeadlineUnknown
        ));
        assert_eq!(d.untracked, 0);
        assert_eq!(
            d.advance(start + TTFT_DEADLINE).len(),
            MAX_PENDING_DEADLINES
        );
        assert_eq!(d.pending_count(), 0);
        d.next = u64::MAX;
        assert!(matches!(d.register(start, None), DeadlineHandle::Untracked));
    }
    #[test]
    fn deadline_union_keeps_actual_cutoff_diagnostics_separate() {
        let start = Instant::now();
        let mut a = Activity::new(start);
        for _ in 0..15 {
            a.record(start, Outcome::DeadlineTimely);
        }
        a.record(start, Outcome::DeadlineBreach);
        a.record(start, Outcome::Timeout); // Later terminal is diagnostic only.
        a.record(start, Outcome::Unknown); // Already-breached unknown isn't union unknown.
        let c = a.context(start, None);
        assert_eq!(c.values[1], Some(1. / 16.));
        assert_eq!(c.counts.deadline_breach, 1);
        assert_eq!(c.counts.upstream_timeout, 1);
        assert_eq!(c.counts.deadline_or_cutoff_unknown, 0);
        assert_eq!(
            a.context(start + Duration::from_secs(60), None).values[1],
            None
        );
    }

    #[test]
    fn causal_buckets_and_unknown_coverage() {
        let now = Instant::now();
        let mut a = Activity::new(now);
        for _ in 0..16 {
            a.record(now, Outcome::FirstText);
            a.record(now, Outcome::DeadlineTimely);
            a.record(now, Outcome::Admitted);
        }
        a.record(now + Duration::from_secs(1), Outcome::Timeout);
        a.record(now + Duration::from_secs(1), Outcome::ActualCutoff);
        assert_eq!(
            a.context(now, Some(0.2)).values,
            [Some(0.2), Some(0.0), Some(0.0)]
        );
        assert_eq!(
            a.context(now + Duration::from_secs(60), None).values,
            [None, None, None]
        );
        for _ in 0..3 {
            a.record(now + Duration::from_secs(1), Outcome::Unknown);
            a.record(now + Duration::from_secs(1), Outcome::DeadlineUnknown);
        }
        assert_eq!(
            a.context(now + Duration::from_secs(1), None).values[1],
            None
        );
    }
    #[test]
    fn untracked_admission_suppresses_cutoff_without_inventing_terminal_outcome() {
        let now = Instant::now();
        let mut activity = Activity::new(now);
        for _ in 0..16 {
            activity.record(now, Outcome::FirstText);
            activity.record(now, Outcome::DeadlineTimely);
            activity.record(now, Outcome::Admitted);
        }
        activity.record(now, Outcome::UntrackedAdmitted);
        let context = activity.context(now, None);
        assert_eq!(context.counts.admitted, 17);
        assert_eq!(context.counts.untracked_admitted, 1);
        assert_eq!(context.counts.unknown_pretext_end, 0);
        assert_eq!(context.counts.upstream_timeout, 0);
        assert_eq!(context.values[1], None);
        assert_eq!(context.values[2], Some(0.0));
    }
}
