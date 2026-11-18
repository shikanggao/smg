//! Bounded first-text deadline observations for exact and censored fit labels.
//! A local deadline is not a native cancellation acknowledgement.
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug)]
pub enum Outcome {
    FirstText,
    Timeout,
    NativeDeadline,
    Disconnect,
    Unknown,
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
            Self::DeadlineTimely => "timely_first_text",
            Self::DeadlineBreach => "deadline_breach",
            Self::ActualCutoff => "actual_cutoff",
            Self::DeadlineUnknown => "unknown_pretext_end",
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
}
