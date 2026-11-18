//! Accepted-work lifecycle with native acknowledgement accounting.
//! Local transport outcomes are not native cancellation acknowledgements.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

pub const METHOD: &str = "accepted_native_ack_ledger_v1";
const MAX_PENDING: usize = 8192;
static NEXT_WORKER_SCOPE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key {
    pub worker: u64,
    pub generation: u64,
    pub request: u64,
    pub attempt: u64,
}

#[derive(Clone, Debug)]
struct Work {
    native_attempt: Option<String>,
    remaining: Option<f64>,
    forwarded: bool,
    last_sequence: u64,
    last_observed: Instant,
}

#[derive(Debug)]
pub struct AcceptedWork {
    worker: u64,
    generation: u64,
    next: u64,
    pending: HashMap<Key, Work>,
    native_attempts: HashMap<String, Key>,
    incomplete: bool,
    exhausted: bool,
}

impl Default for AcceptedWork {
    fn default() -> Self {
        let worker = NEXT_WORKER_SCOPE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .ok();
        Self {
            worker: worker.unwrap_or(0),
            generation: 0,
            next: 0,
            pending: HashMap::new(),
            native_attempts: HashMap::new(),
            incomplete: worker.is_none(),
            exhausted: worker.is_none(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Summary {
    pub pending: usize,
    pub known_work: f64,
    pub unknown_work: usize,
    pub forwarded: usize,
    pub incomplete: bool,
}

impl AcceptedWork {
    /// Called only after admission and selection have succeeded. Rejections
    /// have no handle and therefore cannot add, rollback, or release work.
    pub fn accepted(&mut self, at: Instant, remaining: Option<f64>) -> Option<Key> {
        if self.exhausted || self.pending.len() >= MAX_PENDING || self.next == u64::MAX {
            self.incomplete = true;
            return None;
        }
        self.next += 1;
        let key = Key {
            worker: self.worker,
            generation: self.generation,
            request: self.next,
            attempt: 0,
        };
        self.pending.insert(
            key,
            Work {
                native_attempt: None,
                remaining: remaining.filter(|n| n.is_finite() && *n >= 0.0),
                forwarded: false,
                last_sequence: 0,
                last_observed: at,
            },
        );
        Some(key)
    }

    pub fn forwarded(&mut self, key: Key) {
        if let Some(row) = self.pending.get_mut(&key) {
            row.forwarded = true;
        }
    }

    /// A local failure is rollback evidence only before any upstream send.
    pub fn pre_dispatch_failed(&mut self, key: Key) -> bool {
        if !self.pending.get(&key).is_some_and(|row| !row.forwarded) {
            return false;
        }
        self.remove(key).is_some()
    }

    pub fn first_text(&mut self, key: Key) -> bool {
        // Actual first text proves there is no remaining pre-first-text work.
        self.remove(key).is_some()
    }

    fn remove(&mut self, key: Key) -> Option<Work> {
        let row = self.pending.remove(&key)?;
        if let Some(attempt) = &row.native_attempt {
            self.native_attempts.remove(attempt);
        }
        Some(row)
    }

    fn bind_attempt(&mut self, key: Key, attempt: String) -> bool {
        if attempt.is_empty() || attempt.len() > 256 || self.native_attempts.contains_key(&attempt)
        {
            self.incomplete = true;
            return false;
        }
        let Some(row) = self.pending.get_mut(&key) else {
            return false;
        };
        row.native_attempt = Some(attempt.clone());
        self.native_attempts.insert(attempt, key);
        true
    }

    pub fn attempt_key(&self, attempt: &str) -> Option<Key> {
        self.native_attempts.get(attempt).copied()
    }

    pub fn mark_incomplete(&mut self) {
        self.incomplete = true;
    }

    pub fn mark_unknown(&mut self, key: Key) {
        if let Some(row) = self.pending.get_mut(&key) {
            row.remaining = None;
        }
    }

    pub fn reset(&mut self) {
        self.incomplete |= !self.pending.is_empty();
        if let Some(generation) = self.generation.checked_add(1) {
            self.generation = generation;
        } else {
            self.exhausted = true;
            self.incomplete = true;
        }
        self.pending.clear();
        self.native_attempts.clear();
    }

    pub fn summary(&self) -> Summary {
        let mut summary = Summary {
            pending: self.pending.len(),
            incomplete: self.incomplete,
            ..Summary::default()
        };
        for row in self.pending.values() {
            summary.forwarded += usize::from(row.forwarded);
            match row.remaining {
                Some(work) => summary.known_work += work,
                None => summary.unknown_work += 1,
            }
        }
        summary
    }

    /// This interface accepts only native per-attempt events. No client
    /// disconnect, timeout, threshold crossing, or forecast can call it.
    pub fn native_event(
        &mut self,
        event: NativeEvent,
        now: Instant,
        max_age: f64,
    ) -> Result<f64, &'static str> {
        if event.key.worker != self.worker {
            return Err("worker_identity_changed");
        }
        if self.exhausted {
            return Err("identity_generation_exhausted");
        }
        if event.key.generation != self.generation {
            return Err("generation_changed");
        }
        if !max_age.is_finite()
            || max_age < 0.0
            || event.observed_at > now
            || now.duration_since(event.observed_at).as_secs_f64() > max_age
        {
            return Err("native_event_stale_or_future");
        }
        let Some(row) = self.pending.get_mut(&event.key) else {
            return Err("unknown_or_already_resolved_attempt");
        };
        if !row.forwarded {
            return Err("native_event_before_dispatch");
        }
        if event.sequence <= row.last_sequence || event.observed_at < row.last_observed {
            return Err("out_of_order_native_event");
        }
        match event.kind {
            NativeEventKind::RemainingWork(work) => {
                if !work.is_finite() || work < 0.0 {
                    return Err("invalid_remaining_work");
                }
                if row.remaining.is_some_and(|old| work > old) {
                    // Genuine newer evidence contradicts our smaller amount.
                    // Do not leave that optimistic value usable after error.
                    row.remaining = None;
                    row.last_sequence = event.sequence;
                    row.last_observed = event.observed_at;
                    self.incomplete = true;
                    return Err("remaining_work_rebound");
                }
                row.remaining = Some(work);
                row.last_sequence = event.sequence;
                row.last_observed = event.observed_at;
                Ok(0.0)
            }
            NativeEventKind::CancellationAcknowledged => {
                let remaining = row.remaining.unwrap_or(0.0);
                self.remove(event.key);
                Ok(remaining)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum NativeEventKind {
    RemainingWork(f64),
    CancellationAcknowledged,
}
#[derive(Clone, Copy, Debug)]
pub struct NativeEvent {
    pub key: Key,
    pub sequence: u64,
    pub observed_at: Instant,
    pub kind: NativeEventKind,
}

/// Shared by forecast clones; dropping the last owner before dispatch rolls
/// back once. A send attempt irrevocably disarms local rollback. Uncertain
/// upstream outcomes then require first text or the native ACK interface.
#[derive(Debug)]
pub struct DispatchLease {
    ledger: Arc<Mutex<AcceptedWork>>,
    key: Key,
    native_attempt: String,
}
impl DispatchLease {
    #[cfg(test)]
    pub fn new(ledger: Arc<Mutex<AcceptedWork>>, at: Instant) -> Option<Arc<Self>> {
        Self::allocate(ledger, at, None)
    }

    pub fn with_native_attempt(
        ledger: Arc<Mutex<AcceptedWork>>,
        at: Instant,
        attempt: String,
    ) -> Option<Arc<Self>> {
        Self::allocate(ledger, at, Some(attempt))
    }

    fn allocate(
        ledger: Arc<Mutex<AcceptedWork>>,
        at: Instant,
        attempt: Option<String>,
    ) -> Option<Arc<Self>> {
        // Prompt length is not remaining native work; preserve unknown here.
        let mut work = ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let key = work.accepted(at, None)?;
        let native_attempt = attempt.unwrap_or_else(|| {
            format!(
                "{}:{}:{}:{}",
                key.worker, key.generation, key.request, key.attempt
            )
        });
        if !work.bind_attempt(key, native_attempt.clone()) {
            work.pre_dispatch_failed(key);
            return None;
        }
        drop(work);
        Some(Arc::new(Self {
            ledger,
            key,
            native_attempt,
        }))
    }

    pub fn native_attempt(&self) -> &str {
        &self.native_attempt
    }
    pub fn before_send(&self) -> bool {
        self.ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending
            .get(&self.key)
            .is_some_and(|row| !row.forwarded)
    }
    pub fn forwarded(&self) {
        self.ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .forwarded(self.key);
    }
    pub fn first_text(&self) {
        self.ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .first_text(self.key);
    }

    /// The lease binds the native adapter's event to this selected worker's
    /// ledger and this request attempt. Client transport paths never call it.
    #[cfg(test)]
    pub fn native_event(
        &self,
        sequence: u64,
        observed_at: Instant,
        kind: NativeEventKind,
        now: Instant,
        max_age: f64,
    ) -> Result<f64, &'static str> {
        self.ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .native_event(
                NativeEvent {
                    key: self.key,
                    sequence,
                    observed_at,
                    kind,
                },
                now,
                max_age,
            )
    }
}
impl Drop for DispatchLease {
    fn drop(&mut self) {
        self.ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pre_dispatch_failed(self.key);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn d1_rejected_has_no_credit_and_local_failure_rolls_back_once() {
        let now = Instant::now();
        let mut ledger = AcceptedWork::default();
        assert_eq!(ledger.summary().pending, 0); // admission rejection creates no handle
        let accepted = ledger.accepted(now, Some(100.0)).unwrap();
        assert_eq!(ledger.summary().known_work, 100.0);
        assert!(ledger.pre_dispatch_failed(accepted));
        assert!(!ledger.pre_dispatch_failed(accepted));
        assert_eq!(ledger.summary().known_work, 0.0);
        assert_eq!(ledger.summary().pending, 0);
    }

    #[test]
    fn d1_shared_lease_drop_and_forwarding_have_distinct_release_rules() {
        let ledger = Arc::new(Mutex::new(AcceptedWork::default()));
        let lease = DispatchLease::new(Arc::clone(&ledger), Instant::now()).unwrap();
        let clone = Arc::clone(&lease);
        drop(lease);
        assert_eq!(ledger.lock().unwrap().summary().pending, 1);
        drop(clone);
        assert_eq!(ledger.lock().unwrap().summary().pending, 0);
        let sent = DispatchLease::new(Arc::clone(&ledger), Instant::now()).unwrap();
        sent.forwarded();
        drop(sent); // timeout/disconnect/missing ACK cannot rollback native work
        assert_eq!(ledger.lock().unwrap().summary().pending, 1);
        let text = DispatchLease::new(Arc::clone(&ledger), Instant::now()).unwrap();
        text.forwarded();
        text.first_text();
        text.first_text();
        drop(text);
        assert_eq!(ledger.lock().unwrap().summary().pending, 1);
    }

    #[test]
    fn d2_ack_releases_remaining_once_and_rejects_old_missing_or_reordered_evidence() {
        let now = Instant::now();
        let mut ledger = AcceptedWork::default();
        let key = ledger.accepted(now, Some(100.0)).unwrap();
        ledger.forwarded(key);
        let progress = NativeEvent {
            key,
            sequence: 2,
            observed_at: now + Duration::from_secs(1),
            kind: NativeEventKind::RemainingWork(40.0),
        };
        assert_eq!(
            ledger.native_event(progress, now + Duration::from_secs(1), 2.0),
            Ok(0.0)
        );
        let ack = NativeEvent {
            key,
            sequence: 3,
            observed_at: now + Duration::from_secs(2),
            kind: NativeEventKind::CancellationAcknowledged,
        };
        let old = NativeEvent { sequence: 1, ..ack };
        assert_eq!(
            ledger.native_event(old, now + Duration::from_secs(2), 2.0),
            Err("out_of_order_native_event")
        );
        assert_eq!(ledger.summary().known_work, 40.0); // no ACK yet
        assert_eq!(
            ledger.native_event(ack, now + Duration::from_secs(20), 2.0),
            Err("native_event_stale_or_future")
        );
        assert_eq!(
            ledger.native_event(ack, now + Duration::from_secs(1), 2.0),
            Err("native_event_stale_or_future")
        );
        assert_eq!(
            ledger.native_event(ack, now + Duration::from_secs(2), 2.0),
            Ok(40.0)
        );
        assert_eq!(
            ledger.native_event(ack, now + Duration::from_secs(2), 2.0),
            Err("unknown_or_already_resolved_attempt")
        );
        ledger.reset();
        let replacement = ledger.accepted(now, Some(200.0)).unwrap();
        ledger.forwarded(replacement);
        assert_eq!(
            ledger.native_event(ack, now + Duration::from_secs(2), 2.0),
            Err("generation_changed")
        );
        assert_eq!(ledger.summary().known_work, 200.0);
    }

    #[test]
    fn d2_retry_attempt_and_remaining_work_rebound_fail_closed() {
        let now = Instant::now();
        let mut ledger = AcceptedWork::default();
        let key = ledger.accepted(now, Some(100.0)).unwrap();
        let event = NativeEvent {
            key,
            sequence: 1,
            observed_at: now,
            kind: NativeEventKind::CancellationAcknowledged,
        };
        assert_eq!(
            ledger.native_event(event, now, 1.0),
            Err("native_event_before_dispatch")
        );
        ledger.forwarded(key);
        assert_eq!(
            ledger.native_event(
                NativeEvent {
                    key: Key { attempt: 1, ..key },
                    ..event
                },
                now,
                1.0
            ),
            Err("unknown_or_already_resolved_attempt")
        );
        assert_eq!(
            ledger.native_event(
                NativeEvent {
                    kind: NativeEventKind::RemainingWork(101.0),
                    ..event
                },
                now,
                1.0
            ),
            Err("remaining_work_rebound")
        );
        assert_eq!(ledger.summary().known_work, 0.0);
        assert_eq!(ledger.summary().unknown_work, 1);
        assert!(ledger.summary().incomplete);
    }

    #[test]
    fn d2_wrong_worker_with_colliding_request_and_generation_is_rejected() {
        let now = Instant::now();
        let mut first = AcceptedWork::default();
        let mut second = AcceptedWork::default();
        let a = first.accepted(now, Some(10.0)).unwrap();
        let b = second.accepted(now, Some(10.0)).unwrap();
        assert_eq!((a.request, a.generation), (b.request, b.generation));
        assert_ne!(a.worker, b.worker);
        second.forwarded(b);
        let wrong = NativeEvent {
            key: a,
            sequence: 1,
            observed_at: now,
            kind: NativeEventKind::CancellationAcknowledged,
        };
        assert_eq!(
            second.native_event(wrong, now, 2.0),
            Err("worker_identity_changed")
        );
        assert_eq!(second.summary().known_work, 10.0);
    }

    #[test]
    fn d1_overflow_and_discarded_unconfirmed_work_remain_incomplete() {
        let now = Instant::now();
        let mut ledger = AcceptedWork::default();
        for _ in 0..MAX_PENDING {
            ledger.accepted(now, None).unwrap();
        }
        assert!(ledger.accepted(now, None).is_none());
        assert!(ledger.summary().incomplete);
        ledger.reset();
        assert_eq!(ledger.summary().pending, 0);
        assert!(ledger.summary().incomplete);
        let mut exhausted = AcceptedWork {
            generation: u64::MAX,
            ..AcceptedWork::default()
        };
        exhausted.reset();
        assert!(exhausted.accepted(now, None).is_none());
        assert!(exhausted.summary().incomplete);
    }

    #[test]
    fn d2_lease_native_api_is_bound_to_its_worker_and_attempt() {
        let ledger = Arc::new(Mutex::new(AcceptedWork::default()));
        let now = Instant::now();
        let lease = DispatchLease::new(Arc::clone(&ledger), now).unwrap();
        lease.forwarded();
        assert_eq!(
            lease.native_event(1, now, NativeEventKind::RemainingWork(30.0), now, 2.0),
            Ok(0.0)
        );
        assert_eq!(
            lease.native_event(2, now, NativeEventKind::CancellationAcknowledged, now, 2.0),
            Ok(30.0)
        );
        assert!(lease
            .native_event(3, now, NativeEventKind::CancellationAcknowledged, now, 2.0)
            .is_err());
        assert_eq!(ledger.lock().unwrap().summary().pending, 0);
    }
}
