//! Native HTTP telemetry adapter. Scheduler ACKs are the only cancellation
//! release evidence; a complete inventory's omissions never release a request.
use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use openai_protocol::{
    native_deadline::{NativeCancellationAck, NativeDeadlineTelemetry},
    worker::WorkerLoadResponse,
};

use super::deadline_recovery::{AcceptedWork, NativeEvent, NativeEventKind};

const MAX_ROWS: usize = 8192;
const MAX_EPOCHS: usize = 64;
const MAX_RANKS: usize = 256;

#[derive(Debug, serde::Serialize)]
pub(super) struct DeadlineConfirmation {
    attempt_id: String,
    native_request_id: Option<String>,
    engine_epoch: String,
    profile_id: String,
    dp_rank: i32,
    snapshot_seq: u64,
    hard_ttft_deadline_monotonic_secs: f64,
    deadline_boundary: String,
    deadline_origin: Option<String>,
}

#[derive(Debug)]
struct Rank {
    epoch: String,
    profile: String,
    sequence: u64,
    received: Instant,
    poll_started: Instant,
    telemetry: NativeDeadlineTelemetry,
}

#[derive(Debug)]
struct Binding {
    epoch: String,
    rank: i32,
    native_request_id: Option<String>,
}

#[derive(Debug)]
pub(super) struct State {
    ranks: HashMap<i32, Rank>,
    bindings: HashMap<String, Binding>,
    retired_epochs: HashSet<(i32, String)>,
    pub reason: &'static str,
    pub acknowledged_requests: u64,
    pub released_known_work: f64,
    pub journal_complete: bool,
    pub applied_acknowledgements: Vec<NativeCancellationAck>,
    pub confirmed_deadlines: Vec<DeadlineConfirmation>,
    current_valid: bool,
    timeout_retained_capability: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            ranks: HashMap::new(),
            bindings: HashMap::new(),
            retired_epochs: HashSet::new(),
            reason: "native_telemetry_unavailable",
            acknowledged_requests: 0,
            released_known_work: 0.0,
            journal_complete: true,
            applied_acknowledgements: Vec::new(),
            confirmed_deadlines: Vec::new(),
            current_valid: false,
            timeout_retained_capability: false,
        }
    }
}

impl State {
    pub(super) fn observe(
        &mut self,
        load: Option<&WorkerLoadResponse>,
        expected_rank: Option<usize>,
        poll_started: Instant,
        received: Instant,
        max_age: f64,
        ledger: &mut AcceptedWork,
    ) {
        self.current_valid = false;
        self.timeout_retained_capability = false;
        self.applied_acknowledgements.clear();
        self.confirmed_deadlines.clear();
        self.bindings
            .retain(|attempt, _| ledger.attempt_key(attempt).is_some());
        let Some(load) = load.filter(|l| l.ranks_are_dp_ranks()) else {
            self.reason = "native_telemetry_missing_or_wrong_worker_scope";
            return;
        };
        let mut seen = HashSet::new();
        let mut found = false;
        let mut valid = true;
        for row in &load.loads {
            let Some(telemetry) = &row.deadline_recovery else {
                continue;
            };
            found = true;
            if !seen.insert(row.dp_rank)
                || telemetry.dp_rank != row.dp_rank
                || expected_rank.is_some_and(|rank| usize::try_from(row.dp_rank).ok() != Some(rank))
            {
                self.reason = "native_rank_mismatch";
                return;
            }
            if let Err(reason) =
                self.observe_rank(telemetry, poll_started, received, max_age, ledger)
            {
                self.reason = reason;
                valid = false;
            }
        }
        if !found {
            self.reason = "native_telemetry_unavailable";
        }
        self.current_valid = found && valid;
    }

    /// A positively classified transport timeout supplies no new inventory or
    /// release evidence. Preserve only the previously verified dispatch
    /// capability, with the original timestamps and expiry on every retry.
    pub(super) fn observe_transport_timeout(&mut self, now: Instant, max_age: f64) {
        self.timeout_retained_capability = self.deadline_valid_until(now, max_age).is_some();
        self.current_valid = false;
        self.applied_acknowledgements.clear();
        self.confirmed_deadlines.clear();
        self.reason = if self.timeout_retained_capability {
            "native_transport_timeout_original_capability_only"
        } else {
            "native_transport_timeout_no_valid_capability"
        };
    }

    fn observe_rank(
        &mut self,
        telemetry: &NativeDeadlineTelemetry,
        poll_started: Instant,
        received: Instant,
        max_age: f64,
        ledger: &mut AcceptedWork,
    ) -> Result<(), &'static str> {
        validate(telemetry, poll_started, received, max_age)?;
        let retained_rows: usize = self
            .ranks
            .iter()
            .filter(|(rank, _)| **rank != telemetry.dp_rank)
            .map(|(_, rank)| {
                rank.telemetry.active_requests.len() + rank.telemetry.cancellation_acks.len()
            })
            .sum();
        if retained_rows + telemetry.active_requests.len() + telemetry.cancellation_acks.len()
            > 2 * MAX_ROWS
        {
            ledger.mark_incomplete();
            return Err("native_worker_inventory_overflow");
        }
        if self.ranks.len() >= MAX_RANKS && !self.ranks.contains_key(&telemetry.dp_rank) {
            ledger.mark_incomplete();
            return Err("native_rank_inventory_overflow");
        }
        if self
            .retired_epochs
            .contains(&(telemetry.dp_rank, telemetry.engine_epoch.clone()))
        {
            return Err("retired_native_engine_epoch");
        }
        if let Some(previous) = self.ranks.get(&telemetry.dp_rank) {
            if previous.epoch == telemetry.engine_epoch
                && telemetry.snapshot_seq < previous.sequence
            {
                return Err("duplicate_or_older_native_snapshot");
            }
            if previous.epoch == telemetry.engine_epoch
                && telemetry.snapshot_seq == previous.sequence
            {
                let mut same_snapshot = telemetry.clone();
                same_snapshot.snapshot_age_seconds = previous.telemetry.snapshot_age_seconds;
                if !same_snapshot.legacy_snapshot_eq(&previous.telemetry) {
                    return Err("native_snapshot_changed_without_sequence");
                }
                // An unchanged scheduler state is not new progress or an ACK.
                // Keep its original observation deadline; a repeated poll must
                // never extend freshness, even if it reports a smaller age.
                if !Self::fresh(previous, received, max_age) {
                    return Err("duplicate_native_snapshot_expired");
                }
                self.reason = if self.journal_complete {
                    "native_cancellation_ack_adapter_cached"
                } else {
                    "native_ack_journal_gap"
                };
                return Ok(());
            }
            if previous.epoch != telemetry.engine_epoch {
                if self.retired_epochs.len() >= MAX_EPOCHS {
                    ledger.mark_incomplete();
                    return Err("native_epoch_history_overflow");
                }
                self.retired_epochs
                    .insert((telemetry.dp_rank, previous.epoch.clone()));
            }
            if previous.epoch != telemetry.engine_epoch || previous.profile != telemetry.profile_id
            {
                ledger.mark_incomplete();
                for (attempt, binding) in &self.bindings {
                    if binding.rank == telemetry.dp_rank {
                        if let Some(key) = ledger.attempt_key(attempt) {
                            ledger.mark_unknown(key);
                        }
                    }
                }
            }
            if previous.epoch == telemetry.engine_epoch
                && telemetry.ack_floor_seq > previous.sequence.saturating_add(1)
            {
                self.journal_complete = false;
                ledger.mark_incomplete();
            }
        }

        for row in &telemetry.active_requests {
            let Some(attempt) = &row.attempt_id else {
                continue;
            };
            let Some(key) = ledger.attempt_key(attempt) else {
                continue;
            };
            self.bind(attempt, row.native_request_id.as_deref(), telemetry)?;
            if let Some(deadline) = row.hard_ttft_deadline_monotonic_secs.filter(|_| {
                telemetry.hard_ttft_deadline_supported
                    && telemetry.abort_enforcement_enabled
                    && row.first_text_emitted == Some(false)
            }) {
                self.confirmed_deadlines.push(DeadlineConfirmation {
                    attempt_id: attempt.clone(),
                    native_request_id: row.native_request_id.clone(),
                    engine_epoch: telemetry.engine_epoch.clone(),
                    profile_id: telemetry.profile_id.clone(),
                    dp_rank: telemetry.dp_rank,
                    snapshot_seq: telemetry.snapshot_seq,
                    hard_ttft_deadline_monotonic_secs: deadline,
                    deadline_boundary: telemetry.deadline_boundary.clone(),
                    deadline_origin: telemetry.deadline_origin.clone(),
                });
            }
            if telemetry.remaining_work_source != "uncached_uncomputed_prompt_tokens" {
                ledger.mark_unknown(key);
                continue;
            }
            let Some(remaining) = row.remaining_prefill_work_tokens else {
                ledger.mark_unknown(key);
                continue;
            };
            let observed = received
                .checked_sub(
                    Duration::try_from_secs_f64(telemetry.snapshot_age_seconds)
                        .map_err(|_| "invalid_native_age")?,
                )
                .ok_or("invalid_native_age")?;
            if let Err(reason) = ledger.native_event(
                NativeEvent {
                    key,
                    sequence: telemetry.snapshot_seq,
                    observed_at: observed,
                    kind: NativeEventKind::RemainingWork(remaining),
                },
                received,
                max_age,
            ) {
                if reason != "out_of_order_native_event" {
                    self.reason = reason;
                }
            }
        }
        for ack in &telemetry.cancellation_acks {
            let Some(key) = ledger.attempt_key(&ack.attempt_id) else {
                continue;
            };
            self.bind(&ack.attempt_id, ack.native_request_id.as_deref(), telemetry)?;
            // A fresh scheduler snapshot re-attests a retained terminal ACK.
            // Its actual occurrence time remains in ack_monotonic_secs; do not
            // confuse an old occurrence with a stale observation of that fact.
            let age = telemetry.snapshot_age_seconds;
            let observed = received
                .checked_sub(
                    Duration::try_from_secs_f64(age).map_err(|_| "invalid_native_ack_age")?,
                )
                .ok_or("invalid_native_ack_age")?;
            match ledger.native_event(
                NativeEvent {
                    key,
                    sequence: ack.event_seq,
                    observed_at: observed,
                    kind: NativeEventKind::CancellationAcknowledged,
                },
                received,
                max_age,
            ) {
                Ok(work) => {
                    self.acknowledged_requests = self.acknowledged_requests.saturating_add(1);
                    self.released_known_work += work;
                    self.applied_acknowledgements.push(ack.clone());
                    self.bindings.remove(&ack.attempt_id);
                }
                Err(reason) => self.reason = reason,
            }
        }
        self.ranks.insert(
            telemetry.dp_rank,
            Rank {
                epoch: telemetry.engine_epoch.clone(),
                profile: telemetry.profile_id.clone(),
                sequence: telemetry.snapshot_seq,
                received,
                poll_started,
                telemetry: telemetry.clone(),
            },
        );
        self.reason = if self.journal_complete {
            "native_cancellation_ack_adapter_active"
        } else {
            "native_ack_journal_gap"
        };
        Ok(())
    }

    fn bind(
        &mut self,
        attempt: &str,
        native_request: Option<&str>,
        telemetry: &NativeDeadlineTelemetry,
    ) -> Result<(), &'static str> {
        if let Some(binding) = self.bindings.get_mut(attempt) {
            if binding.epoch != telemetry.engine_epoch || binding.rank != telemetry.dp_rank {
                return Err("native_request_epoch_or_rank_changed");
            }
            if binding
                .native_request_id
                .as_deref()
                .is_some_and(|old| Some(old) != native_request)
            {
                return Err("native_request_identity_changed");
            }
            if binding.native_request_id.is_none() {
                binding.native_request_id = native_request.map(str::to_owned);
            }
        } else {
            if self.bindings.len() >= MAX_ROWS {
                return Err("native_binding_inventory_overflow");
            }
            self.bindings.insert(
                attempt.to_owned(),
                Binding {
                    epoch: telemetry.engine_epoch.clone(),
                    rank: telemetry.dp_rank,
                    native_request_id: native_request.map(str::to_owned),
                },
            );
        }
        Ok(())
    }

    pub(super) fn available(&self, now: Instant, max_age: f64) -> bool {
        self.current_valid
            && self.journal_complete
            && self.ranks.values().any(|rank| {
                rank.telemetry.cancellation_ack_supported && Self::fresh(rank, now, max_age)
            })
    }

    fn fresh(rank: &Rank, now: Instant, max_age: f64) -> bool {
        now.saturating_duration_since(rank.poll_started)
            .as_secs_f64()
            + rank.telemetry.snapshot_age_seconds
            <= max_age
            && now >= rank.received
    }

    pub(super) fn deadline_boundary(&self) -> &str {
        self.ranks.values().next().map_or("unsupported", |rank| {
            rank.telemetry.deadline_boundary.as_str()
        })
    }

    pub(super) fn deadline_origin(&self) -> &str {
        self.ranks
            .values()
            .next()
            .and_then(|rank| rank.telemetry.deadline_origin.as_deref())
            .unwrap_or("unknown")
    }

    pub(super) fn deadline_valid_until(&self, now: Instant, max_age: f64) -> Option<Instant> {
        if !(self.current_valid || self.timeout_retained_capability) || self.ranks.is_empty() {
            return None;
        }
        let mut earliest = None;
        for rank in self.ranks.values() {
            if !Self::fresh(rank, now, max_age)
                || !rank.telemetry.hard_ttft_deadline_supported
                || !rank.telemetry.abort_enforcement_enabled
                || !matches!(
                    rank.telemetry.deadline_boundary.as_str(),
                    "first_nonempty_text" | "first_generated_token"
                )
            {
                return None;
            }
            let remaining = max_age - rank.telemetry.snapshot_age_seconds;
            let expiry = rank
                .poll_started
                .checked_add(Duration::try_from_secs_f64(remaining).ok()?)?;
            earliest = Some(earliest.map_or(expiry, |old: Instant| old.min(expiry)));
        }
        earliest
    }

    pub(super) fn hard_deadlines_present(&self, now: Instant, max_age: f64) -> bool {
        self.current_valid
            && self.confirmed_deadlines.iter().any(|confirmation| {
                self.ranks.get(&confirmation.dp_rank).is_some_and(|rank| {
                    Self::fresh(rank, now, max_age) && rank.epoch == confirmation.engine_epoch
                })
            })
    }
}

fn validate(
    telemetry: &NativeDeadlineTelemetry,
    started: Instant,
    received: Instant,
    max_age: f64,
) -> Result<(), &'static str> {
    if telemetry.schema_version != 1
        || telemetry.engine_epoch.is_empty()
        || telemetry.engine_epoch.len() > 128
        || telemetry.profile_id.is_empty()
        || telemetry.profile_id.len() > 256
        || telemetry.dp_rank < 0
        || telemetry.snapshot_seq == 0
        || !telemetry.cancellation_ack_supported
    {
        return Err("native_contract_unsupported");
    }
    if !telemetry.captured_at_monotonic_secs.is_finite()
        || telemetry.captured_at_monotonic_secs < 0.0
        || !telemetry.snapshot_age_seconds.is_finite()
        || telemetry.snapshot_age_seconds < 0.0
        || !max_age.is_finite()
        || max_age < 0.0
        || started > received
        || telemetry.snapshot_age_seconds + received.duration_since(started).as_secs_f64() > max_age
    {
        return Err("native_snapshot_stale_or_invalid_age");
    }
    if telemetry.active_requests.len() > MAX_ROWS
        || telemetry.cancellation_acks.len() > MAX_ROWS
        || (telemetry.active_inventory_complete
            && telemetry.native_population != telemetry.active_requests.len())
        || telemetry.ack_floor_seq > telemetry.snapshot_seq.saturating_add(1)
    {
        return Err("native_inventory_or_journal_invalid");
    }
    let mut active = HashSet::new();
    for row in &telemetry.active_requests {
        if row.state_seq > telemetry.snapshot_seq
            || (row.attempt_id.is_some() && row.native_request_id.is_none())
            || row
                .native_request_id
                .as_ref()
                .is_some_and(|id| id.is_empty() || id.len() > 512)
            || row
                .remaining_prefill_work_tokens
                .is_some_and(|work| !work.is_finite() || work < 0.0)
            || row
                .hard_ttft_deadline_monotonic_secs
                .is_some_and(|time| !time.is_finite() || time < 0.0)
            || row
                .attempt_id
                .as_ref()
                .is_some_and(|id| id.is_empty() || id.len() > 256 || !active.insert(id.as_str()))
        {
            return Err("invalid_native_request_inventory");
        }
    }
    let mut acknowledgements = HashSet::new();
    for ack in &telemetry.cancellation_acks {
        if ack.attempt_id.is_empty()
            || ack.native_request_id.is_none()
            || ack
                .native_request_id
                .as_ref()
                .is_some_and(|id| id.is_empty() || id.len() > 512)
            || ack.attempt_id.len() > 256
            || !acknowledgements.insert(ack.attempt_id.as_str())
            || active.contains(ack.attempt_id.as_str())
            || ack.event_seq == 0
            || ack.event_seq > telemetry.snapshot_seq
            || ack.event_seq < telemetry.ack_floor_seq
            || !ack.ack_monotonic_secs.is_finite()
            || ack.ack_monotonic_secs < 0.0
            || ack.ack_monotonic_secs > telemetry.captured_at_monotonic_secs
            || !matches!(
                ack.reason.as_str(),
                "hard_ttft_deadline" | "client_abort" | "explicit_abort"
            )
        {
            return Err("invalid_native_cancellation_ack");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use openai_protocol::worker::SchedulerLoadSnapshot;
    use serde_json::json;

    use super::{super::deadline_recovery::DispatchLease, *};

    fn frame(sequence: u64) -> NativeDeadlineTelemetry {
        serde_json::from_value(json!({
            "schema_version":1,"engine_epoch":"engine-a","profile_id":"pinned-config-a","dp_rank":0,
            "snapshot_seq":sequence,"captured_at_monotonic_secs":100.0+sequence as f64,
            "snapshot_age_seconds":0.0,"active_inventory_complete":true,"native_population":0,
            "scheduler_semantics":"continuous_batching","deadline_boundary":"first_nonempty_text",
            "deadline_origin":"engine_http_receive_monotonic",
            "remaining_work_source":"uncached_uncomputed_prompt_tokens",
            "effective_capacity_tokens_per_sec":null,"capacity_source":null,
            "hard_ttft_deadline_supported":true,"cancellation_ack_supported":true,
            "abort_enforcement_enabled":true,"abort_max_lag_seconds":null,"ack_floor_seq":0,
            "active_requests":[],"cancellation_acks":[]
        }))
        .unwrap()
    }

    fn active(frame: &mut NativeDeadlineTelemetry, attempt: &str, remaining: f64) {
        frame.native_population = 1;
        frame.active_requests = vec![serde_json::from_value(json!({"attempt_id":attempt,"native_request_id":"native-exact-id",
            "state_seq":frame.snapshot_seq,"remaining_prefill_work_tokens":remaining,
            "hard_ttft_deadline_monotonic_secs":500.0,"queue_position":null,"state":"prefill","first_text_emitted":false})).unwrap()];
    }

    fn ack(frame: &mut NativeDeadlineTelemetry, attempt: &str) {
        frame.cancellation_acks = vec![serde_json::from_value(json!({"attempt_id":attempt,"native_request_id":"native-exact-id",
            "event_seq":frame.snapshot_seq,"ack_monotonic_secs":frame.captured_at_monotonic_secs,"reason":"hard_ttft_deadline"})).unwrap()];
    }

    fn load(frame: NativeDeadlineTelemetry) -> WorkerLoadResponse {
        WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![SchedulerLoadSnapshot {
                dp_rank: frame.dp_rank,
                deadline_recovery: Some(frame),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn observe(
        state: &mut State,
        ledger: &Arc<Mutex<AcceptedWork>>,
        frame: NativeDeadlineTelemetry,
        at: Instant,
    ) {
        state.observe(
            Some(&load(frame)),
            Some(0),
            at,
            at + Duration::from_millis(1),
            5.0,
            &mut ledger.lock().unwrap(),
        );
    }

    #[test]
    fn native_ack_after_client_observer_drop_releases_real_remaining_once() {
        let at = Instant::now();
        let ledger = Arc::new(Mutex::new(AcceptedWork::default()));
        let lease = DispatchLease::with_native_attempt(
            Arc::clone(&ledger),
            at,
            "globally-unique-attempt".into(),
        )
        .unwrap();
        lease.forwarded();
        let mut state = State::default();
        let mut f = frame(1);
        active(&mut f, lease.native_attempt(), 80.0);
        observe(&mut state, &ledger, f, at + Duration::from_millis(2));
        drop(lease); // disconnect/timeout does not release
        assert_eq!(ledger.lock().unwrap().summary().known_work, 80.0);
        let mut f = frame(2);
        ack(&mut f, "globally-unique-attempt");
        observe(
            &mut state,
            &ledger,
            f.clone(),
            at + Duration::from_millis(4),
        );
        assert_eq!(ledger.lock().unwrap().summary().pending, 0);
        assert_eq!(state.acknowledged_requests, 1);
        assert_eq!(state.released_known_work, 80.0);
        observe(&mut state, &ledger, f, at + Duration::from_millis(6));
        assert!(state.available(at + Duration::from_millis(7), 5.0));
        assert_eq!(state.reason, "native_cancellation_ack_adapter_cached");
        assert!(state.applied_acknowledgements.is_empty());
        assert_eq!(state.acknowledged_requests, 1);
        assert_eq!(state.released_known_work, 80.0);
        let mut f = frame(3);
        ack(&mut f, "globally-unique-attempt");
        f.cancellation_acks[0].event_seq = 2;
        observe(&mut state, &ledger, f, at + Duration::from_millis(8));
        assert_eq!(state.acknowledged_requests, 1);
        assert!(state.available(at + Duration::from_millis(9), 5.0));
    }

    #[test]
    fn transport_timeout_retains_only_original_dispatch_capability() {
        let at = Instant::now();
        let ledger = Arc::new(Mutex::new(AcceptedWork::default()));
        let mut state = State::default();
        observe(&mut state, &ledger, frame(1), at);
        let expiry = state
            .deadline_valid_until(at + Duration::from_secs(1), 5.0)
            .unwrap();
        for second in [2, 3, 4] {
            let now = at + Duration::from_secs(second);
            state.observe_transport_timeout(now, 5.0);
            assert_eq!(state.deadline_valid_until(now, 5.0), Some(expiry));
            assert!(!state.available(now, 5.0));
            assert!(!state.hard_deadlines_present(now, 5.0));
            assert!(state.applied_acknowledgements.is_empty());
            assert!(state.confirmed_deadlines.is_empty());
            assert_eq!(state.acknowledged_requests, 0);
            assert_eq!(state.released_known_work, 0.0);
            assert_eq!(state.ranks[&0].poll_started, at);
        }
        state.observe_transport_timeout(expiry + Duration::from_nanos(1), 5.0);
        assert!(state
            .deadline_valid_until(expiry + Duration::from_nanos(1), 5.0)
            .is_none());
        assert!(!state.timeout_retained_capability);
    }

    #[test]
    fn non_timeout_failure_revokes_retained_capability_and_timeout_cannot_restore_it() {
        let at = Instant::now();
        let ledger = Arc::new(Mutex::new(AcceptedWork::default()));
        let mut state = State::default();
        state.observe_transport_timeout(at, 5.0);
        assert!(state.deadline_valid_until(at, 5.0).is_none());
        observe(&mut state, &ledger, frame(1), at);
        let now = at + Duration::from_secs(1);
        state.observe_transport_timeout(now, 5.0);
        assert!(state.deadline_valid_until(now, 5.0).is_some());
        state.observe(None, Some(0), now, now, 5.0, &mut ledger.lock().unwrap());
        state.observe_transport_timeout(now, 5.0);
        assert!(state.deadline_valid_until(now, 5.0).is_none());
        // A genuinely new valid observation can restore inventory and capability.
        observe(&mut state, &ledger, frame(2), now);
        assert!(state.available(now + Duration::from_millis(2), 5.0));
        let mut unsupported = frame(3);
        unsupported.abort_enforcement_enabled = false;
        observe(
            &mut state,
            &ledger,
            unsupported,
            now + Duration::from_millis(3),
        );
        state.observe_transport_timeout(now + Duration::from_millis(5), 5.0);
        assert!(state
            .deadline_valid_until(now + Duration::from_millis(5), 5.0)
            .is_none());
    }

    #[test]
    fn identical_native_snapshot_retains_only_original_freshness_and_no_new_work() {
        let at = Instant::now();
        let ledger = Arc::new(Mutex::new(AcceptedWork::default()));
        let lease = DispatchLease::with_native_attempt(
            Arc::clone(&ledger),
            at - Duration::from_secs(2),
            "cached-attempt".into(),
        )
        .unwrap();
        lease.forwarded();
        let mut state = State::default();
        let mut f = frame(1);
        f.snapshot_age_seconds = 1.0;
        active(&mut f, "cached-attempt", 80.0);
        observe(&mut state, &ledger, f.clone(), at);
        let original_started = state.ranks[&0].poll_started;
        let original_received = state.ranks[&0].received;
        let original_expiry = state.deadline_valid_until(original_received, 5.0).unwrap();
        assert_eq!(original_expiry, at + Duration::from_secs(4));
        f.snapshot_age_seconds = 3.0;
        observe(&mut state, &ledger, f.clone(), at + Duration::from_secs(2));
        assert!(state.available(at + Duration::from_secs(2), 5.0));
        assert_eq!(state.reason, "native_cancellation_ack_adapter_cached");
        assert_eq!(state.ranks[&0].poll_started, original_started);
        assert_eq!(state.ranks[&0].received, original_received);
        assert_eq!(
            state.deadline_valid_until(at + Duration::from_secs(2), 5.0),
            Some(original_expiry)
        );
        assert_eq!(ledger.lock().unwrap().summary().known_work, 80.0);
        assert_eq!(state.acknowledged_requests, 0);
        assert!(state.applied_acknowledgements.is_empty());
        // A replay claiming age zero cannot extend the original observation.
        f.snapshot_age_seconds = 0.0;
        observe(&mut state, &ledger, f, at + Duration::from_secs(4));
        assert!(!state.available(at + Duration::from_secs(4), 5.0));
        assert_eq!(state.reason, "duplicate_native_snapshot_expired");
        assert!(state
            .deadline_valid_until(at + Duration::from_secs(4), 5.0)
            .is_none());
        assert_eq!(ledger.lock().unwrap().summary().known_work, 80.0);
    }

    #[test]
    fn duplicate_native_snapshot_mismatch_and_invalid_scope_still_fail_closed() {
        let at = Instant::now();
        for invalid in [
            "profile",
            "work",
            "capture",
            "older",
            "malformed",
            "scope",
            "rank",
        ] {
            let ledger = Arc::new(Mutex::new(AcceptedWork::default()));
            let lease =
                DispatchLease::with_native_attempt(Arc::clone(&ledger), at, "attempt".into())
                    .unwrap();
            lease.forwarded();
            let mut state = State::default();
            let mut f = frame(2);
            active(&mut f, "attempt", 80.0);
            observe(&mut state, &ledger, f.clone(), at);
            match invalid {
                "profile" => f.profile_id = "changed-profile".into(),
                "work" => f.active_requests[0].remaining_prefill_work_tokens = Some(70.0),
                "capture" => f.captured_at_monotonic_secs += 1.0,
                "older" => {
                    f = frame(1);
                    active(&mut f, "attempt", 70.0);
                }
                "malformed" => f.native_population = 9,
                "rank" => f.dp_rank = 1,
                _ => {}
            }
            let mut wire = load(f);
            if invalid == "scope" {
                wire.loads[0].worker = Some("another-worker".into());
            }
            state.observe(
                Some(&wire),
                Some(0),
                at + Duration::from_secs(1),
                at + Duration::from_secs(1),
                5.0,
                &mut ledger.lock().unwrap(),
            );
            assert!(
                !state.available(at + Duration::from_secs(1), 5.0),
                "{invalid}"
            );
            assert!(
                state
                    .deadline_valid_until(at + Duration::from_secs(1), 5.0)
                    .is_none(),
                "{invalid}"
            );
            assert_eq!(
                ledger.lock().unwrap().summary().known_work,
                80.0,
                "{invalid}"
            );
            assert_eq!(state.acknowledged_requests, 0, "{invalid}");
        }
    }

    #[test]
    fn complete_empty_inventory_is_not_cancellation_acknowledgement() {
        let at = Instant::now();
        let ledger = Arc::new(Mutex::new(AcceptedWork::default()));
        let lease =
            DispatchLease::with_native_attempt(Arc::clone(&ledger), at, "attempt".into()).unwrap();
        lease.forwarded();
        let mut state = State::default();
        let mut f = frame(1);
        active(&mut f, "attempt", 50.0);
        observe(&mut state, &ledger, f, at);
        observe(&mut state, &ledger, frame(2), at + Duration::from_millis(2));
        assert_eq!(ledger.lock().unwrap().summary().known_work, 50.0);
        assert_eq!(state.acknowledged_requests, 0);
    }

    #[test]
    fn native_epoch_rank_identity_and_journal_gaps_fail_closed() {
        let at = Instant::now();
        for invalid in ["epoch", "rank", "identity", "journal", "future", "stale"] {
            let ledger = Arc::new(Mutex::new(AcceptedWork::default()));
            let lease =
                DispatchLease::with_native_attempt(Arc::clone(&ledger), at, "attempt".into())
                    .unwrap();
            lease.forwarded();
            let mut state = State::default();
            let mut initial = frame(1);
            active(&mut initial, "attempt", 60.0);
            observe(&mut state, &ledger, initial, at);
            let mut f = frame(10);
            ack(&mut f, "attempt");
            match invalid {
                "epoch" => f.engine_epoch = "engine-b".into(),
                "rank" => f.dp_rank = 1,
                "identity" => {
                    f.cancellation_acks[0].native_request_id = Some("different-native-id".into());
                }
                "journal" => {
                    f.ack_floor_seq = 5;
                    f.cancellation_acks.clear();
                }
                "future" => f.cancellation_acks[0].ack_monotonic_secs += 1.0,
                _ => f.snapshot_age_seconds = 10.0,
            }
            observe(&mut state, &ledger, f, at + Duration::from_millis(2));
            assert_eq!(ledger.lock().unwrap().summary().pending, 1, "{invalid}");
            assert_eq!(state.acknowledged_requests, 0, "{invalid}");
            assert!(
                !state.available(at + Duration::from_millis(3), 5.0),
                "{invalid}"
            );
        }
    }

    #[test]
    fn capability_freshness_transport_bound_and_unsupported_defaults_are_explicit() {
        let at = Instant::now();
        let mut state = State::default();
        let mut ledger = AcceptedWork::default();
        assert!(state.deadline_valid_until(at, 5.0).is_none());
        let f = frame(1);
        state.observe(
            Some(&load(f.clone())),
            Some(0),
            at,
            at + Duration::from_secs(1),
            5.0,
            &mut ledger,
        );
        assert!(state
            .deadline_valid_until(at + Duration::from_secs(1), 5.0)
            .is_some());
        assert!(state
            .deadline_valid_until(at + Duration::from_secs(6), 5.0)
            .is_none());
        let mut aged = frame(2);
        aged.snapshot_age_seconds = 4.5;
        state.observe(
            Some(&load(aged)),
            Some(0),
            at,
            at + Duration::from_secs(1),
            5.0,
            &mut ledger,
        );
        assert!(!state.available(at + Duration::from_secs(1), 5.0));
        let mut off = frame(3);
        off.cancellation_ack_supported = false;
        state.observe(Some(&load(off)), Some(0), at, at, 5.0, &mut ledger);
        assert!(!state.available(at, 5.0));
        assert_eq!(state.reason, "native_contract_unsupported");
    }

    #[test]
    fn fresh_snapshot_attests_old_ack_without_changing_its_occurrence_timestamp() {
        let at = Instant::now();
        let ledger = Arc::new(Mutex::new(AcceptedWork::default()));
        let lease =
            DispatchLease::with_native_attempt(Arc::clone(&ledger), at, "old-terminal".into())
                .unwrap();
        lease.forwarded();
        let mut state = State::default();
        let mut f = frame(10);
        ack(&mut f, "old-terminal");
        f.cancellation_acks[0].event_seq = 2;
        f.cancellation_acks[0].ack_monotonic_secs = 1.0;
        observe(&mut state, &ledger, f, at + Duration::from_secs(60));
        assert_eq!(ledger.lock().unwrap().summary().pending, 0);
        assert_eq!(state.acknowledged_requests, 1);
        assert_eq!(state.applied_acknowledgements[0].ack_monotonic_secs, 1.0);
    }

    #[test]
    fn first_text_then_later_decode_abort_ack_or_foreign_attempt_is_idempotent() {
        let at = Instant::now();
        let ledger = Arc::new(Mutex::new(AcceptedWork::default()));
        let lease =
            DispatchLease::with_native_attempt(Arc::clone(&ledger), at, "text-completed".into())
                .unwrap();
        lease.forwarded();
        let mut state = State::default();
        let mut f = frame(1);
        active(&mut f, "text-completed", 20.0);
        observe(&mut state, &ledger, f, at);
        lease.first_text();
        let mut f = frame(2);
        ack(&mut f, "text-completed");
        observe(&mut state, &ledger, f, at + Duration::from_millis(2));
        let mut foreign = frame(3);
        ack(&mut foreign, "unrelated-foreign-attempt");
        observe(&mut state, &ledger, foreign, at + Duration::from_millis(4));
        assert_eq!(state.acknowledged_requests, 0);
        assert_eq!(state.released_known_work, 0.0);
        assert!(state.available(at + Duration::from_millis(5), 5.0));
        assert!(!ledger.lock().unwrap().summary().incomplete);
    }

    #[test]
    fn sglang_pinned_component_producer_frames_are_consumed_as_native_acks() {
        let active: WorkerLoadResponse = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/native_deadline/sglang_active.json"
        )))
        .unwrap();
        let terminal: WorkerLoadResponse = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/native_deadline/sglang_ack.json"
        )))
        .unwrap();
        let at = Instant::now();
        let ledger = Arc::new(Mutex::new(AcceptedWork::default()));
        let lease = DispatchLease::with_native_attempt(
            Arc::clone(&ledger),
            at,
            "sglang-component-attempt-0001".into(),
        )
        .unwrap();
        lease.forwarded();
        let mut state = State::default();
        state.observe(
            Some(&active),
            Some(0),
            at,
            at + Duration::from_millis(1),
            5.0,
            &mut ledger.lock().unwrap(),
        );
        assert_eq!(ledger.lock().unwrap().summary().unknown_work, 1);
        drop(lease);
        assert_eq!(ledger.lock().unwrap().summary().pending, 1);
        state.observe(
            Some(&terminal),
            Some(0),
            at + Duration::from_millis(2),
            at + Duration::from_millis(3),
            5.0,
            &mut ledger.lock().unwrap(),
        );
        assert_eq!(ledger.lock().unwrap().summary().pending, 0);
        assert!(!ledger.lock().unwrap().summary().incomplete);
        assert_eq!(state.acknowledged_requests, 1);
        assert_eq!(state.applied_acknowledgements[0].reason, "client_abort");
        assert_eq!(state.deadline_boundary(), "first_nonempty_text");
    }

    #[test]
    fn vllm_pinned_producer_requires_worker_cleanup_before_native_ack_release() {
        let frames = [
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/native_deadline/vllm_active.json"
            )),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/native_deadline/vllm_abort_pending.json"
            )),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/native_deadline/vllm_ack.json"
            )),
        ];
        let at = Instant::now();
        let ledger = Arc::new(Mutex::new(AcceptedWork::default()));
        let lease = DispatchLease::with_native_attempt(
            Arc::clone(&ledger),
            at,
            "00000000-0000-4000-8000-000000000061".into(),
        )
        .unwrap();
        lease.forwarded();
        let mut state = State::default();
        for (index, json) in frames.into_iter().enumerate() {
            let load: WorkerLoadResponse = serde_json::from_str(json).unwrap();
            let observed = at + Duration::from_secs(index as u64 + 1);
            state.observe(
                Some(&load),
                Some(0),
                observed,
                observed + Duration::from_millis(1),
                5.0,
                &mut ledger.lock().unwrap(),
            );
            if index < 2 {
                assert_eq!(ledger.lock().unwrap().summary().known_work, 50.0);
                assert_eq!(state.acknowledged_requests, 0);
            } else {
                assert_eq!(ledger.lock().unwrap().summary().pending, 0);
                assert_eq!(state.acknowledged_requests, 1);
                assert_eq!(state.released_known_work, 50.0);
            }
        }
    }
}
