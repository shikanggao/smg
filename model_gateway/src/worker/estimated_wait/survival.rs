//! Experimental C2 shadow: fixed-reference work fit, then statistical native
//! predecessor projection. Continuous batching is approximated by gateway
//! admission order. The 1s cleanup prior is an engineering assumption, never an
//! observed release, finite native bound, or admission-policy input.
use std::collections::{HashMap, HashSet, VecDeque};

use openai_protocol::{native_deadline::NativeDeadlineTelemetry, worker::WorkerLoadResponse};
use serde::Serialize;

use super::{basis, decision, pending_work};

pub(super) const METHOD: &str = "stable_work_native_service_composition_C2_raw_fallback_v1";
const LIMIT: usize = 8192;
const LAG_WINDOW: f64 = 600.0;
const ENGINEERING_PRIOR: f64 = 1.0;

fn nonnegative(v: f64) -> bool {
    v.is_finite() && v >= 0.0
}
fn positive(v: f64) -> bool {
    v.is_finite() && v > 0.0
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct Normalization {
    pub z: f64,
    pub selected_x: f64,
    pub admission_capacity: f64,
    pub capacity_reference: f64,
    pub reference_created_at: f64,
    pub basis_epoch: u64,
    pub counter_epoch: u64,
    pub dp_rank: i32,
    pub progress_source: &'static str,
}
#[derive(Clone, Debug, Serialize)]
pub(super) struct Admission {
    pub method: &'static str,
    pub normalization: Option<Normalization>,
    pub normalization_reason: &'static str,
    pub training_snapshot: decision::Snapshot,
    pub predictions: [decision::Prediction; 2],
    pub projection: Projection,
}
#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct RawFallback {
    pub baseline_seconds: f64,
    pub baseline_source: &'static str,
    pub raw_extra_seconds: f64,
    pub total_ttft_seconds: f64,
}
impl RawFallback {
    pub fn new(baseline: f64, source: &'static str, extra: f64) -> Option<Self> {
        let total = baseline + extra;
        (nonnegative(baseline)
            && nonnegative(extra)
            && nonnegative(total)
            && matches!(source, "measured" | "provisional"))
        .then_some(Self {
            baseline_seconds: baseline,
            baseline_source: source,
            raw_extra_seconds: extra,
            total_ttft_seconds: total,
        })
    }
}
impl Admission {
    /// Select only the displayed shadow fallback. The normalized fit and native
    /// service projection retain their original inputs and training provenance.
    pub fn select_raw_fallback(&mut self, raw: Option<RawFallback>) {
        let stable = &self.predictions[1];
        if self.projection.supported
            || !self.training_snapshot.fresh
            || !self.training_snapshot.native_ready
            || !stable.available
            || !nonnegative(stable.total)
            || self.projection.total_ttft_seconds != Some(stable.total)
            || self.projection.stable_work_fallback_seconds != Some(stable.total)
        {
            return;
        }
        if let Some(raw) = raw {
            self.projection.selected_raw_fallback = Some(raw);
            self.projection.total_ttft_seconds = Some(raw.total_ttft_seconds);
        }
    }
}
#[derive(Clone, Debug, Default, Serialize)]
pub(super) struct Projection {
    pub supported: bool,
    pub reason: &'static str,
    pub total_ttft_seconds: Option<f64>,
    pub stable_work_fallback_seconds: Option<f64>,
    /// Present only when the eligible same-admission raw total was selected.
    /// `stable_work_fallback_seconds` still records the unmodified internal fit.
    pub selected_raw_fallback: Option<RawFallback>,
    pub effective_capacity_tokens_per_second: Option<f64>,
    pub gain: Option<f64>,
    pub native_snapshot_seq: Option<u64>,
    pub native_engine_epoch: Option<String>,
    pub native_profile_id: Option<String>,
    pub native_confirmed_deadline_rows: usize,
    pub native_original_freshness_expiry: Option<f64>,
    pub predecessors: usize,
    pub native_known_remaining_rows: usize,
    pub full_prompt_proxy_rows: usize,
    pub unseen_admitted_rows: usize,
    pub first_text_decode_rows: usize,
    pub overdue_retained_rows: usize,
    pub overdue_retained_work_tokens: f64,
    pub input_work_tokens: f64,
    pub served_tokens: f64,
    pub statistically_forecast_removed_tokens: f64,
    pub observed_cleanup_lag_samples: usize,
    pub observed_past_max_cleanup_seconds: Option<f64>,
    pub newest_cleanup_sample_age_seconds: Option<f64>,
    pub engineering_cleanup_prior_seconds: f64,
    pub effective_cleanup_lag_seconds: Option<f64>,
    pub cleanup_lag_is_native_bound: bool,
    pub gain_applied_after_projection: bool,
    pub structural_binding_expiry_at_admission: bool,
    pub structural_expiry_rows: usize,
}
#[derive(Debug, Default)]
pub(super) struct State {
    fit: decision::State,
    basis_epoch: Option<u64>,
    reference: Option<Normalization>,
    pending_expiry_credit: HashMap<u64, bool>,
    pub native: Native,
}
impl State {
    pub fn poll(&mut self, at: f64, profile: Option<u64>, epoch: u64, fresh: bool, max_gap: f64) {
        if self.basis_epoch.is_some_and(|old| old != epoch) {
            self.fit = decision::State::default();
            self.reference = None;
            self.pending_expiry_credit.clear();
        }
        self.basis_epoch = Some(epoch);
        self.fit.poll(at, profile, fresh, max_gap);
    }
    pub fn admission(
        &mut self,
        basis: &basis::Admission,
        pending: Option<&pending_work::Admission>,
        scalar: Option<f64>,
    ) -> Admission {
        let mut training = basis.training_snapshot;
        training.scalar_gain = 1.0;
        let normalization = self.normalize(basis, pending);
        let reason = if normalization.is_some() {
            "fixed_reference_own_x"
        } else {
            "normalization_unavailable_paired_basis_fallback"
        };
        let predictions =
            if let Some(n) = normalization.filter(|_| !basis.paired_fallback_to_original) {
                training.x = n.z;
                let prediction = decision::Snapshot {
                    x: n.z,
                    ..basis.prediction_snapshot
                };
                self.fit.predictions(prediction, prediction.at, true)
            } else {
                training.eligible = false;
                basis.predictions.clone()
            };
        let stable = predictions[1]
            .available
            .then_some(predictions[1].total)
            .or(scalar);
        let mut projection = Projection {
            reason: "stable_normalization_missing",
            total_ttft_seconds: stable,
            stable_work_fallback_seconds: stable,
            engineering_cleanup_prior_seconds: ENGINEERING_PRIOR,
            ..Default::default()
        };
        if let (Some(n), Some(total)) = (normalization, stable) {
            let gain = (total - training.baseline) / n.z;
            projection.gain = positive(gain).then_some(gain);
            let capacity = n.capacity_reference / gain;
            if positive(n.z) && positive(gain) && positive(capacity) {
                projection.effective_capacity_tokens_per_second = Some(capacity);
                if let Err(reason) = self.native.project(
                    training.at,
                    basis.prediction_snapshot.poll_started,
                    training.baseline,
                    capacity,
                    n.admission_capacity,
                    &mut projection,
                ) {
                    projection.reason = reason;
                }
            } else {
                projection.reason = "stable_nonpositive_service_gain";
            }
        }
        Admission {
            method: METHOD,
            normalization,
            normalization_reason: reason,
            training_snapshot: training,
            predictions,
            projection,
        }
    }
    fn normalize(
        &mut self,
        a: &basis::Admission,
        pending: Option<&pending_work::Admission>,
    ) -> Option<Normalization> {
        let p = pending?;
        let s = a.prediction_snapshot;
        let stamp = s.size_basis?;
        let provenance = stamp.provenance;
        if !p.available
            || !s.fresh
            || !s.native_ready
            || !positive(p.capacity)
            || p.capacity_progress_source.as_deref() != Some(provenance.progress_source)
            || !matches!(
                provenance.progress_source,
                "local_compute_tokens" | "prefill_started_requests"
            )
            || provenance.dp_rank < 0
            || self.basis_epoch != Some(stamp.epoch)
        {
            return None;
        }
        let mut n = Normalization {
            z: 0.0,
            selected_x: s.x,
            admission_capacity: p.capacity,
            capacity_reference: p.capacity,
            reference_created_at: s.at,
            basis_epoch: stamp.epoch,
            counter_epoch: provenance.counter_epoch,
            dp_rank: provenance.dp_rank,
            progress_source: provenance.progress_source,
        };
        if let Some(reference) = self.reference {
            if reference.reference_created_at > s.at {
                return None;
            }
            if reference.basis_epoch != n.basis_epoch
                || reference.counter_epoch != n.counter_epoch
                || reference.dp_rank != n.dp_rank
                || reference.progress_source != n.progress_source
            {
                return None;
            }
            n.capacity_reference = reference.capacity_reference;
            n.reference_created_at = reference.reference_created_at;
        } else {
            self.reference = Some(n);
        }
        n.z = s.x * p.capacity / n.capacity_reference;
        nonnegative(n.z).then_some(n)
    }
    pub fn register(&mut self, id: Option<u64>, started: f64, admission: Option<&Admission>) {
        if self
            .fit
            .register(id, started, admission.map(|a| a.training_snapshot))
        {
            if let Some(id) = id {
                self.pending_expiry_credit.insert(
                    id,
                    admission.is_some_and(|a| {
                        a.projection.supported
                            && a.projection.structural_binding_expiry_at_admission
                    }),
                );
            }
        }
    }
    pub fn label(&mut self, label: &decision::Label, terminal: bool) -> &'static str {
        let binding_expiry = self
            .pending_expiry_credit
            .get(&label.id)
            .copied()
            .unwrap_or(false);
        if terminal {
            self.pending_expiry_credit.remove(&label.id);
        }
        if label.exact && binding_expiry {
            if terminal {
                self.fit.terminal_unknown(Some(label.id));
            }
            // Retain any earlier censor row; only the unadjusted exact service
            // interpretation is excluded. The admission flag never changes.
            return "excluded_exact_own_admission_binding_expiry";
        }
        if self
            .fit
            .label(
                label.id,
                label.observed_at,
                label.at,
                label.exact,
                terminal,
                Some(label.elapsed),
            )
            .is_some()
        {
            if label.exact {
                "retained_exact"
            } else {
                "retained_censor"
            }
        } else {
            "untracked_or_ineligible"
        }
    }
    pub fn terminal_unknown(&mut self, id: Option<u64>) {
        self.fit.terminal_unknown(id);
        if let Some(id) = id {
            self.pending_expiry_credit.remove(&id);
        }
    }
}

#[derive(Clone, Debug)]
struct Dispatch {
    at: f64,
    ordinal: u64,
    estimate: pending_work::Estimate,
    identity: Option<(String, String)>,
    seen_native: bool,
    first_text_at: Option<f64>,
}
#[derive(Clone, Debug)]
struct Deadline {
    native: Option<String>,
    deadline: f64,
    observed: f64,
    sequence: u64,
    first_text: bool,
}
#[derive(Clone, Debug)]
struct Snapshot {
    lane: NativeDeadlineTelemetry,
    received: f64,
    available: f64,
    started: f64,
    raw_poll: f64,
    expiry: f64,
    last_age: f64,
}
#[derive(Clone, Debug, Default)]
pub(super) struct Native {
    snapshot: Option<Snapshot>,
    valid: bool,
    reason: &'static str,
    retired: HashSet<(String, String)>,
    admissions: HashMap<String, Dispatch>,
    deadlines: HashMap<String, Deadline>,
    acks: HashMap<String, openai_protocol::native_deadline::NativeCancellationAck>,
    lags: VecDeque<(f64, f64)>,
    ordinal: u64,
    incomplete: bool,
    capacity_key: Option<(u64, u64, String)>,
    capacity_reason: &'static str,
    capacity_poll: Option<(f64, bool, Option<f64>)>,
}
impl Native {
    fn identity(&self) -> Option<(String, String)> {
        self.snapshot
            .as_ref()
            .map(|s| (s.lane.engine_epoch.clone(), s.lane.profile_id.clone()))
    }
    pub fn register(&mut self, attempt: Option<&str>, at: f64, estimate: pending_work::Estimate) {
        let Some(attempt) = attempt else {
            return;
        };
        if self.admissions.len() >= LIMIT
            || self.ordinal == u64::MAX
            || self.admissions.contains_key(attempt)
        {
            self.incomplete = true;
            return;
        }
        self.ordinal += 1;
        self.admissions.insert(
            attempt.to_owned(),
            Dispatch {
                at,
                ordinal: self.ordinal,
                estimate,
                identity: self.identity(),
                seen_native: false,
                first_text_at: None,
            },
        );
    }
    pub fn local_pre_send_rollback(&mut self, attempt: &str) {
        self.admissions.remove(attempt);
    }
    pub fn first_text(&mut self, attempt: Option<&str>, at: f64) {
        if let Some(row) = attempt.and_then(|id| self.admissions.get_mut(id)) {
            row.first_text_at = Some(at);
        }
        if let Some(row) = attempt.and_then(|id| self.deadlines.get_mut(id)) {
            row.first_text = true;
        }
    }
    pub fn raw_poll(&mut self, load: Option<&WorkerLoadResponse>, started: f64, success: bool) {
        let eligible = load
            .filter(|l| success && l.dp_rank_count == 1 && l.loads.len() == 1)
            .and_then(|l| {
                let r = &l.loads[0];
                let p = r.prefill_metric_provenance.as_ref()?;
                let source = r.timely_prefill_progress_source.as_deref()?;
                let key = (
                    r.native_request_counts_profile?,
                    p.counter_epoch,
                    source.to_owned(),
                );
                let valid = r.dp_rank == 0
                    && r.native_request_counts_available == Some(true)
                    && p.schema_version == 1
                    && p.progress_source.as_deref() == Some(source)
                    && !p.counter_reset
                    && matches!(source, "local_compute_tokens" | "prefill_started_requests")
                    && r.learned_prefill_capacity.is_some_and(positive)
                    && r.prefill_capacity_sample_count.unwrap_or(0) > 0
                    && (source != "prefill_started_requests"
                        || p.previous_conversion_size_mean.is_some_and(positive)
                            && p.raw_rate.is_some_and(positive));
                Some((key, valid))
            });
        if let Some((key, _)) = &eligible {
            if self.capacity_key.as_ref().is_some_and(|old| old != key) {
                self.deadlines.clear();
                self.acks.clear();
                self.lags.clear();
                self.valid = false;
                self.reason = "capacity_source_or_profile_reset";
            }
            self.capacity_key = Some(key.clone());
        }
        self.capacity_reason = if load.and_then(|l| l.loads.first()).is_some_and(|r| {
            r.timely_prefill_progress_source.as_deref() == Some("prefill_started_requests")
                && r.prefill_metric_provenance.as_ref().is_some_and(|p| {
                    !p.previous_conversion_size_mean.is_some_and(positive)
                        || !p.raw_rate.is_some_and(positive)
                })
        }) {
            "request_start_capacity_conversion_missing_or_nonpositive"
        } else {
            "same_poll_capacity_unavailable"
        };
        self.capacity_poll = Some((
            started,
            eligible.is_some_and(|(_, valid)| valid),
            load.and_then(|l| l.loads.first())
                .and_then(|r| r.learned_prefill_capacity),
        ));
    }
    pub fn observe(
        &mut self,
        load: Option<&WorkerLoadResponse>,
        accepted: bool,
        started: f64,
        received: f64,
        raw_poll: f64,
        max_age: f64,
    ) {
        // Commit only a wholly validated observation. An invalid late row must
        // not leave an earlier ACK/lag mutation available to the next poll.
        let mut next = self.clone();
        match next.observe_inner(load, accepted, started, received, raw_poll, max_age) {
            Ok(()) => *self = next,
            Err(reason) => {
                self.valid = false;
                self.reason = reason;
            }
        }
    }
    fn observe_inner(
        &mut self,
        load: Option<&WorkerLoadResponse>,
        accepted: bool,
        started: f64,
        received: f64,
        raw_poll: f64,
        max_age: f64,
    ) -> Result<(), &'static str> {
        self.valid = false;
        let load = load.ok_or("native_inventory_missing")?;
        if !load.ranks_are_dp_ranks()
            || load.dp_rank_count != 1
            || load.loads.len() != 1
            || load.loads[0].dp_rank != 0
        {
            return Err("native_rank_scope_unsupported");
        }
        let lane = load.loads[0]
            .deadline_recovery
            .as_ref()
            .ok_or("native_inventory_missing")?;
        if !accepted
            || !lane.active_inventory_complete
            || !lane.cancellation_ack_supported
            || lane.native_population != lane.active_requests.len()
        {
            return Err("native_inventory_unaccepted_or_incomplete");
        }
        let mut native_ids = HashSet::new();
        if lane.snapshot_seq == 0
            || lane.active_requests.iter().any(|r| {
                r.native_request_id
                    .as_ref()
                    .is_none_or(|id| !native_ids.insert(id.as_str()))
            })
        {
            return Err("native_request_identity_missing_or_duplicated");
        }
        let identity = (lane.engine_epoch.clone(), lane.profile_id.clone());
        if self.retired.contains(&identity) {
            return Err("retired_native_identity");
        }
        let old_identity = self.identity();
        let reset = old_identity.as_ref().is_some_and(|old| *old != identity);
        if let Some(old_identity) = old_identity.filter(|_| reset) {
            if self.retired.len() >= 64 {
                self.incomplete = true;
                return Err("native_identity_history_overflow");
            }
            self.retired.insert(old_identity);
            self.deadlines.clear();
            self.acks.clear();
            self.lags.clear();
            self.snapshot = None;
        }
        if let Some(previous) = &mut self.snapshot {
            if received < previous.available || lane.snapshot_seq < previous.lane.snapshot_seq {
                return Err("native_sequence_or_receipt_reversed");
            }
            if lane.snapshot_seq == previous.lane.snapshot_seq {
                let mut comparable = lane.clone();
                comparable.snapshot_age_seconds = previous.lane.snapshot_age_seconds;
                if comparable != previous.lane
                    || lane.snapshot_age_seconds + 1e-6 < previous.last_age
                {
                    return Err("changed_or_rejuvenated_native_duplicate");
                }
                previous.last_age = lane.snapshot_age_seconds;
                previous.raw_poll = raw_poll;
                previous.available = received;
                previous.expiry = previous
                    .expiry
                    .min(previous.started + max_age - previous.lane.snapshot_age_seconds);
                self.valid = true;
                self.reason = "duplicate_retains_original_expiry";
                return Ok(());
            }
        }
        let previous: HashMap<_, _> = self
            .snapshot
            .as_ref()
            .map(|old| {
                old.lane
                    .active_requests
                    .iter()
                    .map(|r| (r.native_request_id.as_deref(), r))
                    .collect()
            })
            .unwrap_or_default();
        let rebound = lane.active_requests.iter().any(|r| {
            previous
                .get(&r.native_request_id.as_deref())
                .filter(|p| p.attempt_id == r.attempt_id)
                .is_some_and(|p| {
                    p.remaining_prefill_work_tokens
                        .zip(r.remaining_prefill_work_tokens)
                        .is_some_and(|(a, b)| b > a)
                })
        });
        if reset || rebound {
            self.snapshot = Some(Snapshot {
                lane: lane.clone(),
                received,
                available: received,
                started,
                raw_poll,
                expiry: started + max_age - lane.snapshot_age_seconds,
                last_age: lane.snapshot_age_seconds,
            });
            self.reason = if reset {
                "native_identity_reset"
            } else {
                "native_remaining_work_rebound"
            };
            return Ok(());
        }
        let active: HashSet<_> = lane
            .active_requests
            .iter()
            .filter_map(|r| r.attempt_id.as_deref())
            .collect();
        for ack in &lane.cancellation_acks {
            if active.contains(ack.attempt_id.as_str()) {
                return Err("ack_before_inventory_release");
            }
            if let Some(old) = self.acks.get(&ack.attempt_id) {
                if old != ack {
                    return Err("changed_repeated_native_ack");
                }
                continue;
            }
            if let Some(prior) = self.deadlines.get(&ack.attempt_id).filter(|d| {
                !d.first_text
                    && d.native == ack.native_request_id
                    && d.observed < received
                    && ack.event_seq > d.sequence
            }) {
                let lag = ack.ack_monotonic_secs - prior.deadline;
                if ack.reason == "hard_ttft_deadline" && nonnegative(lag) && !reset && !rebound {
                    self.lags.push_back((received, lag));
                }
            }
            if self.acks.len() >= LIMIT {
                self.incomplete = true;
                return Err("native_ack_history_overflow");
            }
            self.acks.insert(ack.attempt_id.clone(), ack.clone());
        }
        for row in &lane.active_requests {
            if let Some(known) = row
                .attempt_id
                .as_ref()
                .and_then(|id| self.admissions.get_mut(id))
            {
                known.seen_native = true;
            }
            if let Some((attempt, deadline)) = row
                .attempt_id
                .as_ref()
                .zip(row.hard_ttft_deadline_monotonic_secs)
                .filter(|(_, d)| nonnegative(*d))
            {
                if lane.hard_ttft_deadline_supported && lane.abort_enforcement_enabled {
                    if lane.deadline_boundary != "first_nonempty_text"
                        || lane.deadline_origin.as_deref() != Some("engine_http_receive_monotonic")
                    {
                        return Err("native_deadline_boundary_unsupported");
                    }
                    if let Some(old) = self.deadlines.get_mut(attempt) {
                        if old.native != row.native_request_id || old.deadline != deadline {
                            return Err("changed_attempt_deadline");
                        }
                        old.first_text |= row.first_text_emitted == Some(true);
                    } else {
                        if self.deadlines.len() >= LIMIT {
                            self.incomplete = true;
                            return Err("native_deadline_history_overflow");
                        }
                        self.deadlines.insert(
                            attempt.clone(),
                            Deadline {
                                native: row.native_request_id.clone(),
                                deadline,
                                observed: received,
                                sequence: row.state_seq,
                                first_text: row.first_text_emitted == Some(true),
                            },
                        );
                    }
                }
            }
        }
        while self
            .lags
            .front()
            .is_some_and(|(at, _)| received - at > LAG_WINDOW)
        {
            self.lags.pop_front();
        }
        self.snapshot = Some(Snapshot {
            lane: lane.clone(),
            received,
            available: received,
            started,
            raw_poll,
            expiry: started + max_age - lane.snapshot_age_seconds,
            last_age: lane.snapshot_age_seconds,
        });
        self.valid = !reset && !rebound;
        self.reason = if reset {
            "native_identity_reset"
        } else if rebound {
            "native_remaining_work_rebound"
        } else {
            "native_inventory_available"
        };
        Ok(())
    }
    fn project(
        &self,
        at: f64,
        raw_poll: f64,
        baseline: f64,
        capacity: f64,
        admission_capacity: f64,
        out: &mut Projection,
    ) -> Result<(), &'static str> {
        if self.incomplete {
            return Err("native_history_incomplete_or_overflow");
        }
        if !self.valid {
            return Err(if self.reason.is_empty() {
                "native_telemetry_unavailable"
            } else {
                self.reason
            });
        }
        let s = self.snapshot.as_ref().ok_or("native_inventory_missing")?;
        if s.available > at || s.received > at || at > s.expiry {
            return Err("native_snapshot_stale_or_not_received");
        }
        let matched_capacity = self.capacity_poll.is_some_and(|(poll, valid, cap)| {
            poll == raw_poll
                && valid
                && cap.is_some_and(|v| {
                    (v - admission_capacity).abs() <= 1e-9 * v.abs().max(admission_capacity.abs())
                })
        });
        if s.raw_poll != raw_poll {
            return Err("native_raw_admission_poll_mismatch");
        }
        if !matched_capacity {
            return Err(self.capacity_reason);
        }
        let lane = &s.lane;
        if lane.scheduler_semantics != "continuous_batching"
            || !lane.hard_ttft_deadline_supported
            || !lane.abort_enforcement_enabled
            || lane.deadline_boundary != "first_nonempty_text"
            || lane.deadline_origin.as_deref() != Some("engine_http_receive_monotonic")
        {
            return Err("native_hard_deadline_contract_unavailable");
        }
        let identity = (lane.engine_epoch.clone(), lane.profile_id.clone());
        let mut rows = Vec::with_capacity(lane.active_requests.len());
        let mut active = HashSet::new();
        for row in &lane.active_requests {
            if let Some(attempt) = row.attempt_id.as_deref() {
                active.insert(attempt);
            }
            if row.first_text_emitted == Some(true) {
                out.first_text_decode_rows += 1;
                continue;
            }
            let attempt = row
                .attempt_id
                .as_deref()
                .ok_or("native_predecessor_gateway_order_unknown")?;
            let known = self
                .admissions
                .get(attempt)
                .ok_or("native_predecessor_gateway_order_unknown")?;
            if known.at > at || known.identity.as_ref().is_some_and(|old| *old != identity) {
                return Err("predecessor_identity_or_frozen_cutoff_mismatch");
            }
            if known.first_text_at.is_some_and(|t| t <= at) {
                out.first_text_decode_rows += 1;
                continue;
            }
            let work = if let Some(work) = row
                .remaining_prefill_work_tokens
                .filter(|v| nonnegative(*v))
            {
                out.native_known_remaining_rows += 1;
                work
            } else {
                if !matches!(row.state.as_str(), "waiting" | "prefill") {
                    return Err("unknown_no_text_decode_work");
                }
                out.full_prompt_proxy_rows += 1;
                prompt(known.estimate)?
            };
            out.native_confirmed_deadline_rows +=
                usize::from(row.hard_ttft_deadline_monotonic_secs.is_some());
            let remaining = row.hard_ttft_deadline_monotonic_secs.map(|deadline| {
                deadline
                    - lane.captured_at_monotonic_secs
                    - lane.snapshot_age_seconds
                    - (at - s.received)
            });
            rows.push((known.ordinal, work, remaining));
        }
        for (attempt, known) in &self.admissions {
            if known.at > at
                || active.contains(attempt.as_str())
                || self.acks.contains_key(attempt)
                || known.seen_native
                || known.first_text_at.is_some_and(|t| t <= at)
            {
                continue;
            }
            if known.identity.as_ref().is_some_and(|old| *old != identity) {
                return Err("unseen_dispatch_native_epoch_mismatch");
            }
            out.unseen_admitted_rows += 1;
            out.full_prompt_proxy_rows += 1;
            rows.push((known.ordinal, prompt(known.estimate)?, None));
        }
        rows.sort_unstable_by_key(|r| r.0);
        let samples: Vec<_> = self
            .lags
            .iter()
            .filter(|(t, _)| *t <= at && at - t <= LAG_WINDOW)
            .collect();
        let measured = samples.iter().map(|(_, v)| *v).reduce(f64::max);
        let lag = measured.map_or(ENGINEERING_PRIOR, |v| v.max(ENGINEERING_PRIOR));
        out.observed_cleanup_lag_samples = samples.len();
        out.observed_past_max_cleanup_seconds = measured;
        out.newest_cleanup_sample_age_seconds = samples.last().map(|(t, _)| at - t);
        out.effective_cleanup_lag_seconds = Some(lag);
        out.native_snapshot_seq = Some(lane.snapshot_seq);
        out.native_engine_epoch = Some(lane.engine_epoch.clone());
        out.native_profile_id = Some(lane.profile_id.clone());
        out.native_original_freshness_expiry = Some(s.expiry);
        project_rows(
            rows.iter().map(|(_, work, deadline)| (*work, *deadline)),
            capacity,
            baseline,
            lag,
            out,
        )?;
        Ok(())
    }
}
fn prompt(estimate: pending_work::Estimate) -> Result<f64, &'static str> {
    estimate
        .tokens
        .filter(|v| {
            positive(*v)
                && !estimate.uncached_tokens_known
                && matches!(
                    estimate.source,
                    "utf8_bytes_div4_proxy" | "body_token_ids_full_prompt"
                )
        })
        .ok_or("recorded_prompt_estimate_missing")
}
fn project_rows(
    rows: impl IntoIterator<Item = (f64, Option<f64>)>,
    capacity: f64,
    baseline: f64,
    lag: f64,
    out: &mut Projection,
) -> Result<(), &'static str> {
    if !positive(capacity) || !nonnegative(baseline) || !nonnegative(lag) {
        return Err("invalid_projection_capacity_baseline_or_lag");
    }
    let mut elapsed = 0.0;
    for (work, mut deadline) in rows {
        if !nonnegative(work) || deadline.is_some_and(|d| !d.is_finite()) {
            return Err("unknown_work_or_invalid_deadline");
        }
        if deadline.is_some_and(|d| d < 0.0) {
            out.overdue_retained_rows += 1;
            out.overdue_retained_work_tokens += work;
            deadline = None;
        }
        let service = work / capacity;
        let duration = deadline.map_or(service, |d| service.min((d + lag - elapsed).max(0.0)));
        if deadline.is_some() && duration < service {
            out.structural_binding_expiry_at_admission = true;
            out.structural_expiry_rows += 1;
        }
        let consumed = work.min(duration * capacity);
        elapsed += duration;
        out.predecessors += 1;
        out.input_work_tokens += work;
        out.served_tokens += consumed;
        out.statistically_forecast_removed_tokens += (work - consumed).max(0.0);
    }
    let total = baseline + elapsed;
    if !nonnegative(total) {
        return Err("nonfinite_projection");
    }
    out.supported = true;
    out.reason = "experimental_statistical_projection_engineering_prior_not_bound";
    out.total_ttft_seconds = Some(total);
    Ok(())
}

#[cfg(test)]
mod tests {
    use openai_protocol::{
        common::PromptWorkHint,
        native_deadline::{NativeCancellationAck, NativeDeadlineRequest},
        worker::{PrefillMetricProvenance, SchedulerLoadSnapshot},
    };

    use super::*;
    fn near(a: f64, b: f64) {
        assert!(
            (a - b).abs() <= 1e-8 * a.abs().max(b.abs()).max(1.0),
            "{a} != {b}"
        );
    }
    fn load(seq: u64) -> WorkerLoadResponse {
        let lane: NativeDeadlineTelemetry = serde_json::from_value(serde_json::json!({
            "schema_version":1,"engine_epoch":"engine-a","profile_id":"profile-a","dp_rank":0,
            "snapshot_seq":seq,"captured_at_monotonic_secs":100.0+seq as f64,"snapshot_age_seconds":0.0,
            "active_inventory_complete":true,"native_population":0,"scheduler_semantics":"continuous_batching",
            "deadline_boundary":"first_nonempty_text","deadline_origin":"engine_http_receive_monotonic",
            "remaining_work_source":"native_uncached","effective_capacity_tokens_per_sec":null,"capacity_source":null,
            "hard_ttft_deadline_supported":true,"cancellation_ack_supported":true,"abort_enforcement_enabled":true,
            "abort_max_lag_seconds":null,"ack_floor_seq":0,"active_requests":[],"cancellation_acks":[]
        })).unwrap();
        WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![SchedulerLoadSnapshot {
                dp_rank: 0,
                native_request_counts_available: Some(true),
                native_request_counts_profile: Some(7),
                learned_prefill_capacity: Some(10.0),
                prefill_capacity_sample_count: Some(12),
                timely_prefill_progress_source: Some("local_compute_tokens".into()),
                prefill_metric_provenance: Some(PrefillMetricProvenance {
                    schema_version: 1,
                    progress_source: Some("local_compute_tokens".into()),
                    ..Default::default()
                }),
                deadline_recovery: Some(lane),
                ..Default::default()
            }],
            ..Default::default()
        }
    }
    fn lane(load: &mut WorkerLoadResponse) -> &mut NativeDeadlineTelemetry {
        load.loads[0].deadline_recovery.as_mut().unwrap()
    }
    fn row(attempt: &str, seq: u64, work: Option<f64>, deadline: f64) -> NativeDeadlineRequest {
        NativeDeadlineRequest {
            attempt_id: Some(attempt.into()),
            native_request_id: Some(format!("native-{attempt}")),
            state_seq: seq,
            remaining_prefill_work_tokens: work,
            hard_ttft_deadline_monotonic_secs: Some(deadline),
            queue_position: None,
            state: "prefill".into(),
            first_text_emitted: Some(false),
        }
    }
    fn observe(n: &mut Native, load: &WorkerLoadResponse, received: f64, raw: f64) {
        n.raw_poll(Some(load), raw, true);
        n.observe(Some(load), true, received - 0.01, received, raw, 5.0);
    }
    fn project(n: &Native, at: f64, poll: f64) -> Result<Projection, &'static str> {
        let mut p = Projection {
            engineering_cleanup_prior_seconds: ENGINEERING_PRIOR,
            ..Default::default()
        };
        n.project(at, poll, 2.0, 10.0, 10.0, &mut p)?;
        Ok(p)
    }
    fn assert_actual_goldens(input: &str) {
        let fixture: serde_json::Value = serde_json::from_str(input).unwrap();
        assert!(fixture["samples"].as_array().unwrap().len() >= 10);
        for s in fixture["samples"].as_array().unwrap() {
            let service = &s["service"];
            let k = service["gain"].as_f64().unwrap();
            let capacity = service["capacity_reference"].as_f64().unwrap() / k;
            near(
                capacity,
                service["effective_capacity_tokens_per_second"]
                    .as_f64()
                    .unwrap(),
            );
            let rows = s["predecessors"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| (r[0].as_f64().unwrap(), r[1].as_f64()));
            let mut out = Projection::default();
            project_rows(
                rows,
                capacity,
                service["baseline"].as_f64().unwrap(),
                s["lag"].as_f64().unwrap(),
                &mut out,
            )
            .unwrap();
            let expected = &s["expected"];
            if let Some(binding) = s["structural_binding_expiry"].as_bool() {
                assert_eq!(out.structural_binding_expiry_at_admission, binding);
            }
            near(
                out.total_ttft_seconds.unwrap(),
                expected["total_ttft_seconds"].as_f64().unwrap(),
            );
            near(
                out.input_work_tokens,
                expected["input_remaining_tokens"].as_f64().unwrap(),
            );
            near(
                out.served_tokens,
                expected["served_tokens"].as_f64().unwrap(),
            );
            near(
                out.statistically_forecast_removed_tokens,
                expected["statistically_forecast_removed_tokens"]
                    .as_f64()
                    .unwrap(),
            );
            assert_eq!(
                out.overdue_retained_rows as u64,
                expected["overdue_retained_rows"].as_u64().unwrap()
            );
            near(
                out.overdue_retained_work_tokens,
                expected["overdue_retained_estimated_work_tokens"]
                    .as_f64()
                    .unwrap(),
            );
        }
    }
    #[test]
    fn actual_qwen_c2_and_guarded_prediction_input_goldens() {
        assert_actual_goldens(include_str!("fixtures/survival_c2_qwen.json"));
        assert_actual_goldens(include_str!("fixtures/survival_c2_guarded_qwen.json"));
    }
    #[test]
    fn capacity_gain_applies_before_expiry_and_overdue_work_is_retained() {
        let mut out = Projection::default();
        project_rows([(100.0, Some(5.0))], 10.0 / 2.0, 2.0, 1.0, &mut out).unwrap();
        near(out.total_ttft_seconds.unwrap(), 8.0);
        near(out.served_tokens, 30.0);
        near(out.statistically_forecast_removed_tokens, 70.0);
        assert!(!out.gain_applied_after_projection);
        let mut out = Projection::default();
        project_rows(
            [(100.0, Some(-0.1)), (100.0, Some(5.0))],
            10.0,
            2.0,
            1.0,
            &mut out,
        )
        .unwrap();
        near(out.total_ttft_seconds.unwrap(), 12.0);
        near(out.input_work_tokens, 200.0);
        near(out.statistically_forecast_removed_tokens, 100.0);
        assert_eq!(out.overdue_retained_rows, 1);
        near(out.overdue_retained_work_tokens, 100.0);
    }
    #[test]
    fn duplicate_poll_preserves_original_freshness_and_never_relearns_ack() {
        let mut n = Native::default();
        let first = load(1);
        observe(&mut n, &first, 1.0, 0.9);
        assert!(project(&n, 2.0, 0.9).is_ok());
        n.raw_poll(Some(&first), 3.9, true);
        n.observe(Some(&first), true, 3.99, 4.0, 3.9, 5.0);
        near(n.snapshot.as_ref().unwrap().expiry, 5.99);
        assert!(project(&n, 4.5, 3.9).is_ok());
        assert_eq!(
            project(&n, 6.0, 3.9).unwrap_err(),
            "native_snapshot_stale_or_not_received"
        );
        let mut changed = first.clone();
        lane(&mut changed).abort_enforcement_enabled = false;
        n.observe(Some(&changed), true, 4.4, 4.5, 3.9, 5.0);
        assert_eq!(
            project(&n, 4.6, 3.9).unwrap_err(),
            "changed_or_rejuvenated_native_duplicate"
        );
        assert!(n.lags.is_empty());
    }
    #[test]
    fn lag_is_only_past_hard_ack_prior_deadline_and_first_text_wins() {
        let mut n = Native::default();
        let mut active = load(1);
        lane(&mut active).active_requests = vec![row("a", 1, Some(100.0), 102.0)];
        lane(&mut active).native_population = 1;
        n.register(Some("a"), 0.0, pending_work::Estimate::default());
        observe(&mut n, &active, 1.0, 0.9);
        let before = project(&n, 1.1, 0.9).unwrap();
        assert_eq!(before.observed_cleanup_lag_samples, 0);
        near(before.effective_cleanup_lag_seconds.unwrap(), 1.0);
        let mut ack = load(3);
        lane(&mut ack).cancellation_acks = vec![NativeCancellationAck {
            attempt_id: "a".into(),
            native_request_id: Some("native-a".into()),
            event_seq: 3,
            ack_monotonic_secs: 102.4,
            reason: "hard_ttft_deadline".into(),
        }];
        observe(&mut n, &ack, 3.0, 2.9);
        let after = project(&n, 3.1, 2.9).unwrap();
        assert_eq!(after.observed_cleanup_lag_samples, 1);
        near(after.observed_past_max_cleanup_seconds.unwrap(), 0.4);
        near(after.effective_cleanup_lag_seconds.unwrap(), 1.0);
        let mut repeated = ack.clone();
        lane(&mut repeated).snapshot_seq = 4;
        observe(&mut n, &repeated, 4.0, 3.9);
        assert_eq!(n.lags.len(), 1);
        let frozen = before.total_ttft_seconds;
        n.first_text(Some("a"), 4.1);
        assert_eq!(before.total_ttft_seconds, frozen);
        let mut text = load(5);
        let mut r = row("b", 5, Some(0.0), 106.0);
        r.first_text_emitted = Some(true);
        lane(&mut text).active_requests = vec![r];
        lane(&mut text).native_population = 1;
        observe(&mut n, &text, 5.0, 4.9);
        let mut late = load(7);
        lane(&mut late).cancellation_acks = vec![NativeCancellationAck {
            attempt_id: "b".into(),
            native_request_id: Some("native-b".into()),
            event_seq: 7,
            ack_monotonic_secs: 106.4,
            reason: "hard_ttft_deadline".into(),
        }];
        observe(&mut n, &late, 7.0, 6.9);
        assert_eq!(n.lags.len(), 1);
    }
    #[test]
    fn unknown_decode_and_unmatched_admission_fall_back_but_waiting_proxy_counts() {
        let mut n = Native::default();
        let mut active = load(1);
        lane(&mut active).active_requests = vec![row("a", 1, None, 150.0)];
        lane(&mut active).native_population = 1;
        observe(&mut n, &active, 1.0, 0.9);
        assert_eq!(
            project(&n, 1.1, 0.9).unwrap_err(),
            "native_predecessor_gateway_order_unknown"
        );
        n.register(
            Some("a"),
            1.0,
            pending_work::Estimate::from_hint(PromptWorkHint::BodyTokenIds(100)),
        );
        observe(&mut n, &active, 1.2, 0.9);
        let p = project(&n, 1.3, 0.9).unwrap();
        assert_eq!(p.full_prompt_proxy_rows, 1);
        near(p.input_work_tokens, 100.0);
        let mut decode = load(2);
        let mut r = row("a", 2, None, 150.0);
        r.state = "decode".into();
        lane(&mut decode).active_requests = vec![r];
        lane(&mut decode).native_population = 1;
        observe(&mut n, &decode, 2.0, 1.9);
        assert_eq!(
            project(&n, 2.1, 1.9).unwrap_err(),
            "unknown_no_text_decode_work"
        );
    }
    #[test]
    fn native_epoch_capacity_source_and_remaining_work_rebound_fail_closed() {
        let mut n = Native::default();
        let mut active = load(1);
        lane(&mut active).active_requests = vec![row("a", 1, Some(10.0), 150.0)];
        lane(&mut active).native_population = 1;
        n.register(Some("a"), 0.0, pending_work::Estimate::default());
        observe(&mut n, &active, 1.0, 0.9);
        let mut rebound = active.clone();
        lane(&mut rebound).snapshot_seq = 2;
        lane(&mut rebound).active_requests[0].remaining_prefill_work_tokens = Some(20.0);
        observe(&mut n, &rebound, 2.0, 1.9);
        assert_eq!(
            project(&n, 2.1, 1.9).unwrap_err(),
            "native_remaining_work_rebound"
        );
        let mut restart = load(3);
        lane(&mut restart).engine_epoch = "engine-b".into();
        observe(&mut n, &restart, 3.0, 2.9);
        assert_eq!(project(&n, 3.1, 2.9).unwrap_err(), "native_identity_reset");
        assert!(n.deadlines.is_empty());
        assert!(n.lags.is_empty());
        n.observe(Some(&active), true, 3.9, 4.0, 3.9, 5.0);
        assert_eq!(
            project(&n, 4.1, 3.9).unwrap_err(),
            "retired_native_identity"
        );
        let mut source = load(5);
        source.loads[0]
            .prefill_metric_provenance
            .as_mut()
            .unwrap()
            .counter_epoch = 1;
        n.raw_poll(Some(&source), 4.9, true);
        assert_eq!(n.reason, "capacity_source_or_profile_reset");
        assert!(!n.valid);
    }
    fn admissions(
        at: f64,
        epoch: u64,
        x: f64,
        capacity: f64,
    ) -> (basis::Admission, pending_work::Admission) {
        let provenance = basis::Provenance {
            kind: basis::Kind::Observed,
            dp_rank: 0,
            counter_epoch: 0,
            progress_source: "local_compute_tokens",
            observed_sample_count: 12,
            unshrunk_observed_mean: Some(1000.0),
            configured_prior_tokens: 1024.0,
            configured_prior_samples: 32,
            effective_prior_weight: 0.5,
            blended_queue_tokens_per_request: 1000.0,
        };
        let s = decision::Snapshot {
            size_basis: Some(basis::Stamp { epoch, provenance }),
            at,
            profile: 7,
            baseline: 2.0,
            x,
            source: "live_n",
            original_queue_extra: x,
            live_queue_extra: Some(x),
            scalar_gain: 1.0,
            fresh: true,
            native_ready: true,
            native_waiting: 1.0,
            poll_started: at,
            eligible: true,
            context: decision::Context {
                complete: true,
                ..Default::default()
            },
        };
        let predictions = decision::State::default().predictions(s, at, true);
        let basis = basis::Admission {
            method: basis::METHOD,
            basis_epoch: epoch,
            basis_reason: "verified",
            paired_fallback_to_original: false,
            training_snapshot: decision::Snapshot { x: 1000.0, ..s },
            prediction_snapshot: s,
            paired_control_method: "control",
            paired_control_predictions: predictions.clone(),
            predictions: predictions.clone(),
        };
        let queue = pending_work::Queue::new(
            pending_work::Summary::default(),
            1000.0,
            1.0,
            0.0,
            Some(1000.0),
            true,
            false,
        );
        let pending = pending_work::Admission {
            method: pending_work::METHOD,
            current_inventory_generation: 0,
            queue,
            capacity_progress_source: Some("local_compute_tokens".into()),
            selected_pending_applied: true,
            source_gate_reason: "local_compute_tokens",
            poll_started: at,
            at,
            debt_seconds: 0.0,
            capacity,
            recovery_fraction: 1.0,
            recovery_cap_seconds: None,
            original_x: 1000.0,
            control_x: x,
            candidate_x: x,
            original_raw_total: 1002.0,
            control_raw_total: 2.0 + x,
            candidate_raw_total: 2.0 + x,
            original_predictions: predictions.clone(),
            control_predictions: predictions.clone(),
            predictions,
            available: true,
        };
        (basis, pending)
    }
    #[test]
    fn raw_fallback_changes_only_selected_forecast_and_preserves_learning() {
        let mut original = State::default();
        let mut selected = State::default();
        original.poll(0.0, Some(7), 0, true, 30.0);
        selected.poll(0.0, Some(7), 0, true, 30.0);
        let (basis, pending) = admissions(0.0, 0, 100.0, 10.0);
        for id in 1..=16 {
            let a = original.admission(&basis, Some(&pending), None);
            let mut b = selected.admission(&basis, Some(&pending), None);
            b.select_raw_fallback(RawFallback::new(2.0, "provisional", 42.0));
            near(b.projection.total_ttft_seconds.unwrap(), 44.0);
            near(b.projection.stable_work_fallback_seconds.unwrap(), 102.0);
            let mut changed = serde_json::to_value(&b).unwrap();
            let unchanged = serde_json::to_value(&a).unwrap();
            changed["projection"]["total_ttft_seconds"] =
                unchanged["projection"]["total_ttft_seconds"].clone();
            changed["projection"]["selected_raw_fallback"] = serde_json::Value::Null;
            assert_eq!(changed, unchanged);
            original.register(Some(id), 0.0, Some(&a));
            selected.register(Some(id), 0.0, Some(&b));
            let label = decision::Label {
                id,
                at: 12.0,
                observed_at: 12.0,
                elapsed: 12.0,
                exact: true,
                first_label_at: 12.0,
                snapshot: a.training_snapshot,
            };
            assert_eq!(original.label(&label, true), selected.label(&label, true));
        }
        original.poll(13.0, Some(7), 0, true, 30.0);
        selected.poll(13.0, Some(7), 0, true, 30.0);
        assert_eq!(original.fit.diagnostics(), selected.fit.diagnostics());
        assert_eq!(
            original.pending_expiry_credit,
            selected.pending_expiry_credit
        );
        let (basis, pending) = admissions(14.0, 0, 100.0, 10.0);
        assert_eq!(
            serde_json::to_value(original.admission(&basis, Some(&pending), None)).unwrap(),
            serde_json::to_value(selected.admission(&basis, Some(&pending), None)).unwrap()
        );
    }

    #[test]
    fn raw_fallback_preserves_native_projection_and_unavailable_named_source() {
        let mut state = State::default();
        state.poll(0.0, Some(7), 0, true, 30.0);
        let (basis, pending) = admissions(0.0, 0, 100.0, 10.0);
        let initial = state.admission(&basis, Some(&pending), None);
        for case in 0..5 {
            let mut a = initial.clone();
            match case {
                0 => {
                    project_rows([(100.0, None)], 10.0, 2.0, 1.0, &mut a.projection).unwrap();
                }
                1 => a.predictions[1].available = false,
                2 => a.training_snapshot.fresh = false,
                3 => a.training_snapshot.native_ready = false,
                _ => a.projection.stable_work_fallback_seconds = Some(999.0),
            }
            let before = serde_json::to_value(&a).unwrap();
            a.select_raw_fallback(RawFallback::new(2.0, "measured", 42.0));
            assert_eq!(serde_json::to_value(a).unwrap(), before);
        }
    }

    #[test]
    fn raw_fallback_requires_finite_nonnegative_components_and_known_baseline() {
        for (baseline, source, extra) in [
            (f64::NAN, "measured", 1.0),
            (1.0, "measured", f64::INFINITY),
            (-1.0, "provisional", 42.0),
            (1.0, "measured", -1.0),
            (f64::MAX, "measured", f64::MAX),
            (1.0, "unknown", 42.0),
        ] {
            assert!(RawFallback::new(baseline, source, extra).is_none());
        }
        let zero = RawFallback::new(0.0, "measured", 0.0).unwrap();
        near(zero.total_ttft_seconds, 0.0);
    }

    #[test]
    fn own_x_reference_is_immutable_and_delayed_old_basis_labels_cannot_contaminate() {
        let mut state = State::default();
        state.poll(0.0, Some(7), 0, true, 5.0);
        let (basis, pending) = admissions(0.0, 0, 100.0, 10.0);
        let a = state.admission(&basis, Some(&pending), None);
        near(a.training_snapshot.x, 100.0);
        near(basis.training_snapshot.x, 1000.0);
        near(a.normalization.unwrap().capacity_reference, 10.0);
        state.register(Some(1), 0.0, Some(&a));
        let (basis, pending) = admissions(1.0, 0, 50.0, 20.0);
        let next = state.admission(&basis, Some(&pending), None);
        near(next.training_snapshot.x, 100.0);
        near(next.normalization.unwrap().capacity_reference, 10.0);
        let label = decision::Label {
            id: 1,
            at: 22.0,
            observed_at: 22.0,
            elapsed: 22.0,
            exact: true,
            first_label_at: 22.0,
            snapshot: a.training_snapshot,
        };
        state.label(&label, true);
        assert_eq!(state.fit.diagnostics()["rows"], 1);
        state.poll(23.0, Some(7), 1, true, 5.0);
        state.label(&label, true);
        assert_eq!(state.fit.diagnostics()["rows"], 0);
        assert!(state.reference.is_none());
        let (basis, pending) = admissions(24.0, 1, 100.0, 20.0);
        near(
            state
                .admission(&basis, Some(&pending), None)
                .normalization
                .unwrap()
                .capacity_reference,
            20.0,
        );
    }
    #[test]
    fn missing_basis_keeps_paired_fallback_and_never_invents_a_reference() {
        let mut state = State::default();
        state.poll(0.0, Some(7), 0, true, 5.0);
        let (mut basis, pending) = admissions(0.0, 0, 100.0, 10.0);
        basis.prediction_snapshot.size_basis = None;
        let a = state.admission(&basis, Some(&pending), Some(321.0));
        assert!(a.normalization.is_none());
        assert!(!a.training_snapshot.eligible);
        near(
            a.projection.total_ttft_seconds.unwrap(),
            basis.predictions[1].total,
        );
        assert!(state.reference.is_none());
        assert!(!a.projection.supported);
    }
    #[test]
    fn recovery_labels_can_inflate_next_load_capacity_until_reset_risk_is_explicit() {
        let mut state = State::default();
        state.poll(0.0, Some(7), 0, true, 500.0);
        for id in 0..16 {
            let (basis, pending) = admissions(0.0, 0, 1000.0, 10.0);
            let a = state.admission(&basis, Some(&pending), None);
            state.register(Some(id), 0.0, Some(&a));
            state.label(
                &decision::Label {
                    id,
                    at: 202.0,
                    observed_at: 202.0,
                    elapsed: 202.0,
                    exact: true,
                    first_label_at: 202.0,
                    snapshot: a.training_snapshot,
                },
                true,
            );
        }
        state.poll(202.0, Some(7), 0, true, 500.0);
        let (basis, pending) = admissions(205.0, 0, 1000.0, 10.0);
        let rebound = state.admission(&basis, Some(&pending), None);
        let k = (rebound.predictions[1].total - 2.0) / 1000.0;
        near(k, 0.2);
        let mut fast = Projection::default();
        project_rows([(10000.0, None)], 10.0 / k, 2.0, 1.0, &mut fast).unwrap();
        near(fast.total_ttft_seconds.unwrap(), 202.0);
        let mut actual_rate = Projection::default();
        project_rows([(10000.0, None)], 10.0, 2.0, 1.0, &mut actual_rate).unwrap();
        near(actual_rate.total_ttft_seconds.unwrap(), 1002.0);
        // This documents an unvalidated workload rebound; no hidden C2 guard
        // is added to make the frozen development result appear more robust.
        state.poll(206.0, Some(7), 1, true, 500.0);
        let (basis, pending) = admissions(207.0, 1, 1000.0, 10.0);
        near(
            state.admission(&basis, Some(&pending), None).predictions[1].total,
            1002.0,
        );
    }
    #[test]
    fn bounded_native_history_overflow_keeps_projection_unavailable() {
        let mut native = Native::default();
        for i in 0..LIMIT {
            let key = format!("attempt-{i}");
            native.register(Some(&key), 0.0, pending_work::Estimate::default());
            native.deadlines.insert(
                key.clone(),
                Deadline {
                    native: Some(format!("native-{i}")),
                    deadline: 100.0,
                    observed: 1.0,
                    sequence: 1,
                    first_text: false,
                },
            );
            native.acks.insert(
                key.clone(),
                NativeCancellationAck {
                    attempt_id: key,
                    native_request_id: Some(format!("native-{i}")),
                    event_seq: 2,
                    ack_monotonic_secs: 100.4,
                    reason: "hard_ttft_deadline".into(),
                },
            );
        }
        let mut max_frame = load(1);
        lane(&mut max_frame).active_requests = (0..LIMIT)
            .map(|i| row(&format!("active-{i}"), 1, None, 150.0))
            .collect();
        lane(&mut max_frame).native_population = LIMIT;
        native.snapshot = Some(Snapshot {
            lane: lane(&mut max_frame).clone(),
            received: 1.0,
            available: 1.0,
            started: 0.9,
            raw_poll: 0.9,
            expiry: 5.9,
            last_age: 0.0,
        });
        native.register(Some("overflow"), 0.0, pending_work::Estimate::default());
        assert!(native.incomplete);
        assert_eq!(
            project(&native, 2.0, 0.9).unwrap_err(),
            "native_history_incomplete_or_overflow"
        );
    }
    #[test]
    fn pre_send_rollback_removes_proxy_once_and_timeout_has_no_inventory_credit() {
        let mut n = Native::default();
        let idle = load(1);
        observe(&mut n, &idle, 1.0, 0.9);
        n.register(
            Some("a"),
            1.0,
            pending_work::Estimate::from_hint(PromptWorkHint::BodyTokenIds(100)),
        );
        let p = project(&n, 1.1, 0.9).unwrap();
        assert_eq!(p.unseen_admitted_rows, 1);
        near(p.input_work_tokens, 100.0);
        n.local_pre_send_rollback("a");
        n.local_pre_send_rollback("a");
        assert_eq!(project(&n, 1.2, 0.9).unwrap().predecessors, 0);
        n.observe(None, false, 1.4, 1.5, 0.9, 5.0);
        assert_eq!(
            project(&n, 1.6, 0.9).unwrap_err(),
            "native_inventory_missing"
        );
        assert!(n.lags.is_empty());
        assert!(n.acks.is_empty());
    }
    #[test]
    fn stalled_request_start_conversion_has_a_distinct_fallback_reason() {
        let mut n = Native::default();
        let mut input = load(1);
        input.loads[0].timely_prefill_progress_source = Some("prefill_started_requests".into());
        let p = input.loads[0].prefill_metric_provenance.as_mut().unwrap();
        p.progress_source = Some("prefill_started_requests".into());
        p.previous_conversion_size_mean = Some(1000.0);
        p.raw_rate = Some(0.0);
        observe(&mut n, &input, 1.0, 0.9);
        assert_eq!(
            project(&n, 1.1, 0.9).unwrap_err(),
            "request_start_capacity_conversion_missing_or_nonpositive"
        );
    }
    #[test]
    fn own_binding_expiry_guard_retains_censors_and_prevents_recovery_capacity_on_rebound() {
        let mut state = State::default();
        state.poll(0.0, Some(7), 0, true, 500.0);
        for id in 0..16 {
            let (basis, pending) = admissions(0.0, 0, 1000.0, 10.0);
            let mut a = state.admission(&basis, Some(&pending), None);
            // Its own admission projection takes a binding deadline branch.
            project_rows([(10000.0, Some(200.0))], 10.0, 2.0, 1.0, &mut a.projection).unwrap();
            assert!(a.projection.structural_binding_expiry_at_admission);
            state.register(Some(id), 0.0, Some(&a));
            // A later projection/object mutation cannot alter the saved flag.
            a.projection.structural_binding_expiry_at_admission = false;
            let mut label = decision::Label {
                id,
                at: 100.0,
                observed_at: 100.0,
                elapsed: 100.0,
                exact: false,
                first_label_at: 100.0,
                snapshot: a.training_snapshot,
            };
            assert_eq!(state.label(&label, false), "retained_censor");
            label.at = 202.0;
            label.observed_at = 202.0;
            label.elapsed = 202.0;
            label.exact = true;
            assert_eq!(
                state.label(&label, true),
                "excluded_exact_own_admission_binding_expiry"
            );
        }
        assert_eq!(state.fit.diagnostics()["rows"], 16);
        state.poll(202.0, Some(7), 0, true, 500.0);
        assert_eq!(
            state.fit.diagnostics()["models"][1]["fit"]["exact_count"],
            0
        );
        assert_eq!(
            state.fit.diagnostics()["models"][1]["fit"]["bound_count"],
            16
        );
        let (basis, pending) = admissions(205.0, 0, 1000.0, 10.0);
        let next = state.admission(&basis, Some(&pending), None);
        let k = (next.predictions[1].total - 2.0) / 1000.0;
        near(k, 1.0);
        let mut rebound = Projection::default();
        project_rows([(10000.0, None)], 10.0 / k, 2.0, 1.0, &mut rebound).unwrap();
        near(rebound.total_ttft_seconds.unwrap(), 1002.0);
        assert!(state.pending_expiry_credit.is_empty());
    }
    #[test]
    fn subtraction_residual_cannot_set_structural_expiry_training_flag() {
        let mut projection = Projection::default();
        project_rows([(3.3, Some(100.0))], 0.1, 2.0, 1.0, &mut projection).unwrap();
        assert!(projection.statistically_forecast_removed_tokens > 0.0);
        assert!(!projection.structural_binding_expiry_at_admission);
        assert_eq!(projection.structural_expiry_rows, 0);
    }
}
